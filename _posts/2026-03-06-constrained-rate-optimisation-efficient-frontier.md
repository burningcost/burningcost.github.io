---
layout: post
title: "Constrained Rate Optimisation and the Efficient Frontier"
date: 2026-03-06
categories: [techniques]
tags: [rate-optimisation, efficient-frontier, pricing, python, FCA]
---

Every UK personal lines pricing team we have spoken to runs their rate change process the same way. Someone builds a scenario in Excel. They apply a set of factor adjustments, sum the expected loss ratios across the portfolio, check it against the LR target, and iterate. If the numbers look acceptable, the scenario goes to the underwriting director.

It works. The problems are structural.

The Excel scenario is a single point. The team has picked one combination of factor adjustments and checked whether it satisfies the constraints. They have no idea whether a different combination of adjustments could have hit the same LR with less volume loss, or the same volume with a lower LR. The efficient frontier - the full set of achievable (LR, volume) outcomes - is never computed.

The shadow prices on constraints are never known either. How much volume would you lose if you tightened the LR target by one percentage point? Which constraint is actually binding? What is the regulatory cost of FCA PS21/5 ENBP compliance, in dislocation terms? These are answerable questions. Nobody is answering them.

We built [`rate-optimiser`](https://github.com/burningcost/rate-optimiser) to answer them formally.

---

## The Markowitz analogy

The setup is directly analogous to Markowitz portfolio optimisation.

In portfolio construction, you have a universe of assets. Each has an expected return and a variance. You want to find the portfolio weights - how much to allocate to each asset - that minimise variance for a given expected return target. Solving for many return targets traces the efficient frontier: the Pareto-optimal set of (return, risk) portfolios. The frontier tells you, quantitatively, what trade-off you are making at every point.

In rate optimisation, you have a portfolio of policies. Each has a technical premium, a current premium, and a renewal probability that responds to price changes. The decision variables are multiplicative adjustments to rating factor relativities: you are deciding how much to shift each factor. The objective is minimum dislocation - keep the factor adjustments as close to 1.0 as possible, while meeting the LR and volume constraints.

Solving for many LR targets traces the efficient frontier of achievable (LR, volume) pairs. At each point on the frontier, you get shadow prices: the Lagrange multipliers that tell you the marginal cost of tightening each constraint.

The formal problem is:

```
minimise   Σ_k (m_k - 1)²
subject to E[LR(m)] ≤ LR_target
           E[vol_ratio(m)] ≥ vol_bound
           m_k ∈ [m_k_min, m_k_max]  for all k
           π_i^renewal ≤ π_i^NB_equiv  (ENBP, per channel)
```

The decision variables `m_k` are multiplicative adjustments to each rating factor. A value of 1.05 means factor k's relativities are uniformly scaled up by 5% across all levels - a parallel shift on the log scale. The demand model enters through the volume and LR constraints: `p_i(π_i / π_market_i)` is the renewal probability at the adjusted premium. This makes both constraints nonlinear in `m`, which is why the problem requires SLSQP rather than a linear solver.

---

## A worked example

```python
import polars as pl
from rate_optimiser import (
    PolicyData, FactorStructure, DemandModel,
    RateChangeOptimiser, EfficientFrontier,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

# Load GLM outputs: policy_id, channel, renewal_flag,
# technical_premium, current_premium
df = pl.read_parquet("policies.parquet")

# rate_optimiser works with pandas at its boundary
data = PolicyData(df.to_pandas())

# Describe the multiplicative tariff structure
factor_names = ["f_age_band", "f_ncb", "f_vehicle_group", "f_region", "f_tenure_discount"]
fs = FactorStructure(
    factor_names=factor_names,
    factor_values=df.select(factor_names).to_pandas(),
    renewal_factor_names=["f_tenure_discount"],  # renewal-only; excluded from NB equivalent
)

# Wrap a logistic demand model
params = LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.05)
demand = make_logistic_demand(params)

# Specify constraints
opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)
opt.add_constraint(LossRatioConstraint(bound=0.72))
opt.add_constraint(VolumeConstraint(bound=0.97))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

# Check feasibility before solving
print(opt.feasibility_report())

# Solve
result = opt.solve()
print(result.summary())
```

`result.factor_adjustments` gives you the optimal multiplier for each factor - `{"f_age_band": 1.04, "f_ncb": 1.02, ...}`. `result.shadow_prices` is the number worth reading first.

---

## Shadow prices: the number to put in front of the commercial director

The shadow price on the LR constraint is the Lagrange multiplier: the marginal dislocation cost of tightening the LR target by one unit. When the LR constraint is slack, the shadow price is zero - you are not at the frontier, and tightening the target costs nothing yet. As you approach the knee of the frontier, the shadow price rises steeply.

Tracing the frontier makes this concrete:

```python
frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=20)
print(frontier.shadow_price_summary())
```

```
 lr_target  expected_lr  expected_volume  shadow_lr  shadow_volume
      0.78        0.777            0.973       0.02           0.00
      0.76        0.758            0.971       0.04           0.00
      0.74        0.739            0.968       0.08           0.00
      0.72        0.720            0.963       0.15           0.00
      0.70        0.700            0.954       0.31           0.01
      0.68        0.680            0.937       0.72           0.08
```

At a 72% LR target, the shadow price is 0.15 - modest. At 70%, it has doubled to 0.31. At 68%, it has jumped to 0.72. The knee is between 70% and 68%. The pricing team can show this table to a commercial director and say: "We can push to 70%, but below that the volume cost per LR point gained accelerates sharply." That is a quantified, defensible position. It is not available from a scenario in Excel.

The volume shadow price tells a similar story. At a 97% volume retention bound it is zero throughout the upper LR range - the volume constraint is not binding. It only starts binding below 70% LR, where the required rate increases are large enough to trigger meaningful lapse.

---

## The FCA ENBP constraint

PS21/5 prohibits renewal premiums above the new business equivalent through the same channel. For most UK motor and home portfolios, this is a genuinely binding constraint: renewal customers typically received loyalty discounts that are now prohibited. When you take rate on renewals, you have to be careful which factors you move, because some factor changes apply only to renewals (tenure discounts, NCB-at-renewal adjustments) and are excluded from the NB equivalent calculation.

`ENBPConstraint` enforces this formally and channel-specifically:

```python
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
```

Declare which factors are renewal-only in the `FactorStructure` and the library handles the NB-equivalent calculation automatically. The shadow price on the ENBP constraint tells you the dislocation cost of regulatory compliance: how much harder the optimiser has to work to find a feasible solution because of the ENBP restriction. This number belongs in your PS21/5 impact analysis.

No other open-source tool we are aware of implements this constraint at all. Commercial tools such as Radar Optimiser and Earnix have it in some form, but with opaque solver implementations and no programmatic access to shadow prices.

---

## The stochastic formulation

The deterministic problem uses E[LR] ≤ target. The stochastic formulation, following Branda (2013), is stricter: P(LR ≤ target) ≥ α. The LR must stay below the target with confidence level α, not just in expectation.

Under a normal approximation - appropriate for large books where the CLT applies - this reformulates to:

```
E[LR] + z_α × σ[LR] ≤ target
```

where σ[LR] comes from the variance estimates in your GLM. For a Tweedie claims model you pass the dispersion parameter and power; the library derives the policy-level variance and aggregates.

```python
from rate_optimiser.stochastic import ClaimsVarianceModel, StochasticRateOptimiser

variance_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=data.df["technical_premium"].values,
    dispersion=1.2,
    power=1.5,
)

opt = StochasticRateOptimiser(
    data=data, demand=demand, factor_structure=fs,
    variance_model=variance_model,
    lr_bound=0.72,
    alpha=0.95,
)
result = opt.solve()
```

The stochastic solver will always recommend higher rate than the deterministic one. The difference is the "uncertainty premium" - the additional rate required to maintain the LR target with 95% confidence rather than just in expectation. If your book has high variance (large Tweedie dispersion, concentrated exposure), this can be substantial. If you have a large, diversified portfolio, it will be small. That is not a surprise result, but it is a result the solver quantifies rather than leaves to intuition.

---

## What this is not

The library consumes GLM outputs; it does not fit them. Use statsmodels, CatBoost, or Emblem for the models, then feed the technical premiums and factor values here. It is also an offline rate strategy tool, not a real-time quote engine - Radar Live and Earnix handle individual-level pricing at point of quote.

---

## Getting started

```bash
uv add rate-optimiser

# With stochastic module (requires cvxpy):
uv add "rate-optimiser[stochastic]"
```

Source and issue tracker on [GitHub](https://github.com/burningcost/rate-optimiser). The priority backlog includes a competitive equilibrium module (Lerner index pricing as baseline), Bayesian demand model integration to propagate posterior uncertainty over price elasticity through the optimiser, and a Consumer Duty fair value checker.

The `feasibility_report()` method is the first thing to run before any solve. If your constraints are infeasible at current rates, the solver will tell you - and that is itself useful information about the portfolio.
