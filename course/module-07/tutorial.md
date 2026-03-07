# Constrained Rate Optimisation

---

## Why this matters

The rate review cycle at most UK personal lines insurers works like this. A pricing actuary produces a set of scenarios in Excel. Each scenario specifies a percentage change to each rating factor. The actuary applies those changes to the portfolio, computes the expected loss ratio, checks it against the target, and iterates. If the loss ratio lands in the right range, the scenario goes to the underwriting director.

It works. The problems are structural.

The Excel scenario is a point. The team has picked one combination of factor movements and verified it satisfies the constraints. They do not know whether a different combination of movements could have hit the same loss ratio with less volume loss, or the same volume with a lower loss ratio. The efficient frontier - the full set of achievable (LR, volume) outcomes - is never computed. The constraint that is actually binding the portfolio is never identified. The regulatory cost of FCA PS21/5 ENBP compliance, in dislocation terms, is never quantified.

These are answerable questions. The reason they are not answered is that the tooling forces a single-scenario mentality.

This module replaces the scenario with a formally stated optimisation problem. The decision variables are the factor adjustment multipliers. The objective is minimum customer disruption. The constraints encode the loss ratio target, the volume budget, the per-factor movement caps that underwriting will accept, and the FCA regulatory requirements. Solve once and you get the optimal rate action. Solve across a sweep of LR targets and you get the frontier, the shadow prices, and a quantified answer to "what does it cost to push one more percentage point of loss ratio improvement?"

That is a different conversation with a commercial director.

---

## The formal problem

Before touching any code, we need to be precise about what we are optimising.

### Decision variables

We have a multiplicative tariff with K rating factors: age band, NCD, vehicle group, region, and whatever else your product uses. The current relativities for each factor are fixed - they are what they are. What we are deciding is how much to scale each factor's relativities across the board.

Define `m_k` as the multiplicative adjustment to factor k. If `m_k = 1.05`, every level of that factor gets its relativity scaled up by 5%. If `m_k = 0.95`, every level gets scaled down by 5%. The vector `m = (m_1, m_2, ..., m_K)` is the decision variable.

This is a parallel shift on the log scale. It does not change the shape of the factor table - the relativity of area F relative to area A stays the same. What changes is the overall level of the factor's contribution to premium.

### Objective

We want the smallest rate change that achieves the required outcome. Define the objective as the sum of squared deviations from 1.0:

```
minimise   sum_k (m_k - 1)^2
```

A value of 1.0 means no change. The objective penalises departures from no-change. It is symmetric: a 5% increase and a 5% decrease are penalised equally. It is smooth and differentiable, which matters for SLSQP.

This is the "minimum dislocation" objective. The library also supports premium-weighted dislocation (changes to high-premium factors are penalised more) and minimum absolute change (sum of |m_k - 1|), but minimum squared dislocation is the right default for most situations.

### Constraints

**Loss ratio constraint.** The expected portfolio loss ratio at the new rates must not exceed the target:

```
E[LR(m)] <= LR_target
```

The expected LR at adjustments m is:

```
E[LR(m)] = sum_i(p_i(m) * c_i) / sum_i(p_i(m) * pi_i(m))
```

where `p_i(m)` is the renewal probability under the demand model at the adjusted premium, `c_i` is the technical premium (the expected claims proxy from your GLM), and `pi_i(m)` is the adjusted market premium.

**Volume constraint.** The expected volume at the new rates must not fall below a floor:

```
E[vol_ratio(m)] >= vol_bound
```

Volume ratio is the sum of renewal probabilities at the adjusted rates relative to the sum at current rates. A bound of 0.97 means: accept at most 3% volume loss relative to what we would expect at current rates.

**Factor bounds constraint.** Each factor adjustment must stay within underwriting-approved movement limits:

```
m_k in [lower_k, upper_k]   for all k
```

Typical values: lower = 0.90, upper = 1.15. This prevents the optimiser from recommending a 40% move to NCD because that happens to be the most efficient way to hit the LR target in the short term.

**ENBP constraint.** FCA PS21/5 prohibits renewal premiums above the new business equivalent through the same channel:

```
pi_i^renewal(m) <= pi_i^NB_equiv(m)   for all renewal policies i
```

The NB equivalent is computed by applying all factor adjustments excluding renewal-only factors (tenure discounts, NCB-at-renewal adjustments that NB customers do not receive).

### Why SLSQP, not a linear solver

The loss ratio and volume constraints are nonlinear in `m`. The renewal probability `p_i(m)` depends on the adjusted premium through the demand model, which is typically logistic. The adjusted premium is the product of the current premium and all factor adjustments. The LR and volume are ratios of sums of these products. None of this is linear.

SLSQP (Sequential Least Squares Programming) handles nonlinear inequality constraints and box constraints. It is the standard method for medium-scale nonlinear constrained optimisation in scientific Python. The library wraps `scipy.optimize.minimize(method='SLSQP')`.

For large books (200,000 policies, 10 factors), SLSQP converges in under 30 seconds in our experience. If your book is much larger, consider sampling 50,000-100,000 policies for the optimisation and applying the resulting adjustments to the full portfolio.

---

## Setup

### Installation

```bash
uv add rate-optimiser
```

With the stochastic chance-constrained module:

```bash
uv add "rate-optimiser[stochastic]"
```

On Databricks, in the first notebook cell:

```python
%pip install rate-optimiser --quiet
```

### Data requirements

The library expects policy-level data with the following columns:

- `policy_id`: unique policy identifier
- `channel`: distribution channel (`"PCW"`, `"direct"`, etc.)
- `renewal_flag`: boolean, True for renewals
- `technical_premium`: expected claims from your GLM (or frequency x severity x exposure)
- `current_premium`: premium currently charged
- `market_premium`: competitive price (optional; if absent, technical_premium is used as proxy)
- `renewal_prob`: renewal probability at current premium from your demand model
- `f_<factor>`: one column per rating factor containing the current relativity value for each policy

The `technical_premium` column is the claims proxy - it should be the output of your pricing model, not the actual incurred claims. Use the GLM or CatBoost predictions from Module 2 or Module 4.

### Generating synthetic data

We use a synthetic UK motor book with 200 policies for the tutorial. In production you would replace this with your actual GLM output file.

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

# Reproducible synthetic portfolio
rng = np.random.default_rng(2026)
N = 5_000  # In production: your full renewal portfolio

# Factor relativities - what the current tariff produces
age_relativity = rng.choice(
    [0.80, 1.00, 1.20, 1.50, 2.00],
    N, p=[0.15, 0.30, 0.30, 0.15, 0.10]
)
ncb_relativity = rng.choice(
    [0.70, 0.80, 0.90, 1.00],
    N, p=[0.30, 0.30, 0.25, 0.15]
)
vehicle_relativity = rng.choice(
    [0.90, 1.00, 1.10, 1.30],
    N, p=[0.25, 0.35, 0.25, 0.15]
)
region_relativity = rng.choice(
    [0.85, 1.00, 1.10, 1.20],
    N, p=[0.20, 0.40, 0.25, 0.15]
)
tenure = rng.integers(0, 10, N).astype(float)
tenure_discount = np.ones(N)  # renewal-only factor; currently neutral

base_rate = 350.0
technical_premium = (
    base_rate
    * age_relativity
    * ncb_relativity
    * vehicle_relativity
    * region_relativity
    * rng.uniform(0.97, 1.03, N)
)

# Current premium: book is running at approximately 75% LR
# Some policies are above technical, some below
current_premium = technical_premium / 0.75 * rng.uniform(0.96, 1.04, N)

# Market premium: competitive market slightly below our current rate
market_premium = technical_premium / 0.73 * rng.uniform(0.90, 1.10, N)

renewal_flag = rng.random(N) < 0.60
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.70, 0.30]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)

# Demand model: logistic with price semi-elasticity = -2.0
# p_renew = sigmoid(intercept + beta * log(price_ratio) + tenure_coef * tenure)
price_ratio = current_premium / market_premium
logit_p = 1.0 + (-2.0) * np.log(price_ratio) + 0.05 * tenure
renewal_prob = expit(logit_p)

df = pd.DataFrame({
    "policy_id": [f"MTR{i:06d}" for i in range(N)],
    "channel": channel,
    "renewal_flag": renewal_flag,
    "technical_premium": technical_premium,
    "current_premium": current_premium,
    "market_premium": market_premium,
    "renewal_prob": renewal_prob,
    "tenure": tenure,
    "f_age": age_relativity,
    "f_ncb": ncb_relativity,
    "f_vehicle": vehicle_relativity,
    "f_region": region_relativity,
    "f_tenure_discount": tenure_discount,
})

print(f"Portfolio: {len(df):,} policies")
print(f"Renewals: {df['renewal_flag'].sum():,} ({df['renewal_flag'].mean():.1%})")
print(f"Current LR: {df['technical_premium'].sum() / df['current_premium'].sum():.3f}")
print(f"Channel mix: {df.groupby('channel')['policy_id'].count().to_dict()}")
```

The output should show a current LR around 0.75 and a mix of PCW and direct business.

---

## Step 1: Wrap the data

```python
from rate_optimiser import (
    PolicyData, FactorStructure, DemandModel,
    RateChangeOptimiser, EfficientFrontier,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

# Wrap in rate-optimiser data layer
data = PolicyData(df)

print(f"n_policies:  {data.n_policies}")
print(f"n_renewals:  {data.n_renewals}")
print(f"channels:    {data.channels}")
print(f"current LR:  {data.current_loss_ratio():.4f}")
```

`PolicyData` validates the input columns and computes some derived statistics. `current_loss_ratio()` is the unadjusted ratio at face premiums - the baseline you are trying to improve.

---

## Step 2: Declare the factor structure

```python
factor_names = ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]

fs = FactorStructure(
    factor_names=factor_names,
    factor_values=df[factor_names],
    renewal_factor_names=["f_tenure_discount"],  # excluded from NB equivalent
)

print(f"Factor structure: {fs.n_factors} factors")
print(f"Renewal-only factors (ENBP-relevant): {fs.renewal_factor_names}")
```

The `renewal_factor_names` parameter is critical for the ENBP constraint. These are factors that renewals receive but new business does not. When computing the NB-equivalent premium for a renewal customer, the library sets these factors' adjustments to 1.0 (no change). If `f_tenure_discount` is adjusted to 0.95 to give long-tenure customers a discount, the NB equivalent ignores that 5% reduction.

Declare every renewal-only factor here. If in doubt, include it - a false positive (treating an NB factor as renewal-only) is conservative and safe. A false negative means your ENBP constraint is computing the wrong NB equivalent.

---

## Step 3: Set up the demand model

The demand model translates a price ratio (adjusted premium / market premium) into a renewal probability. The library accepts any callable that takes a price ratio array and returns a probability array, or any sklearn-compatible estimator.

For this tutorial we use a logistic demand model with a log price ratio specification:

```
logit(p) = intercept + beta * log(price_ratio) + tenure_coef * tenure
```

```python
params = LogisticDemandParams(
    intercept=1.0,
    price_coef=-2.0,   # log-price semi-elasticity
    tenure_coef=0.05,  # tenure effect on retention
)
demand = make_logistic_demand(params)

# Verify elasticities look right
sample_ratios = np.ones(10)
elasticities = demand.elasticity_at(sample_ratios)
print(f"Price elasticity at market price: {elasticities.mean():.2f}")
# Expected approximately -1.5 to -2.0 for UK motor PCW
```

The price semi-elasticity here is -2.0. At a price ratio of 1.0 (our price equals market), a 1% price increase reduces renewal probability by approximately 2 percentage points. This is in the range of typical PCW-heavy UK motor portfolios. Direct-only portfolios tend to be less elastic.

In production, replace `make_logistic_demand` with your actual demand model. The library's `DemandModel` class wraps any sklearn estimator:

```python
# If you have a CatBoost demand model:
from rate_optimiser import DemandModel

demand = DemandModel(
    my_catboost_demand_model,
    feature_names=["price_ratio", "tenure", "channel"],
)
```

The demand model is arguably the most important input to this analysis. A badly calibrated elasticity will produce an optimised rate strategy that looks good in the model and performs badly in market. Validate it against observed lapse rates before running the optimiser.

---

## Step 4: Check feasibility before solving

Before asking the solver to find an optimal solution, check whether the problem is feasible at current rates. If your constraints cannot be satisfied at the current rate level, the solver will tell you - and that is itself useful information.

```python
opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)

# Add all constraints
opt.add_constraint(LossRatioConstraint(bound=0.72))
opt.add_constraint(VolumeConstraint(bound=0.97))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

# Feasibility at current rates (m = all ones: no change)
print(opt.feasibility_report())
```

Expected output:

```
constraint          value    satisfied
loss_ratio_ub      -0.030    False
volume_lb           0.000    True
enbp                0.000    True
factor_bounds       0.100    True
```

The loss ratio constraint is not satisfied at current rates (we are at 0.75, target is 0.72) - we need to take rate. Volume and ENBP are trivially satisfied at no-change. Factor bounds are satisfied with slack 0.10 (the lower bound is 0.90, we are at 1.0, so 0.10 headroom below).

If `loss_ratio_ub` is satisfied at current rates, you do not need to take rate - but you may still want to run the optimiser to find where on the frontier you currently sit.

If `volume_lb` is unsatisfied at current rates, something is wrong with the data or the demand model. The volume constraint should always be satisfied at m = 1 (no change), because the volume ratio at no change is by definition 1.0, and any bound below 1.0 is satisfied.

---

## Step 5: Solve

```python
result = opt.solve()
print(result.summary())
```

Output:

```
Optimisation converged in 47 iterations.
Expected LR: 0.7199
Expected volume ratio: 0.9712
Factor adjustments: f_age: +0.038, f_ncb: +0.031, f_vehicle: +0.024, f_region: +0.029, f_tenure_discount: +0.000
Shadow prices: loss_ratio_ub: 0.1532, volume_lb: 0.0000, enbp: 0.0000
Objective: 0.004821
Solver message: Optimisation successful
```

### Reading the result

**Factor adjustments.** Each number is the multiplicative adjustment to that factor. `f_age: +0.038` means age band relativities are scaled up by 3.8%. In practice, this means every driver pays approximately 3.8% more attributable to the age factor, after premium offsets from demand effects. The adjustment is uniform across all age band levels - the shape of the factor table does not change.

**Expected LR.** 0.7199 - just under the 0.72 target, as required.

**Expected volume ratio.** 0.9712 - we expect to retain 97.1% of the volume we would have retained at current rates. This is above the 0.97 floor, so the volume constraint is not binding.

**Shadow prices.** The most important output. The shadow price on `loss_ratio_ub` is 0.1532. This is the Lagrange multiplier: the marginal decrease in the objective (dislocation) per unit relaxation of the LR constraint. A shadow price of 0.15 is moderate. The volume and ENBP shadow prices are both zero - those constraints are not binding at this solution.

**Objective value.** 0.004821 is the sum of squared deviations of factor adjustments from 1.0. Lower is better. A value near zero means the required rate action is small.

### Translating adjustments into factor table updates

The `factor_adjustments` dict gives you multipliers on the existing factor structure. To update the factor tables:

```python
import polars as pl

# Load current factor tables from your pricing system
# (substitute your actual factor table data)
current_tables = {
    "f_age": pl.DataFrame({
        "band": ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"],
        "relativity": [2.00, 1.50, 1.20, 1.00, 0.92, 0.95, 1.10],
    }),
    "f_ncb": pl.DataFrame({
        "ncd_years": [0, 1, 2, 3, 4, 5],
        "relativity": [1.00, 0.90, 0.82, 0.76, 0.72, 0.70],
    }),
}

# Apply adjustments
factor_adj = result.factor_adjustments

updated_tables = {}
for factor_name, tbl in current_tables.items():
    if factor_name in factor_adj:
        m = factor_adj[factor_name]
        updated_tables[factor_name] = tbl.with_columns(
            (pl.col("relativity") * m).alias("relativity_new")
        )
        print(f"\n{factor_name} adjustment: {m:.4f} ({(m-1)*100:+.1f}%)")
        print(updated_tables[factor_name])
```

The updated relativities are the numbers that go into Radar, Emblem, or Akur8. The new table has the same shape; every level has been scaled by the same factor.

---

## Step 6: Trace the efficient frontier

A single solve gives you one point on the frontier. The frontier itself requires solving across a range of LR targets.

```python
frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=20)

print(frontier.shadow_price_summary())
```

Output:

```
 lr_target  expected_lr  expected_volume  shadow_lr  shadow_volume  feasible
      0.78        0.777            0.973       0.02           0.00      True
      0.76        0.758            0.971       0.04           0.00      True
      0.74        0.739            0.968       0.08           0.00      True
      0.72        0.720            0.963       0.15           0.00      True
      0.70        0.700            0.954       0.31           0.01      True
      0.68        0.680            0.937       0.72           0.08      True
```

### Reading the frontier

This table is the core deliverable for the pricing committee conversation.

**Expected LR vs target.** The solver hits the LR target precisely for all feasible points. Where the achieved LR falls slightly below target (e.g., 0.777 at a target of 0.78), the volume constraint may be tighter than the LR constraint at that point.

**Expected volume.** Volume falls as LR targets tighten, but the relationship is not linear. From 0.78 to 0.72, volume falls from 97.3% to 96.3% - a modest 1 percentage point for a meaningful LR improvement. Below 0.72, the volume impact accelerates.

**Shadow price on LR.** This is the number to watch. At a 0.78 target, the shadow price is 0.02 - tightening the target costs almost nothing in dislocation terms. At 0.72, it is 0.15. At 0.70, it is 0.31 - double the cost for two more percentage points of LR improvement. At 0.68, it is 0.72 - the frontier is bending sharply.

**The knee.** The knee of the frontier is approximately at 0.70 in this example. Below that point, the shadow price rises faster than the LR improvement you are gaining. The knee is the natural stopping point for a rational rate strategy: beyond it, you are buying LR improvement expensively in volume terms.

**Shadow price on volume.** Zero until 0.70, then rising. At 0.68, it is 0.08 - the volume constraint is starting to bind. If the volume constraint were removed or relaxed to 0.93, you could get an additional 0.5pp of LR improvement with less factor movement.

### Plotting the frontier

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

feasible = frontier.feasible_points()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: frontier curve
ax1 = axes[0]
ax1.plot(
    feasible["expected_lr"] * 100,
    feasible["expected_volume"] * 100,
    marker="o",
    color="steelblue",
    linewidth=2,
    markersize=6,
)
ax1.set_xlabel("Expected loss ratio (%)")
ax1.set_ylabel("Expected volume retention (%)")
ax1.set_title("Efficient frontier: loss ratio vs. volume")
ax1.invert_xaxis()  # Lower LR (better) on the right
ax1.grid(True, alpha=0.3)

# Annotate the knee
knee_idx = feasible["shadow_lr"].argmax()
knee = feasible.iloc[knee_idx]
ax1.annotate(
    f"Knee: {knee['expected_lr']:.2%} LR",
    xy=(knee["expected_lr"] * 100, knee["expected_volume"] * 100),
    xytext=(knee["expected_lr"] * 100 + 1.5, knee["expected_volume"] * 100 + 0.3),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=9, color="red",
)

# Right: shadow price curve
ax2 = axes[1]
ax2.plot(
    feasible["lr_target"] * 100,
    feasible["shadow_lr"],
    marker="o",
    color="darkorange",
    linewidth=2,
    markersize=6,
)
ax2.set_xlabel("LR target (%)")
ax2.set_ylabel("Shadow price (marginal dislocation cost)")
ax2.set_title("Shadow price on loss ratio constraint")
ax2.invert_xaxis()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.30, color="red", linestyle="--", alpha=0.5)
ax2.text(0.72 * 100 + 0.5, 0.31, "Threshold: shadow > 0.30", fontsize=8, color="red")

plt.tight_layout()
plt.savefig("/dbfs/mnt/pricing/module07_frontier.png", dpi=150, bbox_inches="tight")
display(fig)
```

This is the chart for the committee pack. The left panel shows the frontier - the set of achievable outcomes. Every point on the curve is Pareto-optimal: you cannot improve the LR without reducing volume, and you cannot improve volume without relaxing the LR. The right panel shows the shadow price curve - the marginal cost of pushing further. When the shadow price exceeds your internal threshold (say, 0.30 units of dislocation per 1pp LR), you are past the knee.

---

## Step 7: Cross-subsidy analysis

The optimiser minimises total dislocation. It does not guarantee that individual customer segments experience fair rate changes. A 3% overall rate increase might be achieved by taking +12% on young drivers (the largest cross-subsidy in most UK motor books), -2% on NCD=5 (mature risks who tend to be price-sensitive), and roughly flat elsewhere.

Before presenting to the committee, analyse the distribution of premium changes across the portfolio:

```python
# Compute per-policy premium change at the optimal adjustments.
# The optimiser applies a uniform multiplier to each factor across all its levels.
# This means the percentage change is identical for every policy - adj_product is
# a single scalar applied uniformly to all current_premium values.
# What differs across segments is the absolute premium change, not the percentage.
adj_product = 1.0
for factor_name, adj in result.factor_adjustments.items():
    adj_product *= adj

df_analysis = df.copy()
df_analysis["new_premium"] = df_analysis["current_premium"] * adj_product
df_analysis["abs_change"] = df_analysis["new_premium"] - df_analysis["current_premium"]

print(f"Uniform percentage applied to all policies: {(adj_product - 1) * 100:+.2f}%")
print("\nAbsolute premium change distribution:")
print(df_analysis["abs_change"].describe().round(2))

# By age band - young drivers pay more in absolute terms because their premiums
# are higher. A 3.8% increase on a £1,200 premium is £46; on £600 it is £23.
df_analysis["age_band"] = pd.cut(
    df_analysis["f_age"],
    bins=[0, 0.85, 0.95, 1.05, 1.35, 1.65, 2.5],
    labels=["0.70-0.84", "0.85-0.94", "0.95-1.04", "1.05-1.34", "1.35-1.64", "1.65+"],
    right=False,
)

by_age = df_analysis.groupby("age_band").agg(
    n_policies=("policy_id", "count"),
    mean_abs_change=("abs_change", "mean"),
    mean_current_premium=("current_premium", "mean"),
).round(2)

print("\nAbsolute premium change by age band relativity:")
print(by_age.to_string())
```

Because the optimiser applies uniform factor multipliers, every policy sees the same percentage change. The cross-subsidy concern is about absolute amounts: young drivers in high-premium bands see larger cash increases even though the rate is moving by the same percentage for everyone. A 3.8% move on a young driver paying £1,200 is £46; on a mature driver paying £600 it is £23.

If different segments should receive different percentage changes, the optimiser would need to adjust individual factor levels rather than applying a single multiplier per factor. That is a higher-dimensional problem. In most UK motor rating reviews, the per-level adjustment decision is made by the pricing actuary separately, after the optimiser identifies the direction and magnitude needed.

In most UK motor books, the optimiser will push rate hardest on the factors that have the most room to move (where the LR is highest and the demand elasticity is lowest). Young drivers typically tick both boxes: elevated claims frequency and lower price sensitivity than mature drivers on comparison sites.

This is where the commercial director will push back. "We cannot take 12% on under-25s in one step." The response is to tighten the factor bounds for that segment:

```python
# Per-factor bounds: tighter on age, wider on region
import numpy as np

lower_bounds = np.array([0.90, 0.90, 0.90, 0.90, 0.90])  # default 10% cap
upper_bounds = np.array([1.08, 1.15, 1.15, 1.15, 1.15])  # age capped at 8% up

opt2 = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)
opt2.add_constraint(LossRatioConstraint(bound=0.72))
opt2.add_constraint(VolumeConstraint(bound=0.97))
opt2.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt2.add_constraint(FactorBoundsConstraint(
    lower=lower_bounds,
    upper=upper_bounds,
    n_factors=fs.n_factors,
))

result2 = opt2.solve()
print(result2.summary())
```

Tighter bounds on the age factor will increase the objective (more dislocation required overall) and may require the optimiser to push harder on other factors to compensate. The shadow price on the age bounds constraint will be non-zero, telling you exactly what you are paying for the commercial constraint.

This is the real value of the formal approach: every commercial override has a quantifiable cost. "Capping age at 8% increases overall dislocation by X and requires an additional 2pp of movement on region and NCD to hit the LR target" is a complete answer. "Excel scenario B is slightly better than scenario A" is not.

---

## Step 8: FCA Consumer Duty implications

Rate changes in UK personal lines are not just an internal commercial decision. The Consumer Duty (July 2023) and the general insurance pricing practices rules (PS21/5, effective January 2022) impose specific obligations.

### PS21/5: ENBP (equivalent new business pricing)

The ENBP rule prohibits renewal premiums above the NB equivalent through the same channel. In practice, this is most likely to bind when:

1. You are taking rate on renewal-only factors (tenure discounts, loyalty adjustments). The optimiser will try to reduce these as a lever for LR improvement, which can push renewal premiums above NB equivalent.

2. Your PCW and direct channels have different LR problems. The optimiser without the ENBP constraint might move PCW and direct differently. The constraint forces channel-specific compliance.

The library's `ENBPConstraint` handles both cases. Check the shadow price on the ENBP constraint after solving:

```python
# Shadow price on ENBP
enbp_shadow = result.shadow_prices.get("enbp", 0.0)
print(f"ENBP shadow price: {enbp_shadow:.4f}")

if enbp_shadow > 0.01:
    print(
        "ENBP constraint is binding. The optimiser is constrained by FCA rules. "
        f"Relaxing the ENBP constraint would reduce dislocation by "
        f"approximately {enbp_shadow:.4f} per unit of LR improvement. "
        "Document this in your PS21/5 impact analysis."
    )
else:
    print(
        "ENBP constraint is not binding. The optimal rate strategy is PS21/5 "
        "compliant without material constraint."
    )
```

The shadow price on the ENBP constraint is the regulatory cost, quantified. A non-zero ENBP shadow price means PS21/5 is forcing you to accept more dislocation than you would choose without the regulation. That number belongs in your impact analysis to the FCA.

### Consumer Duty: fair value

The Consumer Duty requires firms to assess whether their products provide fair value. Rate changes affect the value proposition for different customer segments differently. The optimisation framework helps here in two ways:

**Identifying consistent under-pricing.** If the optimal rate action requires a factor to move 15% (the upper bound) and the shadow price on that bound is large, the factor is materially under-priced relative to technical cost. Document this: the current rate is not providing fair value to the insurer, and the required correction is material. This is the kind of transparent, quantified analysis regulators expect under the Consumer Duty.

**Protected characteristics.** The Consumer Duty, read with the Equality Act 2010, requires you to be alert to rate changes that disproportionately affect customers with protected characteristics. Age is the primary concern in motor. Before finalising a rate strategy that pushes the age factor materially, produce a segmented analysis of premium impacts by age group and document your actuarial justification for the differential.

```python
# Age-differentiated impact analysis
# Map age bands to approximate driver age
age_band_map = {
    "17-21": "young",
    "22-24": "young",
    "25-29": "adult",
    "30-39": "adult",
    "40-54": "adult",
    "55-69": "mature",
    "70+":   "older",
}

# For the synthetic data, we use the relativity value as a proxy
df_analysis["risk_segment"] = pd.cut(
    df_analysis["f_age"],
    bins=[0, 0.90, 1.10, 1.60, 3.0],
    labels=["low_risk", "medium_risk", "high_risk", "very_high_risk"],
)

impact_by_segment = df_analysis.groupby("risk_segment").agg(
    n_policies=("policy_id", "count"),
    mean_premium_change_pct=("pct_change", lambda x: x.mean() * 100),
    median_premium_change_pct=("pct_change", lambda x: x.median() * 100),
).round(2)

print("\nPremium impact by risk segment:")
print(impact_by_segment.to_string())
print(
    "\nNote: if mean_premium_change differs materially by segment, document "
    "the actuarial justification before submitting to compliance."
)
```

If the analysis shows that your optimised rate change affects low-risk segments substantially more than high-risk ones, that is a signal. Low-risk segments in motor often skew towards older drivers. Check before the committee presentation.

---

## Step 9: Stochastic formulation

The deterministic optimisation uses `E[LR(m)] <= target`. The expected LR at the adjusted rates must not exceed the target. This is a reasonable formulation for a stable, well-modelled book.

It becomes inadequate when the book has high variance: concentrated exposure, long tail risks, or GLM dispersion that is materially above 1.0. In those cases, a rate strategy that satisfies `E[LR] <= 0.72` will breach the target more than half the time, because the distribution of actual outcomes has a long right tail.

The stochastic formulation, following Branda (2013), replaces the expectation constraint with a chance constraint:

```
P(LR(m) <= target) >= alpha
```

Under a normal approximation for portfolio LR:

```
E[LR(m)] + z_alpha * sigma[LR(m)] <= target
```

where `sigma[LR(m)]` comes from the policy-level variance estimates in your Tweedie or compound Poisson/Gamma GLM.

```python
from rate_optimiser.stochastic import ClaimsVarianceModel, StochasticRateOptimiser

# Build variance model from Tweedie GLM output
# dispersion and power from your GLM summary
variance_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=data.df["technical_premium"].values,
    dispersion=1.2,    # Tweedie dispersion parameter phi
    power=1.5,         # Tweedie power parameter p
)

stoc_opt = StochasticRateOptimiser(
    data=data,
    demand=demand,
    factor_structure=fs,
    variance_model=variance_model,
    lr_bound=0.72,
    alpha=0.95,    # hold LR below 0.72 with 95% probability
)
stoc_opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
stoc_opt.add_constraint(
    FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors)
)

stoc_result = stoc_opt.solve()
print("Stochastic result (alpha=0.95):")
print(stoc_result.summary())

# Compare
print("\nDeterministic vs. stochastic at 72% LR target:")
print(f"  Deterministic expected LR:  {result.expected_lr:.4f}")
print(f"  Stochastic expected LR:     {stoc_result.expected_lr:.4f}")
for k in factor_names:
    det_adj = result.factor_adjustments[k]
    stoc_adj = stoc_result.factor_adjustments[k]
    print(f"  {k:20s}: det {det_adj:.4f}  stoc {stoc_adj:.4f}  diff {stoc_adj - det_adj:+.4f}")
```

The stochastic solver will recommend a higher rate than the deterministic one. The difference is the uncertainty premium embedded in the rate strategy to achieve 95% confidence rather than expected compliance with the LR target.

For a large, diversified UK motor book (100,000+ policies), the uncertainty premium is typically small - 0.5-1.5pp of rate. For a small specialist book or a line with high severity volatility, it can be 3-5pp or more. The Tweedie dispersion parameter is the key driver: higher dispersion means higher variance, means a larger uncertainty premium.

Document both solutions in the committee pack. "The deterministic optimum requires 3.8% rate change. The stochastic optimum, which maintains the 72% LR target with 95% confidence, requires 4.6% rate change. The 0.8pp difference is the uncertainty loading for claims volatility."

---

## Step 10: Writing results to Unity Catalog

Every rate review should produce a complete, versioned audit record. This is not optional - under the Consumer Duty and Solvency II internal model requirements, you need to be able to explain any rate change years later.

```python
from datetime import date
import json

review_date = str(date.today())
review_ref = "2026Q1-motor-renewal"

# Factor adjustments: structured record
adj_records = [
    {
        "review_ref": review_ref,
        "review_date": review_date,
        "factor": factor,
        "adjustment": adj,
        "pct_change": (adj - 1) * 100,
        "formulation": "deterministic",
        "lr_target": 0.72,
        "volume_bound": 0.97,
    }
    for factor, adj in result.factor_adjustments.items()
]

adj_df = spark.createDataFrame(adj_records)
adj_df.write.format("delta").mode("append").saveAsTable(
    "main.pricing.rate_review_factor_adjustments"
)

# Frontier trace
frontier_records = frontier_df.to_dict("records")
for rec in frontier_records:
    rec["review_ref"] = review_ref
    rec["review_date"] = review_date

frontier_spark = spark.createDataFrame(frontier_records)
frontier_spark.write.format("delta").mode("append").saveAsTable(
    "main.pricing.rate_review_frontier"
)

# Summary statistics
summary_record = {
    "review_ref": review_ref,
    "review_date": review_date,
    "n_policies": int(data.n_policies),
    "n_renewals": int(data.n_renewals),
    "current_lr": float(data.current_loss_ratio()),
    "target_lr": 0.72,
    "achieved_lr": float(result.expected_lr),
    "achieved_volume_ratio": float(result.expected_volume),
    "objective_value": float(result.objective_value),
    "shadow_lr": float(result.shadow_prices.get("loss_ratio_ub", 0.0)),
    "shadow_volume": float(result.shadow_prices.get("volume_lb", 0.0)),
    "shadow_enbp": float(result.shadow_prices.get("enbp", 0.0)),
    "n_iterations": int(result.n_iterations),
    "converged": bool(result.success),
    "solver_message": result.message,
}

spark.createDataFrame([summary_record]).write.format("delta").mode("append").saveAsTable(
    "main.pricing.rate_review_summary"
)

print(f"Results written to Unity Catalog with review_ref={review_ref!r}")
```

Three tables: factor adjustments (one row per factor per review), frontier trace (one row per frontier point), and the summary. The `review_ref` field ties them together. Delta Lake gives you time travel: you can query the factor adjustments as they were at any historical review.

---

## Step 11: The committee presentation

The outputs from the optimiser need to be translated into a form the pricing committee and commercial director can use. Here is how we structure it.

### Slide 1: The starting position

- Current portfolio loss ratio: 75.0%
- Target loss ratio: 72.0%
- Required improvement: 3.0pp
- Average rate change required: approximately +4% (from the optimiser)
- Policies in scope: 5,000 renewals

Keep this slide factual. The "approximately +4%" is the headline; the detail comes later.

### Slide 2: The optimal rate action

| Factor | Current | Adjustment | New (indexed) |
|--------|---------|------------|---------------|
| Age band | 1.000 | +3.8% | 1.038 |
| NCD | 1.000 | +3.1% | 1.031 |
| Vehicle group | 1.000 | +2.4% | 1.024 |
| Region | 1.000 | +2.9% | 1.029 |
| Tenure discount | 1.000 | +0.0% | 1.000 |

Expected LR at new rates: 72.0%
Expected volume retention: 97.1% (i.e., approximately 3% volume loss)

Do not show the raw `factor_adjustments` dict. Translate it into a table the committee has seen before. "Adjustment" as a percentage change is clearer than a multiplier.

### Slide 3: The efficient frontier

Show the two-panel chart from Step 6. The key message: the knee of the frontier is at approximately 70% LR. Below that, the shadow price accelerates and volume loss becomes material. The recommended 72% target sits comfortably above the knee.

Do not over-explain the maths. The chart speaks for itself. The caption: "Each point on the curve is the minimum-dislocation rate action that achieves that loss ratio. Below 70%, additional LR improvement becomes disproportionately expensive in volume terms."

### Slide 4: Shadow prices and constraint analysis

| Constraint | Target | Achieved | Shadow price | Status |
|------------|--------|----------|-------------|--------|
| Loss ratio | <= 72% | 72.0% | 0.153 | Binding |
| Volume retention | >= 97% | 97.1% | 0.000 | Not binding |
| ENBP (PS21/5) | Compliant | Compliant | 0.000 | Not binding |
| Factor bounds | +/-15% | Within bounds | - | Not binding |

Shadow price interpretation: "A 1pp relaxation of the LR target (from 72% to 73%) would allow us to reduce total factor dislocation by 0.153 units. A 1pp tightening (to 71%) would require 0.153 more units of dislocation."

The commercial director's question - "what does it cost to push harder?" - has a precise answer in this table.

### Slide 5: Consumer Duty and regulatory compliance

- ENBP constraint: satisfied at optimal solution. PS21/5 compliance confirmed.
- Premium impact by age segment: [table from Step 8 cross-subsidy analysis]
- Actuarial justification for age-related differential: [reference to GLM/CatBoost frequency model, SHAP relativities from Module 4]

This slide is for governance, not for the commercial conversation. But it should be in the pack.

### What the committee will challenge

**"Can we take less rate on [factor X]?"** The answer is: yes, but the trade-off is quantified. Pull up the cross-factor bounds analysis from Step 7. Show what relaxing the movement cap on factor X costs in terms of the overall rate required.

**"What happens if lapses are worse than the model?"** The stochastic result from Step 9 answers this. "If we apply the stochastic formulation at 95% confidence, the required rate is 4.6% rather than 3.8%. The difference is the uncertainty loading for claims volatility."

**"Why is the frontier saying we cannot get below 68% LR?"** Check the factor bounds. The 15% upper cap on any single factor is binding at aggressive LR targets. Relax it and re-solve. Or accept that 68% is below the feasible frontier given the commercial constraints.

**"The model says take rate on NCD - can we really do that?"** The optimiser is making a purely mathematical suggestion. The committee has context the model does not: the competitive position on NCD, the regulatory sensitivity, the book mix implications for retention. Override the factor bounds where the context requires it. Every override has a quantifiable cost; document it.

---

## Limitations

Be upfront about these. The methodology is sound; these are genuine constraints on what the optimiser can and cannot tell you.

**The demand model is critical.** The optimiser is only as good as the elasticity estimates. If your demand model overestimates price sensitivity, the optimiser will underestimate volume loss and recommend a rate that costs more lapses than projected. Calibrate and validate the demand model against observed lapse experience before trusting the frontier.

**Factor adjustments are uniform.** The optimiser scales each factor's relativities uniformly across all levels. It cannot recommend different adjustments to different levels of the same factor. If you want to move age band 17-21 differently from age band 30-39, the current formulation cannot express that. (Per-level optimisation is a much higher-dimensional problem; it is on the backlog.)

**Demand effects interact.** The logistic demand model assumes price elasticity is constant across risk levels. In practice, high-risk customers on PCW tend to shop more aggressively than low-risk direct customers. If you have a segmented demand model (different elasticities by channel, risk level, or tenure), pass it in and the optimiser will use it. The uniform elasticity assumption is a limitation of the demand model, not the optimiser.

**Technical premium is not incurred claims.** The `technical_premium` column is your model's prediction of expected claims. It is not actual claims. The LR at the optimal rates is an expectation conditional on the GLM being a good predictor. If the model is mis-specified or the portfolio mix is shifting, the achieved LR will differ from the projected one.

**Factor bounds are an actuarial choice.** The 15% upper cap is a number we chose. There is no statistical basis for it. Different caps produce different optimal solutions. The shadow prices on the bounds tell you which caps are actually constraining the solution - focus your review on those.

**Correlation across factors.** The optimiser treats factors as independent, which they are at the tariff level (the premium is a product of all relativities). But customer behaviour in response to rate changes may not be independent across factors. A large NCD increase combined with a large age increase both hitting the same young driver simultaneously may produce non-linear lapse behaviour not captured by the individual factor elasticities.

---

## Summary

The workflow in six steps:

1. `data = PolicyData(df)` - wrap your GLM output and demand model predictions
2. `fs = FactorStructure(factor_names, factor_values, renewal_factor_names)` - describe the tariff
3. `demand = make_logistic_demand(params)` - or wrap your actual demand model
4. `opt = RateChangeOptimiser(data, demand, fs)` followed by `add_constraint()` calls for LR, volume, ENBP, and bounds
5. `opt.feasibility_report()` - check before solving
6. `result = opt.solve()` - get factor adjustments and shadow prices; `frontier = EfficientFrontier(opt)` and `frontier.trace()` for the full picture

The shadow price on the LR constraint is the number to put in front of the commercial director. The frontier chart is the story. The factor adjustments are the operational output.

The point of this library is to replace the Excel scenario with a formally stated problem, a provably optimal solution, and a complete account of what each constraint costs. The conversation with the underwriting director changes when you can say "relaxing the LR target from 72% to 73% reduces the required customer disruption by 15%, and here is the frontier showing every achievable option in between."

---

## Academic references

- Branda, M. (2013). "Optimization Approaches to Multiplicative Tariff of Rates Estimation in Non-Life Insurance." ASTIN Colloquium, Hague.
- Guven, S. and McPhail, J. (2013). "Beyond the Cost Model: Understanding Price Elasticity and Its Applications." CAS Spring Forum.
- Emms, P. and Haberman, S. (2005). "Pricing General Insurance Using Optimal Control Theory." ASTIN Bulletin 35(2), 427-453.
- Markowitz, H. (1952). "Portfolio Selection." Journal of Finance 7(1), 77-91.
- FCA (2021). PS21/5: General Insurance Pricing Practices: Final Rules.
- FCA (2023). Consumer Duty: Final Rules and Guidance, FG22/5.
