---
layout: post
title: "Partial Pooling for Thin Rating Cells: Introducing bayesian-pricing"
date: 2026-03-06
categories: [techniques]
tags: [bayesian, hierarchical-models, pymc, credibility, partial-pooling, pricing, python, motor]
---

Every UK motor pricing model has the same problem, buried in the same place: the cells that matter most are the ones with the least data.

Age 17-21. ABI vehicle group 40+. London postcode. Three claims in three years. What's the true expected frequency for that intersection? Your GBM either refused to split on it - pooling it with the trunk - or it split on it and fitted the noise. Your GLM assumes the effect is exactly the product of the young-driver relativity and the sports-car relativity, which is demonstrably wrong. And neither approach tells you how confident to be in the answer.

We built `bayesian-pricing` to solve this properly. The mechanism is partial pooling: thin segments automatically borrow strength from related segments, with the degree of borrowing determined by the data, not by a hyperparameter you tune with cross-validation.

---

## The problem in numbers

A representative UK motor rating model has roughly: 10 driver age bands × 6 NCD levels × 30 vehicle groups × 124 postcode areas × 20 occupation groups. That is 4.5 million theoretical cells. A mid-sized insurer with 1 million policies covers perhaps 3% of them with more than 30 observations. Everything else is thin.

The thin-data problem is worse at interaction level. A cell like "age 17-21 × sports car × London" may have 8 claims in your book, accumulated over 5 years. That's enough to know the cell is high-risk. It is not enough to estimate its frequency to within 30%.

The question is not whether to regularise. You have to regularise. The question is whether the regularisation is calibrated to the data or set by hand.

---

## Why the standard tools get this wrong

**A saturated GLM** - one coefficient per cell - gives you the raw observed rate. For 8 claims on 60 policy-years that's 13.3%. The standard error is ±4.7 percentage points at 95%. You cannot put that in a rate table.

**A main-effects GLM** avoids this by forcing multiplicativity: young-driver relativity × sports-car relativity × London relativity. This works when the interaction really is multiplicative. In practice, for the high-risk intersections that dominate adverse selection, it isn't. A young driver in a sports car in London is not merely the product of three independent effects. The interaction is super-multiplicative, and a main-effects model will systematically underprice it.

**Ridge regularisation** on a GLM does shrinkage, but it applies uniform shrinkage regardless of exposure. A cell with 5,000 policy-years gets the same penalty coefficient as one with 20. That's wrong. The penalty should scale with how noisy the estimate is - which means it should scale with exposure.

**GBMs with `min_data_in_leaf`** avoid splitting on thin cells, which prevents the most egregious overfitting. But they cannot borrow strength across related cells. The "age 17-21, sports car, outside London" leaf with 200 observations knows nothing about the "age 17-21, sports car, London" leaf with 10. And GBMs produce point predictions - there is no calibrated uncertainty interval over the expected loss rate.

---

## The correct answer: partial pooling

The fundamental insight is that neither extreme is right. Treating each cell independently (no pooling) fits noise for thin cells. Treating all cells identically (complete pooling to the portfolio mean) ignores genuine between-segment heterogeneity.

Partial pooling is the middle ground. Each segment's expected loss rate is drawn from a shared population distribution. The population distribution is estimated from all the segments simultaneously. Thin segments, whose own data is noisy, get pulled toward the population mean. Dense segments, whose data is reliable, trust their own experience. The degree of pulling - the credibility factor - is not hand-set. It is the Bayesian posterior: the optimal blend given the ratio of within-segment sampling noise to between-segment signal variance.

Actuaries have been doing a one-dimensional version of this for decades. The Bühlmann-Straub credibility premium is:

```
P_i = Z_i × x̄_i + (1 - Z_i) × μ
```

Where `Z_i = w_i / (w_i + K)` is the credibility factor, `w_i` is the exposure weight, and `K = v/a` is the ratio of within-group variance to between-group variance. This is not an approximation or a heuristic. Under Normal-Normal conjugacy, it is the exact Bayesian posterior mean.

`bayesian-pricing` generalises this to three things Bühlmann-Straub cannot handle: Poisson and Gamma likelihoods (correct for count and severity data), multiple crossed random effects (driver age and vehicle group and postcode area, simultaneously), and full posterior distributions rather than point estimates.

---

## How the model is structured

The frequency model is a Poisson hierarchical model with crossed random effects. For a book with vehicle group and driver age as the rating dimensions:

```
claims_i  ~  Poisson(λ_i × exposure_i)
log(λ_i)  =  α + u_veh[veh_i] + u_age[age_i]

α          ~  Normal(log(portfolio_mean_rate), 0.5)

σ_veh      ~  HalfNormal(0.3)
z_veh_j    ~  Normal(0, 1)                  [one per vehicle group]
u_veh_j    =  z_veh_j × σ_veh              [non-centered]

σ_age      ~  HalfNormal(0.3)
z_age_k    ~  Normal(0, 1)                  [one per age band]
u_age_k    =  z_age_k × σ_age
```

The `σ` parameters are the variance components - they control how much each rating factor is allowed to vary across the book. If `σ_veh` turns out to be 0.1, vehicle group explains little frequency variation and all vehicle segments get pulled strongly toward the portfolio mean. If `σ_veh` is 0.4, vehicle group is a meaningful driver and dense cells get relativities well away from 1.0.

The non-centered parameterisation - expressing group effects as `z × σ` rather than directly as `u ~ Normal(0, σ)` - is mandatory for hierarchical models. The centered version creates a funnel geometry in the posterior when `σ` is small: HMC's step size cannot simultaneously work in the wide mouth and the narrow neck. The non-centered version eliminates this. Every practitioner who has had NUTS produce thousands of divergent transitions in a hierarchical model and didn't know why should read Twiecki's 2017 blog post on this.

---

## Putting it into practice

The library accepts segment-level sufficient statistics - one row per rating cell, with claim count and earned exposure. This is the correct production design: aggregate your book to rating cells before fitting. A book with 500,000 policies and 8,000 non-empty rating cells runs the model on 8,000 rows, making NUTS feasible in 20-40 minutes on a standard machine.

```python
import polars as pl
from bayesian_pricing import HierarchicalFrequency, BayesianRelativities
from bayesian_pricing.frequency import SamplerConfig

# One row per rating cell — aggregate your policy data first
# bayesian_pricing expects pandas at the model boundary; convert from Polars
df = pl.DataFrame({
    "veh_group": ["Supermini", "Supermini", "Sports", "Sports", "Saloon"],
    "age_band":  ["17-21", "31-40", "17-21", "31-40", "31-40"],
    "claims":    [8, 120, 3, 45, 200],
    "exposure":  [60.0, 900.0, 25.0, 350.0, 2000.0],
}).to_pandas()

model = HierarchicalFrequency(
    group_cols=["veh_group", "age_band"],
    prior_mean_rate=0.09,       # set from your portfolio, not from the thin cells
    variance_prior_sigma=0.3,   # log-scale prior on between-segment variation
)

config = SamplerConfig(
    method="nuts",
    draws=1000,
    tune=1000,
    chains=4,
    random_seed=42,
)

model.fit(df, claim_count_col="claims", exposure_col="exposure", sampler_config=config)
```

During model development, use `method="pathfinder"` instead of NUTS. Pathfinder is a variational approximation that runs in seconds rather than minutes. It cannot give you R-hat convergence diagnostics - you cannot use it for final production estimates - but for iterating on priors and model structure it is two orders of magnitude faster.

---

## The output that matters: credibility factors

The `predict()` method returns a DataFrame with the posterior mean claim rate per segment, a credible interval, and a credibility factor:

```python
preds = model.predict()
#   veh_group  age_band     mean     p5      p50     p95   credibility_factor
#   Supermini  17-21      0.1234   0.0812  0.1201  0.1731          0.38
#   Supermini  31-40      0.1341   0.1198  0.1338  0.1490          0.94
#   Sports     17-21      0.1891   0.1102  0.1845  0.2881          0.21
#   Sports     31-40      0.1298   0.1155  0.1294  0.1448          0.89
#   Saloon     31-40      0.0978   0.0912  0.0976  0.1045          0.97
```

The credibility factor is the Bayesian equivalent of `Z_i` in Bühlmann-Straub. A value of 0.21 for Sports/17-21 means that cell has only 25 policy-years of experience - the posterior is 79% portfolio mean, 21% own data. The wide credible interval (0.11 to 0.29) reflects exactly how uncertain we are.

Compare Supermini/31-40 with 900 policy-years: credibility factor 0.94. The posterior is almost entirely data-driven. The credible interval (0.120 to 0.149) is tight.

This is partial pooling doing its job. The thin cell is not overfitting to 3 claims. The dense cell is not being dragged away from its own experience.

---

## Relativities for the rate table

The `BayesianRelativities` class extracts multiplicative relativities in the format actuaries use - the same format as `exp(β)` from a GLM:

```python
rel = BayesianRelativities(model, hdi_prob=0.9)

veh_table = rel.relativities(factor="veh_group")
print(veh_table.table)
#   level      relativity  lower_90pct  upper_90pct  credibility_factor  interval_width
#   Sports          1.524        1.234        1.891           0.71               0.657
#   Saloon          1.000        0.921        1.082           0.94               0.161
#   Supermini       0.819        0.764        0.881           0.89               0.117
```

The 90% credible interval for Sports - 1.23 to 1.89 - is wide because vehicle group alone is not the whole story. The interaction with driver age band matters. `bayesian-pricing` can model this explicitly:

```python
model_with_interaction = HierarchicalFrequency(
    group_cols=["veh_group", "age_band"],
    interaction_pairs=[("veh_group", "age_band")],
    prior_mean_rate=0.09,
)
```

The interaction random effects are a 30 × 10 matrix of offsets (vehicle groups × age bands), each partially pooled toward zero via a shared variance hyperprior. When data are sparse - which they are for most off-diagonal cells - the interaction shrinks to zero, and the model falls back to additive main effects. When data are dense, genuine interactions can be identified.

This is fundamentally different from a GBM split. The GBM either makes the split or does not. The Bayesian model always includes the interaction term but sets its magnitude via the posterior. Thin cells get small interactions; data-rich cells can have large ones.

---

## Identifying thin segments for underwriter review

```python
# Segments where credibility_factor < 0.3 need flagging
thin = rel.thin_segments(credibility_threshold=0.3)
#   factor       level             credibility_factor  relativity
#   age_band     17-21                       0.21        1.84
```

A credibility factor below 0.3 means less than 30% of the estimate comes from the segment's own experience. These segments should not drive large rate changes without actuarial sign-off. The wide credible interval is the quantitative evidence for that caution: the 17-21 age band might really be 1.84× the base, or it might be 1.10×, or 2.60×. The data do not yet tell us which.

This output is the regulatory case for the approach under FCA Consumer Duty. If a pricing model is making significant rate changes to thin segments based on noisy data, that is a fair value risk. The credibility factor flags exactly where that risk sits.

---

## Convergence: the check you must not skip

MCMC results are only valid if the sampler actually explored the posterior. Check before trusting anything:

```python
from bayesian_pricing.diagnostics import convergence_summary, posterior_predictive_check

diag = convergence_summary(model)
# Prints: R-hat: OK (max = 1.004)
#         ESS: OK (min bulk = 621)
#         Divergences: none

ppc = posterior_predictive_check(model, claim_count_col="claims")
# Returns checks on mean, variance, p90, p95
# posterior_predictive_p should be between 0.05 and 0.95 for each
```

R-hat above 1.01 on any parameter means the chains did not mix - the results cannot be used. This almost always means the model is poorly specified or the non-centered parameterisation is not being applied. In `bayesian-pricing` the non-centered parameterisation is the default; if you are extending the model manually in PyMC, make sure to apply it.

If divergences appear after non-centering, increase `target_accept` from 0.8 to 0.95 in `SamplerConfig`. For persistent divergences, check your variance hyperpriors - a `HalfCauchy` hyperprior can allow unrealistically large random effects for thin cells, which paradoxically creates fitting problems.

---

## Where this sits relative to GBMs

We are not suggesting you replace your GBM with this library. For the bulk of your rating factors - vehicle age, driver age as a continuous feature, NCD as a quasi-continuous variable - a GBM on dense cells is the right tool.

The practical architecture for a production book looks like this:

1. CatBoost on the full portfolio for main-effects signal where data are dense
2. Extract segment-level residuals from the GBM
3. `bayesian-pricing` on those residuals to pool the thin-cell adjustments
4. Combine the GBM base prediction with the Bayesian residual adjustment

The GBM knows what it is doing for the 97% of cells with adequate data. The Bayesian model handles the 3% where the GBM is either overfitting or refusing to engage. They are complementary, not competing.

For teams that want the full Bayesian approach without a GBM stage, the library works standalone. Start with Pathfinder for speed, move to NUTS for production, and use the NumPyro backend (`SamplerConfig(nuts_sampler="numpyro")`) if you have GPU resource and a large segment table.

---

## Relationship to the actuarial literature

The connection between hierarchical Bayesian models and credibility theory is older than most pricing actuaries realise. Bühlmann (1967) showed that the credibility premium is the BLUP - best linear unbiased predictor - under the assumed model. What the actuarial community called "credibility" and what statisticians called "empirical Bayes" or "shrinkage estimation" are the same thing in the one-dimensional Normal case.

Ohlsson (2008, *Scandinavian Actuarial Journal*) proved that ridge regression on GLM dummy variables is equivalent to Bühlmann-Straub with `K = λ/v` where `λ` is the ridge penalty. This means that every time you have regularised a GLM, you have been doing implicit credibility. The difference is that in a regularised GLM, `λ` is set by cross-validation - a noisy procedure when cells are thin. In the Bayesian model, the equivalent of `λ` is estimated from the posterior, which propagates its uncertainty correctly.

Krapu et al. (arXiv:2312.07432, 2023) demonstrated that this approach scales. They fitted a 7,756-parameter hierarchical model to 2.6 million Brazilian auto policies using NumPyro on a single Nvidia A10 GPU, achieving an R² of 0.94 on holdout data. For large UK portfolios that need the full policy-level fit rather than the segment-level approach, the NumPyro backend is the practical path.

---

## Install

```bash
uv add "bayesian-pricing[pymc]"
```

The PyMC dependency is optional - the data validation and utility code runs without it, which keeps CI lightweight. The `[pymc]` extra pulls in PyMC 5.x, ArviZ, and PyTensor. For NumPyro GPU inference:

```bash
uv add "bayesian-pricing[numpyro]"
```

Source is on [GitHub](https://github.com/burningcost/bayesian-pricing). The library is at version 0.1.0 - the core frequency and severity models are stable, and we consider the API for `HierarchicalFrequency`, `HierarchicalSeverity`, and `BayesianRelativities` settled. What is not yet built: spatial CAR components for geographic smoothing, temporal random walks for time-varying risk profiles, and the Credibility Transformer integration (Richman, Scognamiglio & Wüthrich, arXiv:2409.16653). Those are on the roadmap.

The thin-cell problem is not going away. If anything it is getting worse as personal lines pricing moves toward more granular segmentation. Partial pooling is the principled answer. Use it.
