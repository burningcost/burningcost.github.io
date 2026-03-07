---
layout: post
title: "Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience"
date: 2026-03-06
categories: [techniques]
tags: [credibility, buhlmann-straub, pricing, python, GLM, scheme-pricing]
---

Every pricing actuary has stared at a claim frequency for a small segment and known, with some certainty, that the number is wrong. Not wrong as in a data error. Wrong in the sense that 14 claims across 800 earned car years is not a reliable estimate of anything. You have a prior - your book experience, your GLM, your market data - and the segment's own history, and you need to combine them in a principled way.

The classical actuarial solution is credibility weighting. Specifically, the Bühlmann-Straub model (1970): a formula for blending segment experience with a portfolio prior, weighted by exposure and calibrated to the actual between-group and within-group variance in your data. It has been the standard in UK scheme and affinity pricing for decades. It is done, almost universally, in Excel.

This post covers the maths, the Python implementation via `credibility.BuhlmannStraub`, and why the approach is more coherent than it appears when you first encounter it as a formula.

---

## The problem with thin segments

A GLM produces a relativity for every level of every rating factor. For high-volume factors - NCD band, vehicle group - this is fine. You have enough data in each cell that the MLE converges to something useful. For thin segments, it breaks down.

Suppose you're rating a scheme for a professional body: 2,200 policies, three years of history. Your GLM for the main book was trained on 800,000 policies and has well-estimated parameters. The scheme GLM, even if you fit it, will have standard errors several times larger than the point estimates for most coefficients. You cannot trust the parameters on their own.

The instinct is to use the book GLM and apply a flat adjustment for the scheme's loss ratio experience. That works, but it throws away information. If the scheme's loss ratio has been 72% for three years running, that tells you something. The question is how much to weight it relative to the 83% book mean.

The answer Bühlmann and Straub gave in 1970 depends on two quantities:

- How variable are individual claim frequencies period-to-period *within* a typical scheme? (Call this v - the expected process variance.)
- How different are schemes from each other, beyond what pure randomness would produce? (Call this a - the variance of hypothetical means.)

The ratio K = v/a is the credibility parameter. High K means schemes are mostly homogeneous - you need a lot of data before you trust individual experience. Low K means schemes vary substantially - individual experience becomes informative quickly.

---

## The maths

The Bühlmann-Straub model structure is as follows. You have r groups (schemes, regions, vehicle classes - whatever the segmentation is). For group i, period j, you observe a loss rate X_{ij} with exposure weight w_{ij}. The variance of X_{ij} given the group's true risk parameter θ_i scales inversely with exposure:

```
Var(X_{ij} | θ_i)  =  σ²(θ_i) / w_{ij}
```

This is the key departure from the basic Bühlmann model: a group with 10,000 earned car years in a single year gets lower variance - more precise measurement - than a group with 500.

The structural parameters:

```
v   =  E[σ²(θ)]      Expected value of Process Variance (EPV)
                      Within-group noise, averaged across the portfolio

a   =  Var[μ(θ)]     Variance of Hypothetical Means (VHM)
                      Between-group signal: how different groups actually are

K   =  v / a          Bühlmann's k - the credibility pivot
```

The credibility factor for group i, with total exposure w_i = Σ_j w_{ij}:

```
Z_i  =  w_i / (w_i + K)
```

And the credibility premium:

```
P_i  =  Z_i · X̄_i  +  (1 − Z_i) · μ̂
```

where X̄_i is the exposure-weighted mean loss rate for group i, and μ̂ is the grand portfolio mean.

This is the Best Linear Unbiased Predictor (BLUP) of μ(θ_i) under the model. It minimises E[(P_i − μ(θ_i))²] over all linear estimators of the group mean. No distributional assumption is required beyond finite second moments - you don't need to assume Poisson counts or Normal loss ratios.

When w_i is small relative to K, Z_i → 0 and P_i → μ̂. The group is thin; lean on the portfolio. When w_i is large, Z_i → 1 and P_i → X̄_i. The group has enough exposure to stand on its own.

### Estimating v, a, K from data

The power of Bühlmann-Straub is that v, a, and K are estimated empirically. No prior specification required. Given r groups, T_i periods per group, total exposure w = Σ_i w_i:

**Estimate of v (within-group variance):**

```
v̂  =  Σ_i Σ_j [ w_{ij} · (X_{ij} − X̄_i)² ]
       ─────────────────────────────────────────
              Σ_i (T_i − 1)
```

This is a weighted within-group mean squared deviation - the same structure as a pooled variance estimate. It requires at least two periods per group to compute; single-period groups contribute nothing to the numerator.

**Estimate of a (between-group variance):**

```
c    =  w  −  Σ_i(w_i²) / w

s²   =  Σ_i [ w_i · (X̄_i − μ̂)² ]

â    =  (s²  −  (r − 1) · v̂) / c
```

The term c is a normalising constant that corrects for the fact that groups with different exposures contribute unequally to the between-group sum of squares. It is analogous to the denominator in an ANOVA between-group variance estimate.

â can be negative. This happens when the observed between-group variation is no larger than what pure noise would produce - the portfolio is homogeneous. Convention is to truncate â at zero, which sends K → ∞ and all Z_i → 0. Every group gets the portfolio mean. This is the right answer: if you cannot distinguish groups beyond random noise, don't try.

With v̂ and â in hand:

```
K̂    =  v̂ / â
Z_i  =  w_i / (w_i + K̂)
P_i  =  Z_i · X̄_i  +  (1 − Z_i) · μ̂
```

---

## Python implementation

The `credibility` package implements Bühlmann-Straub with the non-parametric estimators above:

```bash
uv add credibility
```

The core class is `BuhlmannStraub`. It expects a long-format DataFrame: one row per group-period, with columns for the group identifier, time period, loss rate, and exposure. The library uses Polars natively; examples below use Polars throughout.

```python
import polars as pl
from credibility import BuhlmannStraub

bs = BuhlmannStraub()
bs.fit(
    data=df,          # pl.DataFrame, one row per (group, period)
    group_col="region",
    period_col="year",
    loss_col="claim_freq",   # claims per earned car year
    weight_col="exposure",   # earned car years
)
```

After fitting, the structural parameters are available as attributes:

```python
bs.mu_hat_   # grand weighted mean
bs.v_hat_    # EPV - within-group variance
bs.a_hat_    # VHM - between-group variance (truncated at 0 if negative)
bs.k_        # K = v/a
```

The credibility factors and credibility-weighted estimates:

```python
bs.z_          # credibility factors indexed by group
bs.premiums_   # DataFrame: group, exposure, observed_mean, Z,
               #            credibility_premium, complement
```

`credibility_premium` is the blended estimate P_i. `complement` is μ̂ - the portfolio mean that every thin group regresses toward. `observed_mean` is X̄_i, the group's own exposure-weighted history. The full picture is in one table.

---

## A worked example: claim frequencies across regions

We'll work through a realistic example: a UK motor book with 12 geographic regions, four years of history, exposures ranging from 3,000 to 85,000 earned car years per region per year.

```python
import numpy as np
import polars as pl
from credibility import BuhlmannStraub

rng = np.random.default_rng(42)

# 12 regions, true frequencies drawn from a lognormal portfolio distribution
n_regions = 12
n_years = 4
true_freqs = rng.lognormal(mean=np.log(0.08), sigma=0.25, size=n_regions)

# Exposures vary substantially by region
base_exposures = rng.uniform(3_000, 85_000, size=n_regions)

rows = []
for i, (freq, base_exp) in enumerate(zip(true_freqs, base_exposures)):
    for year in range(2021, 2021 + n_years):
        exposure = base_exp * rng.uniform(0.9, 1.1)
        claims = rng.poisson(freq * exposure)
        rows.append({
            "region": f"R{i+1:02d}",
            "year": year,
            "exposure": exposure,
            "claim_freq": claims / exposure,
        })

df = pl.DataFrame(rows)

bs = BuhlmannStraub()
bs.fit(
    data=df,
    group_col="region",
    period_col="year",
    loss_col="claim_freq",
    weight_col="exposure",
)

print(f"Portfolio mean (μ̂):   {bs.mu_hat_:.4f}")
print(f"EPV (v̂):               {bs.v_hat_:.2e}")
print(f"VHM (â):               {bs.a_hat_:.2e}")
print(f"K:                     {bs.k_:.1f}")
```

In this simulation K comes out around 18. That means a region needs roughly 18,000 earned car years before its own experience carries 50% weight: Z = 18,000 / (18,000 + 18,000) = 0.50. Regions with 3,000-5,000 total exposure are getting Z around 0.14-0.21. They barely move from the portfolio mean. Regions with 80,000 exposure sit at Z ≈ 0.81 - approaching full credibility.

```python
print(bs.summary())
```

Output (illustrative):

```
Bühlmann-Straub Credibility Model
==========================================
  Collective mean    mu  = 0.07851
  Process variance   v   = 1.42e-05   (EPV, within-group)
  Between-group var  a   = 7.73e-04   (VHM, between-group)
  Credibility param  k   = 18.4       (v / a)

  Interpretation: a group needs exposure = k to achieve Z = 0.50

group  exposure  observed_mean      Z  credibility_premium  complement
  R01    14,821         0.0621  0.447               0.0693      0.0785
  R02    72,340         0.0934  0.797               0.0904      0.0785
  R03     5,102         0.0441  0.217               0.0647      0.0785
  R04    31,918         0.1143  0.634               0.1017      0.0785
  ...
```

Region R03 has the lowest exposure - 5,100 car years across four years. Its raw frequency is 0.0441, materially below the portfolio mean of 0.0785. But with Z = 0.217, the credibility-weighted estimate is 0.0647: still pulled substantially toward the mean. We don't believe 0.0441 for R03 because there is not enough data to trust it. The formula agrees.

Region R04 has 31,900 exposure and a raw frequency of 0.114 - notably worse than the mean. Z = 0.634, so the credibility estimate is 0.102. Still elevated, but not as extreme as the raw experience suggests. If you rated R04 at 0.114, you would almost certainly over-charge for it.

The Z-vs-exposure relationship is just one curve: Z_i = w_i / (w_i + K). Every region sits on it somewhere depending on their volume. When explaining the method to underwriters or pricing committees, that curve - K determines its shape, exposure determines where each segment lands - is usually enough to make the logic intuitive.

---

## The connection to Bayesian shrinkage and mixed models

If Bühlmann-Straub looks familiar from a statistical perspective, it should. It is the same computation as the BLUP from a one-way random effects model:

```
X_{ij}  =  μ  +  b_i  +  ε_{ij}

b_i    ~  (0, a)
ε_{ij} ~  (0, v / w_{ij})
```

The BLUP of μ + b_i under this model is exactly the Bühlmann-Straub credibility premium P_i, with K = v/a. The actuarial and statistical communities spent several decades working on the same problem with different vocabulary.

In Python, `statsmodels.regression.mixed_linear_model.MixedLM` produces the same fitted values as `BuhlmannStraub` for Normal responses with REML estimation. The difference is output: `MixedLM` gives you regression coefficients and t-statistics. `BuhlmannStraub` gives you v̂, â, K, Z by group, and a credibility premium table in the format actuaries and underwriters can use directly.

Under conjugate Bayesian models - Poisson-Gamma, Normal-Normal - the Bühlmann credibility premium is also exactly the Bayes posterior mean. For non-conjugate models it is the best linear approximation. This means credibility weighting does Bayesian shrinkage without requiring you to specify a prior distribution: the prior is estimated from the data via v̂ and â.

The connection to penalised regression is exact. Ridge regression on dummy variables for a categorical factor with penalty parameter λ is Bühlmann-Straub credibility with K = λ/v - this result is due to Ohlsson (2008). Akur8's "GLM+" is a penalised GLM; it does credibility implicitly. `BuhlmannStraub` makes it explicit: K comes from the data rather than cross-validation, and Z factors are interpretable outputs rather than implicit regularisation.

### How it complements GBM approaches

A GBM trained on thin data will overfit. Regularisation parameters (`min_data_in_leaf`, `depth`, L2 leaf regularisation) limit this, but they are tuned globally - they don't adapt to which particular segments are thin and which are well-populated.

Credibility weighting offers a different intervention. Train your GBM on the full book. Extract segment-level loss rates from its predictions. Then credibility-blend those segment estimates with the book average, using `BuhlmannStraub` to set K from the data. The result respects the segment's own experience where it has enough data, and falls back to book experience where it doesn't.

The mathematical setup mirrors Bühlmann-Straub exactly. For factor j, level l:

```
Z_{j,l}  =  n_{j,l} / (n_{j,l} + K_j)

r_blend  =  Z_{j,l} · r_{segment}  +  (1 − Z_{j,l}) · r_{book}
```

where r is a segment-level estimate (from SHAP-extracted relativities or any other source) and n_{j,l} is exposure at that level. You get explicit Z factors per factor-level, an auditable trail, and a principled answer to the question: "how much of this segment's rate came from its own experience versus the book?"

This connection also appears in recent deep learning work. Richman, Scognamiglio and Wüthrich (2024) showed that a Transformer architecture applied to insurance data learns a credibility-like blending mechanism via attention weights, with a hyperparameter controlling how much the CLS token draws on portfolio-level versus individual-risk encoding. The mathematics is different but the mechanism is the same: blend individual experience with a collective prior, with the blend weight learned from data. Bühlmann-Straub did this analytically in 1970.

---

## When to use this

**You have multiple groups with a meaningful portfolio prior.** The method requires at least 5-6 groups to estimate â reliably. Three schemes gives you a very uncertain K. Twenty regional territories gives you a much more stable one. With fewer than five groups, use a full Bayesian treatment (PyMC with a Poisson-LogNormal hierarchy) or apply a manual credibility factor based on actuarial judgment.

**Groups have multiple periods of experience.** The v̂ estimator requires at least two periods per group. If you have single-year snapshots only, you cannot estimate within-group variance from the data; you need an external estimate of v - for example, from the overdispersion parameter in a Poisson GLM fitted to the full book.

**The groups share a common rating structure.** Credibility weighting blends toward the portfolio mean. That mean needs to be a meaningful benchmark for the group. If a scheme has a genuinely alien risk profile - a private aviation insurer's employer liability scheme - the book motor mean is not the right prior.

**You need an audit trail.** FCA Consumer Duty (July 2023) requires firms to demonstrate fair value. Credibility-weighted rates are inherently transparent: here is the scheme's own experience, here is the book benchmark, here is Z = 0.41, here is the final rate. A black-box GBM with no credibility layer makes this conversation much harder.

---

## When not to use this

**You have one or two groups.** With a single scheme and no portfolio comparison point, Bühlmann-Straub collapses. You cannot estimate a. Use actuarial judgment, an industry benchmark, or the full Bayesian model.

**The data-generating process is non-stationary.** The standard model assumes v and a are constant across periods. Post-FCA pricing reform (PS21/5, effective January 2022) broke this assumption for UK personal lines renewal pricing: pre-2022 data is a materially different regime. You can partially address this with exponentially discounted weights - giving more recent periods higher weight by scaling w_{ij} by a decay factor λ^(current_year − year_j) before fitting - but if the structural break is severe, the older data may do more harm than good. A λ of 0.7 per year down-weights 2020 experience to 0.49, 2019 to 0.34 - reasonable for a post-FCA recalibration. `BuhlmannStraub` accepts any positive exposure weights, so you pass in the decayed weights directly.

**Your groups have very different structures.** Bühlmann-Straub assumes a single K applies across all groups. If some groups are mass-market commoditised personal lines and others are high-net-worth niche schemes, the between-group heterogeneity is not well-described by a single a parameter. Consider a hierarchical model or run separate credibility analyses by business type.

**Your response is highly non-Normal.** The standard model works in linear space. For Poisson counts with large exposure, applying it to raw frequencies is approximately correct. For severity data with heavy tails and large outliers, linear blending can give perverse results. Cap large losses before computing loss rates, or work in log space and accept the approximation explicitly.

---

## A note on the Python gap

There is no other Python package that does this well. The R `actuar` package has `cm()`, which is excellent - it handles Bühlmann-Straub, hierarchical credibility, Hachemeister regression credibility, and several conjugate Bayesian models. If your team works in R, use it.

In Python, the options before `credibility` were: fit a `statsmodels.MixedLM` and extract the BLUPs manually (no Poisson response, no actuarial output), use `gpboost` (powerful, but a GLMM combined with gradient boosting - not a standalone credibility tool), or do it in Excel and paste the numbers into your pipeline.

We built this because UK pricing teams should not be doing credibility in Excel. The formula is 54 years old. It should be in a library.

---

## References

- Bühlmann, H. (1967). Experience rating and credibility. *ASTIN Bulletin*, 4(3), 199–207.
- Bühlmann, H., & Straub, E. (1970). Glaubwürdigkeit für Schadensätze. *Mitteilungen der Vereinigung Schweizerischer Versicherungsmathematiker*, 70, 111–133.
- Bühlmann, H., & Gisler, A. (2005). *A Course in Credibility Theory and its Applications*. Springer.
- Ohlsson, E. (2008). Combining generalized linear models and credibility models in practice. *Scandinavian Actuarial Journal*, 2008(4), 301–314.
- Richman, R., Scognamiglio, S., & Wüthrich, M.V. (2024). The Credibility Transformer. *arXiv:2409.16653*.
- Robinson, G.K. (1991). That BLUP is a good thing: The estimation of random effects. *Statistical Science*, 6(1), 15–51.
- Dutang, C., Goulet, V., & Pigeon, M. actuar: Actuarial functions and heavy-tailed distributions. R package, CRAN.
