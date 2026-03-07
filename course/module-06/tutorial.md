# Credibility & Bayesian Pricing - The Thin-Cell Problem

---

## Why this matters

You run the pricing model for a UK motor book. You are reviewing the rating factor table for postcode district KT (Kingston upon Thames). Your model has 847 policies in KT in the last 12 months, 11 claims - an observed frequency of 1.30%. The portfolio mean for similar risks is 6.8%. Should KT get a rate reflecting 1.30%? Should it get the portfolio mean? Something in between?

The answer is: something in between, and the correct something depends on two numbers: how variable is KT's true underlying risk year to year (within-group variance), and how different are postcode districts from each other overall (between-group variance). Bühlmann-Straub credibility gives you the principled method to compute the blend.

Scale this up. Great Britain has approximately 2,800 postcode districts. A UK motor book with 1.5 million policies has an average of 535 policies per district. That sounds comfortable, but motor books are not uniformly distributed - inner London districts each have thousands of policies while rural Scottish districts may have under 50. Thin cells are not a rounding problem; they are a structural feature of personal lines rating.

GLMs handle the thin-cell problem indirectly, via parameter constraints and regularisation. A main-effects GLM forces all area relativities onto a single factor table - any area that looks different from the trend in the factor gets pulled toward the line. Ridge-regularised GLMs shrink every coefficient toward zero, regardless of the cell's exposure. LASSO does worse: it forces thin cells to exactly zero, which is wrong - every area has some true effect, it is just hard to measure.

GBMs are the worst offender. Set min_data_in_leaf too low and the model learns noise in KT as signal. Set it too high and the model ignores KT entirely, treating it like the portfolio average. Neither is right.

The most defensible framework for most UK personal lines problems is partial pooling: each cell's estimate is a weighted blend of its own experience and the portfolio mean, where the blend weights are determined by how much information the cell's experience actually contains. That framework is credibility theory, and in its Bayesian formulation it becomes hierarchical modelling. This module covers both.

---

## Classical credibility: Bühlmann-Straub

### The structural parameters

The Bühlmann-Straub model (Bühlmann & Straub, 1970) operates on a dataset of groups (areas, schemes, vehicle classes - any set of segments) observed over multiple periods. For group i in period j, we observe a loss rate X_{ij} with exposure weight w_{ij} (earned car years, policy count, or premium - whatever is appropriate).

Three population-level parameters drive everything:

**mu**: the collective mean. The portfolio-wide expected loss rate, weighted by exposure. This is the prior.

**v (EPV - Expected value of Process Variance)**: within-group variance, averaged over the portfolio. How variable is a group's loss rate from year to year, given its true underlying risk? High v means groups are inherently volatile - even if you knew KT's true risk exactly, its observed rate would jump around.

**a (VHM - Variance of Hypothetical Means)**: between-group variance. How different are groups' true underlying risks from each other? High a means the portfolio is genuinely heterogeneous - KT really is a different risk than, say, OX (Oxford).

The credibility parameter K = v/a. Think of K as: "how many units of exposure does it take before a group's own experience is as informative as the portfolio mean?" Low K (heterogeneous portfolio with stable groups) means you trust individual experience quickly. High K (volatile groups in a homogeneous portfolio) means you need many years of data before you update much from the portfolio average.

### The credibility factor Z

For group i with total exposure w_i = sum_j(w_{ij}), the Bühlmann-Straub credibility factor is:

```
Z_i = w_i / (w_i + K)
```

The credibility-weighted estimate is:

```
P_i = Z_i * X_bar_i + (1 - Z_i) * mu
```

where X_bar_i = sum_j(w_{ij} * X_{ij}) / w_i is the exposure-weighted observed mean for group i.

When w_i is large (plenty of data), Z_i approaches 1 and the estimate trusts the group's own experience. When w_i is small, Z_i approaches 0 and the estimate falls back to the portfolio mean. The speed of that transition is controlled entirely by K.

This is the Best Linear Unbiased Predictor (BLUP) of each group's true underlying mean, within the class of linear estimators. No distributional assumption is required beyond finite second moments. That is what makes Bühlmann-Straub so robust: it works whether your loss rates are Normal, Poisson, Gamma, or nothing you could name.

A note on robustness: the BLUP property holds regardless of distribution, but the quality of the structural parameter estimates (v and a) depends on the data distribution. Very skewed loss rates - common when severity drives thin-cell volatility - will inflate the EPV estimate and reduce all Z values.

### Estimating the structural parameters

You do not need to specify v and a in advance - they are estimated from the data. The standard unbiased estimators:

**Collective mean:**
```
mu_hat = sum_i(w_i * X_bar_i) / w        [where w = sum_i(w_i)]
```

**EPV (v_hat):**
```
v_hat = (1 / sum_i(T_i - 1)) * sum_i sum_j [ w_{ij} * (X_{ij} - X_bar_i)^2 ]
```

This is the within-group sum of squared deviations from the group mean, weighted by exposure, pooled across all groups and periods. It is unbiased for the expected process variance.

One important note on EPV estimation: groups with only one period of data (T_i = 1) contribute zero to the numerator - a group with only one year has no within-group deviation - but they contribute -1 to the denominator sum_i(T_i - 1). This incorrectly reduces the effective denominator. Filter out any group with T_i = 1 before computing the EPV. The `credibility` library handles this internally, but your data preparation should not assume it.

**VHM (a_hat):**
```
c     = w - sum_i(w_i^2) / w
s^2   = sum_i [ w_i * (X_bar_i - mu_hat)^2 ]
a_hat = (s^2 - (r - 1) * v_hat) / c
```

where r is the number of groups. The c term handles unequal group sizes - note that c = sum_i(w_i) - sum_i(w_i^2)/sum_i(w_i), which is the correct denominator for the general case with heterogeneous T_i. Readers familiar with Bühlmann (1967) rather than Bühlmann & Straub (1970) will encounter a simpler symmetric version of c - the formula above is the correct general form. The s^2 term is the between-group sum of squares; subtracting (r-1)*v_hat removes the within-group sampling noise that would otherwise inflate the between-group estimate.

One important detail: a_hat can be negative. This happens when within-group variance dominates - the groups appear similar not because they are truly similar, but because the data are too noisy to distinguish them. By convention, a_hat is truncated at zero. When a_hat = 0, K is infinite and all Z_i = 0 - every group gets the portfolio mean. This is not wrong; it means the data cannot justify any group-level adjustment.

Treat a substantially negative a_hat before truncation as a diagnostic, not just a numerical quirk. If a_hat is, say, -0.003 when v_hat is 0.002, the data are telling you that your grouping structure explains nothing. Posting Z = 0 for every district is the correct decision - but it also means your model produces no geographic differentiation at all, which has direct repricing implications you should raise with the underwriting team.

### Using the credibility library

For UK motor pricing, where we are working in a multiplicative framework (Poisson with log link), we apply Bühlmann-Straub in log-rate space rather than rate space. Applying B-S directly to rates and then converting to relativities introduces a Jensen's inequality bias - because the log of the expected value does not equal the expected value of the log. The correction is straightforward: use `log_transform=True`.

```python
from credibility import BuhlmannStraub

# For frequency data in a multiplicative (log-link) framework, apply B-S in log
# space to avoid the Jensen's inequality bias that would arise from blending rates
# directly and then converting to relativities.
bs = BuhlmannStraub(log_transform=True)
bs.fit(
    data=df,
    group_col="postcode_district",
    value_col="claim_frequency",   # loss rate per unit of exposure
    weight_col="earned_years",
)

# Structural parameters
print(f"mu = {bs.grand_mean:.4f}")
print(f"v  = {bs.v_hat_:.6f}  (EPV)")
print(f"a  = {bs.a_hat_:.6f}  (VHM)")
print(f"K  = {bs.k_:.2f}")

# Credibility factors and estimates
print(bs.results.head(10))
# Columns: group, exposure, obs_mean, Z, credibility_estimate
```

The `results` DataFrame is the deliverable. `Z` is the credibility factor per group. `credibility_estimate` is the blended rate P_i. For KT with 847 policy-years and a K of, say, 1,200: Z = 847 / (847 + 1200) = 0.41. KT's rate would be 41% of its own observed 1.30% and 59% of the portfolio mean 6.8%. That comes to 4.55% - a meaningful adjustment down from the portfolio mean but far from KT's own volatile experience.

For most UK motor portfolios, the Jensen's inequality bias is small in absolute terms. For extreme relativities - thin cells with observed rates far from the mean - it matters enough to use the log-transform as the default.

### The credibility factors DataFrame

The full `results` attribute has everything you need for a pricing committee:

```python
print(bs.credibility_factors)
# group_col: postcode_district
# exposure: earned_years (total per group)
# obs_mean: observed claim frequency
# Z: credibility factor
# credibility_estimate: blended rate
# complement: (1 - Z) * grand_mean - the "borrowed" portion
```

The `credibility_factor` column maps to the segment-level shrinkage. A group with Z = 0.85 is telling you: this segment is large enough that we trust 85% of its own experience, with 15% borrowed from the portfolio. A group with Z = 0.12 is essentially running at the portfolio rate with a 12% nod to its own experience.

The Z interpretation in terms of exposure thresholds (# Z = w/(w+K); solve for w at Z = 0.50, 0.67, 0.90):
- Z = 0.50: need w = K earned years
- Z = 0.67: need w = 2K earned years
- Z = 0.90: need w = 9K earned years

---

## Applying B-S to model residuals, not raw rates

The tutorial above works with raw observed rates. In practice, this is rarely the right application.

If you have already fitted a GLM or GBM on the full dataset, you have a model prediction for each district. The credibility question is: should the district factor from the model be adjusted based on the district's own experience relative to what the model expects? You apply B-S to the district-level residuals - the ratio of observed rate to model-predicted rate - not to the raw observed rates.

```python
# After fitting your GLM or GBM, compute district-level O/E ratios
# (observed over expected = model residuals at segment level)
dist_residuals = (
    df
    .with_columns([
        (pl.col("claim_count") / pl.col("model_predicted_claims")).alias("oe_ratio")
    ])
    .group_by(["postcode_district", "accident_year"])
    .agg([
        pl.col("oe_ratio").mean().alias("oe_ratio"),
        pl.col("earned_years").sum().alias("earned_years"),
    ])
    .filter(pl.col("earned_years") > 0.5)
)

# Apply B-S to the O/E ratios with log_transform=True
bs_residuals = BuhlmannStraub(log_transform=True)
bs_residuals.fit(
    data=dist_residuals.to_pandas(),   # credibility library expects pandas
    group_col="postcode_district",
    value_col="oe_ratio",
    weight_col="earned_years",
)

# The credibility estimate is the district-level adjustment factor to apply on
# top of the model's predictions - multiplicative if log_transform=True
```

This matters because applying B-S to raw rates when a GBM has already partially handled thin cells produces double shrinkage. The GBM's regularisation (min_data_in_leaf, learning rate, L2 leaf penalty) already shrinks thin-cell predictions toward the base rate. If you then apply B-S to the raw observed rates, you are shrinking twice. The correct approach is to apply B-S to the residuals from the fitted model - the remaining district-level signal that the model has not absorbed. This avoids double-shrinkage and gives a clear decomposition: GBM handles the main effects; B-S on residuals handles the district-level adjustment.

---

## Structural parameter stationarity

The structural parameters v and a are estimated from historical data - but from which data? If your training set includes 2020–2022, those years contain the COVID shock: claim frequencies fell sharply in 2020 (less driving), rebounded unevenly in 2021–2022, and the cost of claims per incident inflated due to supply chain disruption. Estimating v from that period inflates the EPV, because some of the within-group year-to-year variance is a portfolio-wide shock rather than genuine group-level volatility. Inflated v inflates K, and inflated K produces Z values that are too low - more shrinkage toward the portfolio mean than the data warrant.

In practice:
- Re-estimate structural parameters annually at each model rebuild, not just when you rebuild the model entirely
- Consider excluding shock years (or applying a reduced weight) when estimating v and a if the portfolio experienced a clear external distortion
- Monitor K stability over time: a substantial upward shift in K without a change in portfolio composition is usually a signal that the EPV estimate has been contaminated by a portfolio-wide event

---

## Correlated group effects

The Bühlmann-Straub model assumes the group hypothetical means theta_i are independently and identically drawn from the same prior. For UK postcodes, this assumption is violated: districts in the same urban area are correlated. KT1, KT2, and KT3 share road networks, parking infrastructure, crime rates, and flood risk. They are not independent draws from a common distribution.

When groups are correlated, flat Bühlmann-Straub underestimates the between-group variance (it misattributes some genuine between-group signal as within-group correlation) and over-shrinks. Districts in the same high-risk urban area will be pulled too far toward the portfolio mean when they should be borrowing strength primarily from their neighbouring districts.

The practical mitigation is the two-level hierarchical model in Exercise 2: districts nest within areas, and the area-level effect is itself partially pooled. This captures the correlation structure without requiring an explicit spatial covariance model. If you are using flat B-S on postcodes, flag this assumption explicitly in your methodology documentation - particularly if your book has concentrated exposure in a few urban areas where the within-area correlation is highest.

---

## The bridge to Bayesian

Bühlmann-Straub is an empirical Bayes method - a fact that is rarely stated clearly in actuarial training, and that motivates everything in the second half of this module.

Specifically, the credibility premium P_i is exactly the posterior mean of a Bayesian model where:

```
X_{ij} | theta_i  ~  Normal(theta_i, v / w_{ij})   [observation model]
theta_i           ~  Normal(mu, a)                   [prior on group mean]
```

The posterior mean of theta_i given the observations is:

```
E[theta_i | data] = Z_i * X_bar_i + (1 - Z_i) * mu
```

where Z_i = w_i / (w_i + K) with K = v/a. That is exactly the Bühlmann-Straub formula. The credibility factor Z is the posterior weight on the likelihood relative to the prior. High Z means the data dominate the prior; low Z means the prior dominates.

Three things follow from this connection:

**First**, Bühlmann-Straub plugs in point estimates of v and a (the structural parameters we estimated above). Full Bayesian treats v and a as uncertain - it places priors on them and integrates over that uncertainty. When you have few groups (say, 5 affinity schemes rather than 124 postcode areas), the uncertainty in K matters a lot. Plugging in a_hat from 5 groups is unreliable; the Bayesian approach propagates that uncertainty into your final estimates.

**Second**, the Normal likelihood in the derivation above is the constraint that makes Bühlmann-Straub give a closed-form answer. For Poisson claim counts or Gamma severity, the conjugate prior no longer gives a tidy linear formula. Full Bayesian handles Poisson and Gamma directly, without approximation.

**Third**, Bühlmann-Straub is inherently one-dimensional - one grouping variable at a time. Full Bayesian handles crossed random effects: vehicle group AND driver age AND area, all simultaneously, each with appropriate partial pooling.

The practical upshot: use Bühlmann-Straub when you have one grouping variable, many groups, and data that are reasonably close to Normal in the log-rate space. Use full Bayesian when you need multiple crossed groupings, proper Poisson/Gamma likelihoods, or uncertainty quantification.

---

## Hierarchical Bayesian models

### Model structure

The `bayesian-pricing` library provides `HierarchicalFrequency` and `HierarchicalSeverity`, built on PyMC 5.x. We work through `HierarchicalFrequency` in detail; the severity model follows the same pattern.

The model for claim counts is a Poisson hierarchical model with crossed random effects:

```
claims_{ij} | lambda_i, exposure_{ij}  ~  Poisson(lambda_i * exposure_{ij})
log(lambda_i) = alpha + u_{area[i]} + u_{veh[i]} + ...
u_{area[k]} ~ Normal(0, sigma_area)
u_{veh[k]}  ~ Normal(0, sigma_veh)
alpha        ~ Normal(log(mu_portfolio), 0.5)
sigma_area   ~ HalfNormal(0.3)
sigma_veh    ~ HalfNormal(0.3)
```

**Prior justification.** A standard deviation of 0.5 on the log scale for alpha means the prior allows the portfolio intercept to range from approximately exp(-0.5) = 0.61x to exp(+0.5) = 1.65x the observed portfolio mean (±1 SD on the log scale). For a UK motor book with a portfolio frequency of 6–8%, this permits the prior to range from about 3.7% to 13% at ±1 SD - a reasonable range that is not unduly informative. If your prior knowledge about the portfolio mean is stronger, narrow this SD; if you are fitting to a genuinely novel book, widen it and check the prior predictive distribution.

A standard deviation of 0.3 on the log scale for sigma_area means the model expects the between-area log-standard deviation to be below about 0.6 log points with high probability (given HalfNormal support is positive). On the rate scale, this implies most areas fall within a range of roughly 0.5x to 2.0x the portfolio mean, which is plausible for UK motor postcode districts. For vehicle group effects - where relativities can reach 3–4x the base rate for exotic vehicles - HalfNormal(0.3) is too restrictive. Adjust the prior to match the factor type: HalfNormal(0.5) or HalfNormal(0.7) for effects with wider expected ranges. Checking the prior predictive distribution (simulated datasets before seeing the data) is the correct way to validate these choices.

Each random effect u_{area[k]} is a log-scale adjustment for area k. When sigma_area is large, the model allows areas to deviate substantially from the global mean. When sigma_area is small, areas are pooled heavily toward the global mean. The data determine sigma_area - that is the Bayesian learning from the between-group signal.

The `HierarchicalFrequency` class takes segment-level sufficient statistics rather than policy-level data. For a dataset of 500,000 policies, grouping to, say, 5,000 area-vehicle combinations and running the Bayesian model on those 5,000 rows is far more practical than passing all 500,000 rows to MCMC. The sufficient statistics for Poisson are total claims and total exposure per segment.

```python
import polars as pl
from bayesian_pricing import HierarchicalFrequency

# Prepare segment-level data using Polars
# Each row is one (area, vehicle_group) combination
segments = (
    df.group_by(["postcode_district", "veh_group"])
    .agg([
        pl.col("claim_count").sum().alias("claims"),
        pl.col("earned_years").sum().alias("exposure"),
    ])
)

hf = HierarchicalFrequency(
    group_cols=["postcode_district", "veh_group"],
    claims_col="claims",
    exposure_col="exposure",
)

# HierarchicalFrequency accepts a pandas DataFrame; bridge here
hf.fit(segments.to_pandas())
```

`fit()` builds the PyMC model and runs NUTS sampling. By default: 4 chains, 1,000 warmup samples, 1,000 posterior samples. On a single-node Databricks cluster, this takes 3–10 minutes depending on the number of segments.

### Non-centered parameterization: why it is non-negotiable

The most common implementation mistake in hierarchical Bayesian models is the centered parameterization. In centered form, the area random effects are directly sampled as:

```python
u_area = pm.Normal("u_area", mu=0, sigma=sigma_area, dims="area")
```

When sigma_area is near zero (all areas are similar), u_area and sigma_area become highly correlated in the posterior - the posterior geometry forms a funnel. NUTS, which uses gradient information to choose its step size, cannot pick a step size that works in both the narrow and wide parts of the funnel simultaneously. The sampler fails to explore the narrow part of the funnel, systematically undersampling the region near sigma → 0, which biases the variance component estimates toward zero. The practical consequence: your credibility factors will be too low - more shrinkage than the data warrant - without any obvious warning that something is wrong.

The non-centered parameterization decouples the raw offsets from the scale:

```python
u_area_raw = pm.Normal("u_area_raw", mu=0, sigma=1, dims="area")
sigma_area = pm.HalfNormal("sigma_area", sigma=0.3)
u_area = pm.Deterministic("u_area", u_area_raw * sigma_area, dims="area")
```

Now u_area_raw is independent of sigma_area in the prior. The funnel disappears. `HierarchicalFrequency` uses non-centered parameterization by default and will warn you if you attempt to override it.

This matters practically. A model run with centered parameterization may appear to converge (the chains complete, the R-hat may even look acceptable in aggregate) but the posterior for the variance components will be biased toward zero, understating the between-group heterogeneity. Your credibility factors will be too low - more shrinkage than the data warrant.

### Reading the posteriors

After fitting, `HierarchicalFrequency` exposes a `posteriors` attribute containing the ArviZ InferenceData object, and a structured results DataFrame:

```python
results = hf.results

# Columns:
# postcode_district, veh_group,
# claims, exposure, observed_rate,
# posterior_mean, posterior_sd,
# credibility_factor,
# lower_90, upper_90
```

`posterior_mean` is the posterior mean of lambda for each segment - the Bayesian equivalent of the credibility-weighted rate. `credibility_factor` maps to the classical Z: it is computed from the posterior moments of the segment-level log-rate as a numerical approximation to the data-versus-prior precision ratio. For conjugate models (Normal-Normal), this equals Z_i exactly. For Poisson-lognormal hierarchical models, the model is not conjugate, so this is a numerical approximation - not the exact posterior precision. For segments with very thin data (Z < 0.10), the Bayesian credibility_factor and the Bühlmann-Straub Z may diverge by more than the Normal approximation alone would suggest. Do not treat them as equivalent outputs for thin groups.

`lower_90` and `upper_90` are the 5th and 95th percentiles of the posterior predictive distribution for each segment's underlying rate. These are what Bühlmann-Straub cannot give you - a full uncertainty band on each segment's rate.

### Convergence diagnostics

Do not use results without checking convergence. The three standard checks, plus a separate check for variance components:

```python
import arviz as az

trace = hf.posteriors

# R-hat: should be < 1.01 for all parameters
rhat = az.rhat(trace)
max_rhat = float(rhat.max().to_array().max())
print(f"Max R-hat: {max_rhat:.3f}  ({'OK' if max_rhat < 1.01 else 'INVESTIGATE'})")

# Effective sample size: should be > 400 for all parameters
ess_bulk = az.ess(trace, method="bulk")
min_ess = float(ess_bulk.min().to_array().min())
print(f"Min ESS (bulk): {min_ess:.0f}  ({'OK' if min_ess > 400 else 'INVESTIGATE'})")

# Variance components need higher ESS - they drive the credibility factors for
# all groups. Underpowered variance component estimation is the most common reason
# Bayesian credibility factors are wrong without the model appearing to fail.
sigma_ess = float(ess_bulk["sigma_postcode_district"].min())
print(f"sigma ESS: {sigma_ess:.0f}  ({'OK' if sigma_ess > 1000 else 'INVESTIGATE - increase n_samples'})")

# Divergences: ideally 0
n_div = int(trace.sample_stats["diverging"].sum())
print(f"Divergences: {n_div}  ({'OK' if n_div == 0 else 'INVESTIGATE'})")
```

If R-hat > 1.01 or ESS < 400: increase the number of samples, check your parameterization, or consider whether the model is misspecified (e.g., too many random effect dimensions for the data available). If you have divergences despite non-centered parameterization: try increasing `target_accept` to 0.95. Persistent divergences after that suggest the model has areas of genuine posterior pathology - typically a random effect variance that is poorly identified from the data.

---

## The shrinkage plot

This is the chart that earns credibility modelling its budget. Plot observed segment rates against posterior means (or Bühlmann-Straub credibility estimates). What you should see:

- Thin segments (low exposure) pulled strongly toward the portfolio mean - the vertical extent of these points compresses toward the horizontal line at mu
- Dense segments (high exposure) with estimates close to their observed rates - these points sit near the 45-degree line
- No segment at an extreme rate with its posterior mean also at that extreme, unless its exposure justifies it

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8))

# Size points by log exposure
log_exposure = np.log1p(results["exposure"])
sizes = 20 + 80 * (log_exposure - log_exposure.min()) / (log_exposure.max() - log_exposure.min())

sc = ax.scatter(
    results["observed_rate"],
    results["posterior_mean"],
    s=sizes,
    alpha=0.6,
    c=results["credibility_factor"],
    cmap="RdYlGn",
    edgecolors="none",
)

# Reference lines
xlim = ax.get_xlim()
ax.plot(xlim, xlim, "k--", alpha=0.3, label="No shrinkage (observed = estimate)")
ax.axhline(hf.grand_mean_, color="grey", linestyle=":", alpha=0.5, label=f"Grand mean = {hf.grand_mean_:.3f}")

plt.colorbar(sc, label="Credibility factor Z")
ax.set_xlabel("Observed claim frequency")
ax.set_ylabel("Posterior mean claim frequency")
ax.set_title("Shrinkage plot: credibility weighting effect\n(point size = log exposure, colour = Z)")
ax.legend()
plt.tight_layout()
display(fig)
```

The colour scale tells the story immediately. Green points (high Z) are near the 45-degree line - they have enough data to trust their own experience. Red points (low Z) are pulled hard toward the horizontal line at the grand mean. The systematic pattern - thin cells pulled toward the mean, dense cells left where they are - is what the pricing committee needs to see.

This plot is more persuasive than any mathematical derivation when you are explaining why the model has revised KT's rate from 1.30% to 4.55%.

---

## Credibility factors from Bayesian posteriors

One of `HierarchicalFrequency`'s outputs is `credibility_factor` - the Bayesian equivalent of Z. Understanding how this maps to classical Bühlmann-Straub Z is important for communication: your pricing committee may know what a credibility factor is, and showing that the Bayesian model produces the same number for the same information should increase their confidence in the approach.

For conjugate models (Normal-Normal), the mapping is exact. For Poisson models with log-normal random effects, the mapping is approximate - the `bayesian-pricing` library computes credibility_factor as a numerical approximation from the posterior moments:

```
Z_i^{Bayes} ≈ posterior_precision_data / (posterior_precision_data + prior_precision)
```

where precisions are computed from the posterior moments of the segment-level log-rate. This approximation is close for well-exposed segments. For very thin groups (Z < 0.10), where the posterior is dominated by the prior and the Normal approximation to the Poisson posterior is least accurate, the Bayesian Z and the B-S Z may diverge for reasons beyond the Normal approximation alone. Do not treat them as interchangeable for thin-group reporting.

To compare:

```python
import polars as pl

# hf.results comes from bayesian-pricing library (pandas DataFrame)
# bs.results comes from credibility library (pandas DataFrame)
# Bridge both to Polars for the join
results_pl = pl.from_pandas(hf.results)
bs_results_pl = pl.from_pandas(
    bs.results[["group", "Z", "credibility_estimate"]]
    .rename(columns={"group": "postcode_district", "Z": "Z_bs",
                     "credibility_estimate": "bs_estimate"})
)

comparison = results_pl.join(bs_results_pl, on="postcode_district", how="inner")

print(comparison.select([
    "postcode_district", "earned_years",
    "Z_bs", "credibility_factor",
    "bs_estimate", "posterior_mean",
]).head(20))
```

You will generally find close agreement on Z values. Differences arise primarily for segments with very low exposure (where the Normal approximation to the Poisson posterior is least accurate) and segments with observed rates far from the portfolio mean (where the log-normal random effect distribution pulls differently from the Normal distribution in the B-S model).

The Bayesian model will produce wider uncertainty intervals for thin segments, because it properly propagates the uncertainty in the variance components (sigma_area, sigma_veh) rather than plugging in point estimates.

---

## When to use which

This decision framework assumes you have already decided that naive observed rates are not appropriate - that some form of partial pooling is needed.

**Use Bühlmann-Straub when:**

- You have one grouping variable (schemes, vehicle classes, postcode districts in isolation)
- You have many groups (at least 5, ideally 20+) - enough to estimate a and v reliably
- Speed matters: you need results in seconds, not minutes
- Your audience is familiar with credibility factors and you want transparent, auditable parameters (v, a, K)
- Your model is downstream of a GLM or GBM that already handles the main effects - you are applying credibility to residuals or segment-level departures
- Regulatory transparency is paramount: Bühlmann-Straub has a 55-year track record in actuarial methodology documentation

The FCA Consumer Duty documentation can cite Bühlmann & Straub (1970) and explain Z without any specialist software. That matters - a documented, auditable methodology is substantially stronger evidence than an undocumented one, even if the undocumented approach uses more sophisticated machinery. Note that an auditable methodology is a necessary but not sufficient condition for Consumer Duty compliance: you will also need to demonstrate calibration and check that credibility adjustments do not correlate with FCA-sensitive variables (e.g., do the districts with the lowest Z values - where we revert most to the portfolio mean - also coincide with areas of high deprivation?).

**Use full Bayesian (HierarchicalFrequency) when:**

- You have multiple crossed grouping variables - driver age AND vehicle group AND area - and you want partial pooling simultaneously on all dimensions
- You have few groups (fewer than 10 schemes) and the uncertainty in structural parameter estimates is material
- You need credible intervals on individual segment rates, not just point estimates - for thin-data pricing decisions or regulatory evidence of fair value
- You are modelling a two-level geographic hierarchy (sector within district within area) with genuine nesting structure
- You want to propagate uncertainty from the credibility stage into downstream calculations (pricing decision, underwriting appetite)

**The case where neither is sufficient:**

GBMs with non-linear interactions across dozens of features, where the interaction structure matters as much as the main effects. The two-stage approach handles this: GBM on the full dataset for main effects + credibility (B-S or Bayesian) on segment-level GBM residuals. The GBM prediction becomes the prior mean; the credibility model estimates departures from it with appropriate pooling.

**On exchangeability.** Both methods assume that group effects are exchangeable - drawn from the same prior. This assumption is more defensible for some groupings than others. Vehicle groups with structurally different risk profiles (commercial vehicles vs private cars) are not exchangeable with each other; you should stratify the analysis rather than pool them. Inner London postcode districts are partially exchangeable with rural Scottish ones (both are draws from the same UK motor market prior) but the correlation structure violates strict independence. Use the nested model (Exercise 2) to handle the most egregious non-exchangeability in geographic data.

---

## Databricks deployment

### PyMC on Databricks

PyMC 5.x runs on the standard Databricks ML runtime (DBR 14.x or later). Install in your notebook:

```python
%pip install "bayesian-pricing[all] @ git+https://github.com/burningcost/bayesian-pricing.git" arviz --quiet
dbutils.library.restartPython()
```

The `[all]` extra installs PyMC, PyTensor, and ArviZ. The first import of PyMC compiles PyTensor graphs, which takes 30–60 seconds. Subsequent cells are faster.

For large segment counts, set the number of MCMC cores to match your cluster:

```python
import pymc as pm

# Check available cores
import multiprocessing
n_cores = multiprocessing.cpu_count()
print(f"Available cores: {n_cores}")

# HierarchicalFrequency uses cores argument
hf = HierarchicalFrequency(
    group_cols=["postcode_district"],
    claims_col="claims",
    exposure_col="exposure",
    n_chains=min(4, n_cores),
    n_samples=1000,
    n_warmup=1000,
)
```

NUTS chains run in parallel. On a 4-core single-node cluster, 4 chains each running 2,000 iterations takes roughly 4x the time of a single chain. With a Databricks cluster having 8 or more cores, you can run 4 chains in parallel comfortably.

### MLflow tracking of posteriors

Every `HierarchicalFrequency` fit should be tracked in MLflow. The key artefacts:

```python
import mlflow
from bayesian_pricing import HierarchicalFrequency

mlflow.set_experiment("/pricing/credibility-bayesian/module06")

with mlflow.start_run(run_name="hierarchical_frequency_v1"):
    hf = HierarchicalFrequency(
        group_cols=["postcode_district"],
        claims_col="claims",
        exposure_col="exposure",
    )
    hf.fit(segments.to_pandas())

    # Log convergence diagnostics
    import arviz as az
    rhat = az.rhat(hf.posteriors)
    max_rhat = float(rhat.max().to_array().max())
    n_div = int(hf.posteriors.sample_stats["diverging"].sum())

    mlflow.log_metric("max_rhat", max_rhat)
    mlflow.log_metric("n_divergences", n_div)
    mlflow.log_metric("n_segments", len(segments))
    mlflow.log_metric("grand_mean", hf.grand_mean_)

    # Log variance components (heterogeneity estimates)
    for dim in hf.group_cols:
        sigma_name = f"sigma_{dim.replace(' ', '_')}"
        if sigma_name in hf.posteriors.posterior:
            sigma_mean = float(hf.posteriors.posterior[sigma_name].mean())
            mlflow.log_metric(f"sigma_{dim}", sigma_mean)

    # Log the posterior as an ArviZ netCDF artefact
    hf.posteriors.to_netcdf("/tmp/posteriors.nc")
    mlflow.log_artifact("/tmp/posteriors.nc", "posteriors")

    # Log the results table (bridge to pandas for CSV export)
    results_path = "/tmp/credibility_results.csv"
    hf.results.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path, "results")
```

The netCDF posterior artefact means you can reload the full posterior at any time without re-running MCMC. This is important: MCMC takes minutes; re-reading from disk takes seconds. Store it.

### Unity Catalog for credibility-weighted estimates

Credibility-weighted factor tables should live in Unity Catalog with the same discipline as GLM or GBM relativities:

```python
import polars as pl
from datetime import date
import json

RUN_DATE = str(date.today())
MODEL_NAME = "hierarchical_freq_v1_module06"

# Hard gate: do not write biased posteriors to Unity Catalog
if max_rhat > 1.01 or n_div > 0:
    raise ValueError(
        f"Model failed convergence: max_rhat={max_rhat:.4f}, divergences={n_div}. "
        "Results not written to Unity Catalog."
    )

# Results with metadata - bridge hf.results (pandas) to Polars
results_out = (
    pl.from_pandas(hf.results)
    .with_columns([
        pl.lit(MODEL_NAME).alias("model_name"),
        pl.lit("hierarchical_poisson").alias("model_type"),
        pl.lit(RUN_DATE).alias("run_date"),
        pl.lit(max_rhat).alias("max_rhat"),
        pl.lit(n_div).alias("n_divergences"),
    ])
)

(
    spark.createDataFrame(results_out.to_pandas())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module06_credibility_estimates")
)

print(f"Written {len(results_out)} Bayesian estimates to main.pricing.module06_credibility_estimates")
```

Run this as a scheduled Databricks Workflow after each data refresh. The convergence gate ensures a model that fails silently - divergences indicate regions of the posterior that were not explored - cannot write bad estimates downstream.

---

## Limitations

**Bühlmann-Straub assumes additive structure in loss rates.** For multiplicative models (Poisson log-link), applying B-S in rate space introduces a small bias via the Jensen's inequality gap. Use `log_transform=True` as the default in the main fitting call (shown above in the Using the credibility library section). For most UK motor portfolios, the bias is small; for extreme relativities (thin cells with observed rates far from the mean), the log-transform matters.

**MCMC is slow and requires diagnostic skill.** A pricing team accustomed to GLMs that run in under a minute will find Bayesian MCMC unsettling. R-hat diagnostics, divergence checks, and ESS requirements are all new concepts. Budget time for this learning curve, and budget compute for exploratory runs.

**The Poisson likelihood assumes observed variance equals expected variance.** If your claim count data are overdispersed (variance > mean) - which is common in insurance due to unobserved risk factors - the Negative Binomial is more appropriate. `HierarchicalFrequency` supports `likelihood="negative_binomial"` for this case. The additional overdispersion parameter is estimated as a global hyperparameter.

If posterior predictive checks (PPCs) show poor calibration - for example, district-level 90% PPC coverage is 75% rather than the expected 90% - the model is misspecified. The options are: switch to Negative Binomial, revisit the random effect prior (widen sigma if the model is over-shrinking extreme districts), or, in cases of severe misspecification, revert to Bühlmann-Straub and document why the Bayesian model did not fit. A warning message in the notebook is not enough; include a decision rule.

**Hierarchical models with few groups are poorly identified at the top level.** If you have 6 postcode areas in your model (Area A through F), the estimate of sigma_area - the between-area variance - is itself highly uncertain. The Bayesian posterior for sigma_area will be wide, and this uncertainty flows into the segment-level credibility factors. This is actually correct behaviour - with 6 groups, you genuinely do not know much about between-group heterogeneity - but it can be uncomfortable for stakeholders expecting precise answers. Report the posterior for sigma_area, not just the sigma_area point estimate.

**Credibility estimates are an input to the pricing decision, not the decision itself.** In production use, credibility-weighted district relativities are typically capped and floored before they enter the rating structure - for example, district relativities cap at 1.5x the portfolio mean and floor at 0.7x. This prevents any single district driving a loss-making or uncompetitive premium, regardless of what the model says. If you implement capping and flooring, document it explicitly: Consumer Duty requires that the methodology choices - including constraints on outputs - are recorded and can be explained.

**Posterior predictive validation is mandatory.** A hierarchical model that converges in MCMC can still be misspecified. Run posterior predictive checks (simulate datasets from the posterior and compare to observed data) before presenting results. `HierarchicalFrequency` provides `posterior_predictive_check()` as a method.
