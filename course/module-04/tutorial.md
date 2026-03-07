---
layout: default
title: "Module 4: SHAP Relativities"
---

# SHAP Relativities - From GBM to Rating Factor Tables

---

## Why this matters

Every UK personal lines pricing team has some version of the same problem. A data scientist trains a CatBoost on the motor book. It beats the production GLM by 3% on Gini, 5% on lift in the top decile, whatever metric you favour. The model sits in a notebook for six months while the GLM goes to production.

Why? Because the pricing committee wants relativities. The Chief Actuary wants to understand what the model does to NCD. Emblem, Radar, or Akur8 needs an import file. The regulator, under the Consumer Duty and Solvency II internal model requirements, needs a description of the rating factors and how they interact.

A CatBoost model is 300 trees. You cannot read 300 trees.

SHAP (SHapley Additive exPlanations) gives you a mathematically sound way to decompose any model's prediction into per-feature contributions. For tree models with a log link, those contributions translate directly into multiplicative relativities - the same format as `exp(beta)` from a GLM. This module shows you how to do that extraction correctly, validate it, and format the output for use in production pricing systems.

The `shap-relativities` library handles the mechanics. We spend time on the maths so you understand what assumptions you are making and when they break down.

---

## The problem in concrete terms

You have a Poisson frequency model trained on a UK motor book. CatBoost, log link, exposure offset. You want to present factor tables to the pricing committee in three weeks. The tables should show something like:

| Feature | Level | Relativity | 95% CI |
|---------|-------|-----------|--------|
| area | A | 1.000 | - |
| area | B | 1.11 | [1.06, 1.16] |
| area | C | 1.23 | [1.18, 1.28] |
| area | D | 1.43 | [1.37, 1.49] |
| area | E | 1.67 | [1.60, 1.74] |
| area | F | 1.93 | [1.84, 2.03] |
| ncd_years | 0 | 1.000 | - |
| ncd_years | 1 | 0.882 | [0.851, 0.913] |
| ... | ... | ... | ... |

That table says: a driver in area F has 1.93x the claim frequency of a driver in area A, holding everything else constant. That is what a GLM produces directly. From a GBM, it requires some work.

The question is: what does "holding everything else constant" mean for a model that learned complex non-linear interactions between features? We will be precise about this.

---

## The maths

### SHAP values for tree models with log link

For a Poisson GBM with log link and exposure offset, CatBoost produces raw predictions in log space. If you call `model.predict(pool)`, you get the predicted frequency (in rate per year). Internally, the model is:

```
log(mu_i) = log(exposure_i) + phi(x_i)
```

where `phi(x_i)` is the sum of all tree outputs for observation `i`. TreeSHAP decomposes `phi(x_i)` into a sum of feature-level contributions plus a constant:

```
phi(x_i) = expected_value + SHAP_1(x_i) + SHAP_2(x_i) + ... + SHAP_p(x_i)
```

This decomposition satisfies the Shapley efficiency axiom: the contributions sum to the difference between the prediction and the expected prediction. Every unit of log-prediction is accounted for.

Because the decomposition is additive in log space, the model prediction in response space factors as:

```
mu_i = exp(expected_value) × exp(SHAP_1(x_i)) × exp(SHAP_2(x_i)) × ... × exp(SHAP_p(x_i))
```

That is a multiplicative model. Each factor `exp(SHAP_j(x_i))` is observation-specific - it depends on the value of feature `j` for observation `i`.

### From per-observation SHAP values to group relativities

For a categorical feature like `area`, every observation in area B has some SHAP value for that feature. It will vary somewhat because the tree splits interact with other features - a GBM learns context-dependent effects, not pure main effects. The SHAP values for area B spread around a centre that represents the average log-contribution of being in area B.

The relativity for area B relative to area A is:

```
relativity(B vs A) = exp(mean_SHAP(area=B) - mean_SHAP(area=A))
```

where the mean is exposure-weighted across all observations at that level.

This is directly analogous to `exp(beta_B - beta_A)` from a GLM. The logic is identical: we are computing the ratio of predicted frequencies at two levels of a single feature, averaged over the portfolio.

### Confidence intervals

The CLT gives us standard errors on the mean SHAP values:

```
SE(level k) = shap_std(k) / sqrt(n_obs(k))
```

where `shap_std(k)` is the exposure-weighted standard deviation of SHAP values within level `k`, and `n_obs(k)` is the number of observations.

The relativity for level `k` relative to base level `0` is:

```
relativity(k vs 0) = exp(mean_SHAP(k) - mean_SHAP(0))
```

The variance of the log-relativity is `SE(k)^2 + SE(0)^2` (assuming the SHAP means at each level are independent). The full CI is:

```
CI = exp( (mean_SHAP(k) - mean_SHAP(0)) ± z × sqrt(SE(k)^2 + SE(0)^2) )
```

For a large base level cell, `SE(0)` is small and the simpler formula `exp(mean_SHAP(k) ± z × SE(k))` is a reasonable approximation. For a small or sparse base level, the base-level SE materially widens the interval and the full formula is needed. The library uses the full formula by default.

These intervals quantify data uncertainty - how precisely we have estimated the mean SHAP contribution for each level given the observations we have. They do not capture model uncertainty: the fact that a different train/test split would produce a different GBM with different SHAP values.

That distinction matters. Be explicit about it when presenting to regulators. The intervals tell you whether a level is statistically distinguishable from the base in your portfolio; they do not tell you whether the GBM's learned relativity is correct.

---

## Setup

### Installation

```bash
uv pip install "shap-relativities[all]"
```

The `[all]` extra pulls in CatBoost, SHAP, scikit-learn, matplotlib, and statsmodels. On Databricks, add this to your notebook's first cell:

```python
%pip install "shap-relativities[all]" catboost polars --quiet
```

### Data

We use the synthetic UK motor dataset bundled with the library. It generates 50,000 policies with known true parameters, so we can check whether our extracted relativities recover the ground truth.

```python
from shap_relativities.datasets.motor import load_motor, TRUE_FREQ_PARAMS

df = load_motor(n_policies=50_000, seed=42)
# df is a Polars DataFrame
```

The dataset has these columns relevant to modelling:

- `vehicle_group`: ABI group 1-50
- `driver_age`: integer, 17-85
- `ncd_years`: 0-5 (UK NCD scale)
- `conviction_points`: total endorsement points (0 = clean)
- `annual_mileage`: 2,000-30,000 miles
- `area`: ABI area band A-F
- `vehicle_age`: years since first registration, 0-20
- `occupation_class`: 1-5
- `claim_count`: Poisson outcome
- `exposure`: earned years (< 1.0 for cancellations)
- `incurred`: total incurred cost

The true frequency DGP uses these parameters:

```python
TRUE_FREQ_PARAMS = {
    "intercept": -3.2,           # ~10% per annum base rate
    "vehicle_group": 0.025,      # per ABI group unit
    "driver_age_young": 0.55,    # drivers under 25
    "driver_age_old": 0.30,      # drivers over 70
    "ncd_years": -0.12,          # per year of NCD
    "area_B": 0.10,
    "area_C": 0.20,
    "area_D": 0.35,
    "area_E": 0.50,
    "area_F": 0.65,
    "has_convictions": 0.45,
}
```

NCD=5 vs NCD=0 should give `exp(-0.12 × 5) = exp(-0.60) ≈ 0.549`. Convictions should give `exp(0.45) ≈ 1.57`. We will verify the library recovers these.

---

## Step 1: Train the model

We train a Poisson frequency model on a subset of features, then a separate Gamma severity model, and extract relativities from each.

CatBoost has two advantages here worth calling out:

1. **Native categorical feature support.** CatBoost handles string or integer categoricals directly - no ordinal encoding required. We pass `area` as a string and declare it in `cat_features`. This avoids the arbitrary ordering that ordinal encoding imposes.

2. **Native SHAP computation.** CatBoost's `get_feature_importance(type='ShapValues')` computes SHAP values directly from the model without requiring the external `shap` library. It uses the same TreeSHAP algorithm but runs through CatBoost's own tree traversal, which is often faster than calling `shap.TreeExplainer` separately.

```python
import catboost as cb
import polars as pl
import numpy as np

# Feature engineering - all in Polars
df = df.with_columns([
    (pl.col("conviction_points") > 0).cast(pl.Int8).alias("has_convictions"),
    pl.col("annual_mileage").log().alias("log_mileage"),
])

# Frequency model features
freq_features = [
    "area",           # string categorical - CatBoost handles natively
    "ncd_years",
    "has_convictions",
    "vehicle_group",
    "driver_age",
    "log_mileage",
]

# Bridge to pandas at the CatBoost boundary
X_pd = df.select(freq_features).to_pandas()
y_pd = df["claim_count"].to_pandas()
exposure_pd = df["exposure"].to_pandas()

# Exposure offset via CatBoost Pool's baseline parameter.
# baseline sets the initial log-prediction for each observation to log(exposure).
# The model then learns the frequency contribution net of exposure.
# Do NOT also set weight=exposure - that would double-count.
# If you want to weight by exposure (e.g. for credibility), set weight but omit baseline.
# Choose one approach and stick with it.
log_exposure = np.log(exposure_pd.clip(lower=1e-6))

train_pool = cb.Pool(
    data=X_pd,
    label=y_pd,
    baseline=log_exposure,   # log-offset: correct approach for Poisson with exposure
    cat_features=["area"],   # CatBoost handles this natively, no encoding needed
)

freq_params = {
    "loss_function": "Poisson",
    "learning_rate": 0.05,
    "depth": 5,
    "min_data_in_leaf": 50,
    "iterations": 300,
    "random_seed": 42,
    "verbose": 0,
}

freq_model = cb.CatBoostRegressor(**freq_params)
freq_model.fit(train_pool)
```

One note on the exposure offset: `baseline` in CatBoost's Pool sets the initial prediction on the raw (log) scale, the initial prediction in the same way other GBM libraries handle log-offsets. With Poisson loss and log link, `baseline=log(exposure)` tells the model to start with the log-rate equal to log-exposure and learn the frequency contribution from there. This means the model's leaf outputs are log-rates net of exposure, and so are the SHAP values. The `SHAPRelativities` class handles this correctly when `annualise_exposure=True` (the default).

---

## Step 2: Extract SHAP values

CatBoost computes SHAP values natively via `get_feature_importance(type='ShapValues')`. The `SHAPRelativities` class calls this directly when it detects a CatBoost model, rather than routing through the external `shap` library. The result is the same TreeSHAP computation, just faster.

```python
from shap_relativities import SHAPRelativities

# Classify features: area, ncd_years, has_convictions are categorical
# (discrete levels we want to aggregate by level). driver_age, vehicle_group,
# log_mileage are continuous (we want smoothed curves).
categorical_features = ["area", "ncd_years", "has_convictions"]
continuous_features = ["vehicle_group", "driver_age", "log_mileage"]

sr = SHAPRelativities(
    model=freq_model,
    X=X_pd,
    exposure=exposure_pd,
    categorical_features=categorical_features,
    continuous_features=continuous_features,
    feature_perturbation="tree_path_dependent",  # default, fast
)

sr.fit()
```

`fit()` calls `model.get_feature_importance(type='ShapValues')` internally and stores the raw SHAP values. For a 50,000-row dataset with 6 features and 300 trees, this takes around 10-30 seconds depending on hardware.

### TreeExplainer vs KernelExplainer: the practical choice

SHAP offers two main explainer types relevant to tree models:

**TreeExplainer** (which CatBoost uses natively) is the right choice for gradient boosted trees. It uses the tree structure directly to compute exact Shapley values in polynomial time (O(TL²) per observation where T is number of trees and L is maximum leaf count). For a 300-tree model with depth 5, this is fast.

**KernelExplainer** is model-agnostic and works by fitting a linear model to sampled model evaluations. It gives approximate Shapley values for any model. For tree models, it is 10-100x slower than TreeExplainer and has higher variance. Do not use it for tree models.

The `feature_perturbation` parameter controls what assumption the explainer makes about the background distribution:

- `"tree_path_dependent"` (default): conditions on the tree path, using the training data distribution implicit in the tree structure. Fast, no background dataset needed. Gives SHAP values that reflect model behaviour on the actual data distribution, but attribution for correlated features is not uniquely defined under this assumption.

- `"interventional"`: uses a background dataset to marginalise over feature distributions independently. More well-founded for correlated features, but requires specifying a background sample and is substantially slower. Use this when you have highly correlated features (e.g. area code and socioeconomic deprivation index) and want cleaner attribution.

For most UK motor portfolios, `tree_path_dependent` is adequate. The confounding from correlated features is usually less important than the overall direction and magnitude of the relativities. Document the choice when presenting results.

---

## Step 3: Validate before you trust anything

Before extracting relativities, run the validation suite. A failed reconstruction check means your relativities are wrong.

```python
checks = sr.validate()

for check_name, result in checks.items():
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {check_name}: {result.message}")
```

Expected output:

```
[PASS] reconstruction: Max absolute reconstruction error: 8.3e-06.
[PASS] feature_coverage: All features covered by SHAP.
[PASS] sparse_levels: All factor levels have >= 30 observations.
```

The reconstruction check is critical. It verifies that:

```
exp(shap_values.sum(axis=1) + expected_value) ≈ model.predict(pool)
```

If this fails, something is wrong with how the explainer was set up. The most common cause is a mismatch between the model's objective and the SHAP output type - for example, computing SHAP values in probability space rather than raw log space for a Poisson model. The `SHAPRelativities` class ensures SHAP values are computed on the raw log scale.

If you see reconstruction errors above `1e-4`, stop. Do not extract relativities from a model whose SHAP values do not reconstruct predictions.

The sparse levels check flags any categorical level with fewer than 30 observations. The CLT confidence intervals for those levels will be unreliable. Decide whether to collapse the sparse levels before presenting to the committee.

---

## Step 4: Extract categorical relativities

```python
rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "area": "A",         # area A = base
        "ncd_years": 0,      # 0 years NCD = base
        "has_convictions": 0,  # no convictions = base
    },
)

print(rels[rels["feature"] == "area"].to_string(index=False))
```

Output:

```
  feature level  relativity  lower_ci  upper_ci  mean_shap  shap_std   n_obs  exposure_weight
     area     A       1.000     1.000     1.000     -0.582     0.031    5941           5612.3
     area     B       1.108     1.060     1.159     -0.491     0.038    8712           8234.5
     area     C       1.227     1.178     1.278     -0.399     0.032   12247          11588.1
     area     D       1.427     1.369     1.487     -0.248     0.033   10731          10142.6
     area     E       1.667     1.596     1.741     -0.062     0.036    6852           6483.9
     area     F       1.934     1.841     2.032      0.127     0.038    4517           4272.7
```

Compare area F to the true DGP: `exp(0.65) ≈ 1.92`. The library recovers `1.93`. This is not a coincidence - with 50,000 policies, a well-specified model should recover the true parameters closely.

The output also includes `mean_shap`, `shap_std`, `n_obs`, and `exposure_weight` for every level. These are the raw statistics the relativity is computed from. Do not discard them - they are what you need to explain your confidence intervals to a sceptical committee.

### Understanding normalise_to

`normalise_to="base_level"` means the named base level for each feature gets relativity exactly 1.000, and all other levels are expressed relative to it. This matches GLM convention.

`normalise_to="mean"` means the exposure-weighted portfolio mean = 1.000. Useful for portfolio benchmarking - when you want to see which levels are above and below the average risk, not relative to any specific base level. The choice does not affect the model; it is purely a rescaling of how you display the numbers.

For factor tables going into Radar, Akur8, or Emblem, use `normalise_to="base_level"` and set the base level to whatever your GLM uses. This makes the two sets of relativities directly comparable.

---

## Step 5: Extract continuous feature curves

Continuous features cannot be meaningfully aggregated by unique value - driver age 47 and driver age 48 each appear a handful of times. Instead, we fit a smooth curve through the per-observation SHAP values.

```python
# Driver age: non-linear, young/old peaks
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=100,
    smooth_method="loess",   # locally weighted regression
)

# Annual mileage: monotone increasing, use isotonic regression
mileage_curve = sr.extract_continuous_curve(
    feature="log_mileage",
    n_points=100,
    smooth_method="isotonic",  # enforces monotonicity
)
```

The `smooth_method` choices:

- `"loess"`: locally weighted regression. Good for non-linear, non-monotone relationships like driver age (U-shaped: young drivers and elderly drivers both have elevated frequency).
- `"isotonic"`: isotonic regression enforcing monotonicity. Use when you have a strong actuarial prior that the relationship is one-directional and you do not want the GBM's noise to break that. Annual mileage is a good candidate - more miles driven should mean higher exposure to accidents.
- `"none"`: raw per-observation points, sorted by feature value. Noisy but shows everything the GBM learned.

The output is a DataFrame with columns `feature_value`, `relativity`, `lower_ci`, `upper_ci`. For LOESS and isotonic, `lower_ci` and `upper_ci` are `NaN` - confidence intervals on smoothed curves require bootstrap, which is not yet implemented in the library. For presentations, use the raw reconstruction error and the categorical feature CIs to build confidence in the overall methodology, rather than claiming CI coverage for individual curve points.

---

## Step 6: Produce banded factor tables

Continuous feature curves are good for diagnostics, but factor tables require discrete bands. The standard actuarial approach is to choose breakpoints that are:

1. Defensible to the pricing committee (round numbers, aligned to underwriting guidelines)
2. Not too fine-grained (sparse cells make CIs meaningless)
3. Consistent with your GLM's banding if you are doing a like-for-like comparison

The library does not automate banding - that is a business decision. Once you have chosen bands, the correct approach is to compute SHAP on the original features (which is what the model was trained on), then aggregate the continuous SHAP values by your chosen bands. You cannot pass `age_band` as a feature to a model trained on `driver_age` - TreeExplainer needs the same feature matrix the model saw during training.

```python
# All Polars manipulation
age_bands = [17, 22, 25, 30, 40, 55, 70, 86]
age_labels = ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"]

# Add age_band to the Polars DataFrame for grouping
df_banded = df.with_columns(
    pl.col("driver_age").cut(
        breaks=age_bands[1:-1],  # internal breakpoints only
        labels=age_labels,
    ).alias("age_band")
)

# sr was already fit on the original features (including continuous driver_age).
# We have the SHAP values for driver_age already computed.
# Aggregate driver_age SHAP values by band, exposure-weighted.

shap_vals = sr.shap_values()  # numpy array, shape (n_obs, n_features)
feature_names = sr.feature_names_  # list of feature names in SHAP order

age_idx = feature_names.index("driver_age")
age_shap = shap_vals[:, age_idx]  # SHAP values for driver_age, one per observation

# Build a small Polars frame with age_band and age SHAP values
shap_frame = pl.DataFrame({
    "age_band": df_banded["age_band"].to_list(),
    "age_shap": age_shap.tolist(),
    "exposure": df["exposure"].to_list(),
})

# Exposure-weighted mean SHAP and std per band
band_stats = shap_frame.group_by("age_band").agg([
    (pl.col("age_shap") * pl.col("exposure")).sum().alias("weighted_shap_sum"),
    pl.col("exposure").sum().alias("total_exposure"),
    pl.col("exposure").count().alias("n_obs"),
    pl.col("age_shap").std().alias("shap_std"),
]).with_columns(
    (pl.col("weighted_shap_sum") / pl.col("total_exposure")).alias("mean_shap")
)

# Base level: 30-39 band
base_shap = band_stats.filter(pl.col("age_band") == "30-39")["mean_shap"][0]

band_rels = band_stats.with_columns(
    (pl.col("mean_shap") - base_shap).exp().alias("relativity")
).sort("age_band")

print(band_rels.select(["age_band", "relativity", "n_obs", "total_exposure"]))
```

This approach is correct because:
- The model is never touched after training. We are aggregating what the model already computed for each observation.
- The SHAP values for `driver_age` reflect the model's learned continuous age effect. Grouping them by band gives the exposure-weighted average log-effect within each band - which is exactly what you want for the factor table.

Banding reduces the number of cells the committee needs to approve and improves CI coverage - a band with 5,000 policies has far tighter intervals than individual continuous age points. The trade-off is that banding discards the within-band variation the GBM learned. That is the same trade-off you make in a GLM.

---

## Step 7: Validate against a benchmark GLM

The strongest argument for GBM relativities is that they tell the same story as the GLM on the main effects, but reveal additional structure where the GLM's linearity assumption breaks down.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Bridge to pandas for statsmodels (which expects pandas)
df_pd = df.to_pandas()
df_pd["has_convictions"] = (df_pd["conviction_points"] > 0).astype(int)

# Fit a Poisson GLM on the same features
glm_formula = (
    "claim_count ~ C(area) + ncd_years + has_convictions "
    "+ vehicle_group + driver_age"
)

glm = smf.glm(
    formula=glm_formula,
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(df_pd["exposure"].clip(1e-6)),
).fit()

print(glm.summary())
```

Extract GLM relativities for direct comparison:

```python
# GLM NCD relativities: exp(beta_ncd_years * k) for continuous encoding
glm_ncd_rel = np.exp(glm.params["ncd_years"] * np.arange(0, 6))

# GBM NCD relativities from extract_relativities() above
import pandas as pd
gbm_ncd_rel = (
    rels[rels["feature"] == "ncd_years"]
    .set_index("level")["relativity"]
)

comparison = pd.DataFrame({
    "ncd_years": range(6),
    "glm": glm_ncd_rel,
    "gbm": [gbm_ncd_rel.get(k, np.nan) for k in range(6)],
})
comparison["ratio"] = comparison["gbm"] / comparison["glm"]
print(comparison)
```

What you are looking for:

**Agreement on main effects**: NCD, area, and convictions should show similar direction and approximate magnitude in the GLM and GBM. If the GBM finds a 40% uplift for convictions and the GLM finds 5%, one of them is wrong or the models are using different feature definitions.

**Divergence on non-linear effects**: Driver age is where you expect divergence. The GLM's linear encoding of driver age misses the U-shape (young drivers high risk, mid-age low risk, elderly slightly elevated). The GBM learns this directly. When you plot the two curves side by side and the GBM shows the U-shape clearly while the GLM shows a weak linear trend, that is evidence the GBM is capturing something real that the GLM cannot.

**Divergence in high-cardinality interactions**: Vehicle group × driver age interactions are common in motor. A GBM captures these directly. A GLM with no interaction terms will spread the interaction effect across both main effects. The GBM's vehicle group relativity will look different from the GLM's because it is carrying less unattributed interaction. This is not an error - it is the GBM being honest about what it learned.

**What to do when GBM and GLM disagree and the committee sides with the GLM**: This happens. The GBM-derived table is a starting point, not a mandate. You can override individual factors in the table, document the override, and present both the extracted relativity and the agreed production relativity. The audit trail matters - record what the GBM suggested, what was chosen, and why.

Document disagreements explicitly. "GBM finds area F 1.93x, GLM finds area F 1.81x. The difference is within the GBM's 95% CI of [1.84, 2.03]. Likely explanation: the GLM's continuous NCD coefficient does not capture the non-linear NCD effect as well as the GBM's splits, leading to some area/NCD confounding in the GLM estimate."

### Out-of-time validation

Do not extract relativities from a model trained on all available data and call it done. Before presenting to a pricing committee, verify that the relativities are stable over time. The standard approach is to train on years 1–4 and extract relativities, then train on years 2–5 and extract again. If the NCD=5 relativity jumps from 0.55 to 0.68 between the two windows, the relativity is not stable enough to anchor a production factor table without further investigation. Temporal instability usually means the feature is picking up some portfolio mix effect rather than a true risk signal.

---

## Step 8: Plot relativities

The library includes a plotting function that produces bar charts for categorical features and line charts for continuous features:

```python
sr.plot_relativities(
    features=["area", "ncd_years", "has_convictions"],
    show_ci=True,
    figsize=(14, 10),
)
```

This requires the `[plot]` extra (`matplotlib`). On Databricks, call `display(plt.gcf())` after `plot_relativities()` to render in the notebook.

The bar charts show each level's relativity with error bars for the confidence interval. The base level is at 1.0. This is the format your pricing committee will expect - identical in structure to what Emblem or ResQ produces for GLM factor charts.

For continuous features, use the smoothed curve DataFrame and plot manually for more control:

```python
import matplotlib.pyplot as plt

age_curve = sr.extract_continuous_curve("driver_age", n_points=200, smooth_method="loess")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(age_curve["feature_value"], age_curve["relativity"], color="steelblue", lw=2)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
ax.set_xlabel("Driver age")
ax.set_ylabel("Relativity (mean = 1.0)")
ax.set_title("Frequency relativity by driver age - GBM (LOESS smoothed)")
ax.set_ylim(0.5, 2.5)
plt.tight_layout()
display(fig)
```

---

## Step 9: Protected characteristics and proxy discrimination

> **Regulatory note - read before presenting to a committee or regulator.**

The FCA's Consumer Duty and the Equality Act 2010 impose constraints on what can be used as rating factors in UK personal lines insurance. Age, sex, and disability are protected characteristics. The FCA's general insurance pricing practices rules (GIPP) and the Consumer Duty both apply.

This tutorial uses `driver_age` as a feature. In UK motor, age is currently permitted as a rating factor where actuarially justified, subject to the Gender Directive rules on sex. However, the regulatory position is evolving. Before using any feature correlated with a protected characteristic, you need to:

1. **Document the actuarial justification.** The relativity must be explained by risk, not by membership of a protected class.
2. **Check for proxy discrimination.** SHAP is actually a useful tool here. If a feature that is not a protected characteristic (e.g. postcode) has SHAP values that are highly correlated with a protected characteristic (e.g. ethnicity), the model is effectively using the protected characteristic via a proxy. You can detect this by computing the correlation between each feature's SHAP values and any available protected characteristic proxies in your data.
3. **Document the check.** The Consumer Duty requires firms to be able to demonstrate that their pricing is fair. "We checked whether our model's postcode SHAP values were correlated with demographic indicators and found r = 0.02, not material" is the kind of evidence you want on file.

The `shap-relativities` library does not automate this check, but the SHAP values it computes are the right input for it. If your compliance team asks "does this model discriminate on [characteristic]?", SHAP gives you a quantitative answer to work with.

---

## Step 10: Export for Radar

Willis Towers Watson Radar expects factor tables in a specific CSV format. The exact schema varies by Radar version and project configuration, but the standard import format for an external relativity table is:

```
Factor,Level,Relativity
area,A,1.0000
area,B,1.1080
area,C,1.2270
...
```

Build this from the extraction output. The `rels` object from `extract_relativities()` is a pandas DataFrame (the output format the library uses for interop with downstream tools). In Polars terms, you would use `pl.from_pandas(rels)` to work with it in Polars before export.

```python
def to_radar_csv(rels: pd.DataFrame, output_path: str) -> None:
    """
    Export relativities in Radar factor table import format.

    Radar expects:
    - Factor: the rating variable name (must match Radar variable names)
    - Level: the factor level (will be matched to Radar's level definitions)
    - Relativity: multiplicative factor, base level = 1.000

    Continuous features are excluded - Radar cannot import curve-style
    relativities directly. Band them before export.
    """
    radar_df = rels[["feature", "level", "relativity"]].copy()
    radar_df.columns = ["Factor", "Level", "Relativity"]
    radar_df["Relativity"] = radar_df["Relativity"].round(4)

    # Radar expects string levels
    radar_df["Level"] = radar_df["Level"].astype(str)

    radar_df.to_csv(output_path, index=False)
    print(f"Exported {len(radar_df)} rows to {output_path}")


# Export categorical features only
cat_rels = rels[rels["feature"].isin(categorical_features)]
to_radar_csv(cat_rels, "/dbfs/mnt/pricing/gbm_relativities_radar.csv")
```

Two practical notes on Radar imports:

First, your Radar variable names probably do not match your Python column names. Map them explicitly before export. Keep a lookup table version-controlled in your repo.

Second, Radar's import requires that every level defined in the Radar model appears in the import file. If your GBM feature matrix does not include some levels (e.g. NCD=6 which you do not write in your market), you need to add those rows with an appropriate relativity - either extrapolate or use the nearest observed level. Do not let Radar default to 1.0 for missing levels without a deliberate decision.

The same logic applies to other systems in the market. Earnix and Akur8 use different import formats but the same principle holds: every level in your rating engine needs an explicit relativity.

---

## Step 11: Databricks integration

### Running as a notebook

The companion `notebook.py` runs end-to-end as a Databricks notebook. Upload it to your workspace:

```bash
databricks workspace import notebook.py /Workspace/pricing/module-04-shap-relativities \
  --format SOURCE --language PYTHON
```

Or use the Databricks UI: Workspace > Import > drag the file.

The notebook uses `%pip install "shap-relativities[all]" catboost polars --quiet` in the first cell. On Databricks Runtime 14.x and later, this installs into the current session without requiring a cluster restart. Databricks Free Edition is sufficient for running the exercises.

### Writing results to Unity Catalog

Do not write your relativities to a local file and call it done. Write them to Unity Catalog so there is a permanent, versioned record of what relativities were extracted from which model at which date.

```python
from datetime import date
import json

# rels is a pandas DataFrame - convert to Spark for Delta write
rels_with_meta = rels.copy()
rels_with_meta["model_run_date"] = str(date.today())
rels_with_meta["model_name"] = "freq_catboost_v3"
rels_with_meta["n_policies"] = len(df)

spark.createDataFrame(rels_with_meta).write.format("delta").mode("overwrite").saveAsTable(
    "main.pricing.gbm_relativities"
)

# Validation results
validation_records = [
    {
        "check": name,
        "passed": result.passed,
        "value": result.value,
        "message": result.message,
        "model_run_date": str(date.today()),
        "model_name": "freq_catboost_v3",
    }
    for name, result in sr.validate().items()
]

spark.createDataFrame(validation_records).write.format("delta").mode("append").saveAsTable(
    "main.pricing.gbm_relativity_validation_log"
)

# Serialise the SHAPRelativities object itself for auditability
sr_state = sr.to_dict()
with open("/dbfs/mnt/pricing/sr_state_freq_catboost_v3.json", "w") as f:
    json.dump(sr_state, f)
```

The `.to_dict()` / `.from_dict()` serialisation stores the SHAP values themselves, not just the derived relativities. This means you can reconstruct the full `SHAPRelativities` object later, re-run validation, change the banding, or re-normalise to a different base level - without re-running the GBM or SHAP computation.

### Scheduling with Databricks Workflows

Once the pipeline is working, schedule it as a Databricks Workflow job. A sensible trigger is weekly, after your claims data refresh. The job should:

1. Load the current fitted model from MLflow Model Registry
2. Load the current portfolio from Unity Catalog
3. Run `SHAPRelativities.fit()` and `validate()`
4. Fail the job if reconstruction check fails (`result.passed == False`)
5. Write results to Unity Catalog with the run date
6. Optionally trigger a Radar import via API

### Monitoring relativity stability

When this pipeline runs weekly, you need to decide what counts as a material change in a relativity. A NCD=5 relativity that moves from 0.55 to 0.54 between runs is noise. A move from 0.55 to 0.48 is worth investigating - has the data quality changed? Is new business mix shifting? Is the model overfitting to a recent batch of policies?

Set alert thresholds on the relativities table. A sensible approach is to flag any relativity that moves by more than 5% week-on-week, and any relativity that has drifted more than 15% from the version loaded into the rating engine. Alert the pricing team. Do not let the automation run silently past a material drift.

The reconstruction check as a job gate prevents silent technical failures. The drift alert catches cases where the computation is technically correct but the answer has changed materially.

---

## Limitations

Be upfront about these with your pricing committee and regulator. The methodology is sound; these are genuine constraints that you manage rather than hide.

**Correlated features.** SHAP attribution for correlated features is not unique under `tree_path_dependent`. If area code and a socioeconomic deprivation index are both in your model and highly correlated, some of each feature's true attribution will be allocated to the other. The total effect of the correlated cluster is correct; the individual attributions are not. Use `feature_perturbation="interventional"` to mitigate this, or - better - think carefully about whether you need both features in the model.

**Interaction effects.** TreeSHAP allocates interaction effects to individual features by default. A vehicle group × driver age interaction will be partly attributed to vehicle group and partly to driver age, blended into each feature's SHAP values. The extracted relativities for each feature include their share of the interaction. This means the GBM's vehicle group relativity is not directly comparable to the GLM's vehicle group relativity if the GLM has no interaction term.

**CLT intervals only.** The library's confidence intervals capture data uncertainty, not model uncertainty. Two bootstrap refits of the GBM will give different SHAP values. We have not quantified that variation. For regulatory presentations, be precise: "95% data confidence interval" rather than "95% confidence interval."

**Log-link only.** Exponentiating SHAP values only gives multiplicative relativities for log-link objectives (Poisson, Tweedie, Gamma). Do not use this library with a linear-link model.

**Claims development.** SHAP relativities extracted today will change as IBNR and case reserve development comes through. For long-tail lines, the relativities at 12 months development will differ from those at 36 months. If you are training on reported claims, build in a development margin or explicitly document the development point. Training on ultimate estimates (where available) is cleaner.

**Features not in the rating structure.** Real GBMs often include features like "days since last policy change" that are useful for prediction but cannot appear in a factor table because underwriting systems do not capture them at quote. Do not silently absorb their SHAP contribution into the base rate without recording that you have done so. Make a deliberate decision: either exclude the feature from the model, or document the absorption.

**Risk relativities only.** The factor tables extracted here are pure risk relativities - they reflect expected claim cost only. Expense loadings, profit margins, and reinsurance costs are applied downstream. Make this explicit when presenting to the pricing committee. "These are risk-only relativities" should appear on every slide.

**Severity and combined models.** This tutorial covers frequency in depth. Severity relativities can be extracted identically from a Gamma severity model. For severity, you should weight observations by claim count, and consider truncating large losses at a defined threshold before fitting - large individual claims add noise to the severity SHAP attribution for features that have no real causal relationship to severity. Combining frequency and severity into pure premium relativities requires mSHAP (Lindstrom et al., 2022), which composes the two sets of SHAP values in prediction space. That is the subject of the next module.

---

## Summary

The workflow in five steps:

1. `sr = SHAPRelativities(model, X, exposure, categorical_features=..., continuous_features=...)`
2. `sr.fit()` - compute SHAP values via CatBoost's native TreeSHAP
3. `sr.validate()` - check reconstruction before trusting anything
4. `sr.extract_relativities(normalise_to="base_level", base_levels=...)` - get factor table
5. `sr.extract_continuous_curve(feature, smooth_method="loess")` - get continuous curves; aggregate by band by grouping the raw SHAP values

The output is a DataFrame with columns `feature`, `level`, `relativity`, `lower_ci`, `upper_ci`, `mean_shap`, `shap_std`, `n_obs`, `exposure_weight`. Every number needed to explain and challenge the relativities is in that table.

The point of this library is not to make GBMs look like GLMs. GBMs are better models in most situations and the relativities will reflect that - they will show non-linear patterns, interactions, and effects that the GLM cannot capture. The point is to give the pricing committee and regulator the same transparent, auditable artefact they get from a GLM, so that "we can't explain it" is no longer the reason the GBM sits unused in a notebook.
