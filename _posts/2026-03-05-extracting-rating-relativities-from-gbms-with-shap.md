---
layout: post
title: "Extracting Rating Relativities from GBMs with SHAP"
date: 2026-03-05
categories: [techniques]
tags: [shap, gbm, catboost, relativities, pricing, python]
---

Every UK pricing team we've spoken to is in some version of the same situation: a GBM sitting on a server somewhere outperforming the production GLM, but nobody can get the relativities out of it. The regulator wants a factor table. Radar needs an import file. The head of pricing wants to challenge the model in terms they recognise.

So the GBM sits in a notebook. The GLM goes to production. And the team loses the lift.

We built `shap-relativities` to close that gap. It extracts multiplicative rating relativities from CatBoost models using SHAP values - the same format as `exp(beta)` from a GLM, with confidence intervals, exposure weighting, and a validation check that the numbers actually reconstruct the model's predictions.

---

## Why partial dependence doesn't cut it

The obvious alternative to SHAP is a partial dependence plot. Fix all other features at their observed values, vary the feature of interest, average the predictions. You get a curve. It shows you something.

What it doesn't show you is how much of each individual prediction is attributable to that feature. PDPs show marginal effects averaged across the portfolio. They don't decompose predictions. If you have two features that are correlated, a PDP for one of them will absorb some of the other's effect - and you won't know which direction.

SHAP has a different guarantee. The Shapley axioms - efficiency, symmetry, null player, linearity - mean that SHAP values sum exactly to the model output. For a CatBoost Poisson model:

```
log(μ_i) = expected_value + SHAP_area_i + SHAP_ncd_i + SHAP_age_i + ...
```

Every prediction is fully decomposed into per-feature contributions in log space. That's the foundation everything else builds on.

---

## The maths, briefly

For a Poisson GBM with log link, the model predicts `log(frequency)`. SHAP values are additive in log space. To get a multiplicative relativity for area = "B" relative to area = "A":

1. For each policy with area = "B", extract `SHAP_area`. Average across policies in that level, weighted by exposure. Call this `mean_shap("B")`.
2. Do the same for area = "A". Call this `mean_shap("A")`.
3. Relativity = `exp(mean_shap("B") - mean_shap("A"))`.

This is directly analogous to `exp(β_B - β_A)` from a GLM. The base level gets relativity 1.0 by construction.

Confidence intervals come from the CLT on those weighted means:

```
SE = shap_std / sqrt(n_obs)
CI = exp(mean_shap ± z * SE - base_shap)
```

These are data uncertainty intervals - they quantify how precisely we've estimated each level's mean SHAP contribution given the portfolio. They do not capture model uncertainty from the GBM fitting process itself. That distinction matters, and we come back to it in the limitations section.

---

## A worked example

We'll train a CatBoost frequency model on synthetic UK motor data with a known data-generating process, then extract relativities and compare them to the true parameters.

```python
import polars as pl
from catboost import CatBoostRegressor, Pool
from shap_relativities import SHAPRelativities

# Assume df is a Polars DataFrame with: exposure, claims, area (A/B/C/D),
# ncd_years (0-5), vehicle_age (numeric), driver_age (numeric)

features = ["area", "ncd_years", "vehicle_age", "driver_age"]
cat_features = ["area", "ncd_years"]

# CatBoost requires pandas or numpy arrays; convert at the boundary
X = df.select(features).to_pandas()
y = df["claims"].to_numpy()
exposure = df["exposure"].to_numpy()

# Build a Pool with categorical features declared explicitly
train_pool = Pool(
    data=X,
    label=y,
    weight=exposure,
    cat_features=cat_features,
)

# Train a Poisson frequency model
model = CatBoostRegressor(
    loss_function="Poisson",
    learning_rate=0.05,
    depth=6,
    iterations=500,
    verbose=0,
)
model.fit(train_pool)

# Extract relativities
sr = SHAPRelativities(
    model=model,
    X=X,
    exposure=exposure,
    categorical_features=cat_features,
)
sr.fit()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area": "A", "ncd_years": 0},
)
```

`rels` is a DataFrame with one row per (feature, level) combination:

| feature | level | relativity | lower_ci | upper_ci | n_obs | exposure_weight |
|---------|-------|-----------|----------|----------|-------|-----------------|
| area | A | 1.000 | 1.000 | 1.000 | 12847 | 10203.4 |
| area | B | 1.183 | 1.141 | 1.227 | 9832 | 7891.2 |
| area | C | 0.921 | 0.889 | 0.954 | 8104 | 6512.7 |
| area | D | 1.347 | 1.298 | 1.399 | 6219 | 4978.1 |
| ncd_years | 0 | 1.000 | 1.000 | 1.000 | 8741 | 6982.3 |
| ncd_years | 1 | 0.887 | 0.856 | 0.920 | 7923 | 6341.1 |
| ... | ... | ... | ... | ... | ... | ... |

For the validation test we ran on synthetic data where we knew the true DGP parameters, extracted relativities matched the true multiplicative parameters to within 2-3% across all area and NCD levels after 50,000 training policies. That's within the confidence intervals, which is exactly what we'd want to see.

---

## Exposure weighting matters

Without exposure weighting, a postcode with three policies in the training set gets the same vote as one with 30,000. In motor pricing, exposure periods vary (mid-term adjustments, cancellations), fleet accounts represent single policies but large exposures, and rare categories often have fewer than 30 observations.

`SHAPRelativities` takes `exposure` as a numpy array of earned policy years. Every weighted mean SHAP computation uses these as observation weights. The CLT standard error divides by `sqrt(n_obs)` - not `sqrt(exposure)` - because CLT applies to the count of independent observations, not their weight. This is the correct treatment.

If you don't pass exposure, all observations are weighted equally. That's fine for a balanced dataset, but for insurance portfolios it's almost never the right call.

The `validate()` method includes a sparse-levels check that warns when any categorical level has fewer than 30 observations - the point at which the CLT assumption becomes shaky:

```python
checks = sr.validate()
# Returns: {"reconstruction": ..., "feature_coverage": ..., "sparse_levels": ...}

print(checks["reconstruction"])
# CheckResult(passed=True, value=8.3e-06, message="Max absolute error: 8.3e-06")

print(checks["sparse_levels"])
# CheckResult(passed=False, value=4.0,
#   message="4 levels have < 30 observations: ncd_years=5 (n=17), ...")
```

The reconstruction check verifies that `exp(shap.sum(axis=1) + expected_value)` matches the model's own predictions within `1e-4`. If this fails, the explainer was set up incorrectly - almost always a mismatch between the model's link function and the SHAP output type.

---

## Continuous features

For continuous features like `vehicle_age` and `driver_age`, level-by-level aggregation doesn't make sense - every value is distinct. `SHAPRelativities` handles these differently: it returns per-observation SHAP values and provides a smoothed curve via `extract_continuous_curve()`:

```python
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=100,
    smooth_method="loess",  # or "isotonic" for monotone
)
# Returns DataFrame: feature_value, relativity, lower_ci, upper_ci
```

LOESS smoothing (via statsmodels) works well for most features. Use `smooth_method="isotonic"` when you have a strong theoretical prior that the relativity should be monotone - younger drivers are higher risk, age can only increase relativity up to a point. Isotonic regression will enforce that without adding parametric assumptions about the shape.

---

## Limitations you should document

Three things to be honest about when presenting SHAP relativities to regulators or pricing committees.

**Correlated features.** SHAP attribution for correlated features is not uniquely defined under the default `tree_path_dependent` method. Area and vehicle deprivation index will share attribution in a way that depends on tree split order. You can switch to `feature_perturbation="interventional"` and pass a background dataset - this corrects for correlations using marginalisation - but it's substantially slower and requires a representative background sample. Document the choice you made.

**Interaction effects.** TreeSHAP allocates interaction effects back to individual features by default. If area and vehicle age have an interaction in the model, some of that interaction gets attributed to area and some to vehicle age, but not in a way that cleanly separates main effects from interactions. `shap_interaction_values()` gives pure main effects, but it's O(n × p²) and quickly becomes infeasible on large portfolios.

**Model uncertainty.** The CLT intervals capture data uncertainty - how well we've estimated each level's mean SHAP contribution. They say nothing about whether the GBM itself is well-specified, whether it would give different relativities on a different data split, or whether the feature contributions are stable across refits. For a full uncertainty picture you'd need to bootstrap across model refits. We haven't implemented that yet; it's on the roadmap.

---

## What's next: mSHAP for two-part models

Most frequency×severity models can't be analysed with this approach directly. You can extract relativities from the frequency model and the severity model separately, but you can't simply multiply them together and get a valid decomposition of the pure premium. The scales are different, the link functions may differ, and the portfolio averages don't cancel cleanly.

The correct approach is mSHAP (multiplicative SHAP), proposed by Lindstrom et al. (2022), which works for two-part models by combining SHAP values in prediction space rather than log space. We're building this as a second module. For now: extract freq and severity relativities separately, present them side by side, and be explicit that the pure premium relativities require a further combining step.

---

## Getting started

```bash
uv add shap-relativities
```

The library supports CatBoost models with log-link objectives - Poisson, Tweedie, Gamma. If you pass a model with a linear link, the SHAP values will be in linear space and the `exp()` transformation will give you nonsense.

For cases where you don't need the intermediate object:

```python
from shap_relativities import extract_relativities

rels = extract_relativities(
    model=cb_model,
    X=X_train,
    exposure=exposure,
    categorical_features=["area", "ncd_years"],
    base_levels={"area": "A", "ncd_years": 0},
)
```

Source and issue tracker on [GitHub](https://github.com/burningcost/shap-relativities). If you find a case where extracted relativities diverge materially from your expected model behaviour, the `validate()` method is the first place to look.
