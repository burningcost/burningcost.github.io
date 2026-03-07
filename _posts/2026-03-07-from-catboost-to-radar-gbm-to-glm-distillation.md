---
layout: post
title: "From CatBoost to Radar in 50 Lines of Python"
date: 2026-03-07
categories: [techniques]
tags: [insurance-distill, catboost, glm, radar, emblem, distillation, surrogate, factor-tables, uk-personal-lines]
description: "An open-source Python library that distils GBM models into multiplicative GLM factor tables for Radar, Emblem, and other rating engines. The first open-source solution for the most common deployment problem in UK pricing."
author: Burning Cost
---

You have built a CatBoost model on three years of motor data. You ran your validation properly: out-of-time holdout, exposure-weighted Gini, double-lift chart. The model beats your existing GLM by 3 Gini points. The pricing committee is interested. Then someone asks the obvious question: how do we get it into Radar?

There is no good answer to that question today. You cannot load a fitted CatBoost model into Radar. Radar needs factor tables: one table per rating variable, with a relativity for each level. CatBoost produces an ensemble of decision trees. These are not the same thing, and no conversion exists that is both automatic and open.

The standard workarounds are unpleasant. You can attempt to manually read the GBM's partial dependence plots and hand-code factor tables, which loses most of the model's discrimination power and introduces error. You can rebuild the model inside Radar's native GBM fitter, which abandons your Python training pipeline and the feature engineering that made the model good. You can buy Akur8, which is a SaaS platform that builds its own transparent ML from scratch inside its own environment - it does not accept your existing fitted CatBoost model.

WTW's answer is Layered GBM in Radar: a patent-pending two-layer structure where one GBM captures main effects and a second captures interactions. It produces interpretable outputs, but it is a proprietary Radar format, not a standard multiplicative factor table. It also requires you to rebuild your model inside Radar.

We think there is a better approach. Last month we published `insurance-distill`, an open-source Python library that distils any scikit-learn-compatible GBM into multiplicative GLM factor tables that a rating engine can consume. This post explains how it works and how to use it.

## The idea: fit a GLM on GBM predictions

The core insight is that you do not need the GLM to learn from raw claims data. Your GBM has already processed that data and produced smooth, noise-reduced predictions for every policy in your training set. Use those predictions as the target for a GLM.

This is sometimes called a pseudo-response or surrogate approach. You generate GBM predictions across the training set, then fit a Poisson or Gamma GLM with a log link where the response variable is the GBM's output and the exposure is your actual earned exposure. The GLM learns to approximate the GBM's structure. The result is a set of multiplicative GLM coefficients - one set per rating variable and level - that you can load directly into a rating engine.

There are two things to do before fitting the GLM. First, you need to bin continuous variables. A rating engine does not accept a continuous age variable; it needs discrete levels with a relativity for each. Second, those bins should not be arbitrary: they should reflect where the GBM actually changes its predictions. A young driver at 17 and one at 24 have very different risk profiles in a GBM; a driver at 30 and one at 33 may be effectively identical. The bins should respect that structure.

The binning approach we use is a CART decision tree fit on each variable's GBM predictions individually. The tree's split points become the bin boundaries. This is fast, requires no hyperparameter tuning beyond a maximum bin count, and finds boundaries that are statistically meaningful relative to the GBM's learned response. For variables where you want a monotone factor (no-claims discount, years of driving experience), we offer an isotonic regression alternative that finds change-points in a monotone fit.

Once features are binned, the GLM is a standard one-hot encoded regression fit using glum - a Poisson/Gamma GLM solver purpose-built for the kind of large, sparse problems that insurance pricing produces. The GLM is fit with a log link throughout, which means the factor tables are multiplicative by construction. That is what Radar expects.

## In practice

Here is the full workflow on a motor frequency model:

```python
from catboost import CatBoostRegressor
from insurance_distill import SurrogateGLM

# fitted_catboost: your trained CatBoostRegressor
surrogate = SurrogateGLM(
    model=fitted_catboost,
    X_train=X_train,        # Polars DataFrame
    y_train=y_train,        # actual claim counts
    exposure=exposure_arr,  # earned car-years
    family="poisson",
)

surrogate.fit(
    max_bins=10,
    interaction_pairs=[("driver_age", "region")],
)
```

That is it for fitting. `X_train` is a Polars DataFrame with your rating variables. `exposure_arr` is a numpy array of earned car-years. The `family="poisson"` argument tells the GLM which distribution to use; for a severity model you would pass `family="gamma"`.

The `interaction_pairs` argument handles two-way interactions. Where you know from domain knowledge (or from your GBM's SHAP interaction values) that two variables interact materially, pass the pair and the library will include a cross-classified interaction term in the GLM. You can include as many pairs as you want; include too many and the deviance ratio will not improve much, which is a useful diagnostic signal.

For variables where you want per-variable binning method control:

```python
surrogate.fit(
    max_bins=10,
    binning_method="tree",          # default
    method_overrides={
        "ncd_years": "isotonic",    # monotone NCD factor
        "vehicle_age": "quantile",  # equal-frequency fallback
    },
)
```

## Validation: how much do you lose?

The GLM surrogate will not match the GBM's Gini coefficient exactly. That is expected and acceptable. The question is how much you lose.

After fitting, `surrogate.report()` returns a `DistillationReport` with four metrics:

```python
report = surrogate.report()
print(report.metrics.summary())
# Gini (GBM):              0.3241
# Gini (GLM surrogate):    0.3087
# Gini ratio:              95.2%
# Deviance ratio:          0.9143
# Max segment deviation:   8.3%
# Mean segment deviation:  2.1%
# Segments evaluated:      312
```

The **Gini ratio** is the most important number. It tells you what fraction of the GBM's discrimination the GLM retains. Above 90% is generally acceptable for a surrogate that will be deployed into a rating engine. Above 95% is excellent. In our testing on UK motor data, a 10-variable model with 5-10 bins per variable and 2-3 interaction terms typically lands between 92% and 97% Gini retention.

The **deviance ratio** is the GLM analogue of R-squared, measuring how well the GLM explains the GBM's predictions. Values above 0.90 are good. Below 0.85 suggests the GLM structure is not capturing something important - often a missing interaction term.

The **segment deviation** metrics are operationally the most relevant. For each unique combination of factor levels (each cell in the rating grid), we compute the relative difference between the GBM's average prediction and the GLM's average prediction. Max deviation of 8.3% means the worst-case cell is off by 8.3% relative to the GBM. Mean deviation of 2.1% means the typical cell is within 2%. If max deviation is below 10%, the factor tables are a faithful representation of the GBM's output. If it is above 15-20%, you likely need more bins or additional interaction terms.

The report also includes a **double-lift chart**: rows sorted by the ratio of GBM prediction to GLM prediction, grouped into deciles. A flat line across deciles indicates the GLM and GBM agree on risk ordering throughout the distribution. Slope indicates where the GLM is systematically under- or over-pricing relative to the GBM. This is the same double-lift chart format used in Radar and Emblem workflows.

```python
# Access the lift chart as a Polars DataFrame
print(report.lift_chart)
# shape: (10, 5)
# columns: decile, avg_gbm, avg_glm, ratio_gbm_to_glm, exposure_share
#
# decile  avg_gbm  avg_glm  ratio_gbm_to_glm  exposure_share
#      1    0.041    0.043             0.953           0.100
#      2    0.057    0.059             0.966           0.100
#    ...      ...      ...               ...             ...
#     10    0.218    0.211             1.033           0.100
```

Ratios between 0.95 and 1.05 across all deciles are excellent. Ratios outside 0.90-1.10 for the top or bottom decile - where high- and low-risk policies sit - warrant attention.

## Inspecting and exporting factor tables

The factor tables are the deliverable. You can inspect a single variable:

```python
driver_age_table = surrogate.factor_table("driver_age")
print(driver_age_table)
# shape: (8, 3)
# level             log_coefficient  relativity
# [-inf, 21.00)               0.412       1.510
# [21.00, 25.00)              0.218       1.244
# [25.00, 35.00)              0.000       1.000   <- base level
# [35.00, 50.00)             -0.071       0.931
# [50.00, 62.00)             -0.093       0.911
# [62.00, 70.00)             -0.018       0.982
# [70.00, 79.00)              0.088       1.092
# [79.00, +inf)               0.244       1.277
```

The `relativity` column is `exp(log_coefficient)`. The base level - [25.00, 35.00) here - always has `relativity = 1.0`. Everything else is expressed relative to it. This is the convention used by Radar, Emblem, and most other UK personal lines rating engines.

To export all tables as CSV files for import into your rating engine:

```python
surrogate.export_csv(
    "output/factors/",
    prefix="motor_freq_",
)
# Writes:
#   motor_freq_driver_age.csv
#   motor_freq_vehicle_value.csv
#   motor_freq_ncd_years.csv
#   ... (one file per variable)
#   motor_freq_base.csv  (intercept / base rate)
```

Each CSV has three columns: `level`, `log_coefficient`, `relativity`. The base factor CSV contains the model intercept, which corresponds to the base pure premium before multiplicative factors are applied.

For direct Radar formatting, `format_radar_csv()` converts a factor table DataFrame to the two-column format (FeatureName, Relativity) that Radar expects when you rebuild a table manually:

```python
from insurance_distill import format_radar_csv

radar_csv = format_radar_csv(driver_age_table, "driver_age")
with open("radar_driver_age.csv", "w") as f:
    f.write(radar_csv)
```

There is no direct Radar API for programmatic import. That is a Radar limitation, not ours. The CSV output gives you a clean source to paste from or import via Radar's factor table editor.

## Why not just rebuild the model in Radar?

The honest answer is that sometimes you should. If your CatBoost model's performance advantage over a native Radar GLM is marginal - say 1-2 Gini points - and your team is already comfortable with the Radar workflow, rebuilding inside Radar may be the right choice.

`insurance-distill` is useful when:

- Your feature engineering is non-trivial Python: text features, external data sources, postcode-level lookups that are hard to replicate in Radar's data preparation layer.
- You need reproducible, version-controlled model artefacts in a Python MLOps pipeline. Factor tables in CSV, committed to Git, are simpler to audit than Radar model files.
- You have a GBM that is genuinely 3-5+ Gini points ahead of what a GLM can produce, and you want to recover as much of that performance as possible within a rating engine.
- Your team's development capacity is in Python, not in Radar's scripting environment.

In any of those situations, distillation is more productive than rebuilding.

## The competitive context

As of March 2026, there is no other Python open-source package that accepts an externally-fitted CatBoost model and outputs Radar-compatible GLM factor tables.

The academic methods that `insurance-distill` implements have existed since Henckaerts et al. (2019, 2022) developed MAIDRR and Lindholm and Palmquist (2024) published a LASSO-based variant in the Annals of Actuarial Science. The R package `maidrr` implements Henckaerts' method but it is R-only, single-researcher, and was flagged as under development as of early 2026. There is no comparable Python implementation.

WTW's Layered GBM in Radar is the closest commercial analogue. It layers two GBMs to produce interpretable outputs, but the result is a Radar-proprietary format, not a portable factor table. You cannot take a Layered GBM out of Radar and put it somewhere else.

Akur8 builds transparent ML from within its own platform. It has a WTW integration announced in August 2023. It does not accept external models. Pricing teams that have already built and validated a CatBoost model in Python cannot use Akur8 to deploy it.

The gap is real. We built `insurance-distill` because we needed it ourselves, and because we think it belongs in the open-source Python ecosystem rather than locked inside a commercial platform.

## Implementation notes

The library uses glum for GLM fitting. glum is a generalised linear model solver developed by Quantco, purpose-built for the large, sparse design matrices that insurance pricing produces. On a motor book with 500,000 policies and 15 rating variables at 8 bins each, glum is measurably faster than statsmodels - on the order of 10-100x, depending on the problem structure. Coefficient estimates are identical to statsmodels for the unregularised case.

We use Polars throughout for data handling. The aggregation operations in segment deviation computation and lift chart generation are faster and more memory-efficient in Polars than in pandas for the group-by patterns we use. The GLM fitting itself uses numpy arrays internally, as glum requires, so the Polars dependency does not touch the core numerical path.

The library supports Poisson (frequency), Gamma (severity), and Tweedie (pure premium) families. CatBoost and any other sklearn-compatible model with a `.predict()` method are supported. For CatBoost classifiers, pass `predict_method="predict_proba"` and the library will use the positive class probability as the pseudo-response.

The regularisation parameter `alpha` on `SurrogateGLM` controls L2 shrinkage on the GLM coefficients. The default is 0.0 (unregularised). For high-cardinality categorical variables or a large number of interaction terms, a small positive alpha (0.001-0.01) can prevent overfitting to sparse cells.

## Installation

```bash
uv add insurance-distill
```

With CatBoost support:

```bash
uv add "insurance-distill[catboost]"
```

Python 3.10 or later. The library requires polars >= 0.20, numpy >= 1.24, scikit-learn >= 1.3, and glum >= 2.0.

The source is at [github.com/burningcost/insurance-distill](https://github.com/burningcost/insurance-distill). The `README.md` has a worked example on synthetic motor data. Issues and pull requests welcome.

One thing the library does not do: it does not tell you whether the Gini retention on your specific dataset is acceptable. A 93% Gini ratio on a 0.28 Gini model retains more absolute discrimination than a 97% ratio on a 0.12 Gini model. The right threshold depends on your book, your rating structure, and what the pricing committee considers material. That judgement remains yours.
