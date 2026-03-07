# Module 4: SHAP Relativities

Part of **Modern Insurance Pricing with Python and Databricks**.

---

## The problem this module solves

You have a CatBoost model that outperforms your production GLM. The loss ratio lift is real. The problem is that neither your pricing committee nor Radar can work with a black-box gradient boosting model. They need a factor table: one row per (feature, level) combination, relativity relative to a base level, with confidence intervals. The same format as `exp(beta)` from a GLM.

This module teaches you to get that table from the GBM using SHAP values. The approach is mathematically sound, not a heuristic. The output is reviewable by a pricing actuary, submittable to the FCA, and importable into Radar or Emblem.

---

## What you will build

- A CatBoost Poisson frequency model and Gamma severity model on synthetic UK motor data, with correct exposure handling and MLflow tracking.
- SHAP-derived multiplicative relativities for categorical features (area, NCD) and smoothed curves for continuous features (driver age, vehicle group).
- A factor table Excel workbook formatted for a pricing committee.
- A Radar-compatible CSV export.
- A Delta Lake history table for rate version control.

---

## Prerequisites

- Comfortable with GLM frequency-severity pricing. You do not need to know the econometrics, but you should know what a relativity is.
- Basic Python. You will be reading and modifying code, not writing it from scratch.
- Access to a Databricks workspace. Databricks Free Edition is sufficient for the exercises.

---

## Contents

| File | Description | Estimated time |
|------|-------------|----------------|
| `00-overview.md` | Module overview, objectives, prerequisites | 10 min |
| `01-why-shap-relativities.md` | The production problem and why SHAP solves it | 30 min |
| `02-setup.md` | Installation, notebook setup, dataset | 20 min |
| `03-training-the-gbm.md` | CatBoost freq and severity training | 45 min |
| `04-extracting-relativities.md` | SHAP extraction pipeline | 45 min |
| `05-regulatory-tables.md` | Committee formatting, proxy discrimination, IBNR | 30 min |
| `06-radar-export.md` | Radar/Emblem export, version control, drift monitoring | 30 min |
| `07-exercises.md` | Five exercises with worked solutions | 45 min |

---

## Key technical decisions

**CatBoost.** CatBoost handles categorical features natively (no ordinal encoding needed), has built-in SHAP support that is faster than the generic `shap` library, and its Poisson objective handles exposure via a proper log-offset rather than a sample weight. These are practical advantages, not preferences.

**Exposure as offset, not weight.** In a Poisson frequency model, exposure enters the log-linear predictor as an offset: `log(exposure)` is added to `log(lambda)`. It is not a sample weight on the likelihood. Setting both `baseline=log(exposure)` and `weight=exposure` simultaneously double-counts exposure and produces wrong predictions. This is covered in section 3 and demonstrated in Exercise 1.

**Polars for data manipulation.** All DataFrame operations use Polars. Conversion to pandas happens only at the CatBoost `Pool` boundary. The `shap-relativities` library accepts Polars DataFrames natively.

**SHAP on original features, band aggregation separately.** Continuous features like driver age are passed to SHAP as continuous variables (what the model was trained on). Banding for the factor table is a post-hoc aggregation step on the SHAP values, not a re-specification of the model. Passing a banded feature to an explainer trained on the continuous feature produces wrong SHAP values.

---

## The `shap-relativities` library

This module uses `shap-relativities`, an open-source Python library for extracting multiplicative rating relativities from GBMs. Install via:

```bash
uv pip install 'shap-relativities[catboost]==0.1.0'
```

Source: https://github.com/burningcost/shap-relativities

The library outputs Polars DataFrames with columns: `feature`, `level`, `relativity`, `lower_ci`, `upper_ci`, `mean_shap`, `shap_std`, `n_obs`, `exposure_weight`.

---

## Part of the MVP bundle

This module is included in the £295 MVP bundle alongside Module 1 (Databricks for Pricing Teams), Module 2 (GLMs in Python), and Module 6 (Credibility and Bayesian Pricing). Individual module: £79.
