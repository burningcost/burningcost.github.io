# Module 3: GBMs for Insurance Pricing

Part of **Modern Insurance Pricing with Python and Databricks**.

---

## What this module covers

You built a frequency-severity GLM in Module 2. This module adds a CatBoost GBM to the same dataset and teaches you how to train, validate, and track it properly - then compare it honestly against the GLM.

The comparison is the point. GBMs almost always show better hold-out performance than GLMs on the same data. What this module teaches is what to do with that: when the lift is real and worth acting on, when it reflects overfitting, how to validate correctly using temporal splits (not random splits), and what the production decision actually looks like in practice.

We use CatBoost throughout. We use `insurance-cv` for cross-validation. We track everything in MLflow and register the best model in the model registry.

---

## What you will be able to do after this module

- Train a CatBoost Poisson frequency model and Gamma severity model on insurance data, handling exposure correctly
- Explain why temporal cross-validation matters for insurance pricing and how to implement it with `insurance-cv`
- Tune CatBoost hyperparameters for insurance data using Optuna, and understand what each key parameter actually does
- Track experiments in MLflow: parameters, metrics, model artefacts, and the comparison between GBM and GLM runs
- Register a model in the Databricks model registry using aliases rather than deprecated stages
- Compare GBM and GLM performance properly: Gini, calibration curves, double lift charts, and out-of-sample deviance
- State clearly when you would use a GBM, when a GLM, and when to run both in tandem

---

## Prerequisites

- Module 1 completed: you have a working Databricks workspace with Unity Catalog and a Delta table with the synthetic motor policy data
- Module 2 completed: you have a fitted frequency-severity GLM, logged to MLflow, with factor tables in Unity Catalog. This module builds the GBM as a direct comparator to that GLM
- Comfortable with Poisson and Gamma GLM theory - you know what the deviance measures, what an exposure offset is, and why we model frequency and severity separately
- Basic Python: you can modify a function and follow a data pipeline

---

## Estimated time

4-5 hours for the tutorial and exercises. The notebook trains end-to-end in about 25-35 minutes on a standard Databricks single-node cluster (Standard_DS3_v2 or equivalent), including hyperparameter search.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | The full written tutorial. Read this before running the notebook. |
| `notebook.py` | Databricks notebook. Import this into your workspace and run cell by cell. |

---

## Key technical decisions

**CatBoost, not LightGBM or XGBoost.** CatBoost handles categorical features without ordinal encoding. Its ordered boosting algorithm is specifically designed to reduce overfitting on smaller datasets - relevant because most UK personal lines books have 50,000-500,000 policies, not ten million. Its symmetric tree structure also makes SHAP value computation faster than alternatives, which matters in Module 4. The practical reason we picked it: CatBoost's native `cat_features` parameter means we pass the column names and the library handles everything, rather than preprocessing with label encoders that need to be versioned alongside the model.

**Polars for data manipulation.** All DataFrame operations use Polars. We convert to a CatBoost `Pool` at training time; the Pool accepts numpy arrays and lists, so the conversion is a one-liner. Pandas does not appear.

**`insurance-cv` for cross-validation.** Random cross-validation on insurance data produces optimistic metrics because it mixes policies from different development years in train and validation sets. A model trained on policies from 2021-2023 and validated on a random 20% of the same period will look better than it performs on 2024 data. `insurance-cv` implements walk-forward splits that respect policy year boundaries and accept IBNR development buffers. Install via `uv add insurance-cv`.

**Exposure as offset, not weight.** This is the same rule as Module 2, and it is easy to get wrong. In a Poisson frequency model, exposure enters as `log(exposure)` added to the linear predictor. Setting `sample_weight=exposure` instead gives a weighted likelihood that is not the same thing. CatBoost's Poisson objective uses `baseline` for the log-offset. Setting both `baseline` and `sample_weight` to exposure simultaneously double-counts exposure and produces wrong predictions. The notebook demonstrates this.

---

## Part of the MVP bundle

This module is part of the £295 MVP bundle (modules 1, 2, 4, 6). Individual module: £79.

See [burningcost.github.io/course](https://burningcost.github.io/course/) for the full curriculum.
