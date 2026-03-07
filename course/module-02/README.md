# Module 2: GLMs in Python - The Bridge from Emblem

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## What this module covers

You currently build your GLMs in Emblem. The output - a set of multiplicative factor tables, deviance statistics, and actual-versus-expected charts - is exactly what you need. The problem is that Emblem is a black box from a reproducibility standpoint: the model lives in a project file, the code that produced it is not version-controlled, and re-running a historical model requires the exact software version and project configuration you used at the time.

This module shows you how to replicate that workflow in Python, running on Databricks. We use `statsmodels` for the GLM itself - it gives you the same iteratively reweighted least squares (IRLS) algorithm Emblem uses, the same deviance statistics, the same z-scores. On synthetic data without manual overrides, the output matches Emblem's to four decimal places given identical data and factor encodings. Real Emblem models often have manual overrides that must be validated separately - the tutorial covers how to identify them. We use Polars for all data manipulation; Pandas only appears at the point where statsmodels requires it.

Alongside the GLM fitting, we cover the entire surrounding workflow: loading claims data from Delta tables, encoding rating factors, handling exposure offsets, running diagnostics, and exporting factor tables to a format Radar can import.

---

## Prerequisites

- Comfortable with GLM theory: you know what a link function is, what the deviance statistic measures, and what `exp(beta)` gives you
- You have used Emblem or a comparable tool for frequency/severity modelling - you know what a factor table looks like and what "base level" means
- Basic Python: you can read a function, understand a list comprehension, and follow a data pipeline
- Module 1 completed, or equivalent: you have a Databricks workspace and know how to run a notebook

You do not need to know statsmodels before this module. We introduce the relevant API as we go.

---

## Estimated time

4-5 hours for the tutorial plus exercises. The notebook runs end-to-end in under 20 minutes on a small Databricks cluster.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial - read this first |
| `notebook.py` | Databricks notebook - full GLM workflow on synthetic UK motor data |
| `exercises.md` | Four hands-on exercises with full solutions |

---

## What you will be able to do after this module

- Fit a frequency-severity GLM in Python using statsmodels, producing output numerically consistent with Emblem
- Encode rating factors correctly: understand base level selection, aliasing, and how Python's dummy encoding differs from Emblem's default behaviour
- Handle exposure offsets properly - the most common source of wrong results when moving from Emblem to Python
- Run the standard diagnostic suite: deviance residuals, actual-versus-expected by factor level, double lift charts
- Validate that your Python GLM reproduces Emblem's published relativities, within floating-point tolerance, and diagnose manual overrides when it does not
- Export factor tables as CSVs that Radar can import directly
- Log your GLM to MLflow and register it in Unity Catalog's model registry
- Describe what the PS 21/5 pricing practices rules and FCA Consumer Duty requirements mean for your GLM audit trail and how this workflow satisfies them

---

## Why this is worth your time

The case for moving GLMs from Emblem to Python is not that Python produces better GLMs. Emblem fits GLMs correctly. The case is everything around the GLM: reproducibility, version control, automation, and the ability to integrate your GLM workflow with the rest of your modelling stack.

PS 21/5 (effective January 2022) and Consumer Duty (PS 22/9, effective July 2023) together require pricing teams to demonstrate that their models are explainable, that historical model outputs can be reproduced, and that fair value decisions are documented and auditable. A GLM living in an Emblem project file on a shared drive satisfies none of these requirements without significant additional process. A GLM fit in a Databricks notebook, with inputs from Delta tables, outputs logged to MLflow, and results written to Unity Catalog, satisfies all of them automatically.

The practical benefit: once your GLM is Python code, you can schedule it to run automatically after each data refresh, version it alongside your feature engineering code, diff two model versions to see exactly what changed, and hand it to a GBM development pipeline without any manual re-coding. Module 3 depends on this foundation.
