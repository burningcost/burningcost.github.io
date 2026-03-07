---
layout: page
title: "Modern Insurance Pricing with Python and Databricks"
description: "A practitioner-written course for UK personal lines pricing teams. Eight modules covering GLMs, GBMs, SHAP relativities, conformal prediction intervals, credibility, and constrained rate optimisation on Databricks."
permalink: /course/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Course",
  "name": "Modern Insurance Pricing with Python and Databricks",
  "description": "A practitioner-written course for UK personal lines pricing teams. Eight modules covering GLMs, GBMs, SHAP relativities, conformal prediction intervals, credibility, and constrained rate optimisation on Databricks.",
  "url": "https://burningcost.github.io/course/",
  "provider": {
    "@type": "Organization",
    "name": "Burning Cost",
    "url": "https://burningcost.github.io"
  },
  "author": {
    "@type": "Organization",
    "name": "Burning Cost",
    "url": "https://burningcost.github.io"
  },
  "educationalLevel": "Professional",
  "teaches": "Insurance pricing using Python, Databricks, GLMs, GBMs, SHAP, credibility theory, and constrained rate optimisation",
  "inLanguage": "en-GB",
  "offers": [
    {
      "@type": "Offer",
      "name": "MVP bundle",
      "price": "295",
      "priceCurrency": "GBP"
    },
    {
      "@type": "Offer",
      "name": "Full course",
      "price": "495",
      "priceCurrency": "GBP"
    },
    {
      "@type": "Offer",
      "name": "Individual module",
      "price": "79",
      "priceCurrency": "GBP"
    }
  ],
  "hasCourseInstance": [
    {
      "@type": "CourseInstance",
      "name": "Module 1: Databricks for Pricing Teams",
      "description": "Unity Catalog for pricing data, cluster configuration, Delta tables, and MLflow experiment tracking."
    },
    {
      "@type": "CourseInstance",
      "name": "Module 2: GLMs in Python - The Bridge from Emblem",
      "description": "Replicating Emblem in Python with statsmodels: offset terms, variance functions, one-way and two-way analysis."
    },
    {
      "@type": "CourseInstance",
      "name": "Module 3: GBMs for Insurance Pricing",
      "description": "CatBoost with Poisson, gamma, and Tweedie objectives. Walk-forward cross-validation with IBNR buffers."
    },
    {
      "@type": "CourseInstance",
      "name": "Module 4: SHAP Relativities",
      "description": "Extracting multiplicative relativities from GBMs using SHAP values in the format actuarial reviewers expect."
    },
    {
      "@type": "CourseInstance",
      "name": "Module 5: Conformal Prediction Intervals",
      "description": "Distribution-free prediction intervals for insurance GBMs, calibrated to your own holdout data."
    },
    {
      "@type": "CourseInstance",
      "name": "Module 6: Credibility and Bayesian Pricing",
      "description": "Buhlmann-Straub credibility in Python and its relationship to hierarchical models and partial pooling."
    },
    {
      "@type": "CourseInstance",
      "name": "Module 7: Constrained Rate Optimisation",
      "description": "Linear programming for rate changes that meet a target loss ratio and respect movement caps."
    },
    {
      "@type": "CourseInstance",
      "name": "Module 8: End-to-End Pipeline (Capstone)",
      "description": "A complete motor frequency and severity pipeline from Delta ingestion through to rate optimisation output."
    }
  ]
}
</script>

# Modern Insurance Pricing with Python and Databricks

**A practitioner-written course for UK personal lines pricing teams.**

---

## The problem with how pricing teams learn Databricks

Most actuaries and pricing analysts learn Databricks from the same place: generic data science tutorials aimed at software engineers doing retail churn or ad-click models. Those tutorials cover the Databricks UI and Delta Lake and MLflow in the abstract. They do not cover Poisson deviance as a loss function, IBNR buffers in cross-validation, SHAP relativities in the format that a pricing meeting will actually accept, or how to build a constrained rate optimisation that respects a target loss ratio and a maximum movement cap simultaneously.

You can piece it together. People do. But it takes six months of wasted effort, and you end up with notebooks that work but that no-one on the team can maintain, because they were written by someone learning two things at once.

This course teaches Databricks for insurance pricing specifically. Every module is grounded in real personal lines problems - motor frequency and severity, home peril models, NCD elasticity - not toy datasets from Kaggle.

---

## Who this is for

Pricing actuaries and analysts at UK personal lines insurers who:

- Are using Databricks, or will be within the next year
- Already know GLMs (you do not need to be an expert, but you should know what a log link is and why it is there)
- Can write basic Python - loops, functions, DataFrames - but are not software engineers
- Are tired of being the person who adapts a generic tutorial to their specific problem and hopes for the best

This is not an introductory course to either Python or insurance pricing. If you need that, there are better places to start. This course assumes you price things for a living and want to do it in Python on Databricks properly.

---

## What you will need

- A Databricks workspace. The [Free Edition](https://www.databricks.com/product/pricing/databricks-free-edition) works for all exercises - no company approval needed to start
- Python basics: Polars or PySpark DataFrames, functions, loops
- GLM knowledge: you should understand what a link function is and have built at least one frequency or severity model, even in another tool

---

## The course

Eight modules. Each module is a written tutorial plus a Databricks notebook you can run directly. The notebooks use synthetic data that mirrors real personal lines structure - motor policies with realistic exposure, claim counts, and development patterns. All code uses Polars for data wrangling and CatBoost for gradient boosting - the best tools for the job in 2026.

### Available now - MVP bundle

The four MVP modules are published. Each has a written tutorial and a Databricks notebook.

---

#### [Module 1: Databricks for Pricing Teams](/course/module-01/)

What Databricks actually is (not the marketing version), and how to set it up for a pricing project rather than a generic data pipeline. Unity Catalog for pricing data, cluster configuration, Delta tables as a replacement for flat-file data passes, MLflow experiment tracking from first principles. The goal: a clean, reproducible workspace that a second analyst can pick up without a two-hour handover.

**Files:** [README](/course/module-01/README.md) - [Tutorial](/course/module-01/tutorial.md) - [Notebook](/course/module-01/notebook.py)

---

#### [Module 2: GLMs in Python - The Bridge from Emblem](/course/module-02/)

How to replicate what Emblem does in Python, transparently. `statsmodels` GLMs with offset terms, variance functions, one-way and two-way analysis, aliasing detection, and model comparison. We also cover the gap between `statsmodels` and what sklearn's GLM implementation does - and when the difference matters. By the end, you can build and validate a frequency model in Python that a traditional actuarial reviewer can follow.

**Files:** [README](/course/module-02/README.md) - [Tutorial](/course/module-02/tutorial.md) - [Notebook](/course/module-02/notebook.py)

---

#### [Module 4: SHAP Relativities](/course/module-04/)

SHAP values as a replacement for GLM relativities. How to extract them, how to aggregate them into a format that looks like a traditional relativities table, and how to explain them to someone who has spent twenty years using Emblem. We cover the cases where SHAP relativities are honest and the cases where they are misleading - interaction effects, correlated features, and what to do when the SHAP waterfall plot does not match the underwriter's intuition. Includes coverage of protected characteristics and proxy discrimination detection using SHAP - essential for FCA Consumer Duty compliance. Uses our open-source [`shap-relativities`](https://github.com/burningcost/shap-relativities) library.

**Files:** [README](/course/module-04/README.md) - [Tutorial](/course/module-04/tutorial.md) - [Notebook](/course/module-04/notebook.py)

---

#### [Module 6: Credibility and Bayesian Pricing](/course/module-06/)

Classical credibility (Buhlmann-Straub) in Python, and its relationship to mixed models and partial pooling. When to use credibility weighting versus a hierarchical GLM. Practical applications: capping thin segments, stabilising NCD factors, blending a new model with an incumbent rate. Uses our open-source [`credibility`](https://github.com/burningcost/credibility) library. We also cover the cases where credibility gives you false comfort - specifically, what it does not protect you from when the underlying exposure mix is shifting.

**Files:** [README](/course/module-06/README.md) - [Tutorial](/course/module-06/tutorial.md) - [Notebook](/course/module-06/notebook.py)

---

### Coming later

#### Module 3: GBMs for Insurance Pricing

CatBoost from a pricing perspective. Poisson objective for frequency, gamma for severity, Tweedie for pure premium. Hyperparameter tuning calibrated to insurance data - why the defaults from generic tutorials are wrong for insurance and what to use instead. CatBoost's native handling of categorical features means no more manual ordinal encoding - no encoding gymnastics required. Walk-forward cross-validation with IBNR buffers using our open-source [`insurance-cv`](https://github.com/burningcost/insurance-cv) library, so you are not lying to yourself about out-of-sample performance.

#### Module 5: Conformal Prediction Intervals

Prediction intervals for insurance models that are statistically honest - not confidence intervals for the mean, but intervals for individual risk predictions. Conformal prediction on top of a trained GBM, calibrated to your own holdout data. How to use these intervals to flag uncertain risks and to set minimum premium floors. Uses our [`insurance-conformal`](https://github.com/burningcost/insurance-conformal) library, which implements the variance-weighted non-conformity score from Manna et al. (2025) - producing intervals roughly 30% narrower than the naive approach with identical coverage guarantees.

#### Module 7: Constrained Rate Optimisation

Building a rate change that meets a target loss ratio, respects a maximum movement cap per cell, and minimises cross-subsidy across rating factors simultaneously. Linear programming formulation, `scipy.optimize` and `PuLP` in Databricks, and how to structure the constraints so that the optimiser produces something a pricing actuary and a commercial director will both accept. Uses our open-source [`rate-optimiser`](https://github.com/burningcost/rate-optimiser) library. This is the module most courses do not have.

#### Module 8: End-to-End Pipeline (Capstone)

A complete motor frequency and severity pipeline: data ingestion from Delta, feature engineering with a reproducible transform layer, walk-forward CV, CatBoost training with MLflow tracking, SHAP relativities, conformal intervals, rate optimisation, and a final output table that feeds a rating engine. The notebook is designed to be a working template for a real project, not a demo.

---

## Pricing

The MVP bundle - modules 1, 2, 4, and 6 - covers the core Databricks setup, the GLM bridge, SHAP relativities, and credibility. It is the sequence most teams need first.

| Bundle | Modules | Price |
|---|---|---|
| MVP bundle | 1, 2, 4, 6 | £295 |
| Full course | All 8 modules | £495 |
| Individual module | Any one | £79 |

These are one-time prices. You get the notebook files and the written tutorials. No subscription. No expiry.

---

## Why trust us on this

We have written open-source tools for every topic this course covers.

- [`insurance-cv`](https://github.com/burningcost/insurance-cv) - temporally-correct cross-validation for insurance pricing models, with IBNR buffer support and sklearn compatibility.
- [`insurance-conformal`](https://github.com/burningcost/insurance-conformal) - distribution-free prediction intervals for insurance GBMs, implementing the variance-weighted non-conformity score from Manna et al. (2025).
- [`shap-relativities`](https://github.com/burningcost/shap-relativities) - SHAP values aggregated into multiplicative relativities tables in the format actuarial reviewers expect.
- [`credibility`](https://github.com/burningcost/credibility) - Buhlmann-Straub credibility in Python, with mixed-model equivalence checks.
- [`rate-optimiser`](https://github.com/burningcost/rate-optimiser) - constrained rate change optimisation for UK personal lines.
- [`bayesian-pricing`](https://github.com/burningcost/bayesian-pricing) - hierarchical Bayesian models for thin-data segments, with Buhlmann-Straub credibility factor output.

We have also written at length about [why standard k-fold cross-validation is wrong for insurance data](/2026/03/06/why-your-cross-validation-is-lying-to-you.html) - the kind of detail that only comes from having had to fix it in production.

The course teaches you to use these tools properly on Databricks. We built them because we needed them. We wrote this course because the documentation alone is not enough.

We are pricing practitioners, not data science generalists who have read the insurance Wikipedia page.

---

## Get the MVP bundle

Email [pricing.frontier@gmail.com](mailto:pricing.frontier@gmail.com?subject=MVP%20Bundle%3A%20Modern%20Insurance%20Pricing%20with%20Python%20and%20Databricks) with the subject line pre-filled. We will send you the bundle files and invoice. Waitlist members get first access at the launch price.

**[Buy the MVP bundle - £295](mailto:pricing.frontier@gmail.com?subject=MVP%20Bundle%3A%20Modern%20Insurance%20Pricing%20with%20Python%20and%20Databricks)**

<!-- TODO: Replace mailto link with Gumroad payment page when payment processing is set up -->

---

*Questions or suggestions on the curriculum? Email [pricing.frontier@gmail.com](mailto:pricing.frontier@gmail.com) or open an issue on our [GitHub](https://github.com/burningcost).*
