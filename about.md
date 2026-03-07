---
layout: page
title: About
permalink: /about/
---

Burning Cost builds open-source Python tools for UK personal lines and commercial pricing teams.

The name comes from a basic actuarial concept: burning cost is claims incurred divided by premium earned. Simple, direct, no mystification. That is how we think about tooling.

---

## What we have built

Thirteen Python libraries covering the full pricing workflow.

**Model interpretation**

- [`shap-relativities`](https://github.com/burningcost/shap-relativities) - extract multiplicative rating factor tables from CatBoost models using SHAP values, in the same format as exp(beta) from a GLM

**Validation**

- [`insurance-cv`](https://github.com/burningcost/insurance-cv) - temporally-correct walk-forward cross-validation with IBNR buffer support and sklearn-compatible scorers
- [`insurance-conformal`](https://github.com/burningcost/insurance-conformal) - distribution-free prediction intervals for insurance GBMs, implementing the variance-weighted non-conformity score from Manna et al. (2025)

**Techniques**

- [`credibility`](https://github.com/burningcost/credibility) - Buhlmann-Straub credibility in Python with mixed-model equivalence checks
- [`bayesian-pricing`](https://github.com/burningcost/bayesian-pricing) - hierarchical Bayesian models for thin-data pricing segments
- [`insurance-interactions`](https://github.com/burningcost/insurance-interactions) - detecting and quantifying interaction effects that a main-effects GLM cannot see
- [`insurance-causal`](https://github.com/burningcost/insurance-causal) - causal inference for insurance pricing; separating genuine risk signal from confounded association
- [`insurance-spatial`](https://github.com/burningcost/insurance-spatial) - BYM2 spatial models for postcode-level territory ratemaking

**Commercial**

- [`rate-optimiser`](https://github.com/burningcost/rate-optimiser) - constrained rate change optimisation; the efficient frontier between loss ratio target and movement cap constraints
- [`insurance-demand`](https://github.com/burningcost/insurance-demand) - price elasticity and conversion modelling, integrated with rate optimisation

**Compliance**

- [`insurance-fairness`](https://github.com/burningcost/insurance-fairness) - proxy discrimination detection and FCA Consumer Duty documentation support

**Infrastructure**

- [`insurance-datasets`](https://github.com/burningcost/insurance-datasets) - synthetic personal lines datasets with realistic exposure, claim count, and development structure, for testing and teaching
- [`burning-cost`](https://github.com/burningcost/burning-cost) - the Burning Cost CLI; orchestration for pricing model pipelines

---

## The problem we are solving

UK pricing teams have been building GBMs for years, mostly CatBoost. The models are better than the production GLMs. But many teams are still taking the GLM to production, because the GBM outputs are not in a form that a rating engine, regulator, or pricing committee can work with.

The issue is not technical skill. It is tooling. There is no standard Python library that extracts a multiplicative relativities table from a GBM. There is no standard library that does temporally-correct walk-forward cross-validation with IBNR buffers. There is no standard library that builds a constrained rate optimisation a pricing actuary can challenge.

We wrote those libraries because we needed them. Then we kept going.

---

## Training course

We also run a training course - [Modern Insurance Pricing with Python and Databricks](/course/) - for pricing actuaries and analysts who want to use these tools properly. Eight modules, written from first principles for insurance, not adapted from generic data science tutorials.

---

## Contact

**Email:** [pricing.frontier@gmail.com](mailto:pricing.frontier@gmail.com)

**GitHub:** [github.com/burningcost](https://github.com/burningcost)
