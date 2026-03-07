# burningcost.github.io

![License: MIT](https://img.shields.io/badge/license-MIT-green)

Insurance pricing education and open-source tooling for UK actuaries and pricing teams.

The site at [burningcost.github.io](https://burningcost.github.io) publishes worked examples, methodology explainers, and links to all open-source libraries.

## Libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burningcost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burningcost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burningcost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burningcost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burningcost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [credibility](https://github.com/burningcost/credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [rate-optimiser](https://github.com/burningcost/rate-optimiser) | Constrained rate change optimisation with FCA PS21/5 compliance |
| [insurance-demand](https://github.com/burningcost/insurance-demand) | Conversion, retention, and price elasticity modelling |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burningcost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-causal](https://github.com/burningcost/insurance-causal) | Double Machine Learning for causal pricing inference |
| [insurance-monitoring](https://github.com/burningcost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

**Spatial**

| Library | Description |
|---------|-------------|
| [insurance-spatial](https://github.com/burningcost/insurance-spatial) | BYM2 spatial territory ratemaking for UK personal lines |

## Site structure

Built with Jekyll. Posts are in `_posts/`. The `course/` directory contains structured learning materials.
