---
layout: post
title: "How Much of Your GLM Coefficient Is Actually Causal?"
date: 2026-03-07
categories: [techniques]
tags: [causal-inference, double-machine-learning, DML, catboost, pricing, FCA, python, motor]
description: "GLM coefficients measure association, not causation. How Double Machine Learning isolates the causal effect of rating factors from confounding, and why this matters for FCA-compliant pricing."
---

Your GLM has a price elasticity coefficient of -0.045. Your pricing team uses it to optimise renewal offers. Your renewal optimisation is wrong.

Not wrong as in miscoded. Wrong as in: the coefficient does not measure what you think it measures. It is an estimate of the association between price changes and renewal, not the causal effect of price on renewal. Those are different quantities, often materially so, and the difference has direct consequences for how you price, what you tell the FCA, and how you compete.

We built [`insurance-causal`](https://github.com/burningcost/insurance-causal) to measure the difference. This post explains why it matters and how the library works.

---

## The confounding problem in plain terms

A motor insurer running renewals has observational data: for each renewing policy, they have the price change applied, the renewal decision, and the rating factors used to price the policy.

The problem is that price changes are not random. The same pricing model that generates them also uses the rating factors. High-risk customers receive larger premium increases. High-risk customers are also more likely to lapse - for reasons entirely unrelated to price. Poor driving history increases both the price change and the baseline lapse probability.

When you run a GLM with price change as a covariate, you get a coefficient that blends the genuine causal price effect with the risk-driven lapse effect. You cannot disentangle them from within the GLM. The result: the naive coefficient overstates true price sensitivity, because the model is partly attributing risk-driven lapse to price.

The same structure appears everywhere in insurance pricing:

**Age and urban driving.** Young drivers are disproportionately urban. An age coefficient in a motor frequency GLM absorbs some of the urban driving effect, because age and urban exposure are correlated. You cannot tell from the GLM alone how much of the age relativity is actually an age effect versus a proxy for urban driving.

**Telematics harsh braking.** Drivers who brake harshly in city traffic show a strong association with claim frequency. Is that harsh braking causing the claims, or is it an urban driving proxy? The GLM coefficient on harsh braking will absorb both effects, in proportions you cannot observe.

**Channel effects.** Aggregator-sourced business frequently shows lower first-year loss ratios. Is aggregator distribution genuinely reducing claims? Or is it that aggregator customers are a different risk profile, and that difference is not fully captured by your rating factors?

For each of these, pricing teams argue endlessly about how much of the association is "real causation" versus confounding. The classical response is educated judgment and factor stability tests. Both are honest but imprecise. There is a better method.

---

## Double Machine Learning: the two-step

Double Machine Learning (DML), introduced by Chernozhukov et al. in their 2018 paper in *The Econometrics Journal*, solves this by explicitly separating the causal question from the prediction question.

The setup assumes a data generating process:

```
Y = θ₀ · D + g₀(X) + ε
D = m₀(X) + V
```

Where Y is the outcome (renewal, claim frequency), D is the treatment (price change, telematics score), X is the vector of observed confounders (rating factors), and θ₀ is the causal parameter you want. The functions g₀(X) and m₀(X) are unknown and potentially highly nonlinear - the "nuisance" functions.

The naive approach - regress Y on D and X using a flexible ML model - fails because regularisation in the ML model biases the estimate of θ₀. DML's solution is to partial out the confounders from both Y and D separately, then regress the residuals on each other:

1. Fit E[Y|X] using CatBoost (with 5-fold cross-fitting). Compute residuals Ỹ = Y - Ê[Y|X].
2. Fit E[D|X] using CatBoost (with 5-fold cross-fitting). Compute residuals D̃ = D - Ê[D|X].
3. Regress Ỹ on D̃ via OLS. The coefficient is θ̂.

The key mathematical property - Neyman orthogonality - means that errors in the nuisance estimates (steps 1 and 2) produce second-order bias in θ̂, not first-order. When CatBoost converges at the rates achievable on typical insurance datasets, the resulting bias in θ̂ is negligible relative to its standard error. Step 3 is just OLS, which gives valid standard errors and confidence intervals.

The cross-fitting in steps 1 and 2 ensures that nuisance estimation errors are asymptotically independent of the score. The result: θ̂ is √n-consistent and asymptotically normal. You get a real confidence interval on the causal effect.

### Why CatBoost for the nuisance models

The nuisance models need to be flexible enough to capture genuinely nonlinear confounding. A 2024 systematic evaluation (arXiv:2403.14385) found that gradient boosted trees outperform LASSO in the DML nuisance step when confounding is nonlinear - which it is for insurance data with postcode effects and age-vehicle interactions.

We use CatBoost specifically because it handles high-cardinality categoricals (postcode band, vehicle group, occupation class) natively without label encoding. Its ordered boosting also reduces target leakage from categoricals with many levels. The nuisance model defaults are 500 trees, depth 6, learning rate 0.05 - more conservative than a predictive model, appropriate for the debiasing goal.

---

## The confounding bias report

The single output that makes this library immediately useful for a pricing team is the confounding bias report. It answers the question they are already arguing about.

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(
        column="pct_price_change",
        scale="log",
    ),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims",
                 "postcode_band", "vehicle_group"],
    cv_folds=5,
)

model.fit(df)

report = model.confounding_bias_report(naive_coefficient=-0.045)
print(report)
```

```
  treatment         outcome  naive_estimate  causal_estimate    bias  bias_pct
  pct_price_change  renewal         -0.0450          -0.0230  -0.022     -95.7%
```

The naive estimate overstates the true causal effect by 95.7%. Pricing decisions made using -0.045 are substantially biased.

What this means operationally: the genuine causal elasticity is -0.023, not -0.045. A renewal pricing model optimised on -0.045 will act as if price is twice as powerful a lever as it really is. It will undercut on high-risk segments (thinking that doing so recovers retention that would otherwise be lost to price), when actually those customers' lapse propensity is largely driven by risk characteristics, not price. The optimisation is working on a false premise.

You can pass a fitted GLM directly instead of a manual coefficient:

```python
report = model.confounding_bias_report(glm_model=fitted_statsmodels_glm)
```

The library extracts the treatment coefficient from statsmodels, glum, or sklearn linear models automatically.

---

## Treatment types

The library handles the three treatment structures that appear most often in insurance pricing.

**Price change** (continuous, the renewal pricing problem):

```python
from insurance_causal.treatments import PriceChangeTreatment

treatment = PriceChangeTreatment(
    column="pct_price_change",   # proportional: 0.05 = 5% increase
    scale="log",                 # log(1+D) transform; θ is a semi-elasticity
    clip_percentiles=(0.01, 0.99),
)
```

**Binary treatment** (channel, discount flag):

```python
from insurance_causal.treatments import BinaryTreatment

treatment = BinaryTreatment(
    column="is_aggregator",
    positive_label="aggregator",
    negative_label="direct",
)
```

**Continuous treatment** (telematics score):

```python
from insurance_causal.treatments import ContinuousTreatment

treatment = ContinuousTreatment(
    column="harsh_braking_score",
    standardise=True,   # coefficient = effect of 1 SD increase
)
```

---

## Outcome types

Insurance outcomes are not Gaussian. The library handles the distributions that actually appear in the data:

```python
CausalPricingModel(
    outcome_type="binary",      # renewal indicator, conversion
    outcome_type="poisson",     # claim count, with exposure handling
    outcome_type="continuous",  # log loss cost
    outcome_type="gamma",       # claim severity (log-transformed internally)
)
```

For claim frequency, set `exposure_col`:

```python
model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    exposure_col="earned_years",
    treatment=ContinuousTreatment(column="harsh_braking_score", standardise=True),
    confounders=["age_band", "postcode_band", "annual_mileage",
                 "vehicle_group", "pct_urban_mileage", "ncb_years"],
)
model.fit(df)
ate = model.average_treatment_effect()
print(ate)
```

```
Average Treatment Effect
  Treatment: harsh_braking_score
  Outcome:   claim_count
  Estimate:  0.0061
  Std Error: 0.0021
  95% CI:    (0.0020, 0.0102)
  p-value:   0.0038
  N:         42,500
```

The causal effect of a one-standard-deviation increase in harsh braking score on claim frequency is +0.6 percentage points, after controlling for urban mileage percentage, postcode, age, vehicle group, and NCB. The naive GLM coefficient on the same feature, without controlling for urban mileage correctly, was +1.8 percentage points. Most of that association is geography confounding.

---

## CATE by segment

Average treatment effects can conceal heterogeneity that matters for pricing. The price elasticity for a 20-year-old in their first year of driving is probably different from that of a 45-year-old with five years' NCB. `cate_by_segment()` estimates the causal effect separately within subgroups:

```python
cate = model.cate_by_segment(df, segment_col="age_band")
```

This fits a separate DML model per segment and returns a DataFrame with point estimates and confidence intervals for each. It is computationally expensive - proportionally to the number of segments - but gives segment-level inference rather than just a portfolio average.

For risk-decile analysis:

```python
from insurance_causal.diagnostics import cate_by_decile

cate = cate_by_decile(
    model, df,
    score_col="predicted_frequency",
    n_deciles=10,
)
```

A consistent pattern on motor renewal data is that the causal price elasticity is larger in the lower risk deciles than the upper deciles. High-risk customers lapse at high rates regardless of price; the price lever is most effective on the lower-risk, longer-tenure segment. A renewal optimiser built on a single pooled elasticity misses this entirely.

---

## Sensitivity to unobserved confounders

The critical limitation of DML is that it is only as good as the assumption that all relevant confounders are in the `confounders` list. In insurance, this assumption is always imperfect. Actual annual mileage (rather than stated), attitude to risk, and claim reporting behaviour are all relevant and never fully observed.

The sensitivity analysis module quantifies the fragility of a result to violations of this assumption:

```python
from insurance_causal.diagnostics import sensitivity_analysis

ate = model.average_treatment_effect()
report = sensitivity_analysis(
    ate=ate.estimate,
    se=ate.std_error,
    gamma_values=[1.0, 1.25, 1.5, 2.0, 3.0],
)
print(report[["gamma", "conclusion_holds", "ci_lower", "ci_upper"]])
```

The Rosenbaum parameter Γ (gamma) represents the odds ratio of treatment assignment for two units with identical observed confounders. Γ = 1 is no unobserved confounding. Γ = 2 means an unobserved factor doubles the treatment odds for some units relative to comparable units.

If `conclusion_holds` becomes False at Γ = 1.25 - a very mild violation - the result is fragile. If it holds to Γ = 2.0, the result is robust to moderate unobserved confounding. Run this before any causal estimate influences a pricing decision.

---

## Three applications where this matters now

### FCA outcomes testing

Consumer Duty (July 2023) requires insurers to demonstrate that products provide fair value. For pricing, this means being able to show that rating factors reflect genuine risk differentials rather than protected-characteristic proxies.

The FCA's concern: if a pricing factor is substantially a proxy for a protected characteristic, then the premium loading applied for that factor includes a hidden protected-characteristic loading, which is discriminatory. A naive GLM coefficient cannot answer this question. It conflates the genuine risk effect with any confounding by demographics.

DML provides a stronger answer. The causal estimate of a rating factor's effect, controlling for potential demographic confounders, is what you want to report to the FCA. "The GLM coefficient on vehicle group is X" is a correlation. "The DML estimate of vehicle group's causal effect on claim frequency, controlling for age, postcode, and occupation, is Y with 95% CI [a, b]" is evidence.

### Competitive advantage on confounded relativities

If your competitors' age relativities are partly urban-driving relativities, there is a segment where they are systematically overpricing: suburban young drivers. A suburban 22-year-old is priced as if they were a generic 22-year-old, with a significant portion of the age loading actually tracking urban driving behaviour. Your DML estimate separates the genuine age effect from the urban driving effect.

The result is a pricing basis that is more accurate in the suburbs. You can compete aggressively for suburban young drivers because you know the genuine risk level, not the one inflated by urban driving confounding. Your competitors may not.

### Pricing decisions under mix shift

Confounded coefficients misallocate premium when the portfolio mix shifts. If your age coefficient is partly an urban-driving coefficient, and your portfolio mix shifts toward more rural business (say, through a new distribution agreement with a regional broker), your GBM will misprice the new business using a relativity calibrated on a different mix.

A causal estimate is stable to mix shift: it measures the effect of age on claims holding constant urban driving, so changes in the urban-rural portfolio mix do not cause the age relativity to be miscalibrated.

---

## What the library does not solve

**The bad controls problem.** Including a mediator - a variable causally downstream of treatment - as a confounder blocks the causal channel you are trying to measure. NCB is a good example: it is partly caused by claim history, which is the thing you want to measure. If NCB is both a confounder and a mediator for the treatment you are studying, including it will attenuate your estimate.

Think carefully about the causal graph before specifying confounders. This is the hardest part of applying DML, and no library solves it. We are building a lightweight DAG validator for v0.2 that checks for common bad-control patterns before running the model.

**Near-deterministic treatment.** If price changes are nearly entirely determined by the pricing model, the residualised treatment D̃ will have near-zero variance. There is very little exogenous variation to identify the causal effect, and the confidence interval will be wide. The solution is to include genuinely exogenous variation: manual underwriting adjustments, competitive market shocks, or timing effects outside the model.

**Unobserved confounders.** DML cannot adjust for what you did not observe. The sensitivity analysis tells you how fragile your result is, but it cannot fix the underlying data limitation.

---

## Getting started

```bash
uv add insurance-causal
```

Dependencies: `doubleml`, `catboost`, `polars`, `pandas`, `scikit-learn`, `scipy`, `numpy`.

The [README](https://github.com/burningcost/insurance-causal) covers the full API. The recommended starting point for most teams is the price elasticity use case on renewal data - it is the application where the confounding bias is most consistently material and the commercial consequences most direct.

On Databricks Free Edition, DML with CatBoost nuisance models and 5-fold cross-fitting runs in 5-15 minutes on 100k observations with a standard cluster. Use `cv_folds=3` for exploratory work.

---

## The argument for doing this now

The actuarial profession's honest answer to causal questions has always been: "We cannot prove causation from observational data. We use educated judgment." That is correct but incomplete.

DML does not prove causation. It estimates causal effects under the assumption that all relevant confounders are observed, with a valid confidence interval on the estimate, and a sensitivity analysis that characterises how robust the conclusion is to violations of that assumption. That is substantially more than "educated judgment" and substantially better than a GLM correlation.

The academic literature on causal inference in insurance is thin. A 2023 survey (arXiv:2307.16427) reviewed 45 papers across banking, finance, and insurance and concluded "the application remains in its infancy." The methodological tooling is now mature, the regulatory context is creating demand, and teams that can quantify confounding bias will make better pricing decisions than those that cannot. The gap between method and industry practice is the commercial opportunity.

We built `insurance-causal` because the tools existed (DoubleML, CatBoost) but the insurance-specific interface did not. The confounding bias report - one method call, a table showing naive GLM coefficient vs DML causal estimate vs implied bias - is the output that makes this immediately actionable for a pricing team.

Source and issue tracker on [GitHub](https://github.com/burningcost/insurance-causal).
