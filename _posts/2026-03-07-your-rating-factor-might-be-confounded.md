---
layout: post
title: "Your Rating Factor Might Be Confounded"
date: 2026-03-07
categories: [techniques]
tags: [causal-inference, double-machine-learning, DML, confounding, pricing, python, motor, elasticity]
---

When a pricing actuary adds a factor to a GLM and reads off `exp(β)`, what they get is the association between that factor and loss cost, holding other factors constant. That sounds like a causal estimate. It is not.

Consider a concrete UK motor example. Engine size is in your GLM. The coefficient is positive: larger engines are associated with higher claim frequency. But engine size correlates with the type of driver who buys the car, their age, their postcode, how they use the vehicle, whether they have telematics. A GLM that includes all those factors does partial adjustment, but the adjustment is correct only if the confounders are included linearly and the model is correctly specified. In practice, the GLM coefficient on engine size is part causal effect and part residual confounding from everything engine size correlates with that the GLM has not fully captured.

This matters the moment someone asks: "If we reduce the weight on engine size in the rate table, what happens to our loss ratio?"

That is a causal question. The GLM coefficient gives you the wrong answer for it - not directionally wrong, necessarily, but quantitatively wrong in ways that are hard to characterise without a method that is actually designed to answer causal questions.

Double Machine Learning is that method. We built `insurance-causal` to bring it into the insurance pricing workflow.

---

## Why the GLM coefficient is biased

The standard econometrics framing: you want to estimate the effect of a treatment D on an outcome Y, controlling for confounders X. You fit:

```
Y = β · D + γ · X + ε
```

The OLS estimate of β is unbiased if the confounders X are included linearly and completely. In insurance, neither condition holds.

The confounders are not linear. Age affects loss cost through a non-monotone curve - young drivers are high risk, middle-aged drivers are lower risk, elderly drivers start rising again. Vehicle deprivation index interacts with area and vehicle type in ways a linear term cannot capture. When the confounding relationship is nonlinear and you model it linearly, the residual confounding leaks into your estimate of β. The GLM coefficient absorbs it.

The confounders are not complete. You observe postcode but not whether the driver commutes through central Manchester. You observe annual mileage (self-reported) but not actual miles driven. You observe NCD but not the full claims history that generated it. Every omitted variable that correlates with your factor of interest biases its coefficient.

The practical consequence: when management asks "what is the true effect of engine size on claims?" and you show them the GLM coefficient, you are showing them an estimate that is biased by an unknown amount in an uncertain direction. You cannot, from the GLM output alone, tell them how large the bias is.

Double Machine Learning tells you.

---

## The DML approach

The foundational paper is Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey and Robins (2018), "Double/Debiased Machine Learning for Treatment and Structural Parameters", _The Econometrics Journal_, 21(1): C1-C68. The mathematical result they prove is: you can isolate the causal effect of D on Y from observational data if you correctly partial out the confounders, and you can use ML to do the partialling out without the regularisation bias destroying your inference.

The key procedure is three steps:

**Step 1.** Fit a flexible model (we use CatBoost, but any gradient booster or neural net works) to predict Y from X. Subtract: this is the residualised outcome, Ỹ = Y - E[Y|X].

**Step 2.** Fit a separate flexible model to predict D from X. Subtract: residualised treatment, D̃ = D - E[D|X].

**Step 3.** Regress Ỹ on D̃ with OLS. The coefficient is θ̂.

The intuition: D̃ is the part of the treatment that cannot be explained by the confounders. It is the exogenous variation in D - the variation that is not systematically correlated with risk characteristics. Regressing the residualised outcome on residualised treatment isolates the causal channel.

The mathematical guarantee (the "debiasing" in the name): even though the ML models in steps 1 and 2 introduce regularisation bias, those biases cancel in the final regression at order δ² rather than δ. When ML convergence rates are √n or faster - which they are for CatBoost on most insurance datasets - the bias in θ̂ is negligible relative to its standard error. You get a valid confidence interval.

**Cross-fitting** prevents overfitting bias: the nuisance models in steps 1 and 2 are trained on held-out folds, so the residuals used in step 3 are not on data the ML model has already seen. Use 5-fold cross-fitting as the default.

---

## What confounding looks like in practice

Take price elasticity on motor renewals. You want to know: if we increase the renewal premium by 5%, how many customers will we lose?

The observable data is: historical renewal decisions, with the premium charged and whether the customer renewed. You fit a model. The coefficient on price looks like an elasticity estimate. It is not.

The confounding structure: customers who received large premium increases were, in general, higher-risk customers - that is why you increased their premium. Higher-risk customers lapse for a variety of reasons, including but not limited to price. If you regress renewal indicator on price change, you are picking up both the causal effect of price on renewal probability and the correlation between premium increases and all the other reasons high-risk customers lapse.

The naive GLM estimate of price elasticity will overstate price sensitivity. You will think customers are more price-elastic than they are. Your renewal pricing strategy will give back margin unnecessarily.

The DML estimate partials out the risk factors that drove the price change - age, vehicle, postcode, NCD, prior claims - from both the renewal outcome and the price change itself. What remains in D̃ is the exogenous variation: price changes that were not mechanically driven by risk profile. Changes from manual underwriting decisions, bulk re-rating cycle timing, competitor market movements. Regressing the residualised renewal indicator on residualised price change estimates the true causal elasticity.

In Guelman and Guillén (2014, _Expert Systems with Applications_, 41(2)), applying causal methods to motor renewal data found that the propensity-score-adjusted elasticity was materially lower than the naive estimate - roughly half the magnitude in their Spanish motor dataset. That finding is directionally consistent with every commercial team we have spoken to who has done this comparison in UK data.

---

## Using the library

```bash
uv add insurance-causal
```

```python
import polars as pl
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="renewed",
    outcome_type="binary",
    treatment=PriceChangeTreatment(
        column="pct_price_change",
        scale="log",
    ),
    confounders=["age_band", "vehicle_group", "postcode_band", "ncd", "prior_claims"],
    nuisance_model="catboost",   # used for both E[Y|X] and E[D|X]
    cv_folds=5,
)

model.fit(df_train, exposure_col="policy_years")

ate = model.average_treatment_effect()
```

`ate` gives you the average causal effect of price on renewal probability - a coefficient you can interpret as an elasticity, with a confidence interval:

```
ATE: -0.023  (95% CI: -0.031 to -0.015)
Interpretation: a 1% price increase causes a 2.3% reduction in renewal probability
```

The confidence interval is computed from the asymptotic normality of the OLS estimate in step 3. It is a valid frequentist interval, not a bootstrap approximation.

---

## The confounding bias report

The feature that makes this useful to a pricing team - rather than just interesting to a statistician - is the comparison between the naive GLM estimate and the DML causal estimate:

```python
model.confounding_bias_report()
```

```
Factor: pct_price_change
  Naive GLM coefficient:  -0.047
  DML causal estimate:    -0.023
  Implied confounding:    +0.024
  Interpretation: confounding inflates apparent price sensitivity by ~2x

Factor: telematics_harsh_braking
  Naive GLM coefficient:  +0.181
  DML causal estimate:    +0.062
  Implied confounding:    +0.119
  Interpretation: two-thirds of the observed association is geography confounding
```

This table answers the question that pricing committees argue about endlessly: "How much of this coefficient is real?" For price elasticity, the confounding bias in this example is large enough to materially affect renewal pricing decisions. For the telematics factor, the result is more striking: if the causal effect is genuinely 0.062 rather than 0.181, a pricing factor weighted at 0.181 is doing the wrong thing and should be reduced.

The confounding bias report is the output that justifies doing this analysis. It converts an abstract methodological point - "GLM coefficients are biased" - into a table a head of pricing can use.

---

## Heterogeneous treatment effects

For price elasticity in particular, the average treatment effect is the wrong thing to target. Different customers have different price sensitivity. A 15-year loyal customer and a first-year aggregator customer should not be assumed to have the same elasticity, and your pricing strategy should reflect that.

`CausalForestDML` (Wager and Athey, 2018, _Journal of the American Statistical Association_, 113(523)) extends DML to heterogeneous treatment effects. The DML residualisation happens first - same two steps - then a causal forest is fitted on the residuals to estimate τ(x), the treatment effect for each individual customer:

```python
from insurance_causal.estimators import HeterogeneousTreatmentEffect

hte = HeterogeneousTreatmentEffect(
    outcome="renewed",
    treatment=PriceChangeTreatment(column="pct_price_change"),
    confounders=["age_band", "vehicle_group", "postcode_band", "ncd", "prior_claims"],
    heterogeneity_features=["time_on_book", "ncd", "age_band"],
    nuisance_model="catboost",
)

hte.fit(df_train)
cate = hte.effect(df_test)
# Returns DataFrame with point estimate and CI per policy
```

The `cate` DataFrame gives you an individual-level causal effect estimate. The commercial use case: the renewal targeting rule becomes "offer a discount when τ̂(x) × expected_margin > discount_cost." You offer discounts where they are causally effective, not where customers happen to have high observed loyalty scores.

---

## Sensitivity to unobserved confounding

The most common challenge to any causal analysis from observational data is: "What about confounders you have not measured?" This is a valid question and the right response to it is not defensiveness - it is a sensitivity analysis.

Rosenbaum bounds quantify how strong an unobserved confounder would need to be to overturn the conclusion:

```python
sensitivity = model.sensitivity_to_unobserved_confounders(gamma_range=[1.1, 1.5, 2.0, 3.0])
print(sensitivity)
```

```
Gamma=1.1: estimate range [-0.031, -0.014]  - sign stable
Gamma=1.5: estimate range [-0.039, -0.006]  - sign stable
Gamma=2.0: estimate range [-0.048, +0.002]  - sign borderline
Gamma=3.0: estimate range [-0.061, +0.016]  - sign could flip
```

Gamma is the odds ratio of treatment assignment for an unobserved binary confounder. A value of 1.5 means: "If there is an unobserved variable that makes customers twice as likely to receive a price increase, and also affects renewal probability by some amount, how does that change our estimate?" If the sign is stable to Gamma=2.0 but flips at Gamma=3.0, the result is reasonably robust to moderate unobserved confounding but fragile to severe confounding.

For price elasticity, where the treatment (price change) is nearly determined by the pricing model itself, we recommend presenting the sensitivity analysis alongside the point estimate. A regulator or a pricing committee that sees "our estimate is -0.023 and it holds up to Gamma=2.0" is better informed than one that sees "-0.023 with no caveats."

---

## What DML cannot fix

Three things to be honest about.

**Near-deterministic treatment.** In motor renewal, the price change is largely determined by the pricing model - which means it is largely determined by the risk factors in X. The residualised treatment D̃ has low variance when the treatment is nearly a deterministic function of X. Low variance in D̃ means a noisy estimate of θ̂. The practical consequence: if your pricing model is very accurately determining prices from risk factors, the exogenous variation in price is small, and the DML standard error is wide. You might get a statistically valid estimate with a confidence interval so wide it is not actionable. The fix is to identify genuinely exogenous price variation - from manual overrides, competitor market events, bulk re-rating timing - and use those as instruments. That is the PLIV (partially linear IV) extension, also supported by the library.

**Bad controls.** If you include a mediator as a confounder, you block the causal channel. NCD is a confounder for some questions (it causes both price and loss cost) but a mediator for others (it is caused by prior claims, which also cause future claims). Partialling out NCD when NCD is a mediator produces a wrong estimate - the "bad control" problem. The library includes a DAG specification module that checks for this before fitting:

```python
from insurance_causal import CausalDAG

dag = CausalDAG()
dag.add_treatment("pct_price_change")
dag.add_outcome("renewed")
dag.add_confounder("age_band")
dag.add_confounder("ncd")  # will flag ncd as a potential mediator
dag.validate()
# Warning: ncd is downstream of prior_claims which may affect renewed directly.
# Consider whether ncd is a confounder or mediator for this treatment.
```

**Unobserved confounders.** DML is only as good as the assumption that all relevant confounders are observed. If attitude to risk, actual driving patterns, or claim reporting behaviour are correlated with your factor of interest and not in the data, the causal estimate is still biased. The sensitivity analysis bounds how bad this could be, but it cannot eliminate it. This is not a weakness unique to DML - it applies to every observational analysis. What DML gives you is a cleaner estimate conditional on the observed confounders, plus the tools to quantify vulnerability to unobserved ones.

---

## When to care about this

Not every pricing question requires a causal analysis. If you are building a predictive model to rank risks - to separate high-risk from low-risk policies - the GLM or GBM coefficient is fine. You want prediction accuracy, not causal identification.

Causal analysis becomes important when the question is about intervention - when you are asking what happens if you change something. The situations in UK motor pricing where this matters:

**Renewal pricing decisions.** The whole point of estimating price elasticity is to set renewal prices that maximise retention-weighted margin. A biased elasticity estimate leads to systematically wrong pricing strategy. If you are overestimating price sensitivity by a factor of two, you are giving back margin to customers who would have renewed anyway.

**Factor weighting decisions.** When management asks "should we reduce the weight on vehicle group in our rating structure?", the answer depends on the causal effect of vehicle group on claims, not the GLM association. If 60% of the vehicle group coefficient is confounded by age and postcode, reducing the weight by 60% would not move the loss ratio at all.

**Regulatory defence of pricing factors.** Under FCA Consumer Duty and PS21/5, an insurer defending a pricing factor needs more than "it correlates with claims in our data." A causal estimate with a confidence interval and a sensitivity analysis is a substantially stronger regulatory argument. "We estimate the causal effect is 1.4× with 95% CI 1.2-1.7, and this holds up to unobserved confounders with Gamma=2.0" is harder to challenge than "exp(β) from our GLM is 1.4."

**Telematics factor selection.** The specific question of which telematics features have genuine causal standing - as opposed to being correlated with geography, driving occasion type, or other factors - is one DML is well-suited to. A telematics feature with a large observed association and a small DML estimate is a proxy, not a cause. It probably should not be in the rating structure.

---

## Getting started

```bash
uv add insurance-causal
```

Source and issue tracker on [GitHub](https://github.com/burningcost/insurance-causal).

Start with the confounding bias report. Run it on a factor you already have in your GLM and think you understand. If the DML estimate and the GLM coefficient are within 10% of each other, confounding is not a major problem for that factor - the GLM is doing a reasonable job. If they diverge materially, you have evidence that the factor's GLM coefficient is carrying confounding bias, and you should think carefully about how that affects the decisions it is used for.

The right use of this library is not to replace your GLM. It is to audit the causal standing of the factors in your GLM - to find out which coefficients you can trust for intervention decisions and which ones are proxies. Most pricing models have a mix of both. Knowing which is which is commercially valuable.
