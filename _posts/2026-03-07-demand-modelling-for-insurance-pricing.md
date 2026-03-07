---
layout: post
title: "Demand Modelling for Insurance Pricing"
date: 2026-03-07
categories: [techniques]
tags: [demand, elasticity, DML, conversion, retention, survival, FCA, GIPP, pricing, python, motor]
description: "How to build a demand model for UK personal lines pricing: conversion, retention, price elasticity, and demand curves. Covers FCA GIPP requirements and the tools that make it tractable."
---

Every UK personal lines pricing team we know has a technical premium model. Most of them have a good one. Few of them have a demand model.

The technical premium tells you what a risk costs. It does not tell you whether the customer will accept the price you set for that risk. On a price comparison website, the gap between your technical premium and the price you quote is not a free variable - it is a commercial decision with a predictable volume consequence, and many teams are making it without quantifying that consequence.

The result is pricing decisions that leave money on the table in both directions. Overpriced risks convert at rates lower than they should. Underpriced risks convert well but generate inadequate margins. Without a demand model, you cannot tell which situation you are in, or by how much.

We built [`insurance-demand`](https://github.com/burningcost/insurance-demand) to close this gap.

---

## Two distinct models, not one

The demand problem in insurance splits cleanly into two questions:

**Conversion modelling**: Given a quote we have issued for new business, what is the probability the customer binds? This is a function of our price, our competitive position, the risk characteristics, and the channel.

**Retention modelling**: Given a renewal invitation we have sent to an existing customer, what is the probability they stay? This is a function of the renewal price, the price change from last year, tenure, payment method, and claim history.

These look superficially similar but they are not the same problem. New business conversion is driven primarily by relative price - on a PCW, rank position matters more than absolute price, and being second rather than first can significantly reduce your conversion rate for that risk. Renewal is driven primarily by price change and customer inertia. A customer who has been with you for six years and pays by direct debit is not evaluating you the same way a new PCW visitor is.

The library handles both with separate classes. Use `ConversionModel` for new business and `RetentionModel` for renewals. They share an interface but have different underlying logic.

---

## Conversion modelling

```bash
uv add insurance-demand
```

The conversion model wraps a logistic GLM or CatBoost classifier, depending on how much data you have and what you want to do with the output. Logistic GLM for interpretability and regulatory documentation. CatBoost when the data is large enough (50k+ quotes) and you want better predictive accuracy.

```python
import polars as pl
from insurance_demand import ConversionModel
from insurance_demand.datasets import load_motor_quotes

df = load_motor_quotes()  # 200k synthetic quotes with known DGP

model = ConversionModel(
    base_estimator="logistic",
    price_col="quoted_price",
    technical_premium_col="technical_premium",
    feature_cols=["age_band", "vehicle_group", "ncd_years", "area", "channel"],
    price_transform="log_ratio",  # models log(price / technical_premium)
)

model.fit(df.filter(pl.col("quote_date") < "2025-01-01"))
probs = model.predict_proba(df.filter(pl.col("quote_date") >= "2025-01-01"))
```

The `price_transform="log_ratio"` option is the right default. You are modelling how conversion responds to pricing above or below technical, not to the absolute price level. A quoted price of £800 means something completely different for a risk with a technical premium of £700 versus one with a technical premium of £400. The ratio removes this ambiguity.

For the marginal price effect at current levels:

```python
elasticity = model.marginal_effect(df)
# Series of dP/d(price) at current quoted price for each row
```

This is the naive marginal effect from the logistic model. It is a useful first pass. It is also biased, for reasons we will come to.

If you have competitor prices from an aggregator data feed, include rank position and the price ratio to the cheapest competitor:

```python
model = ConversionModel(
    base_estimator="catboost",
    price_col="quoted_price",
    technical_premium_col="technical_premium",
    feature_cols=["age_band", "vehicle_group", "ncd_years", "area", "channel",
                  "rank_position", "price_ratio_to_cheapest"],
    price_transform="log_ratio",
)
```

In 2024, 63% of UK motor insurance switchers used a PCW. If your conversion data is primarily PCW data, a model without rank position is misspecified. Rank is not a continuous substitute for price - visibility effects at the top of results pages are difficult to capture with a smooth logistic curve on price alone.

---

## Retention modelling

The retention problem has a survival analysis dimension that conversion does not. A renewal logistic model predicts whether the customer renews at their next anniversary. That is useful. What it does not give you is the full tenure distribution - the probability that a customer who has been with you for two years stays for a third, a fourth, a fifth year - which is what a customer lifetime value model needs.

For renewal probability at next anniversary, the logistic model is sufficient:

```python
from insurance_demand import RetentionModel
from insurance_demand.datasets import load_motor_renewals

df_renewals = load_motor_renewals()  # 100k synthetic renewal records

model = RetentionModel(
    model_type="logistic",
    price_col="renewal_price",
    price_change_col="price_change_pct",
    feature_cols=["tenure_years", "ncd_years", "payment_method",
                  "claim_count_3yr", "channel"],
    event_col="lapsed",
)

model.fit(df_renewals)
renewal_probs = model.predict_proba(df_renewals)
```

For a CLV model, use the survival variants. Cox proportional hazards is the semi-parametric standard; Weibull AFT is appropriate when you want a clean parametric hazard function shape:

```python
model = RetentionModel(
    model_type="survival_weibull",
    tenure_col="tenure_years",
    event_col="lapsed",
    price_col="renewal_price",
    price_change_col="price_change_pct",
    feature_cols=["ncd_years", "payment_method", "claim_count_3yr", "channel"],
)

model.fit(df_renewals)

# P(still active) at years 1, 2, 3, 5 for each policy
survival_curve = model.predict_survival(df_renewals, times=[1, 2, 3, 5])
```

The survival model is better when your observation window includes mid-term censoring - policies that have not yet reached renewal and whose outcome is unknown. The logistic model treats these as complete observations (which they are not). The survival model handles censoring correctly.

For UK motor post-PS21/11, survival models have become more relevant because the FCA's GIPP remedies shifted the industry focus from renewal price extraction to customer lifetime value. If you cannot charge renewing customers more than new customers for the same risk, the commercial game is about retention efficiency rather than inertia extraction. CLV models require multi-period renewal probabilities. Survival analysis provides them.

---

## The problem with naive elasticity estimates

Before going further, a qualification.

Both `ConversionModel.marginal_effect()` and `RetentionModel.marginal_effect()` return naive estimates of price sensitivity. They measure the association between price and conversion/retention in the data. They do not measure the causal effect of changing price.

The difference is material. In observational insurance data, prices are not randomly assigned. High-risk customers receive higher prices. High-risk customers may also have systematically different price sensitivity than low-risk customers - they have fewer alternatives, so they are less elastic. When you regress conversion on price without accounting for this, the estimated price coefficient absorbs both the genuine price effect and the risk-driven demand effect. You get a coefficient that overstates true price sensitivity for high-risk segments.

The standard fix - include technical premium as a control variable - helps but does not fully solve this. It removes the variation explained by your risk model, but your risk model is not perfect. Any residual risk variation that is correlated with both price and demand will still bias the estimate.

This matters for pricing decisions. A renewal optimiser built on an overstated price elasticity will act as if price is a stronger lever than it really is. It will discount more aggressively than it should, and the retention uplift it expects will not materialise.

---

## Unbiased elasticity via Double Machine Learning

The `ElasticityEstimator` class implements Double Machine Learning (DML), from Chernozhukov et al.'s 2018 paper in *The Econometrics Journal*. DML removes confounding bias by residualising both the outcome and the treatment on the set of observed confounders, then regressing the outcome residuals on the treatment residuals.

The algorithm, concretely:

1. Fit E[conversion | confounders] using CatBoost. Compute residuals: Ỹ = conversion - predicted\_conversion.
2. Fit E[log\_price\_ratio | confounders] using CatBoost. Compute residuals: D̃ = log\_price\_ratio - predicted\_log\_price\_ratio.
3. Regress Ỹ on D̃ via OLS. The coefficient is the debiased elasticity estimate.

Both steps use 5-fold cross-fitting to ensure nuisance estimation errors are asymptotically independent. The result is a √n-consistent estimate with a valid confidence interval.

```python
from insurance_demand import ElasticityEstimator

est = ElasticityEstimator(
    outcome_col="converted",
    treatment_col="log_price_ratio",
    feature_cols=["age_band", "vehicle_group", "ncd_years", "area",
                  "channel", "month"],
    n_folds=5,
    heterogeneous=False,  # global elasticity
)

est.fit(df_quotes)
print(est.summary())
```

```
Price Elasticity (DML)
  Treatment: log_price_ratio
  Outcome:   converted
  Estimate:  -0.312
  Std Error:  0.021
  95% CI:    (-0.353, -0.271)
  N:          187,432
```

The interpretation: a 10% increase in the price-to-technical-premium ratio reduces conversion probability by approximately 3.1 percentage points at the average conversion rate. With a 95% confidence interval of (-3.5pp, -2.7pp), this estimate is precise enough to use in an optimiser.

For segment-level elasticity, set `heterogeneous=True`. This uses `CausalForestDML` from the `econml` library under the hood:

```python
est = ElasticityEstimator(
    outcome_col="converted",
    treatment_col="log_price_ratio",
    feature_cols=["age_band", "vehicle_group", "ncd_years", "area", "channel"],
    n_folds=5,
    heterogeneous=True,
)

est.fit(df_quotes)
per_customer_elasticity = est.effect(df_quotes)
```

Young drivers and PCW customers will come out more elastic. Customers with high NCD and long tenure will come out less elastic. These are useful inputs to a segmented pricing decision. A single portfolio-average elasticity loses all of this, and optimisers that work from a single number will allocate discount budgets poorly.

The DML approach requires volume: minimum 50,000 quotes with meaningful price variation. It also requires that technical premium was stored at quote time - not retroactively recalculated. If your technical premium has been rebased since the quotes were written, the log\_price\_ratio will be wrong and the elasticity estimate will be biased in a way DML cannot fix.

---

## Demand curves and optimal pricing

Once you have elasticity estimates, the `DemandCurve` class converts them into callable price-to-probability functions. These are the inputs a rate optimiser needs to compute the expected volume consequence of any given price change.

```python
from insurance_demand import DemandCurve

curve = DemandCurve.from_estimator(
    estimator=est,
    base_conversion_rate=0.18,  # current book average
    form="semi_log",            # P(convert) = base * exp(theta * log_price_ratio)
)

# Probability of conversion at different price levels
import polars as pl
prices = pl.Series("price_ratio", [0.9, 1.0, 1.1, 1.2, 1.3])
print(curve.predict(prices))
# [0.217, 0.180, 0.149, 0.124, 0.102]
```

For single-segment profit maximisation:

```python
from insurance_demand import OptimalPrice

opt = OptimalPrice(
    demand_curve=curve,
    technical_premium=650.0,
    cost_per_policy=45.0,       # acquisition + admin
    enbp_ceiling=None,          # new business: no ENBP constraint
)

result = opt.solve()
print(f"Optimal price: £{result.optimal_price:.0f}")
print(f"Expected conversion: {result.expected_conversion:.1%}")
print(f"Expected margin: £{result.expected_margin:.0f}")
```

For renewal pricing, the ENBP ceiling is hard:

```python
opt = OptimalPrice(
    demand_curve=renewal_curve,
    technical_premium=650.0,
    cost_per_policy=30.0,
    enbp_ceiling=710.0,         # new business price via same channel
)

result = opt.solve()
# result.optimal_price will never exceed 710.0
```

The `OptimalPrice` class maximises expected profit per policy, where expected profit = (price - technical premium - cost) x P(conversion or renewal). This is a simple but correct single-policy formulation. For portfolio-level optimisation with factor-level constraints, use `rate-optimiser` and feed the demand curves in as inputs.

---

## FCA compliance: the ENBP checker

PS21/11 requires that renewal prices do not exceed the Equivalent New Business Price for the same risk profile through the same channel. The rule has been in force since January 2022. The FCA's July 2025 evaluation (EP25/2) confirmed it has substantially eliminated price-walking, and also that multi-firm reviews are ongoing.

The `ENBPChecker` audits a renewal portfolio against a new business price table:

```python
from insurance_demand.compliance import ENBPChecker

checker = ENBPChecker(
    new_business_price_col="nb_price",
    renewal_price_col="renewal_price",
    channel_col="channel",
    tolerance=0.0,  # strict: renewal must be <= NB price
)

violations = checker.check(df_renewals)
print(violations.shape[0], "policies with renewal_price > ENBP")
print(violations.select(["policy_id", "channel", "renewal_price",
                          "nb_price", "excess"]))
```

The ENBP constraint binds by channel. A customer who originally came via Confused.com has an ENBP calculated from your Confused.com new business price for that risk, not your direct price. UK insurers who quote differently across channels - which is common practice - must compute ENBP on a channel-specific basis.

The FCA's MS18/1 market study found that motor insurance customers with 5+ years' tenure were paying on average around 85% more than equivalent new customers in 2018. The ENBP rule made that illegal. The demand modelling that was used to identify and exploit inelastic customers is now only useful for the opposite purpose: identifying customers who need a retention discount before they lapse.

---

## What DML cannot fix

Three honest limitations.

**Data quality.** DML residualises on observed confounders. If technical premium was retroactively recalculated, if competitor prices are not in your dataset, or if channel attribution is unreliable, the residualisation will be wrong. Garbage in, garbage out - but more precisely than before.

**Unobserved confounders.** If you raised prices in Q3 2024 because of reinsurance cost pressures while also running a targeted direct mail campaign in the same quarter, and campaign exposure is not in your dataset, the price effect and campaign effect are confounded. DML cannot separate them. The `sensitivity_analysis()` method in the estimator tells you how large unobserved confounding would need to be to overturn your result - run it before taking elasticity estimates to a pricing committee.

**Near-zero treatment variation.** DML identifies elasticity from the variation in `log_price_ratio` after removing the part explained by confounders. If your loading decisions have been very uniform - if you applied a single commercial loading to the whole book for a long period - there is very little variation left after residualisation, and the confidence interval on the elasticity estimate will be wide. You need rate variation to estimate price effects.

---

## Getting started

```bash
uv add insurance-demand

# For DML elasticity estimation:
uv add "insurance-demand[dml]"

# For survival-based retention models:
uv add "insurance-demand[survival]"

# Everything:
uv add "insurance-demand[dml,survival]"
```

Source and issue tracker on [GitHub](https://github.com/burningcost/insurance-demand).

The minimum viable starting point: fit a `ConversionModel` on your quote data and run `model.predict_proba()` on your current book. Plot predicted conversion against the actual bind rate by price decile. If they match, your GLM is adequate. If they diverge above the 7th or 8th decile, you have a non-linearity that a logistic GLM is not capturing and CatBoost will.

After that: run `ElasticityEstimator` on a year of PCW quote data. Compare the DML estimate with `ConversionModel.marginal_effect()`. If they are materially different - and they often are - the difference is the confounding bias you have been pricing with. That number, concretely, is how wrong your current pricing assumption is.

The libraries that commercial platforms sell for significant annual fees are doing the same maths. The methodology is not proprietary. What they sell is the integration, the UI, and the professional services. `insurance-demand` is the methodology in an auditable Python package with no vendor lock-in and an API that reads like sklearn.
