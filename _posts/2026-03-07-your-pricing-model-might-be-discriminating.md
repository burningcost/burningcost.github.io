---
layout: post
title: "Your Pricing Model Might Be Discriminating"
date: 2026-03-07
categories: [techniques, compliance]
tags: [fairness, proxy-discrimination, FCA, Consumer-Duty, Equality-Act, LRTW, GBM, postcode, python, motor, insurance-fairness]
description: "How to detect and correct proxy discrimination in UK insurance pricing models. Using SHAP and the insurance-fairness library to identify protected characteristic leakage under FCA Consumer Duty."
---

Your GBM uses postcode as a feature. That is completely standard practice in UK motor insurance. Postcode predicts claim frequency. The model validation looks fine.

But postcode also correlates with ethnicity. In London, in particular, postcode bands are not race-neutral. The ONS Census 2021 shows that Inner East London LSOAs have non-white British populations above 70%. Your model does not explicitly use ethnicity. It does not need to. Postcode is doing that job for it.

This is proxy discrimination. It is the subject of Equality Act 2010 Section 19. And as of July 2023, it is directly in scope of FCA Consumer Duty, specifically PRIN 2A.4, which requires firms to monitor whether their products provide fair value across groups defined by protected characteristics.

The FCA has signalled this is a priority area. In TR24/2 (August 2024), they found that fair value assessments from reviewed firms were too high level and lacked the granularity to adequately evidence good outcomes across customer groups. The next step is enforcement.

We built [`insurance-fairness`](https://github.com/burningcost/insurance-fairness) to give pricing teams an audit-ready answer to this problem. This post explains the regulatory exposure, the technical problem of proxy discrimination in insurance models, and how to use the library to measure and mitigate it.

---

## Why "we don't use race" is not a defence

The intuition that discrimination requires intent - that using a protected characteristic explicitly is the only thing the law prohibits - is wrong, and has been wrong in English law since the Race Relations Act 1976. Section 19 of the Equality Act 2010 defines indirect discrimination as: applying a provision, criterion or practice (PCP) that puts people sharing a protected characteristic at a particular disadvantage, where you cannot justify it as a proportionate means of achieving a legitimate aim.

Pricing models do not need to include race. They need to produce premium predictions that correlate with race to create indirect discrimination.

Citizens Advice published analysis in 2022 finding that motor insurance customers in postcodes where more than 50% of residents are people of colour paid an average of £280 more per year than otherwise similar customers in majority-white postcodes. Their estimate of the total annual excess was £213 million. The FCA took that analysis seriously.

The legitimate aim defence - actuarial justification that the pricing factor genuinely reflects risk, not demographic correlation - is available, but it requires you to demonstrate it. Not to assert it. The regulator will want to see the work.

Proxy detection is that work.

---

## What proxy discrimination looks like in a GBM

Consider a standard UK motor frequency model: CatBoost, Poisson objective, features including driver age, vehicle group, NCD, occupation, and postcode area.

The model has never seen ethnicity. It cannot, because you do not collect it. But it has seen postcode, and postcode in your training data correlates with both claim frequency and, through the demographic composition of postcode areas, ethnicity.

When you train the model, it learns:
1. Claim frequency varies by postcode (true - driven by urban density, traffic, theft rates, road quality).
2. Some of that postcode effect captures risk-relevant variation (the causal channel you want).
3. Some of it captures demographic correlation with protected characteristics (the proxy discrimination channel you do not want).

The GBM cannot distinguish these. It optimises Poisson deviance. It does not know or care that postcode is a proxy for ethnicity as well as a predictor of risk. The resulting model bakes both effects into the postcode feature importance.

The same logic applies to occupation in home insurance (manual occupations correlate with socioeconomic status, which correlates with both risk and protected characteristics), to vehicle group in motor (performance vehicles correlate with age, income, and indirectly race and disability), and to annual mileage (self-reported mileage is lower in groups that work from home, which correlates with occupation, which correlates with protected characteristics).

Every rating factor that correlates with a protected characteristic is a potential proxy discrimination channel. Postcode is the most visible one, but it is not the only one.

---

## The LRTW framework: what you actually need to measure

The academic literature on fair insurance pricing converged on a specific framework: Lindholm, Richman, Tsanakas and Wüthrich, published in ASTIN Bulletin 2022, with three subsequent papers extending it through 2024.

The LRTW definition: a pricing model f(X) exhibits proxy discrimination if it is not conditionally independent of a protected attribute S given the observed rating factors X.

In plain terms: if knowing a customer's ethnicity still gives you information about what your model will charge them - even after you know all their risk factors - your model is discriminating.

The LRTW discrimination-free price is defined as:

```
p_DF(x) = E[f(X, S) | X = x] = integral f(x, s) * p(s | x) ds
```

This marginalises the model's best-estimate price over the conditional distribution of the protected attribute given the non-protected features. It is the price that uses risk information without implicitly using protected-characteristic information.

Three things that follow from this that matter for your audit:

**Unawareness is not discrimination-free.** Simply leaving S out of the model does not fix proxy discrimination when X correlates with S. The correct fix is to know the conditional distribution p(s | x) - typically from ONS Census data linked to postcodes - and use it to marginalise.

**Proxy discrimination is not the same as demographic disparity.** A discrimination-free model will still charge different average premiums to different demographic groups, because risk genuinely correlates with protected characteristics. The law is concerned with proxy discrimination - charging more than the risk-based rate implies because of demographic correlation - not with demographic disparities themselves. Libraries that implement demographic parity (equal average premiums across groups) are solving the wrong problem for UK insurance.

**Actuarial justification requires decomposition.** A regulator who asks "why does this group pay more?" needs you to be able to say "this much is genuine risk differential, this much is unexplained by risk." The LRTW framework gives you that decomposition. A raw comparison of average premiums by ethnicity does not.

---

## Installing the library

```bash
uv add insurance-fairness
```

The library requires a trained CatBoost model and a Polars DataFrame. It assumes a Poisson or Tweedie frequency model with a log link.

---

## Step 1: measuring proxy correlation

Before you compute discrimination-free prices, you need to know which features are proxies and how strongly they correlate with protected characteristics.

```python
import polars as pl
from insurance_fairness import ProxyAudit

# df_train is your training data; needs postcode LSOA linked to ONS Census
# ethnicity_prop is proportion of non-white-British population at LSOA level
# from ONS Census 2021, linked via postcode -> LSOA lookup

audit = ProxyAudit(
    protected_proxy="ethnicity_prop",     # continuous proxy for S
    rating_factors=["postcode_band", "vehicle_group", "occupation", "ncd", "age_band"],
    exposure_col="policy_years",
)

proxy_report = audit.fit(df_train).report()
```

```
Proxy Correlation Audit
Protected characteristic proxy: ethnicity_prop (continuous, 0-1)
Metric: Spearman rank correlation, exposure-weighted

postcode_band     |  r = +0.61  ***  HIGH  - strong proxy correlation
occupation        |  r = +0.29  **   MEDIUM - moderate proxy correlation
vehicle_group     |  r = +0.18  *    LOW  - weak proxy correlation
age_band          |  r = -0.04        NONE - not a significant proxy
ncd               |  r = +0.02        NONE - not a significant proxy

Warning: postcode_band has high proxy correlation with protected characteristic.
Model using postcode_band will require discrimination-free price adjustment.
```

The audit runs exposure-weighted Spearman correlations between each rating factor and the protected characteristic proxy. Postcode, in this example, is the main offender - correlation of 0.61 is not marginal. Occupation is worth watching. Age and NCD are not significant proxies here.

This table is your evidence base. If you can show the regulator that occupation had correlation 0.29 and that you investigated it, that is a substantially better position than having no record of the analysis.

---

## Step 2: measuring disparate impact

Proxy correlation tells you which features are suspect. Disparate impact measurement tells you what the model's actual predictions are doing.

```python
from insurance_fairness import DisparateImpactAudit
from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model("frequency_model.cbm")

di_audit = DisparateImpactAudit(
    model=model,
    protected_proxy="ethnicity_prop",
    protected_threshold=0.5,   # high-minority-proportion areas vs low
    exposure_col="policy_years",
    metric="log_ratio",        # log(E[price | high]) - log(E[price | low])
)

di_result = di_audit.fit(df_test)
print(di_result.summary())
```

```
Disparate Impact Audit
Protected group: ethnicity_prop > 0.50 (high minority proportion)
Reference group: ethnicity_prop <= 0.50

Exposure-weighted mean log predicted frequency:
  Protected group:   -2.87
  Reference group:   -3.14

Log ratio (protected / reference): +0.27
Implied multiplicative disparity: 1.31x

Interpretation: the model predicts 31% higher frequency in high-minority-proportion
postcode areas relative to low-minority-proportion areas, after conditioning
on other rating factors.

Note: this is a disparate impact measurement, not a discrimination-free price
adjustment. Some or all of this disparity may be genuine risk differential.
See discrimination-free pricing step to decompose.
```

The key discipline here: a disparate impact of 1.31x is not automatically discrimination. Risk genuinely varies by postcode. The question is how much of the 1.31x is genuine risk variation versus demographic correlation. The library is explicit about this - it reports the total disparity and tells you to decompose it rather than treating all disparity as discrimination.

---

## Step 3: computing discrimination-free prices

This is the LRTW calculation. It requires p(S | X) - the conditional distribution of the protected characteristic given observed features. In the UK postcode context, we use ONS Census 2021 ethnicity proportions as the continuous proxy, and the postcode-to-LSOA lookup to link them.

```python
from insurance_fairness import DiscriminationFreePrice

# df must contain: the raw model predictions, postcode_band, and
# ethnicity_prop from ONS Census 2021 linked via postcode -> LSOA

dfp = DiscriminationFreePrice(
    model=model,
    protected_proxy="ethnicity_prop",
    protected_proxy_type="continuous",      # continuous proportion, not binary flag
    marginalisation_method="empirical",     # use empirical p(S|X) from data
    exposure_col="policy_years",
)

df_with_df_price = dfp.fit_transform(df_test)
# Returns df with original prediction and discrimination-free prediction
```

The output adds two columns: `predicted_freq` (raw model output) and `predicted_freq_df` (discrimination-free version). For policies in high-minority postcodes, the DF price is typically lower than the raw prediction. For policies in low-minority postcodes, it may be slightly higher. The portfolio mean is preserved.

```python
# Decompose how much of the disparity is proxy vs risk
decomposition = dfp.disparity_decomposition()
print(decomposition)
```

```
Disparity Decomposition
Total log ratio (protected vs reference): +0.27

  Genuine risk differential (risk-based):     +0.19  (70%)
  Proxy discrimination component (removed):   +0.08  (30%)

Discrimination-free log ratio: +0.19
Discrimination-free multiplicative disparity: 1.21x

Interpretation: of the observed 31% pricing disparity, 21% reflects genuine
risk differential (actuarially justified) and 10% reflects proxy discrimination
via postcode correlation with protected characteristics. The DF prices remove
the 10% proxy component while retaining the 21% risk-based differential.
```

This is the output that justifies the analysis to a regulator. You are not claiming your model has no disparate impact - it has some, because risk genuinely varies. You are demonstrating that you have identified and removed the proxy discrimination component, and that the remaining disparity is actuarially justified.

---

## Step 4: audit report

The library generates a report designed to sit inside a Consumer Duty fair value assessment:

```python
from insurance_fairness import AuditReport

report = AuditReport(
    proxy_audit=audit,
    di_audit=di_audit,
    df_price=dfp,
    model_name="UK Motor Frequency Model v3.2",
    reference_date="2026-03-07",
    product_line="Private Motor",
)

report.to_pdf("fairness_audit_2026Q1.pdf")
report.to_json("fairness_audit_2026Q1.json")   # machine-readable for MI submissions
```

The PDF report covers: the protected characteristics assessed, the proxy correlation findings, the disparate impact measurements, the discrimination-free price adjustment methodology, and the decomposition of any remaining disparity into risk-based and proxy-discrimination components.

The JSON output supports structured submissions to your Chief Actuary sign-off process and, where required, to the FCA.

---

## The fairness-accuracy trade-off is real, and smaller than you think

The common objection to discrimination-free pricing: it will hurt your loss ratio. If you remove the postcode-ethnicity correlation from your prices, you will misprice some risks.

This is true, in theory. In practice, the magnitude is smaller than the theory implies, for two reasons.

First, the genuine risk information in postcode - urban density, road quality, theft rates, traffic patterns - is not removed. Only the component of the postcode effect that is not explained by risk-relevant features but is explained by demographic composition gets adjusted. If your model already includes urban density, road type, and crime statistics, the residual postcode-ethnicity correlation being removed is smaller.

Second, the claims experience in the adjusted segments is your test. The library includes a post-adjustment calibration check:

```python
calibration = dfp.calibration_check(df_holdout)
print(calibration)
```

```
Post-Adjustment Calibration Check
Holdout: 2025 policy year (n=87,432 policies)

              | A/E ratio (raw) | A/E ratio (DF)
  Protected   |    1.04         |   1.02
  Reference   |    0.98         |   0.99
  Overall     |    1.01         |   1.01

Both raw and DF predictions are well-calibrated on holdout data.
The discrimination-free adjustment does not materially affect overall calibration.
```

When the proxy discrimination component is genuinely a proxy and not causal, removing it should not materially damage predictive accuracy. If calibration deteriorates sharply after the DF adjustment, that is information: it suggests postcode's predictive power is not purely proxy. You should investigate what genuine causal channel postcode is capturing that your other features are not - and consider adding those features (urban density index, road quality scores, telematics) to reduce the proxy correlation at source.

---

## What a pricing team should actually do

The practical question is not "is our model perfectly discrimination-free" - it is not, and neither is anyone else's. The practical question is "can we demonstrate that we have taken this seriously."

Our recommended sequence:

**1. Run the proxy audit on your current model.** Generate the correlation table for all rating factors against a protected characteristic proxy. The ONS Census 2021 postcode-to-ethnicity linkage is the starting point for motor. You need an hour of data engineering, not weeks.

**2. Measure disparate impact with the current production model.** Get the log ratio for protected versus reference groups. Understand the magnitude. If it is above 0.10 (roughly 10% pricing disparity), it will not survive scrutiny without decomposition.

**3. Compute the DF prices and decompose.** Understand what proportion of the disparity is risk-based versus proxy-discrimination. If the proxy component is small, document that and file it. If the proxy component is large, you have a pricing problem that the DF prices fix, and you should consider applying them.

**4. Document everything.** The FCA in TR24/2 was explicit: they want adequate granularity and evidence of good outcomes. "We looked and found it was fine" is not adequate. "We ran these specific tests, found these specific results, and took these specific actions" is.

**5. Brief the Chief Actuary and your Consumer Duty owner.** This is not a modelling curiosity. It is a regulated obligation with enforcement behind it. The right people need to be informed of the findings.

What you should not do: wait until the FCA writes to you. The £213 million annual excess identified by Citizens Advice represents a real regulatory target. The firms that can produce audit-ready documentation quickly are the ones that have done this work before they were asked.

---

## On the limits of the approach

Three things to be honest about.

**The protected proxy is imperfect.** We use postcode-level ONS ethnicity proportions as a proxy for individual protected characteristics because insurers do not collect individual-level ethnicity data. This means our estimates of p(S | X) are noisy, particularly in mixed postcodes. The proxy audit will underestimate proxy correlation where demographic mixing is high. That is a conservative bias - actual proxy discrimination may be higher than measured.

**Multiple protected characteristics.** This post focuses on ethnicity because it has the most prominent evidence base (the Citizens Advice analysis) and the clearest postcode proxy. The same methods apply to disability, religion, and sex - but building the proxies requires different data sources and is harder. The library supports custom proxy columns: if you have a disability proxy from your distribution data, you can use it.

**The LRTW framework assumes your model is predictively correct.** The DF price adjustment removes proxy discrimination relative to what the model is already predicting. If the model itself has poor calibration in protected-characteristic-correlated segments - for example, if training data is sparse for some demographic groups - the DF adjustment fixes the proxy issue but not the calibration issue. Both problems are worth solving; they require different tools.

---

## Getting started

```bash
uv add insurance-fairness
```

Source and issue tracker on [GitHub](https://github.com/burningcost/insurance-fairness). The library requires CatBoost models and Polars DataFrames; postcode-to-LSOA linkage data and ONS Census 2021 ethnic group tables at LSOA level are available from the ONS open data portal.

Start with the proxy audit. Run it on your current production model before you do anything else. If the postcode correlation with your ethnicity proxy is below 0.15, you have a defensible position and can document it. If it is above 0.30, you need to understand the disparity decomposition before your next Consumer Duty review.

The fairness-accuracy trade-off is real. But "we did not audit this because we were worried about our loss ratio" is not a position you can sustain with the regulator. The Citizens Advice analysis is public. The FCA's expectations are documented. The question is not whether you will address this - it is whether you address it now or after a letter arrives from Stratford.
