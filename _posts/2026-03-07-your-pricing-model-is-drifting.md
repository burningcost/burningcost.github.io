---
layout: post
title: "Your Pricing Model is Drifting (and You Probably Can't Tell)"
date: 2026-03-07
categories: [monitoring]
tags: [insurance-monitoring, drift, PSI, Gini, calibration, motor, FCA, python]
description: "PSI and aggregate A/E are not enough. A three-layer monitoring framework - feature drift, segmented calibration, and a formal Gini test - that tells you whether to recalibrate or refit."
---

Picture a motor pricing team in late 2022. Their BI frequency model was trained on 2018-2020 data and deployed in early 2021. Eighteen months on, the portfolio stability dashboard is green: PSI on the model score is 0.08, comfortably below the 0.10 threshold. The aggregate A/E ratio is 1.02 - essentially perfect. Nobody is losing sleep.

Meanwhile, bodily injury claims for small road traffic accidents have dropped by roughly 50% since the Civil Liability Act 2021 came into force on 31 May 2021. The Official Injury Claim portal and the fixed whiplash tariff have reshaped the BI claims landscape fundamentally. The model, trained on a world that no longer exists, is systematically overpricing customers whose risk profile includes higher whiplash exposure. Other segments are being underpriced to compensate. The aggregate A/E washes out to 1.02 because the errors offset. The PSI is clean because the portfolio composition has not changed - the same customers are buying, just at wrong prices.

This is concept drift. It is the most dangerous failure mode for a pricing model, and the most commonly missed.

---

## Why the standard monitoring setup misses it

Most pricing teams who monitor at all run two checks: PSI on the model score, and an aggregate A/E ratio. These are not wrong - they are just insufficient.

PSI (Population Stability Index) measures whether the distribution of your model's predictions has shifted relative to training. It answers the question: are we scoring a different kind of customer? It does not answer: does the model still correctly describe reality for the customers we are scoring.

The aggregate A/E ratio measures whether your model's overall level is right. A/E = 1.00 means your model is exactly calibrated in aggregate. But aggregate balance is consistent with severe segment-level miscalibration, as the whiplash example shows. A 10% overestimate in one segment cancelled by a 10% underestimate in another delivers a perfect A/E = 1.00, while leaving meaningful pricing errors in both.

Neither metric tells you whether your model's rank ordering - the fundamental thing a pricing model is for - has degraded.

---

## The drift taxonomy

Three distinct things can go wrong with a deployed pricing model, and they require different responses.

**Covariate shift** (virtual drift) is when P(X) changes but the true relationship P(Y|X) stays fixed. Your portfolio mix has shifted: more young drivers, fewer rural risks, a new affinity scheme bringing in a different profile. The model is still correct; it is just being applied to a different population than it was trained on. PSI and CSI will be elevated. A global recalibration - adjusting the intercept to restore aggregate balance - is often sufficient.

**Concept drift** (real drift) is when P(Y|X) changes - the true relationship between risk features and claims frequency or severity has structurally shifted. The whiplash reform is a clean example: a driver's age, vehicle type and NCD band predict a different BI claims frequency in 2022 than they did in 2019, because the legal environment governing what a BI claim looks like has changed. Claims inflation in 2022-23, with CPI above 10% for much of the year, is another: severity models trained before 2021 systematically underestimate current repair and hire costs. Concept drift cannot be fixed with recalibration. The model needs refitting on recent data.

**Prior probability shift** is simpler: overall claims frequency or severity shifts uniformly across all segments. Post-lockdown normalisation of driving frequency in 2021-22 is a reasonable example. The relative risk ordering remains valid and the model just needs its overall level correcting - a straightforward recalibration.

The reason this taxonomy matters operationally: recalibration takes an afternoon. A full model refit takes two to eight weeks of data scientist time, plus governance, plus potentially regulatory notification. Diagnosing the wrong type of drift in either direction is expensive.

---

## What to actually monitor: three layers

### Layer 1: Feature drift (PSI/CSI per rating factor)

Compute PSI on the model score distribution. Then compute CSI - the same PSI formula applied feature by feature. This gives you a signal per rating factor for whether that feature's distribution has shifted.

The critical modification for insurance: weight by exposure, not policy count. A portfolio where young drivers account for 15% of policies but 30% of earned exposure needs the young driver bin's contribution to PSI weighted by their exposure share, not their headcount. Standard PSI from credit scoring does not do this. Our `insurance-monitoring` library computes exposure-weighted PSI and CSI throughout.

```python
from insurance_monitoring.drift import psi, csi
import polars as pl

# Exposure-weighted PSI on model score
score_psi = psi(
    reference=train_scores,
    current=monitor_scores,
    exposure_weights=monitor_exposure,  # current-period exposure
    n_bins=10,
)
# 0.08 -> green; 0.10-0.25 -> amber; >0.25 -> red

# CSI across all continuous rating factors
feature_drift = csi(
    reference_df=train_features,    # polars or pandas DataFrame
    current_df=monitor_features,
    features=["driver_age", "vehicle_age", "ncd_years"],
)
# Returns a DataFrame: one row per feature, with csi value and band ('green'/'amber'/'red')
print(feature_drift)
# shape: (3, 3)
# ┌─────────────┬──────────┬───────┐
# │ feature     ┆ csi      ┆ band  │
# ╞═════════════╪══════════╪═══════╡
# │ driver_age  ┆ 0.031    ┆ green │
# │ vehicle_age ┆ 0.143    ┆ amber │
# │ ncd_years   ┆ 0.019    ┆ green │
# └─────────────┴──────────┴───────┘
```

Thresholds: PSI/CSI below 0.10 is stable; 0.10-0.25 warrants investigation; above 0.25 requires action. These originate from 1990s credit scoring practice and are not formally calibrated for insurance, but they remain the industry standard.

### Layer 2: Calibration (segmented A/E ratios)

An aggregate A/E is necessary but not sufficient. You need A/E by each key rating factor.

Track A/E separately for claim frequency and claim severity. A combined A/E can mask exactly offsetting errors: 10% frequency underestimate combined with 10% severity overestimate gives a combined A/E of 1.00, with the wrong decomposition. For a GLM-based pricing model using separate Poisson and Gamma components, this matters.

The standard alert thresholds from industry practice: A/E outside 0.90-1.10 on a mature accident quarter triggers investigation; A/E outside 0.85-1.15 for two or more consecutive quarters triggers action.

One complication that catches teams out: IBNR. Using notified claims as your actuals for recent accident periods systematically understates A/E because claims take 12-18 months to develop. Either restrict your A/E analysis to accident periods that are 12+ months old - which introduces a lag - or apply development factors to project incurred claims to ultimate.

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci
import numpy as np

# Aggregate A/E with exact Poisson confidence interval
result = ae_ratio_ci(
    actual=df["claims_ultimate"].to_numpy(),   # IBNR-adjusted counts
    predicted=df["predicted_frequency"].to_numpy(),
    exposure=df["exposure"].to_numpy(),        # earned car-years
)
print(f"A/E = {result['ae']:.3f}  95% CI [{result['lower']:.3f}, {result['upper']:.3f}]")
# A/E = 1.087  95% CI [1.041, 1.136]

# Segmented A/E by driver age band
ae_by_age = ae_ratio(
    actual=df["claims_ultimate"].to_numpy(),
    predicted=df["predicted_frequency"].to_numpy(),
    exposure=df["exposure"].to_numpy(),
    segments=df["driver_age_band"].to_numpy(),
)
print(ae_by_age)
# ┌──────────┬─────────┬──────────┬──────────┬────────────┐
# │ segment  ┆ actual  ┆ expected ┆ ae_ratio ┆ n_policies │
# ╞══════════╪═════════╪══════════╪══════════╪════════════╡
# │ 17-25    ┆ 312.0   ┆ 269.4    ┆ 1.158    ┆ 2891       │
# │ 26-50    ┆ 1847.0  ┆ 1761.2   ┆ 1.049    ┆ 18420      │
# │ 51-70    ┆ 891.0   ┆ 874.5    ┆ 1.019    ┆ 9103       │
# │ 71+      ┆ 198.0   ┆ 197.3    ┆ 1.004    ┆ 1986       │
# └──────────┴─────────┴──────────┴──────────┴────────────┘
```

### Layer 3: Discrimination (Gini over time, with a statistical test)

This is the layer most teams skip, and it is the most informative.

The Gini coefficient - equivalent to 2 * (AUROC - 0.5) for binary outcomes - measures whether your model still rank-orders risks correctly. A Gini of 0.42 at training that has drifted to 0.35 in production is a different problem from a Gini that has held steady at 0.42 while your A/E has drifted from 1.00 to 0.88. One requires refitting. The other requires recalibration.

The key advance in arXiv 2510.04556 (December 2025) is providing an asymptotic normality result for the insurance Gini that enables formal hypothesis testing. Under the null that Gini has not changed, `sqrt(n) * (G_new - G_old) / sigma` is asymptotically standard normal. This gives you a z-statistic and p-value rather than a subjective judgement about whether the Gini has "changed enough".

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

# Current Gini
g_current = gini_coefficient(
    actual=current_df["claims_occurred"].to_numpy(),
    predicted=current_df["predicted_frequency"].to_numpy(),
    exposure=current_df["exposure"].to_numpy(),
)

# Drift test against training baseline
result = gini_drift_test(
    reference_gini=0.42,         # stored from training evaluation
    current_gini=g_current,
    n_reference=125_000,         # training sample size
    n_current=len(current_df),
    reference_actual=train_df["claims_occurred"].to_numpy(),
    reference_predicted=train_df["predicted_frequency"].to_numpy(),
    current_actual=current_df["claims_occurred"].to_numpy(),
    current_predicted=current_df["predicted_frequency"].to_numpy(),
)

print(f"Current Gini: {result['current_gini']:.3f}")
print(f"z-statistic: {result['z_statistic']:.2f}, p-value: {result['p_value']:.3f}")
# Current Gini: 0.389
# z-statistic: -2.14, p-value: 0.032
```

A p-value of 0.032 means you can reject the null that your model's discrimination is unchanged. That is a refit signal, not a recalibration signal.

From the paper's application to the FreMTPL2freq dataset with controlled drift: shifting 200 claims between age groups generated z = -2.38, p = 0.017. Shifting 100 claims: z = -1.28, p = 0.201. The test requires economically meaningful drift to trigger, which is the right property for a test that may generate expensive operational responses.

---

## Running everything at once

For routine monitoring you do not want to orchestrate PSI, A/E and Gini separately. `MonitoringReport` runs the full three-layer check in one call and returns a traffic-light summary with a recommendation:

```python
from insurance_monitoring import MonitoringReport
import polars as pl

report = MonitoringReport(
    reference_actual=train_claims,
    reference_predicted=train_predicted,
    current_actual=current_claims,
    current_predicted=current_predicted,
    exposure=current_exposure,
    reference_exposure=train_exposure,
    feature_df_reference=train_features,
    feature_df_current=current_features,
    features=["driver_age", "vehicle_age", "ncd_years"],
    score_reference=train_scores,
    score_current=current_scores,
)

print(report.recommendation)
# 'RECALIBRATE'  |  'REFIT'  |  'NO_ACTION'  |  'INVESTIGATE'

print(report.to_polars())
# metric               value    band
# ae_ratio             1.087    amber
# gini_current         0.389    red
# gini_p_value         0.032    red
# score_psi            0.078    green
# csi_driver_age       0.031    green
# csi_vehicle_age      0.143    amber
# recommendation       NaN      REFIT
```

The recommendation logic follows the decision tree from arXiv 2510.04556: if the Gini z-test is red, return `REFIT` regardless of A/E. If the A/E is red but Gini is stable, return `RECALIBRATE`. This is not the only valid decision framework, but it is a defensible starting point and you can override the thresholds to match your portfolio and monitoring cadence.

---

## Reading the diagnostic

The three layers are most useful when read in combination.

**PSI clean, A/E drifted globally, Gini stable**: prior probability shift. Recalibrate the intercept. This is the post-whiplash-reform scenario for BI frequency: claim frequency has dropped uniformly, the model's relative ranking is fine, just update the level.

**CSI elevated for specific features, A/E drifted in those segments, Gini stable**: covariate shift. Portfolio mix has changed. Recalibrate, or refit the affected interaction terms if segment-level A/E deviation is large.

**Gini z-test significant, A/E also drifted**: concept drift in the ranking. The model's fundamental ability to discriminate has changed. Full refit required.

**PSI clean, A/E drifted in specific segments only, Gini stable**: local concept drift. The relative risk relationship has changed for a specific segment. Investigate; partial refit or manual override of affected rating factors.

The whiplash scenario falls into the first category. Appropriate response: recalibrate the BI frequency model intercept, document the regulatory trigger (Civil Liability Act 2021), record that refit was not required because discrimination was stable. That is a defensible audit trail under FCA Consumer Duty (PRIN 2A, effective July 2023), which requires firms to evidence that pricing models continue to deliver fair value across customer segments. It is also the kind of documented decision record that PRA SS1/23 model risk governance calls for.

Contrast that with claims severity inflation in 2022-23. CPI peaked at 11.1% in October 2022. Vehicle repair costs and credit hire rates ran well ahead of general inflation. A severity model trained on 2019-20 data would show elevated A/E across most segments - not concentrated in a specific rating factor - alongside likely Gini degradation as the rank ordering of expected costs shifted. That combination points to full refit, not recalibration.

---

## Cadence

Our recommendation: monthly PSI and segmented A/E on mature accident periods (accident months 12+ old). Quarterly Gini z-test. Annual full model validation.

Monthly monitoring on immature data is a trap. Partially developed claims look like an A/E of 0.6-0.7 not because the model is wrong but because the claims have not yet reported. Use development factors or restrict to mature periods.

Quarterly is the natural cadence for the Gini test: volumes are large enough to have statistical power, and quarterly rate review cycles provide a natural decision point. A monitoring system that recommends refitting every month will be ignored within two cycles.

The annual full model validation should include: backtesting the model on out-of-time data from the most recent policy year; recomputing one-way and two-way analyses to check that rating factor relativities still align with observed loss ratios; and a fresh Gini benchmark to update the baseline against which future quarterly tests are measured.

---

## The library

[`insurance-monitoring`](https://github.com/burningcost/insurance-monitoring) implements the three-layer framework described above:

- Exposure-weighted PSI and CSI, with per-feature traffic-light output
- Segmented A/E ratios with optional IBNR adjustment via development factor tables
- Gini coefficient with the asymptotic z-test from arXiv 2510.04556
- Murphy score decomposition (calibration, resolution, uncertainty) to support the refit vs recalibrate decision
- Quarterly monitoring report template suitable for model risk committees

Install with `uv add insurance-monitoring`.

The only thing we do not supply is the will to run it quarterly. That part is on you.
