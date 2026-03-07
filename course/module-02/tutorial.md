# GLMs in Python - The Bridge from Emblem

---

## Why Emblem is not the problem

Let us be clear about something before we start. Emblem fits GLMs correctly. It uses IRLS, the same algorithm statsmodels uses. It handles exposure offsets. It produces deviance statistics, factor charts, and actual-versus-expected plots. If you are a pricing actuary who has been fitting frequency-severity models in Emblem for ten years, you are not doing it wrong.

The problem is the infrastructure around the model. The Emblem project file is not version-controlled. Nobody commits it to Git. The data extract you fed it lives on a network drive with a name like `motor_extract_final_v2_ACTUAL.csv`. When the FCA asks you to reproduce the relativities from your Q3 2023 renewal cycle, you go hunting for the right combination of software version, project file, and data extract - and you hope nothing has changed.

This matters now more than it did five years ago. PS 21/5 (the general insurance pricing practices rules, effective January 2022) banned price walking and introduced explicit audit trail requirements for pricing decisions. Consumer Duty (PS 22/9, effective July 2023) extended this further to require demonstrable fair value - meaning the FCA wants to walk into your office, ask about any price charged to any customer in the past three years, and have you show them the model, the inputs, and the decision trail in under an hour. "We can reconstruct what our model said and why" is not optional. It is table stakes.

Moving your GLM to Python and Databricks solves this. The model code is version-controlled. The training data is a Delta table with time travel. The fitted model is logged to MLflow with the parameters, metrics, and artefacts. The factor tables go to Unity Catalog. Running the model from six months ago means checking out the relevant Git tag and pointing at the Delta table at that timestamp. That is reproducibility you can demonstrate to a regulator.

The GLM itself is not the hard part. The encoding, the validation, and the export are where the work is. That is what this module covers.

---

## The workflow

We are building a motor frequency-severity model: a Poisson GLM for claim frequency and a Gamma GLM for average severity, both with log link and exposure offset. The pure premium estimate is the product: `frequency × severity`.

The data pipeline:

1. Load policy data from a Delta table (Polars)
2. Prepare features: encode factors, handle missing values, create interactions (Polars)
3. Fit GLMs (statsmodels - requires Pandas, but only at model-fit time)
4. Run diagnostics: deviance residuals, A/E by factor level, lift charts (Polars + matplotlib)
5. Validate against Emblem: check our relativities match the published Emblem output
6. Export factor tables for Radar import
7. Log to MLflow, register in Unity Catalog

We use a synthetic UK motor dataset throughout, with known true parameters, so we can verify our GLM is recovering the right answers.

---

## Data

We generate a synthetic UK motor portfolio with 100,000 policies.

```python
import polars as pl
import numpy as np
from datetime import date

rng = np.random.default_rng(seed=42)
n = 100_000

# Rating factors - UK motor conventions
areas = ["A", "B", "C", "D", "E", "F"]
area = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])

vehicle_group = rng.integers(1, 51, size=n)  # ABI group 1-50
ncd_years = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age = rng.integers(17, 86, size=n)
conviction_flag = rng.binomial(1, 0.06, size=n)
exposure = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

# True log-frequency parameters (GLM intercept + log-linear effects)
INTERCEPT = -3.10
TRUE_PARAMS = {
    "area_B": 0.10, "area_C": 0.20, "area_D": 0.35,
    "area_E": 0.50, "area_F": 0.65,
    "vehicle_group": 0.018,   # per ABI group unit above 1
    "ncd_years": -0.13,       # per year of NCD
    "young_driver": 0.55,     # age < 25
    "old_driver": 0.28,       # age > 70
    "conviction": 0.42,
}

# Generate log-mu for each policy
log_mu = (
    INTERCEPT
    + np.where(area == "B", TRUE_PARAMS["area_B"], 0)
    + np.where(area == "C", TRUE_PARAMS["area_C"], 0)
    + np.where(area == "D", TRUE_PARAMS["area_D"], 0)
    + np.where(area == "E", TRUE_PARAMS["area_E"], 0)
    + np.where(area == "F", TRUE_PARAMS["area_F"], 0)
    + TRUE_PARAMS["vehicle_group"] * (vehicle_group - 1)
    + TRUE_PARAMS["ncd_years"] * ncd_years
    + np.where(driver_age < 25, TRUE_PARAMS["young_driver"], 0)
    + np.where(driver_age > 70, TRUE_PARAMS["old_driver"], 0)
    + TRUE_PARAMS["conviction"] * conviction_flag
    + np.log(exposure)
)

freq_rate = np.exp(log_mu - np.log(exposure))  # annualised frequency
claim_count = rng.poisson(freq_rate * exposure)

# Severity DGP: Gamma with mean around £3,500, vehicle group effect only.
# NCD reflects driver behaviour and correlates with claim frequency,
# not individual claim size. Including it in the severity model would
# capture frequency effects through the back door.
sev_log_mu = (
    np.log(3500)
    + 0.012 * (vehicle_group - 1)
)
true_mean_sev = np.exp(sev_log_mu)
shape_param = 4.0  # coefficient of variation = 1/sqrt(4) = 0.5

has_claim = claim_count > 0
avg_severity = np.where(
    has_claim,
    rng.gamma(shape_param, true_mean_sev / shape_param),
    0.0
)

df = pl.DataFrame({
    "policy_id": np.arange(1, n + 1),
    "area": area,
    "vehicle_group": vehicle_group,
    "ncd_years": ncd_years,
    "driver_age": driver_age,
    "conviction_flag": conviction_flag,
    "exposure": exposure,
    "claim_count": claim_count,
    "avg_severity": avg_severity,
    "incurred": avg_severity * claim_count,
})

print(f"Portfolio: {len(df):,} policies")
print(f"Exposure: {df['exposure'].sum():,.0f} earned years")
print(f"Claims: {df['claim_count'].sum():,} ({df['claim_count'].sum() / df['exposure'].sum():.3f}/year)")
print(f"Total incurred: £{df['incurred'].sum() / 1e6:.1f}m")
```

This gives us a portfolio with a known data-generating process. We will use this to verify our GLM recovers the true parameters.

The true NCD=5 vs NCD=0 relativity is `exp(-0.13 × 5) = exp(-0.65) ≈ 0.522`. The conviction uplift is `exp(0.42) ≈ 1.52`. The area F relativity over area A is `exp(0.65) ≈ 1.92`. These are our benchmarks.

---

## Factor encoding: what Emblem does and what Python does by default

This is where most migration projects go wrong, and it is worth spending time on.

### Base levels and reference categories

In a GLM, every categorical factor must have a reference category. For a factor with k levels, the model estimates k-1 coefficients. The reference category has no coefficient - its effect is absorbed into the intercept. The relativity for each non-reference level is `exp(beta)` relative to the reference level.

Emblem calls the reference category the "base level." Choosing it is a deliberate actuarial decision: you pick a level that is (a) well-populated so the intercept is stable, and (b) makes business sense as the "standard" risk. For NCD, base = NCD 0 is conventional. For area, base = area A or whichever band your portfolio uses as a baseline.

Python's `statsmodels`, using `C(factor)` in formula syntax, picks the base level automatically. By default, it uses the **first level alphabetically or numerically**. This means:

- For area: `A` is base (correct by convention - matches Emblem's typical default)
- For NCD: `0` is base (correct)
- For vehicle group, if encoded as a factor: `1` is base (correct)
- For any factor with alphabetically/numerically non-obvious ordering, you need to specify explicitly

If your Emblem model uses area A as base and your Python model also defaults to area A, the relativities will match. If there is any discrepancy in base level selection, every relativity for that factor will be off by a constant multiplier. This is the most common validation failure when people first move from Emblem to Python - not the GLM fitting, the base level.

How to specify the base level explicitly in statsmodels formula syntax:

```python
# C(factor) uses alphabetical first as base - area A in this case
# C(factor, Treatment("D")) explicitly sets area D as base
formula = "claim_count ~ C(area) + C(ncd_years) + C(conviction_flag) + vehicle_group"

# To pin ncd_years base to 0:
formula = "claim_count ~ C(area) + C(ncd_years, Treatment(0)) + C(conviction_flag, Treatment(0)) + vehicle_group"
```

In Polars, you encode factors as `Categorical` or `Enum` dtype, but the actual dummy encoding for statsmodels happens when you call `.to_pandas()`. If you use `patsy` (the formula engine behind statsmodels), it reads the column dtype and orders levels accordingly. Factors encoded as `pl.Enum` with a specified ordering will preserve that ordering through the Pandas conversion.

```python
# Encode area as Enum with explicit ordering
area_order = ["A", "B", "C", "D", "E", "F"]
df = df.with_columns(
    pl.col("area").cast(pl.Enum(area_order)).alias("area")
)

# When converted to pandas, area A will be the natural first level
df_pd = df.to_pandas()
```

### Sum-to-zero encoding

The examples here use treatment (dummy) coding throughout. This matches Emblem's default, which is why the relativities align directly. For the record: statsmodels supports `C(area, Sum)` for sum-to-zero (deviation) coding, where each level coefficient measures the deviation from the grand mean rather than the deviation from a base level. This changes the interpretation of the intercept but not the fitted values or the relativities between levels. You will encounter sum-to-zero coding in textbooks and some GLM papers. For Emblem migration work, stick with treatment coding.

### Aliasing

Aliasing occurs when one factor level is an exact linear combination of others - usually when you accidentally include a level for every category, making the design matrix rank-deficient. Emblem raises an error in this case ("linear dependency detected"). statsmodels drops aliased columns silently by default when using formula API, which can mask the problem.

The most common cause in practice: you have a `young_driver` flag and `driver_age < 25` as separate columns, and you include both. If `young_driver` is defined as exactly `driver_age < 25`, they are perfectly collinear. One will be dropped. You will not see an error - you will see a NaN coefficient and a warning in `model.summary()`.

Always check for NaN coefficients in the fitted model before presenting results:

```python
nan_params = glm_freq.params[glm_freq.params.isna()]
if len(nan_params) > 0:
    print("WARNING: Aliased parameters detected:")
    print(nan_params)
```

### Sparse factor levels in production data

Real claims extracts will have factor levels with very few policies: area codes that appear in the policy file but not in the training data, occupation codes with 2 claims, NCD levels with 12 policies. Emblem consolidates sparse levels automatically, but Python will estimate a separate coefficient for every level unless you intervene.

A practical pattern for grouping sparse levels in Polars by exposure threshold:

```python
# Identify area levels with fewer than 50 earned years of exposure
area_exposure = (
    df
    .group_by("area")
    .agg(pl.col("exposure").sum().alias("total_exposure"))
)

sparse_areas = area_exposure.filter(pl.col("total_exposure") < 50)["area"].to_list()
print(f"Sparse area levels: {sparse_areas}")

# Group sparse levels into "Other"
df = df.with_columns(
    pl.when(pl.col("area").is_in(sparse_areas))
    .then(pl.lit("Other"))
    .otherwise(pl.col("area"))
    .alias("area")
)
```

The threshold (50 earned years here) is a business judgement. A level with fewer than about 30-50 years of exposure will produce a relativity with such a wide confidence interval that it is essentially noise. Merge it with the nearest adjacent level or a generic "Other" bucket, and document the consolidation in your model notes. Any level that shows up in live rating data but was not in your training data needs a fallback relativity before you deploy.

### Interaction terms

Emblem supports interaction terms through its UI - you define an interaction between two factors and it generates the cross-level factor. In statsmodels formula syntax:

```python
# Main effects only
"claim_count ~ C(area) + C(ncd_years)"

# Area × NCD interaction (all cross-level combinations)
"claim_count ~ C(area) * C(ncd_years)"

# Interaction only, no main effects (unusual)
"claim_count ~ C(area):C(ncd_years)"
```

The interaction formula generates `(k_area - 1) × (k_ncd - 1)` additional coefficients. For area (6 levels) × NCD (6 levels), that is 25 interaction parameters on top of the 5 area and 5 NCD main effects. With 100,000 policies, this is fine. With a UK homeowner book on 200 occupancy types × 15 construction types × 30 property ages, you will run out of data quickly.

The aliasing issue becomes acute with interactions. If you have a full vehicle group × driver age interaction and sparse combinations (few 17-year-olds in ABI group 50), some cells may have zero or near-zero claims. These cells will produce extreme relativities or NaN coefficients. Check cell counts before including interactions:

```python
cell_counts = (
    df
    .group_by(["vehicle_group", "driver_age"])
    .agg(pl.col("exposure").sum().alias("exposure"),
         pl.col("claim_count").sum().alias("claims"))
    .filter(pl.col("claims") < 5)
    .sort("claims")
)
print(f"Cells with fewer than 5 claims: {len(cell_counts):,}")
```

---

## Fitting the frequency GLM

With the feature encoding understood, fitting the GLM is straightforward. We move to Pandas at this point because statsmodels requires it.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# Convert to Pandas for statsmodels
# We keep the Polars DataFrame for all subsequent data manipulation
df_pd = df.to_pandas()
df_pd["log_exposure"] = np.log(df_pd["exposure"].clip(lower=1e-6))

freq_formula = (
    "claim_count ~ "
    "C(area) + "
    "C(ncd_years, Treatment(0)) + "
    "C(conviction_flag, Treatment(0)) + "
    "vehicle_group"
)

glm_freq = smf.glm(
    formula=freq_formula,
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

print(glm_freq.summary())
```

Key points on the call signature:

**`family=sm.families.Poisson(link=sm.families.links.Log())`** - Poisson family with canonical log link. This is what Emblem uses for frequency. For severity, you will use `sm.families.Gamma(link=sm.families.links.Log())` - also a log link, not the canonical reciprocal link. Emblem uses log link for severity by default; if yours differs, change here.

**`offset=df_pd["log_exposure"]`** - this is the exposure offset. It enters the linear predictor as `log(exposure)` with a fixed coefficient of 1. This means the model is fitting annualised frequency rates, not raw claim counts. Every policy's claim count is divided by its exposure before the GLM sees it, in effect. We will return to this in detail below.

**The formula string** - `C(area)` tells patsy to dummy-encode area with area A as base (alphabetical first). `C(ncd_years, Treatment(0))` explicitly sets NCD=0 as base. `vehicle_group` is treated as continuous (linear effect per ABI group unit). If you want vehicle group as a factor, use `C(vehicle_group)`.

### Reading the model output

The `glm_freq.summary()` output looks like this (abbreviated):

```
                 Generalized Linear Model Regression Results
================================================================================
Dep. Variable:           claim_count   No. Observations:               100000
Model:                           GLM   Df Residuals:                    99988
Model Family:                Poisson   Df Model:                            11
Link Function:                   Log   Scale:                          1.0000
Method:                         IRLS   Log-Likelihood:                 -74382.
Date:                                  Deviance:                        98476.
                                        Pearson chi2:                 1.34e+05
Converged:                        True   Pseudo R-squ. (CS):             0.2814
=================================================================================
                          coef    std err          z      P>|z|  [0.025  0.975]
---------------------------------------------------------------------------------
Intercept              -3.0847      0.052    -59.33      0.000  -3.187  -2.983
C(area)[T.B]            0.0991      0.024      4.11      0.000   0.052   0.146
C(area)[T.C]            0.1978      0.022      8.97      0.000   0.154   0.241
...
C(ncd_years, ...)[T.1] -0.1249      0.032     -3.91      0.000  -0.188  -0.062
C(ncd_years, ...)[T.5] -0.6408      0.028    -22.84      0.000  -0.696  -0.585
vehicle_group           0.0179      0.001     17.90      0.000   0.016   0.020
```

The `[T.B]` suffix means "treatment contrast: level B relative to the reference level (A)." The coefficient for area B is 0.0991, so `exp(0.0991) ≈ 1.104` - a 10.4% frequency uplift in area B versus area A. The true parameter is 0.10, so `exp(0.10) = 1.105`. Our model has recovered it.

### Extracting relativities

```python
import polars as pl

def extract_freq_relativities(glm_result, base_levels: dict) -> pl.DataFrame:
    """
    Extract multiplicative relativities from a fitted statsmodels GLM.
    Returns a Polars DataFrame with columns:
        feature, level, log_relativity, relativity, se, lower_ci, upper_ci
    """
    records = []
    params = glm_result.params
    conf_int = glm_result.conf_int()

    for param_name, coef in params.items():
        if param_name == "Intercept":
            continue

        lo = conf_int.loc[param_name, 0]
        hi = conf_int.loc[param_name, 1]
        se = glm_result.bse[param_name]

        # Parse the patsy parameter name: "C(area)[T.B]" -> feature="area", level="B"
        if "[T." in param_name:
            feature_part = param_name.split("[T.")[0]
            level_part = param_name.split("[T.")[1].rstrip("]")
            # Strip patsy formula wrapping
            if feature_part.startswith("C("):
                feature_part = feature_part[2:].split(",")[0].split(")")[0].strip()
        else:
            # Continuous feature
            feature_part = param_name
            level_part = "continuous"

        records.append({
            "feature": feature_part,
            "level": level_part,
            "log_relativity": coef,
            "relativity": np.exp(coef),
            "se": se,
            "lower_ci": np.exp(lo),
            "upper_ci": np.exp(hi),
        })

    rels = pl.DataFrame(records)

    # Add base level rows for completeness
    base_rows = []
    for feat, base_level in base_levels.items():
        base_rows.append({
            "feature": feat,
            "level": str(base_level),
            "log_relativity": 0.0,
            "relativity": 1.0,
            "se": 0.0,
            "lower_ci": 1.0,
            "upper_ci": 1.0,
        })

    return pl.concat([pl.DataFrame(base_rows), rels]).sort(["feature", "level"])


freq_rels = extract_freq_relativities(
    glm_freq,
    base_levels={"area": "A", "ncd_years": "0", "conviction_flag": "0"},
)
print(freq_rels.filter(pl.col("feature") == "area"))
```

Output:

```
shape: (6, 7)
┌─────────┬───────┬──────────────────┬────────────┬──────────┬────────────┬────────────┐
│ feature ┆ level ┆ log_relativity   ┆ relativity ┆ se       ┆ lower_ci   ┆ upper_ci   │
│ ---     ┆ ---   ┆ ---              ┆ ---        ┆ ---      ┆ ---        ┆ ---        │
│ str     ┆ str   ┆ f64              ┆ f64        ┆ f64      ┆ f64        ┆ f64        │
╞═════════╪═══════╪══════════════════╪════════════╪══════════╪════════════╪════════════╡
│ area    ┆ A     ┆ 0.0              ┆ 1.0        ┆ 0.0      ┆ 1.0        ┆ 1.0        │
│ area    ┆ B     ┆ 0.099078         ┆ 1.104087   ┆ 0.024103 ┆ 1.057512   ┆ 1.152659   │
│ area    ┆ C     ┆ 0.197783         ┆ 1.218452   ┆ 0.022041 ┆ 1.167091   ┆ 1.271968   │
│ area    ┆ D     ┆ 0.348912         ┆ 1.417238   ┆ 0.021502 ┆ 1.359491   ┆ 1.477468   │
│ area    ┆ E     ┆ 0.499124         ┆ 1.647089   ┆ 0.023118 ┆ 1.572861   ┆ 1.724508   │
│ area    ┆ F     ┆ 0.648319         ┆ 1.912042   ┆ 0.027204 ┆ 1.814162   ┆ 2.015709   │
└─────────┴───────┴──────────────────┴────────────┴──────────┴────────────┴────────────┘
```

Area F relativity: 1.912. True value: `exp(0.65) = 1.916`. We are within 0.2%.

---

## Exposure handling - the most important thing to get right

Exposure handling is where the most significant errors occur in GLM migrations, and where Emblem's behaviour needs to be understood precisely.

### What an offset does

An exposure offset in a GLM with log link enters the linear predictor as a term with a fixed coefficient of exactly 1:

```
log(E[Y_i]) = offset_i + X_i * beta
            = log(exposure_i) + X_i * beta
```

This means:

```
E[Y_i] = exposure_i × exp(X_i * beta)
```

The model is fitting the rate per unit exposure (per earned policy year), and the claim count for each policy is that rate multiplied by its exposure. A policy with 0.5 earned years should generate half as many expected claims as an otherwise identical policy with 1.0 earned years. The offset enforces this.

Without the offset, the model fits raw claim counts. A policy with 6 months of exposure will look like it has fewer claims than an identical full-year policy, and the model will learn that (wrongly) as a predictor. You will get biased coefficients and wrong relativities.

### Earned exposure vs written exposure

**Written exposure** is the exposure at policy inception: a 12-month policy written on 1 July has 1.0 written policy year.

**Earned exposure** is the exposure that has been "earned" by the time of analysis: that same policy, analysed on 31 December, has earned 0.5 policy years. Claims from 1 July to 31 December are covered by 0.5 earned years.

For frequency modelling, always use earned exposure. The claim count during a period is proportional to the earned exposure during that period, not the written exposure. Using written exposure gives the correct answer only when your analysis date is after all policies have run to completion - which it rarely is in practice.

If your extract comes from a system that provides policy-period exposure rather than earned exposure, you need to calculate it. The calculation depends on the policy term, the inception date, and the analysis date. For a 12-month policy incepted on date `d_start`, analysed at date `d_analysis`:

```python
from datetime import date

def earned_exposure(
    policy_start: date,
    policy_end: date,
    analysis_date: date,
) -> float:
    """
    Calculate earned exposure (in years) at analysis_date
    for a policy covering [policy_start, policy_end].
    """
    earned_end = min(policy_end, analysis_date)
    if earned_end <= policy_start:
        return 0.0
    earned_days = (earned_end - policy_start).days
    policy_days = (policy_end - policy_start).days
    return earned_days / policy_days  # fraction of policy year earned
```

In Polars (vectorised), the same calculation using correct date arithmetic:

```python
analysis_date_lit = pl.lit(analysis_date)

df = df.with_columns(
    (
        (
            pl.col("policy_end").clip(upper_bound=analysis_date_lit) - pl.col("policy_start")
        ).dt.total_days()
        /
        (
            pl.col("policy_end") - pl.col("policy_start")
        ).dt.total_days()
    )
    .clip(lower_bound=0.0, upper_bound=1.0)
    .alias("earned_exposure")
)
```

The key expression is `(end - start).dt.total_days()` - date subtraction returns a Duration, and `.dt.total_days()` converts it to an integer. The numerator clips `policy_end` to `analysis_date` so that policies running beyond the analysis date contribute only their earned fraction.

### The clipping trap

Always clip exposure to a small positive value before taking the log:

```python
log_exposure = np.log(df_pd["exposure"].clip(lower=1e-6))
```

Zero or negative exposure values exist in real bordereaux data: mid-term cancellations, data entry errors, policies voided ab initio. A zero-exposure policy has no information to contribute to the model. Log(0) is negative infinity. If you pass it to statsmodels, IRLS will attempt to compute the working weights for that observation and produce NaN values that corrupt the entire fit.

Filter out zero-exposure policies before model fitting:

```python
df_model = df.filter(pl.col("exposure") > 0)
```

If you have negative exposures, these are data errors - investigate them rather than filtering them silently:

```python
neg_exposure = df.filter(pl.col("exposure") < 0)
if len(neg_exposure) > 0:
    print(f"WARNING: {len(neg_exposure)} policies with negative exposure")
    print(neg_exposure.describe())
```

### Development patterns: IBNR and truncation

If your claims data is not fully developed - i.e. you are working with claims that are still open and may be reserved rather than paid - the average severity from your claims extract is not the ultimate average severity. It is the paid-to-date severity, which understates the ultimate.

Emblem users typically apply development factors outside the tool (in Excel, before importing to Emblem) and work with ultimate-developed data. Do the same in your Python pipeline. Do not try to model undeveloped severity; model developed severity or use a chain ladder development in your data preparation step.

For frequency: IBNR (incurred but not reported) claims mean your claim count is also understated for recent periods. Frequency models typically use a development or lag factor, or are restricted to experience periods where IBNR is negligible (e.g. accident years that are 24 months developed or older).

---

## Fitting the severity GLM

### Large loss truncation

Before fitting a severity GLM on motor data, you need to decide what to do about large personal injury claims. Bodily injury claims on UK motor books are typically 10-100x the average accidental damage claim. An untruncated Gamma severity model will be driven by whichever risk characteristics correlate with the handful of catastrophic PI claims in your portfolio - not by the systematic differences in average claim cost.

Standard practice is to cap large losses at £100k-£250k and model the excess separately (as a frequency of large losses, typically at a higher truncation point), or to separate PI claims from property damage and model them independently. The right approach depends on your book. What you should not do is feed a Gamma GLM raw uncapped incurred amounts and trust the area relativities you get out.

For the synthetic data here we have no large PI exposure - the severity DGP is a simple Gamma with mean £3,500. In a real model, add this as a pre-processing step:

```python
LARGE_LOSS_THRESHOLD = 100_000  # £100k cap - adjust for your book

df_sev = df_sev.with_columns(
    pl.col("incurred").clip(upper_bound=LARGE_LOSS_THRESHOLD * pl.col("claim_count"))
    .alias("incurred_capped")
)
# Log the number of capped policies
n_capped = df_sev.filter(
    pl.col("incurred") > LARGE_LOSS_THRESHOLD * pl.col("claim_count")
).shape[0]
print(f"Policies capped at £{LARGE_LOSS_THRESHOLD/1000:.0f}k: {n_capped}")
```

Document the cap in your MLflow run parameters. It is a modelling assumption that materially affects your severity relativities, and anyone who tries to reproduce your results without knowing the cap will not match your numbers.

### The severity formula

Severity GLMs are fit on claimed policies only - you condition on `claim_count > 0`. The response is average severity per claim (total incurred divided by claim count for that policy).

NCD is not in the severity formula. NCD reflects driver behaviour and correlates with claim frequency - drivers with zero NCD have more accidents. But conditional on a claim occurring, the claim cost is not systematically different between NCD=0 and NCD=5 drivers. Including NCD in the severity model would capture frequency effects through the back door, double-counting the NCD signal. The frequency model picks it up correctly; the severity model should not.

```python
# Severity data: claimed policies only
df_sev = df.filter(pl.col("claim_count") > 0)
df_sev = df_sev.with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
)

df_sev_pd = df_sev.to_pandas()

sev_formula = (
    "avg_severity ~ "
    "C(area) + "
    "vehicle_group"
)

glm_sev = smf.glm(
    formula=sev_formula,
    data=df_sev_pd,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    var_weights=df_sev_pd["claim_count"],  # weight by claim count
).fit()

print(glm_sev.summary())
```

The severity GLM uses claim count as the variance weight, not as an offset. The distinction matters:

- **Variance weights** (`var_weights`): scale the variance of each observation. For a Gamma GLM, this is appropriate because the average severity from a policy with 3 claims has lower variance than from a policy with 1 claim - you are averaging 3 severities rather than 1.
- **Frequency weights** (`freq_weights`): replicate observations. This is not what you want here.
- **Offset** (`offset`): fixed contribution to the linear predictor. Used in frequency models for log-exposure.

If you use `var_weights=claim_count`, policies with multiple claims have proportionally more influence on the severity fit. This is correct: a policy with 5 claims tells us more about the severity distribution than a policy with 1 claim.

### The Gamma GLM shape parameter

The Gamma GLM has a shape parameter (dispersion parameter) that Emblem estimates and reports. statsmodels estimates it automatically - it appears in `glm_sev.scale` after fitting. For a Gamma GLM with log link:

```python
print(f"Gamma scale parameter (phi): {glm_sev.scale:.4f}")
print(f"Coefficient of variation: {np.sqrt(glm_sev.scale):.3f}")
```

A coefficient of variation (CV) of 0.5-0.8 is typical for motor accidental damage severity distributions. If your estimated CV is below 0.3 or above 2.0, check your data - you may have extreme outliers pulling the distribution. If the CV is unusually low (say 0.1), the most common explanation is that your severity data has already been capped or censored at some limit. If the CV is above 1.5, the Gamma assumption may be wrong for your data and you should investigate whether you have a mixture of claim types (small property damage plus large PI claims) that would be better separated.

---

## Diagnostics

Running a GLM without diagnostics is not modelling - it is curve fitting. These checks tell you whether the model is well-specified and where it fails.

### Deviance residuals

The deviance residual for observation i is:

```
d_i = sign(y_i - mu_i) × sqrt(2 × (y_i × log(y_i / mu_i) - (y_i - mu_i)))
```

for the Poisson family. The total deviance is `sum(d_i^2)`. A well-fitting model has deviance close to the residual degrees of freedom.

```python
# statsmodels gives residuals directly
resid_deviance = glm_freq.resid_deviance

# For Poisson GLMs, scale is fixed at 1.0 (no free dispersion parameter),
# so dividing by sqrt(scale) is a no-op. The line below is conventional
# but produces the same numbers as resid_deviance for Poisson.
# It does matter for quasi-Poisson and Gamma, where scale > 1.
# Properly standardised residuals would divide by sqrt(1 - h_ii) where
# h_ii is the hat matrix diagonal (leverage), but for large datasets
# the leverage correction is small.
resid_std = resid_deviance / np.sqrt(glm_freq.scale)

# Plot residuals vs fitted values
fitted_vals = glm_freq.fittedvalues

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs fitted
axes[0].scatter(np.log(fitted_vals), resid_std, alpha=0.1, s=5, color="steelblue")
axes[0].axhline(0, color="black", linestyle="--", lw=1)
axes[0].axhline(2, color="red", linestyle="--", lw=1, alpha=0.5)
axes[0].axhline(-2, color="red", linestyle="--", lw=1, alpha=0.5)
axes[0].set_xlabel("log(fitted frequency)")
axes[0].set_ylabel("Deviance residual")
axes[0].set_title("Residuals vs Fitted - Frequency GLM")

# QQ plot of deviance residuals
from scipy import stats
stats.probplot(resid_std, dist="norm", plot=axes[1])
axes[1].set_title("Normal QQ - Deviance Residuals")

plt.tight_layout()
display(fig)
```

What to look for:
- Residuals should show no strong pattern against fitted values. A funnel shape (residuals increasing with fitted) suggests overdispersion: your Poisson variance assumption is wrong.
- More than 5% of residuals outside ±2 suggests either genuine overdispersion or a systematic missing feature. Investigate the policies with residuals outside ±3 individually - they are often data errors or extreme risks that do not belong in the model.
- The QQ plot for Poisson GLM deviance residuals is not expected to be perfectly normal (the Poisson is discrete), but the upper tail should not be dramatically heavier than normal. Heavy upper tails indicate overdispersion.

### A note on overdispersion

For real UK motor data, the Poisson model will almost always be overdispersed - the deviance will be materially above the residual degrees of freedom. A deviance/df ratio above 1.3 is common; ratios above 2.0 are not unusual on books with bodily injury cover, because large PI claims inflate claim count variance beyond what Poisson predicts.

When this happens, you have two options:

**Quasi-Poisson**: same point estimates as Poisson, but standard errors are inflated by `sqrt(deviance/df)` to account for overdispersion. Use this when you want conservative confidence intervals but do not want to change the relativities themselves.

```python
glm_freq_quasi = smf.glm(
    formula=freq_formula,
    data=df_pd,
    family=sm.families.quasi.Quasipoisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()
print(f"Dispersion estimate: {glm_freq_quasi.scale:.3f}")
```

**Negative Binomial**: models overdispersion explicitly with an additional dispersion parameter. The coefficients will differ slightly from Poisson because the likelihood is different. More appropriate when you expect genuine extra-Poisson variation in the data-generating process, not just noise.

The choice between them is a modelling judgement. For a first migration from Emblem (which uses Poisson), quasi-Poisson is the lower-risk option: relativities are identical to the Poisson model, only the standard errors change.

### Actual vs Expected by factor level

This is the diagnostic Emblem shows you in its factor charts, and it is the single most useful check for a pricing model. For each level of each rating factor, you compute the ratio of observed claims to predicted claims (the A/E ratio). A well-specified model should have A/E close to 1.0 for all levels.

```python
def ae_by_factor(
    df: pl.DataFrame,
    fitted_values: np.ndarray,
    feature: str,
) -> pl.DataFrame:
    """
    Compute actual vs expected claim counts by factor level.
    Returns a Polars DataFrame with actual, expected, ae_ratio, and exposure.
    """
    result = (
        df
        .with_columns(
            pl.Series("fitted_freq", fitted_values).alias("expected_claims")
        )
        .group_by(feature)
        .agg([
            pl.col("claim_count").sum().alias("actual_claims"),
            pl.col("expected_claims").sum().alias("expected_claims"),
            pl.col("exposure").sum().alias("exposure"),
        ])
        .with_columns(
            (pl.col("actual_claims") / pl.col("expected_claims")).alias("ae_ratio")
        )
        .sort(feature)
    )
    return result


ae_area = ae_by_factor(df, glm_freq.fittedvalues, "area")
print(ae_area)
```

Output for a well-fitting model:

```
shape: (6, 5)
┌──────┬───────────────┬─────────────────┬──────────────┬──────────┐
│ area ┆ actual_claims ┆ expected_claims  ┆ ae_ratio     ┆ exposure │
│ ---  ┆ ---           ┆ ---             ┆ ---          ┆ ---      │
│ str  ┆ i64           ┆ f64             ┆ f64          ┆ f64      │
╞══════╪═══════════════╪═════════════════╪══════════════╪══════════╡
│ A    ┆ 1842          ┆ 1843.2          ┆ 0.999        ┆ 9991.3   │
│ B    ┆ 3419          ┆ 3421.7          ┆ 0.999        ┆ 17984.1  │
│ C    ┆ 5921          ┆ 5918.3          ┆ 1.000        ┆ 24972.4  │
│ D    ┆ 7834          ┆ 7832.1          ┆ 1.000        ┆ 21953.2  │
│ E    ┆ 6288          ┆ 6290.2          ┆ 1.000        ┆ 14962.1  │
│ F    ┆ 4897          ┆ 4894.8          ┆ 1.001        ┆ 9977.3   │
└──────┴───────────────┴─────────────────┴──────────────┴──────────┘
```

A/E ratios this close to 1.0 are expected because area is in the model - the model is calibrated to area. The more important A/E check is for **factors not in the model**. If you omit vehicle group from the frequency GLM and the A/E for large vehicle groups is consistently above 1.0 while small groups are below 1.0, that is evidence of a missing feature.

```python
# Check A/E by a factor not in the model: driver age bands
df_diag = df.with_columns(
    pl.when(pl.col("driver_age") < 25).then(pl.lit("17-24"))
    .when(pl.col("driver_age") < 35).then(pl.lit("25-34"))
    .when(pl.col("driver_age") < 50).then(pl.lit("35-49"))
    .when(pl.col("driver_age") < 65).then(pl.lit("50-64"))
    .otherwise(pl.lit("65+"))
    .alias("age_band")
)

ae_age = ae_by_factor(df_diag, glm_freq.fittedvalues, "age_band")
print(ae_age)
```

If the 17-24 band shows A/E of 1.45, the model is materially underpricing young drivers. Add age to the model.

### Double lift charts

A lift chart shows how well the model ranks risks. A double lift chart compares your new model against a benchmark (the existing Emblem model, or a naïve rate). Sort policies by the ratio `new_model_rate / benchmark_rate`. Bin into deciles. Plot the actual loss ratio in each decile.

If the new model is better than the benchmark, the left deciles (policies the new model rates lower than the benchmark) should have lower actual loss ratios, and the right deciles (policies the new model rates higher) should have higher actual loss ratios. A flat chart means the models agree. A reverse-sloping chart means the new model is worse.

```python
def double_lift_chart(
    df: pl.DataFrame,
    new_fitted: np.ndarray,
    benchmark_fitted: np.ndarray,
    claim_count_col: str = "claim_count",
    exposure_col: str = "exposure",
    n_bins: int = 10,
) -> pl.DataFrame:
    """
    Build a double lift chart comparing new_fitted against benchmark_fitted.
    Sort by ratio new/benchmark, bin into n_bins, compute observed frequency.
    """
    return (
        df
        .with_columns([
            pl.Series("new_fitted", new_fitted),
            pl.Series("benchmark_fitted", benchmark_fitted),
        ])
        .with_columns(
            (pl.col("new_fitted") / pl.col("benchmark_fitted")).alias("model_ratio")
        )
        .with_columns(
            pl.col("model_ratio")
            .rank()
            .floordiv(len(df) / n_bins)
            .clip(0, n_bins - 1)
            .cast(pl.Int32)
            .alias("decile")
        )
        .group_by("decile")
        .agg([
            pl.col(claim_count_col).sum().alias("actual_claims"),
            pl.col(exposure_col).sum().alias("exposure"),
            pl.col("new_fitted").sum().alias("new_expected"),
            pl.col("benchmark_fitted").sum().alias("benchmark_expected"),
            pl.col("model_ratio").mean().alias("mean_ratio"),
        ])
        .with_columns(
            (pl.col("actual_claims") / pl.col("exposure")).alias("observed_rate"),
            (pl.col("new_expected") / pl.col("exposure")).alias("new_rate"),
            (pl.col("benchmark_expected") / pl.col("exposure")).alias("benchmark_rate"),
        )
        .sort("decile")
    )
```

---

## Validating against Emblem output

This is the critical check: does your Python GLM reproduce Emblem's published relativities?

Emblem exports a factor table as a CSV - typically with columns like `Factor`, `Level`, `Relativity`, `SE`, `LowerCI`, `UpperCI`. To validate, you need to:

1. Feed your Python GLM the same data as Emblem (same extract, same development)
2. Use the same base levels
3. Use the same formula structure (same factors, same continuous/categorical treatment)
4. Compare `exp(beta)` from Python to Emblem's relativities column

The tolerances you should accept:

- **Identical data, identical specification**: relativities should match to 4+ decimal places. If they do not, there is a specification mismatch.
- **Same data, minor specification differences** (e.g. Emblem rounds vehicle group to bands, Python uses continuous): you expect differences that reflect the specification difference. Document them.
- **Different data vintage**: some differences are expected from the additional data. Do not try to match these; instead verify the sign and approximate magnitude are consistent.

```python
def compare_to_emblem(
    python_rels: pl.DataFrame,
    emblem_path: str,
    tolerance: float = 0.001,
) -> pl.DataFrame:
    """
    Compare Python GLM relativities to an Emblem CSV export.

    Emblem CSV format:
        Factor, Level, Relativity, SE, LowerCI, UpperCI

    Returns a comparison DataFrame with a 'match' column.
    """
    emblem_rels = pl.read_csv(
        emblem_path,
        schema_overrides={"Level": pl.Utf8, "Relativity": pl.Float64},
    )

    comparison = (
        python_rels
        .rename({"feature": "Factor", "level": "Level", "relativity": "Python_Rel"})
        .join(
            emblem_rels.rename({"Relativity": "Emblem_Rel"}).select(["Factor", "Level", "Emblem_Rel"]),
            on=["Factor", "Level"],
            how="inner",
        )
        .with_columns([
            ((pl.col("Python_Rel") - pl.col("Emblem_Rel")).abs()).alias("abs_diff"),
            ((pl.col("Python_Rel") / pl.col("Emblem_Rel") - 1).abs()).alias("rel_diff"),
        ])
        .with_columns(
            (pl.col("rel_diff") < tolerance).alias("match")
        )
        .sort(["Factor", "Level"])
    )

    n_matched = comparison["match"].sum()
    n_total = len(comparison)
    print(f"Matched: {n_matched}/{n_total} relativities within {tolerance*100:.1f}% tolerance")

    if n_matched < n_total:
        mismatches = comparison.filter(~pl.col("match"))
        print("\nMismatches:")
        print(mismatches.select(["Factor", "Level", "Python_Rel", "Emblem_Rel", "rel_diff"]))

    return comparison
```

### Common reasons for mismatches

**Base level differs.** Emblem defaults to the highest-exposure level as base; Python defaults to alphabetical first. If you have not pinned the base level explicitly, every relativity will be uniformly off by a constant.

**Continuous vs categorical encoding.** Emblem often auto-detects whether a factor should be treated as continuous or categorical. Python's formula interface requires you to be explicit. If Emblem fit vehicle group as categorical (27 dummies for groups 1-27, 28-50 collapsed) and your Python model treats it as continuous, the relativities will not match.

**Missing level consolidation.** Emblem consolidates sparse levels automatically (e.g. NCD=6 merged into NCD=5 if NCD=6 has fewer than 30 claims). Python will estimate a separate coefficient for every level unless you merge them manually. If Emblem merged levels and Python did not, the merged levels will look like mismatches.

**IRLS convergence settings.** Both tools use IRLS but with different default convergence criteria. On pathological data (very sparse cells, extreme outliers), one may converge to a slightly different local solution. Check `glm_freq.converged` and `glm_freq.nit`. If `nit` equals the maximum iterations (default 100), IRLS did not converge and your estimates are not reliable.

**Exposure definition.** Emblem sometimes uses policy count as the denominator when "earned exposure" is requested but the data does not have intra-policy exposure splits. Verify the column.

**Emblem manual overrides.** If the validation fails despite identical data and identical specification, the most likely explanation is a manual override in Emblem that was never documented. Someone clicked on a factor level in the Emblem UI and typed in a relativity. It happens constantly on live books.

Check whether any factor-level relativities in Emblem's export are suspiciously round numbers - 1.000, 0.850, 1.250. These are usually overrides, not estimated parameters. A likelihood-based GLM will almost never produce a relativity of exactly 0.850 to three decimal places. If you see a cluster of round-number relativities, treat the entire factor with scepticism and talk to whoever built the original Emblem model before concluding there is a coding error in your Python model.

---

## Exporting to Radar format

Willis Towers Watson Radar imports external relativity tables as CSVs. The format:

```
Factor,Level,Relativity
area,A,1.0000
area,B,1.1041
area,C,1.2185
...
```

The factor name must match the rating variable name in the Radar model exactly. Radar is case-sensitive. If your Python column is `area` and your Radar variable is `Area`, the import will fail silently - Radar will apply a relativity of 1.0 for all levels. Check the exact names in your Radar project before export.

```python
def to_radar_csv(
    rels: pl.DataFrame,
    output_path: str,
    factor_name_map: dict | None = None,
    decimal_places: int = 4,
) -> None:
    """
    Export relativities in Radar factor table import format.

    factor_name_map: dict mapping Python column names to Radar variable names.
                     e.g. {"area": "PostcodeArea", "ncd_years": "NCDYears"}
    """
    radar_df = rels.select(["feature", "level", "relativity"])

    if factor_name_map:
        radar_df = radar_df.with_columns(
            pl.col("feature").replace(factor_name_map).alias("feature")
        )

    radar_df = (
        radar_df
        .rename({"feature": "Factor", "level": "Level", "relativity": "Relativity"})
        .with_columns(
            pl.col("Relativity").round(decimal_places).alias("Relativity")
        )
    )

    radar_df.write_csv(output_path)
    print(f"Exported {len(radar_df)} factor table rows to {output_path}")


# Export frequency relativities for Radar
# Only export categorical factors - continuous features need banding first
cat_rels = freq_rels.filter(pl.col("level") != "continuous")

to_radar_csv(
    cat_rels,
    "/dbfs/mnt/pricing/outputs/freq_relativities_radar.csv",
    factor_name_map={
        "area": "PostcodeArea",
        "ncd_years": "NCDYears",
        "conviction_flag": "ConvictionFlag",
    },
)
```

One practical issue: Radar requires every level that exists in the policy file to appear in the import table. If your rating variable has NCD=6 (some insurers allow it for advanced drivers) but your Python model merged NCD=5 and NCD=6 into a single level, you need to add a NCD=6 row to the export. Decide the relativity (usually the NCD=5 relativity, or a mild discount beyond it) and add it explicitly:

```python
# Add NCD=6 with same relativity as NCD=5
ncd5_rel = cat_rels.filter(
    (pl.col("feature") == "ncd_years") & (pl.col("level") == "5")
)["relativity"].item()

ncd6_row = pl.DataFrame({
    "feature": ["ncd_years"],
    "level": ["6"],
    "log_relativity": [np.log(ncd5_rel)],
    "relativity": [ncd5_rel],
    "se": [0.0],
    "lower_ci": [ncd5_rel],
    "upper_ci": [ncd5_rel],
})

cat_rels_full = pl.concat([cat_rels, ncd6_row])
```

Do not let this happen quietly. Write a comment, add it to your model documentation, and flag it in the Radar export log. Any deviation from the fitted model is a business decision, not a technical detail.

---

## Running on Databricks

### Loading data from Delta tables

Rather than CSV exports, the production pattern loads data from Delta tables registered in Unity Catalog. This gives you time travel (query the data as it was at model-fit time) and full lineage tracking.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Load current policy data
df_spark = spark.table("main.pricing.motor_policies")

# Convert to Polars for manipulation
df = pl.from_pandas(df_spark.toPandas())
```

The `spark.table()` call automatically uses the current version of the Delta table. To use the version from a specific date (for model reproducibility), use the SQL time travel syntax:

```python
df_spark = spark.sql(
    "SELECT * FROM main.pricing.motor_policies TIMESTAMP AS OF '2024-03-15T00:00:00'"
)
```

Or equivalently with `.load()`:

```python
df_spark = (
    spark.read
    .format("delta")
    .option("timestampAsOf", "2024-03-15T00:00:00")
    .load("dbfs:/user-hive-metastore/main/pricing/motor_policies")
)
```

Note: `.table()` cannot be chained after `.option()` on a DataFrameReader - use `.load("path")` or the SQL approach above.

This is the reproducibility mechanism. Your model log should record the Delta table version number or the `timestampAsOf` string used when fitting. With that information, anyone can reconstruct the exact training dataset.

```python
# Record the table version used for training
table_version = spark.sql(
    "DESCRIBE HISTORY main.pricing.motor_policies LIMIT 1"
).first()["version"]

print(f"Training data version: {table_version}")
# Log this to MLflow - shown below
```

### Logging to MLflow

```python
import mlflow

mlflow.set_experiment("/pricing/motor-glm")

with mlflow.start_run(run_name="freq_glm_v2") as run:
    # Log parameters
    mlflow.log_params({
        "model_type": "Poisson_GLM",
        "formula": freq_formula,
        "n_policies": len(df),
        "training_data_version": table_version,
        "training_date": str(date.today()),
        "base_levels": str({"area": "A", "ncd_years": "0", "conviction_flag": "0"}),
    })

    # Log metrics
    mlflow.log_metrics({
        "deviance": glm_freq.deviance,
        "null_deviance": glm_freq.null_deviance,
        "pseudo_r2": 1 - (glm_freq.deviance / glm_freq.null_deviance),
        "aic": glm_freq.aic,
        "n_params": len(glm_freq.params),
        "converged": int(glm_freq.converged),
        "n_iterations": glm_freq.nit,
    })

    # Log the relativities as a CSV artefact
    rels_path = "/tmp/freq_relativities.csv"
    freq_rels.write_csv(rels_path)
    mlflow.log_artifact(rels_path, artifact_path="factor_tables")

    # Log the statsmodels model summary as a text artefact
    summary_path = "/tmp/glm_freq_summary.txt"
    with open(summary_path, "w") as f:
        f.write(str(glm_freq.summary()))
    mlflow.log_artifact(summary_path, artifact_path="diagnostics")

    run_id = run.info.run_id
    print(f"MLflow run ID: {run_id}")
```

MLflow does not have a native serialiser for statsmodels GLM objects, but you can save them using Python's `pickle`:

```python
import pickle

model_path = "/tmp/glm_freq_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(glm_freq, f)

mlflow.log_artifact(model_path, artifact_path="model")
```

A caveat on the pickle: statsmodels pickle files are not guaranteed to be forward-compatible across statsmodels version upgrades. If you load the pickle after upgrading statsmodels you may get an error. For long-term reproducibility, the formula string, the training data version, and the coefficient CSV are more reliable. Those three things let you refit the model from scratch without any pickle compatibility concerns. The pickle is useful for short-term scoring pipelines; the coefficient CSV is what you actually want for archival.

### Writing factor tables to Unity Catalog

```python
from datetime import date

# Add metadata columns
rels_with_meta = freq_rels.with_columns([
    pl.lit(str(date.today())).alias("model_run_date"),
    pl.lit("freq_glm_v2").alias("model_name"),
    pl.lit(run_id).alias("mlflow_run_id"),
    pl.lit(table_version).alias("training_data_version"),
    pl.lit(len(df)).alias("n_policies_trained"),
])

# Write to Delta table via Spark
spark.createDataFrame(rels_with_meta.to_pandas()).write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("main.pricing.glm_relativities")

print(f"Written {len(rels_with_meta)} rows to main.pricing.glm_relativities")
```

Using `mode("append")` rather than `mode("overwrite")` means every model run adds to the history. You can query the history of any factor's relativity over time:

```python
# How has the area F relativity changed across model runs?
area_f_history = spark.sql("""
    SELECT model_run_date, model_name, relativity
    FROM main.pricing.glm_relativities
    WHERE feature = 'area' AND level = 'F'
    ORDER BY model_run_date
""")
display(area_f_history)
```

This is the audit trail that both PS 21/5 and Consumer Duty require. When the regulator asks "how has your area F rating changed over the past two years and why," you have the complete answer in a queryable table.

---

## FCA Consumer Duty: what this workflow actually gives you

The FCA wants to be able to walk into your office and reconstruct any price charged to any customer in the past three years. The question is whether you can show them the model, the inputs, and the decision trail. This workflow makes that possible without a three-week excavation of network drives.

The underlying regulatory framework has two distinct layers that pricing actuaries sometimes conflate:

**PS 21/5 (General Insurance Pricing Practices, effective January 2022)** banned price walking - the practice of charging renewing customers more than equivalent new customers - and introduced explicit requirements for firms to implement systems and controls around pricing. This is where the audit trail obligation comes from. Your pricing must be deliberate, documented, and free of systematic renewal discrimination.

**PS 22/9 (Consumer Duty, effective July 2023)** extended the framework to require firms to demonstrate positive fair value outcomes. It is not enough to show that you did not price-walk; you must show that the price a customer pays is appropriately related to the risk they represent, and that your pricing process produces fair outcomes for different customer groups.

These are separate regulatory interventions, each with its own requirements. The workflow in this module supports both:

**Explainability (Consumer Duty).** You must be able to explain what each rating factor does to the price and why. Your GLM factor tables are the explanation. The MLflow artefacts and this tutorial are the documentation. When the FCA asks "why does a 23-year-old in area F pay significantly more than a 45-year-old in area A," you point to the NCD and area relativities, explain the statistical basis, and demonstrate the model was validated against out-of-sample data.

**Reproducibility (PS 21/5 + Consumer Duty).** Historical pricing decisions must be reproducible. "The model was in Emblem but we upgraded versions and the project file is corrupted" is not acceptable. Delta table time travel + MLflow model registry + Git-versioned code means any model run can be reproduced exactly.

**Change management (PS 21/5).** When relativities change materially between cycles, you need to document why. Because your relativities are in a Delta table with timestamps, you can run an automated diff between consecutive model versions and flag any relativity that changed by more than a threshold:

```python
# Compare current model relativities to previous cycle
# Rename before joining to avoid ambiguous column references
current = (
    spark.table("main.pricing.glm_relativities")
    .filter("model_run_date = '2024-09-30'")
    .withColumnRenamed("relativity", "current_relativity")
)

previous = (
    spark.table("main.pricing.glm_relativities")
    .filter("model_run_date = '2024-03-31'")
    .withColumnRenamed("relativity", "prev_relativity")
)

from pyspark.sql.functions import col, abs as spark_abs

changes = (
    current.join(previous, ["feature", "level"], "left")
    .withColumn(
        "pct_change",
        (col("current_relativity") / col("prev_relativity") - 1) * 100,
    )
    .filter(spark_abs(col("pct_change")) > 5)
)

display(changes)
```

Any change above 5% in any factor level should be accompanied by an explanation in your model change log. This is not additional work created by Python - it is work that should have been happening in Emblem anyway, now automated.

---

## Tweedie models: when you want one model instead of two

The frequency-severity split (Poisson frequency × Gamma severity) is the standard UK personal lines approach. An alternative is the Tweedie pure premium model, which models the total incurred amount directly using a compound Poisson-Gamma distribution.

The Tweedie GLM fits one model, not two. Its power parameter `p` controls the compound distribution: `p = 1` is Poisson, `p = 2` is Gamma, and `1 < p < 2` is the compound Poisson-Gamma relevant for insurance. In statsmodels:

```python
# Tweedie pure premium model in statsmodels
# Power p=1.5 is a reasonable starting point for UK motor pure premiums
# Compute pure_premium in Polars before converting to pandas
df_pp = df.with_columns(
    (pl.col("incurred") / pl.col("exposure")).alias("pure_premium")
)
df_pp_pd = df_pp.to_pandas()
df_pp_pd["log_exposure"] = np.log(df_pp_pd["exposure"].clip(lower=1e-6))

pp_formula = (
    "pure_premium ~ "
    "C(area) + "
    "C(ncd_years, Treatment(0)) + "
    "C(conviction_flag, Treatment(0)) + "
    "vehicle_group"
)

glm_tweedie = smf.glm(
    formula=pp_formula,
    data=df_pp_pd,
    family=sm.families.Tweedie(
        var_power=1.5,
        link=sm.families.links.Log(),
    ),
    offset=df_pp_pd["log_exposure"],
).fit()

print(f"Tweedie GLM deviance: {glm_tweedie.deviance:,.1f}")
print(f"Pseudo R²: {1 - glm_tweedie.deviance/glm_tweedie.null_deviance:.4f}")

# Extract pure premium relativities
pp_rels = extract_freq_relativities(
    glm_tweedie,
    base_levels={"area": "A", "ncd_years": "0", "conviction_flag": "0"},
)
print("\nTweedie area relativities:")
print(pp_rels.filter(pl.col("feature") == "area"))
```

When to use Tweedie vs frequency-severity split:

**Tweedie** is simpler (one model, not two) and handles the zero-inflated pure premium directly. It is appropriate when you only want a pure premium prediction and are not trying to understand whether a risk is high frequency or high severity.

**Frequency-severity split** gives you more diagnostic power. You can see whether your area F uplift is driven by frequency (more accidents) or severity (more expensive accidents). That distinction matters: high-frequency/low-severity areas have a different risk management profile from low-frequency/high-severity areas, and reinsurance structuring (frequency excess of loss vs large loss per-risk) is designed around that distinction. The split also allows different sets of rating factors for frequency and severity - occupation might be a strong frequency predictor but irrelevant for severity, and forcing it into a combined Tweedie model is wrong.

The Tweedie also requires you to commit to a value of `p`. Usually it is fixed at 1.5 or estimated separately. When the regulator asks why you chose `p=1.5`, you need an answer. The frequency-severity split has a cleaner actuarial justification: claims arrive as a Poisson process and each claim has a Gamma-distributed cost.

We recommend the frequency-severity split for production personal lines pricing. The Tweedie is useful for exploratory work or when you genuinely do not need the separate frequency and severity relativities.

---

## Putting it together: a complete motor GLM pipeline

The full end-to-end pipeline in summary:

```python
# 1. Load from Delta (Spark -> Polars)
df = pl.from_pandas(spark.table("main.pricing.motor_policies").toPandas())

# 2. Prepare features (Polars)
df = (
    df
    .filter(pl.col("exposure") > 0)
    .with_columns([
        pl.col("area").cast(pl.Enum(["A","B","C","D","E","F"])),
        pl.col("ncd_years").cast(pl.Int32),
        pl.col("conviction_flag").cast(pl.Int32),
    ])
)

df_pd = df.to_pandas()
df_pd["log_exposure"] = np.log(df_pd["exposure"].clip(lower=1e-6))

# 3. Fit frequency GLM (statsmodels)
glm_freq = smf.glm(
    formula="claim_count ~ C(area) + C(ncd_years, Treatment(0)) + C(conviction_flag, Treatment(0)) + vehicle_group",
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()
assert glm_freq.converged, "Frequency GLM did not converge"

# 4. Fit severity GLM (statsmodels, on claims only)
# NCD excluded: it is a frequency signal, not a severity driver
# Compute avg_severity in Polars before converting to pandas
df_sev_pd = (
    df
    .filter(pl.col("claim_count") > 0)
    .with_columns(
        (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
    )
    .to_pandas()
)

glm_sev = smf.glm(
    formula="avg_severity ~ C(area) + vehicle_group",
    data=df_sev_pd,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    var_weights=df_sev_pd["claim_count"],
).fit()
assert glm_sev.converged, "Severity GLM did not converge"

# 5. Extract relativities (Polars)
freq_rels = extract_freq_relativities(glm_freq, {"area": "A", "ncd_years": "0", "conviction_flag": "0"})
sev_rels = extract_freq_relativities(glm_sev, {"area": "A"})

# 6. Diagnostics
ae_area_freq = ae_by_factor(df, glm_freq.fittedvalues, "area")
ae_area_sev = ae_by_factor(
    df.filter(pl.col("claim_count") > 0),
    glm_sev.fittedvalues,
    "area",
)

# 7. Export to Radar
to_radar_csv(freq_rels.filter(pl.col("level") != "continuous"),
             "/dbfs/mnt/pricing/freq_rels.csv",
             factor_name_map={"area": "PostcodeArea", "ncd_years": "NCDYears", "conviction_flag": "ConvictionFlag"})

# 8. Log to MLflow and write to Unity Catalog
with mlflow.start_run(run_name="motor_glm_pipeline"):
    mlflow.log_params({...})
    mlflow.log_metrics({...})
    spark.createDataFrame(freq_rels.to_pandas()).write.format("delta").mode("append").saveAsTable("main.pricing.glm_relativities")
```

---

## Gotchas when moving from Emblem

We have worked with several teams on this migration. These are the problems that actually bite people.

**Emblem's automatic base level selection.** Emblem picks the most credible level as the base - usually the highest-exposure level. statsmodels picks alphabetically. Forgetting to align these is the number one source of "why don't the numbers match."

**Emblem's missing value handling.** Emblem treats missing values as a separate level, "Unknown," and estimates a relativity for it. statsmodels drops rows with missing values unless you handle them explicitly. If 3% of your policies have missing vehicle group and Emblem is pricing them as "Unknown," while Python is dropping them from the training set, your models are fit on different data.

```python
# Check for missing values before fitting
missing_report = df.null_count()
print(missing_report)

# Options:
# 1. Impute with the mode or a designated "Unknown" level
df = df.with_columns(
    pl.col("vehicle_group").fill_null(strategy="mean").cast(pl.Int32)
)
# 2. Create an "Unknown" level for categorical factors
df = df.with_columns(
    pl.col("area").fill_null("Unknown")
)
```

**Credibility-weighted relativities.** Emblem has a credibility option that shrinks sparse level relativities toward 1.0 using a Bühlmann-Straub weighting. By default in statsmodels, you get maximum likelihood estimates - no shrinkage. If Emblem's published relativities show NCD=6 at 1.000 (plausible but sparse), while your Python estimate is 0.78 with a wide CI, Emblem may have applied credibility weighting. Module 6 covers this properly.

**Iterative development patterns vs one-shot.** Emblem allows interactive factor-by-factor variable addition with immediate visual feedback. Python requires you to re-fit the full model when you add a factor. On 100,000 rows, this is fast (< 5 seconds). On 2 million rows with 50 dummy variables, IRLS takes longer. Build the iterative development loop explicitly - a function that re-fits and re-runs diagnostics on demand - rather than running ad-hoc notebook cells.

**The deviance statistic and likelihood ratio tests.** Emblem reports "change in scaled deviance" when you add a factor. statsmodels reports total deviance. To get the chi-squared test for adding a factor, compute the deviance difference manually - this is the likelihood ratio test:

```python
# Fit base model and extended model
glm_base = smf.glm(
    "claim_count ~ C(area) + C(ncd_years, Treatment(0))",
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

glm_extended = smf.glm(
    "claim_count ~ C(area) + C(ncd_years, Treatment(0)) + vehicle_group",
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

# Likelihood ratio test for adding vehicle_group
lr_stat = glm_base.deviance - glm_extended.deviance
df_diff = glm_base.df_resid - glm_extended.df_resid
from scipy import stats
p_value = stats.chi2.sf(lr_stat, df_diff)
print(f"LR chi-squared: {lr_stat:.2f}, df: {df_diff}, p-value: {p_value:.4f}")
```

---

## Summary

The Python GLM workflow produces output that is numerically consistent with Emblem when given the same data and the same specification. On synthetic data without manual overrides, the relativities match to four decimal places. On real Emblem models, validate any overrides explicitly before declaring a match - see the section on Emblem manual overrides above.

The difference is not in the model - it is in the surrounding infrastructure: version control, reproducibility, auditability, and integration with the rest of the modelling stack.

The critical steps, in order:

1. Get the exposure right. Use earned exposure, clip at a small positive number, filter out zeros.
2. Match the base levels to Emblem's explicitly. Do not rely on defaults.
3. Handle missing values deliberately - decide between dropping, imputing, or creating an "Unknown" level.
4. Truncate large losses before the severity GLM. Document the cap.
5. Exclude NCD from the severity formula - it is a frequency signal.
6. Check for aliased parameters and non-convergence before trusting results.
7. Run A/E diagnostics for factors not in the model as well as those that are.
8. Check deviance/df - if materially above 1, consider quasi-Poisson or negative binomial.
9. Validate against Emblem's published relativities on matched data, accounting for any manual overrides.
10. Log everything to MLflow and Unity Catalog before exporting to Radar.

The model is the easy part.
