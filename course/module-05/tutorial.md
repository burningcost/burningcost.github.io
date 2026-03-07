# Conformal Prediction Intervals for Insurance Pricing Models

---

## Why this matters

Every UK pricing team has a version of the same conversation. A data scientist shows the pricing committee a CatBoost model that outperforms the production GLM by 4 points of Gini. The committee approves it. The model goes to production.

Six months later, the reinsurance team asks: "Which risks should we be most uncertain about?" The underwriting team asks: "How do we know when to refer a non-standard risk for review rather than letting the model price it automatically?" The reserving team asks: "What is the range of plausible outcomes, not just the expected value?"

The model cannot answer any of these questions. It gives a number. One number, for every risk, regardless of how well-supported that number is.

The standard fix is bootstrap confidence intervals or parametric intervals from the Tweedie dispersion parameter. Both approaches depend on distributional assumptions. If the model is misspecified - and every model is misspecified to some degree - the intervals are wrong in a way you cannot easily characterise. They might be fine. They might be systematically narrow for the largest risks in the book. You will not know without a diagnostic you may not have run.

Conformal prediction offers a different kind of guarantee. It does not assume anything about the error distribution. It produces prediction intervals that contain the true outcome at least 90% of the time - not in expectation conditional on the model being correctly specified, but as a finite-sample guarantee that holds for any model and any data distribution, as long as one condition is met: the calibration data must be exchangeable with the test data.

That condition has a straightforward interpretation for insurance: calibrate on recent experience, test on more recent experience. The guarantee breaks down if you calibrate on 2023 and apply intervals to 2019 data. We will be explicit about where the guarantee holds and where it does not.

The `insurance-conformal` library implements split conformal prediction with a variance-weighted non-conformity score designed for the heteroscedastic structure of insurance claims. The key result, from Manna et al. (2025), is approximately 30% narrower intervals than the standard approach with identical coverage guarantees.

---

## Prediction intervals, not confidence intervals

Before the code, a distinction that matters.

A **confidence interval for the mean** says: the expected loss cost for risks of this type falls in [L, U] with 95% probability. It narrows as you add data. With enough observations, it collapses to a point.

A **prediction interval for an individual observation** says: the actual loss outcome for this specific policy falls in [L, U] with 90% probability. It does not narrow arbitrarily as you add data - individual outcomes have irreducible variance. For a Tweedie model, that variance scales with the expected value. You can estimate the mean very precisely with a large dataset; individual outcomes still vary.

The pricing use cases in this module all require prediction intervals. When you ask "should I refer this risk to an underwriter?", the relevant quantity is the range of plausible outcomes for this individual policy, not the confidence interval for the mean across all similar policies. When you ask "what is the range of reserve requirements for this portfolio?", you are aggregating individual prediction intervals, which is different from computing a confidence interval for the portfolio mean.

Conformal prediction produces prediction intervals. The coverage guarantee is for individual predictions. Keep this distinction clear when presenting to a pricing committee or reserving team.

---

## How split conformal prediction works

The algorithm is simple. The sophistication is in choosing the right non-conformity score.

You have a fitted model. You hold out a calibration set - data the model has never seen, set aside before training. For each observation in the calibration set, compute a non-conformity score: a number that measures how wrong the model was for that observation. Call the scores s_1, s_2, ..., s_n.

For a target miscoverage rate α, you want intervals that miss the true value at most α of the time. The calibration quantile q is the ⌈(1 - α)(n + 1)⌉ / n quantile of the calibration scores. For a new test observation, the prediction interval is:

```
[lower, upper] = {y : score(y, ŷ) ≤ q}
```

That is, the set of outcomes that would not be labelled non-conforming given the calibration distribution.

The coverage guarantee:

```
P(y_test ∈ [lower, upper]) ≥ 1 - α
```

holds unconditionally, for any model and any data distribution, as long as the calibration scores and the test score are exchangeable - roughly, drawn from the same distribution in the same order.

For a linear score function like the absolute residual, this reduces to:

```
[lower, upper] = [ŷ - q, ŷ + q]
```

Fixed-width intervals. Every risk gets the same width regardless of its expected value. For insurance data with variance that scales with the mean, this is wrong. A £300 miss on a £300 risk is catastrophic. A £300 miss on a £30,000 risk is noise.

The correct score for Tweedie data is the locally-weighted Pearson residual:

```
score(y, ŷ) = |y - ŷ| / ŷ^(p/2)
```

Dividing by ŷ^(p/2) normalises by the model's expected standard deviation at each risk level. The calibration quantile q is then on a variance-stabilised scale. The interval for a new observation is:

```
[lower, upper] = [ŷ - q × ŷ^(p/2), ŷ + q × ŷ^(p/2)]
```

Intervals widen proportionally with risk size. A large risk gets a wide interval. A small risk gets a narrow one. Both reflect the actual variance structure.

The practical effect, demonstrated on a synthetic UK motor book in Manna et al. (2025): pearson_weighted intervals are approximately 30% narrower than raw residual intervals with identical 90% coverage. The interval width reduction comes entirely from the score function correctly accounting for heteroscedasticity.

---

## Setup

### Installation

```bash
uv add "insurance-conformal[catboost]"
```

The `[catboost]` extra pulls in CatBoost and enables auto-detection of the Tweedie power from the model's loss function parameter. For plotting:

```bash
uv add "insurance-conformal[all]"
```

On Databricks, add to the first cell of your notebook:

```python
%pip install "insurance-conformal[catboost]" polars --quiet
```

### Data

We use the same synthetic UK motor dataset from Module 4 - 50,000 policies with known true parameters. This lets us verify that the intervals have correct coverage. For this module, the key columns are:

- `claim_count`, `exposure`: Poisson frequency outcome
- `incurred`: total incurred cost, the pure premium target
- `vehicle_group`, `driver_age`, `ncd_years`, `area`, `conviction_points`, `annual_mileage`: features

We will work primarily with a Tweedie pure premium model - the direct target for most UK personal lines pricing teams.

```python
import polars as pl
import numpy as np
import catboost as cb

# For this module we use the synthetic motor dataset
# In practice, replace this with your own data pipeline
from insurance_conformal.datasets import load_motor_synthetic

df = load_motor_synthetic(n_policies=50_000, seed=42)
```

### Temporal split

The single most important practical decision in this module. We split data temporally, not randomly.

```python
from insurance_conformal.utils import temporal_split

X = df.select([
    "vehicle_group", "driver_age", "ncd_years",
    "area", "conviction_points", "annual_mileage"
]).to_pandas()

y = df["incurred"].to_pandas()

# 60% train, 20% calibration, 20% test - ordered by accident_year
X_train, X_cal, X_test, y_train, y_cal, y_test = temporal_split(
    X, y,
    calibration_frac=0.20,
    test_frac=0.20,
    date_col="accident_year",
)

print(f"Training:    {len(X_train):,} policies")
print(f"Calibration: {len(X_cal):,} policies")
print(f"Test:        {len(X_test):,} policies")
```

Why temporal rather than random? The exchangeability condition. Conformal prediction requires that the calibration scores and the test score come from the same distribution in the same ordering. In insurance, the relevant ordering is time - claims patterns, inflation, and exposure mix all drift year over year. If you calibrate on a random 20% sample that mixes all years, you are calibrating on data from the same temporal distribution as the test set, which satisfies the condition technically. But you are also hiding any temporal trend in residuals, which is useful diagnostic information.

A temporal split where calibration covers the most recent year before test makes the exchangeability assumption visible and testable. If coverage in the test period is materially below the calibration period's coverage, temporal drift is the first thing to investigate.

---

## Step 1: Train the model

We train a Tweedie pure premium model on the training split. The Tweedie distribution with variance power p between 1 and 2 is the standard choice for combined frequency-severity modelling of non-zero-inflated pure premiums.

```python
from catboost import CatBoostRegressor, Pool

# Feature preparation
cat_features = ["area"]
numeric_features = [
    "vehicle_group", "driver_age", "ncd_years",
    "conviction_points", "annual_mileage"
]

train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_features,
)

cal_pool = Pool(
    data=X_cal,
    label=y_cal,
    cat_features=cat_features,
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=cat_features,
)

# Tweedie pure premium model
# variance_power=1.5 is typical for UK motor;
# Poisson frequency-only would use variance_power=1.0,
# Gamma severity-only would use loss_function="Gamma"
tweedie_params = {
    "loss_function": "Tweedie:variance_power=1.5",
    "eval_metric": "Tweedie:variance_power=1.5",
    "learning_rate": 0.05,
    "depth": 5,
    "min_data_in_leaf": 50,
    "iterations": 400,
    "cat_features": cat_features,
    "random_seed": 42,
    "verbose": 0,
}

model = CatBoostRegressor(**tweedie_params)
model.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=50)

print(f"Best iteration: {model.best_iteration_}")
print(f"Test RMSE: {np.sqrt(np.mean((model.predict(test_pool) - y_test.values)**2)):.4f}")
```

The model is trained on training data only. The calibration and test pools are never passed to `fit()` except as an `eval_set` for early stopping - and even that is marginal. Strictly, early stopping on the calibration set means the model has used the calibration set to make a fitting decision. For conformal prediction purposes, this is a minor violation of the held-out assumption; in practice its effect on coverage is negligible. If you want to be strict, use a separate validation set for early stopping and keep calibration genuinely untouched.

### Why Tweedie, not frequency-severity separately

For conformal prediction, the pure premium target has a practical advantage: you have one model, one set of calibration residuals, one interval. Frequency-severity produces two models. To get a prediction interval on pure premium from frequency and severity models separately, you need to compose the two sets of intervals - that requires either Monte Carlo simulation or assumptions about the dependence structure between frequency and severity. Tweedie sidesteps this.

The pure premium approach does sacrifice some interpretability - you cannot easily separate the frequency and severity components of uncertainty. For underwriting decisions, "the total loss cost interval is [£200, £1,400]" is often more useful than separate frequency and severity intervals anyway.

---

## Step 2: Wrap with InsuranceConformalPredictor

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",  # variance-weighted score
    distribution="tweedie",
    tweedie_power=1.5,
)
```

The `tweedie_power` parameter tells the score function what power to use in the normalisation denominator ŷ^(p/2). If you omit it, the library attempts to read it from CatBoost's `loss_function` parameter. For `Tweedie:variance_power=1.5`, auto-detection reads `1.5`. If detection fails, the library warns and defaults to p=1.5, which is a reasonable prior for UK motor.

Pass `tweedie_power` explicitly if:
- Your loss function string is in a format the parser does not recognise
- You are using a non-CatBoost model where there is no loss function string to parse
- You want the code to be self-documenting (recommended)

---

## Step 3: Calibrate

```python
cp.calibrate(X_cal, y_cal)
```

Under the hood, this:
1. Generates predictions on `X_cal`
2. Computes the non-conformity score for each calibration observation: `|y - ŷ| / ŷ^(p/2)`
3. Stores the sorted score distribution

The result is a single quantile per coverage level. For `alpha=0.10`, you get the 90th percentile of the calibration scores. That quantile is the threshold. Any new observation whose score exceeds the threshold is in the "non-conforming" 10%.

Calibration is fast - it is just scoring and sorting a set of predictions you already have. The computation is dominated by generating the predictions from the CatBoost model, which takes a few seconds for 10,000 observations.

### What calibration set size do you need?

The coverage guarantee holds for any calibration set size, but the margin of error on coverage depends on n. With n calibration observations, the true coverage is within roughly `1 / sqrt(n)` of the target, with high probability.

| Calibration n | Margin of error on coverage (approximately) |
|---------------|---------------------------------------------|
| 500           | ±4.5pp                                      |
| 1,000         | ±3.2pp                                      |
| 2,000         | ±2.2pp                                      |
| 5,000         | ±1.4pp                                      |
| 10,000        | ±1.0pp                                      |

For a 90% target, you need to be sure you are not actually getting 85%. With 1,000 calibration observations, ±3.2pp means the interval [86.8%, 93.2%] for the true coverage. For most insurance applications, 2,000-5,000 calibration observations gives you enough certainty about the coverage level without wasting too much training data.

If your calibration set is under 500, treat the intervals as approximate and be conservative about claiming the stated coverage level.

---

## Step 4: Generate prediction intervals

```python
intervals = cp.predict_interval(X_test, alpha=0.10)

print(intervals.head(10))
```

```
     lower   point    upper
0   0.0121  0.0845   0.3291
1   0.0034  0.0231   0.0901
2   0.1820  1.2742   4.9621
3   0.0089  0.0621   0.2421
4   2.1340  14.921  58.212
5   0.0011  0.0076   0.0296
6   0.0352  0.2460   0.9593
7   0.0445  0.3111   1.2130
8   0.0081  0.0568   0.2215
9   0.8230  5.7590  22.443
```

`alpha=0.10` gives 90% prediction intervals. The units are whatever you trained on - if `y` is annual pure premium in pounds, the intervals are in pounds.

The lower bound is clipped at zero unconditionally. Insurance losses are non-negative, and intervals with negative lower bounds are not useful as risk communication tools.

Notice row 4: predicted pure premium £14.92 with interval [£2.13, £58.21]. That is a high-risk observation in a sparse part of the feature space - the model is uncertain. Row 5: predicted £0.008 with interval [£0.001, £0.030]. Low risk, narrow interval. The variance-weighted score is doing its job: intervals scale with the risk level.

### Generating intervals for a specific coverage level

The `alpha` parameter can be anything between 0 and 1. Common choices:

- `alpha=0.10` - 90% intervals: standard for uncertainty flagging
- `alpha=0.05` - 95% intervals: conservative, appropriate for reserving inputs
- `alpha=0.20` - 80% intervals: wider tolerance, useful for soft minimum premium floors

You can generate multiple sets of intervals from a single calibrated predictor:

```python
intervals_90 = cp.predict_interval(X_test, alpha=0.10)
intervals_95 = cp.predict_interval(X_test, alpha=0.05)
intervals_80 = cp.predict_interval(X_test, alpha=0.20)
```

No recalibration needed. The calibration step stores the full score distribution; different `alpha` values just read different quantiles from it.

---

## Step 5: Validate coverage before trusting the intervals

The marginal coverage guarantee says the average miss rate across all test observations will be at most α. That average can conceal serious problems in the tails.

Consider this: overall coverage is 90.2%. Sounds fine. If coverage in the top risk decile is 68%, you have a material problem - the portfolio's largest risks have intervals that miss one claim in three. A reserving team using these intervals to bracket outcomes would be systematically surprised by the largest risks. An underwriting team using the intervals to set referral thresholds would be under-referring the cases with the most uncertainty.

Run the coverage-by-decile diagnostic before using the intervals for any purpose:

```python
diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
print(diag)
```

```
   decile  mean_predicted  n_obs  coverage  target_coverage
0       1          0.0234    400     0.923             0.90
1       2          0.0512    400     0.910             0.90
2       3          0.0891    400     0.912             0.90
3       4          0.1534    400     0.908             0.90
4       5          0.2671    400     0.901             0.90
5       6          0.4622    400     0.897             0.90
6       7          0.8034    400     0.893             0.90
7       8          1.4210    400     0.891             0.90
8       9          2.5871    400     0.894             0.90
9      10          5.8340    400     0.898             0.90
```

This is what well-calibrated intervals look like with `pearson_weighted`. Coverage is flat across deciles - the variance-weighted score is correctly accounting for the data's variance structure.

For comparison, run the same diagnostic with the naive absolute residual score. You would see something like:

```
   decile  mean_predicted  n_obs  coverage  target_coverage
0       1          0.0234    400     0.973             0.90
1       2          0.0512    400     0.961             0.90
...
7       8          1.4210    400     0.871             0.90
8       9          2.5871    400     0.814             0.90
9      10          5.8340    400     0.723             0.90
```

Marginal coverage: 90.3%. Top decile coverage: 72.3%. The aggregate number hides both failures simultaneously - low-risk policies are grossly overtreated, large risks are badly undercovered. This is the pattern that `pearson_weighted` eliminates.

### Reading the coverage plot

```python
fig = cp.coverage_plot(X_test, y_test, alpha=0.10)
fig.savefig("/dbfs/mnt/pricing/coverage_by_decile.png", dpi=150)
```

The plot shows the coverage series with Wilson score confidence bands - which correctly propagate finite-sample uncertainty in the coverage estimate. With 400 observations per decile, those bands are roughly ±4pp. A reading of 72% coverage is 18pp below target and several standard deviations outside the confidence band. A reading of 87% coverage with 4pp bands is in the acceptable zone.

What you are looking for:
- All deciles within 5pp of target: good, use the intervals
- Monotone decline from low to high deciles: residual heteroscedasticity, try switching to `deviance` score
- Non-monotone: coverage high in middle, low at both ends, suggesting distribution shift between calibration and test data. Investigate the temporal split first

### Full summary

```python
cp.summary(X_test, y_test, alpha=0.10)
```

This prints marginal coverage, mean interval width, interval width distribution by decile, and flags any decile more than 5pp off target.

---

## Step 6: Practical application - flagging uncertain risks

The most immediate operational use of prediction intervals is flagging risks for underwriting referral. A risk with a very wide interval relative to its point estimate is one the model is uncertain about. That uncertainty might be benign (thin data, not a bad risk) or it might be a signal to look harder.

The natural measure of relative uncertainty is the interval-to-point ratio:

```python
intervals_90 = cp.predict_interval(X_test, alpha=0.10)

intervals_90["point_estimate"] = intervals_90["point"]
intervals_90["interval_width"] = intervals_90["upper"] - intervals_90["lower"]
intervals_90["relative_width"] = (
    intervals_90["interval_width"] / intervals_90["point"].clip(lower=1e-6)
)

# Flag top 10% by relative width for review
width_threshold = intervals_90["relative_width"].quantile(0.90)
intervals_90["flag_for_review"] = intervals_90["relative_width"] > width_threshold

n_flagged = intervals_90["flag_for_review"].sum()
print(f"Flagged for review: {n_flagged:,} ({100 * n_flagged / len(intervals_90):.1f}%)")
print(f"\nFlagged risk profile:")
print(intervals_90[intervals_90["flag_for_review"]][
    ["lower", "point", "upper", "relative_width"]
].describe())
```

In production, pass the flag to your quote system. Risks with `flag_for_review=True` trigger a referral instead of an automated quote. The threshold is a business decision: 10% referral rate, 15%, 5%. Start conservative (10%) and tune based on actual referral outcomes.

### What drives wide intervals?

Relative width is highest for:
- Observations in sparse regions of the feature space (thin training data for this risk type)
- High predicted value risks where absolute variance is inherently larger
- Risks with unusual feature combinations not well-represented in training

You can investigate which features correlate with wide intervals by joining the interval frame back to the feature matrix:

```python
import pandas as pd

flagged = pd.concat([
    X_test.reset_index(drop=True),
    intervals_90.reset_index(drop=True)
], axis=1)

# Mean relative width by area band - which areas have the most uncertain pricing?
print(
    flagged.groupby("area")["relative_width"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
)
```

Actionable result: if area F has mean relative width 3.2x the portfolio average, that is evidence the model is thinner there. The pricing team should look at whether the data quality in area F is lower, or whether a specific vehicle/driver combination in area F is dominating the thin cell.

---

## Step 7: Minimum premium floors

Prediction interval upper bounds give you a principled minimum premium floor. The logic: the upper bound of a 90% interval is the loss outcome exceeded only 10% of the time, assuming the calibration period was representative. Setting your minimum premium at the 90% upper bound means you expect the premium to be insufficient no more than 10% of the time for that risk - not that it covers losses in bad years, which requires a different argument about what a bad year means for the portfolio.

This is different from the conventional approach, which typically sets floors based on a percentage of a GLM-derived technical premium or a fixed floor from historical experience. The conformal approach is data-driven and risk-specific: a high-volatility risk gets a higher floor, not a percentage of a lower mean estimate.

```python
# intervals_95 for a more conservative floor
intervals_95 = cp.predict_interval(X_test, alpha=0.05)

# Proposed minimum premium: upper bound of 95% interval
# (the outcome exceeded only 5% of the time)
min_premium_floor = intervals_95["upper"]

# Compare to point estimate
ratio_to_point = min_premium_floor / intervals_95["point"]
print(f"Median floor / point estimate ratio: {ratio_to_point.median():.2f}")
print(f"90th percentile floor / point estimate ratio: {ratio_to_point.quantile(0.90):.2f}")
```

For a well-calibrated Tweedie model on UK motor data, the 95% upper bound is typically 3-5x the point estimate for most risks, reflecting the right-skew of the Tweedie distribution. This ratio will be higher for large risks (where absolute variance is greater) and for thin-cell risks (where the model is less certain).

In practice, you will not set the minimum premium equal to the upper bound - that is far too conservative for a competitive market. But the upper bound gives you an objective anchor. "Our minimum premium is 1.5x the technical premium but never below the 80th percentile of the prediction interval" is a more principled floor construction than a fixed multiplier applied uniformly.

```python
# Practical floor: max of (1.5 x point estimate, 80th percentile upper bound)
intervals_80 = cp.predict_interval(X_test, alpha=0.20)

practical_floor = np.maximum(
    1.5 * intervals_80["point"],
    intervals_80["upper"]  # 80% upper bound
)
```

Document the rationale. Consumer Duty requires firms to be able to explain pricing decisions. "Minimum premium is the higher of 150% of technical premium and the 80th percentile prediction bound from our calibrated Tweedie model, using conformal prediction intervals with validated 90% coverage across all risk deciles" is auditable. A fixed floor of £350 applied uniformly is not, unless it is backed by equivalent analysis.

---

## Step 8: Reserve range estimates

Prediction intervals for individual risks aggregate to portfolio-level range estimates. This is conceptually straightforward but requires care: individual prediction intervals are for individual outcomes. Summing them naively assumes perfect positive dependence (all risks simultaneously hit their upper bounds), which overstates the portfolio variance.

For a portfolio of n independent risks with intervals [L_i, U_i], the portfolio-level range is:

```python
# Point estimate: sum of individual point estimates
portfolio_point = intervals_90["point"].sum()

# Naive range (assumes perfect correlation - conservative)
portfolio_lower_naive = intervals_90["lower"].sum()
portfolio_upper_naive = intervals_90["upper"].sum()

# Independence-based range.
# For symmetric distributions, the 90% interval width equals 2 * 1.645 * sd, which
# lets you recover sd from the interval. But Tweedie intervals are asymmetric - the
# upper bound is materially further from the point estimate than the lower bound.
# Applying the symmetric formula to asymmetric intervals understates portfolio variance,
# particularly for books with heavy concentration in large risks.
#
# A more honest approach: use the naive (perfect-correlation) range as the upper bound
# and present the true range as lying somewhere between independence and perfect correlation.
# The code below shows both; do not present the independence range as a precise estimate.
#
# Alternative: simulate using the Tweedie distribution at the calibrated quantile scale.
# That is more accurate but requires additional implementation.

# Approximate independence range (use with caution for Tweedie data)
# The symmetric normal approximation is used here for illustration only.
# It will understate variance for right-skewed, heavy-tailed books.
approx_sd = (intervals_90["upper"] - intervals_90["lower"]) / (2 * 1.645)
portfolio_sd = np.sqrt((approx_sd**2).sum())
portfolio_lower_indep = portfolio_point - 1.645 * portfolio_sd
portfolio_upper_indep = portfolio_point + 1.645 * portfolio_sd

print(f"Portfolio point estimate:       £{portfolio_point:,.0f}")
print(f"Naive (perfect corr) 90% range: £{portfolio_lower_naive:,.0f} - £{portfolio_upper_naive:,.0f}")
print(f"Independence range (approx):     £{portfolio_lower_indep:,.0f} - £{portfolio_upper_indep:,.0f}")
```

In practice, insurance risks are not independent - they share weather events, economic cycles, inflation shocks. The true portfolio range lies between the independence and perfect-correlation bounds.

The independence range above uses a symmetric normal approximation to extract individual standard deviations from the interval widths. This is a rough approximation: Tweedie prediction intervals are right-skewed, not symmetric. The formula underestimates portfolio variance for books with heavy concentration in large risks. For a more reliable independence-based range, simulate individual outcomes from the Tweedie distribution parameterised at each policy's point estimate and calibrated quantile.

Present both bounds to the reserving team and be explicit: the naive range assumes everything goes wrong together; the independence range assumes diversification and should be treated as approximate for Tweedie data.

### Segmented reserve ranges

The more operationally useful analysis is reserve ranges by portfolio segment - by area, by vehicle group, by book year:

```python
import pandas as pd

segment_frame = pd.concat([
    X_test.reset_index(drop=True)[["area"]],
    intervals_90.reset_index(drop=True)
], axis=1)

segment_summary = segment_frame.groupby("area").agg(
    n_risks=("point", "count"),
    total_point=("point", "sum"),
    total_lower=("lower", "sum"),
    total_upper=("upper", "sum"),
).assign(
    upper_lower_ratio=lambda df: df["total_upper"] / df["total_lower"]
)

print(segment_summary.sort_values("upper_lower_ratio", ascending=False))
```

Segments with high upper/lower ratios have the most variance in their loss outcomes. For a reinsurance team setting treaty terms by geographic area, this is directly actionable: the areas with the widest reserve ranges are the candidates for aggregate stop-loss cover.

---

## Step 9: Integration with Databricks

### Running as a notebook

The companion `notebook.py` covers the full workflow. Upload to your workspace:

```bash
databricks workspace import notebook.py \
    /Workspace/pricing/module-05-conformal-intervals \
    --format SOURCE --language PYTHON
```

First cell:

```python
%pip install "insurance-conformal[catboost]" polars --quiet
```

On Databricks Runtime 14.x and later, this installs into the current session without a cluster restart.

### Writing intervals to Unity Catalog

Do not write intervals to a local file. Write to Delta Lake so there is a permanent, versioned record tied to the model version.

```python
from datetime import date
import mlflow

# Log the conformal predictor state alongside the model
with mlflow.start_run(run_name="conformal_calibration"):
    mlflow.log_param("nonconformity_score", "pearson_weighted")
    mlflow.log_param("tweedie_power", 1.5)
    mlflow.log_param("alpha", 0.10)
    mlflow.log_param("calibration_n", len(X_cal))
    mlflow.log_param("calibration_coverage", cp.calibration_coverage_)
    mlflow.log_metric("marginal_coverage_test", cp.summary(X_test, y_test, alpha=0.10)["marginal_coverage"])

# Write intervals with metadata
intervals_to_write = intervals_90.copy()
intervals_to_write["model_run_date"] = str(date.today())
intervals_to_write["model_version"] = "tweedie_catboost_v2"
intervals_to_write["alpha"] = 0.10
intervals_to_write["nonconformity_score"] = "pearson_weighted"

spark.createDataFrame(intervals_to_write).write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("main.pricing.conformal_intervals")

# Write coverage diagnostics
diag["model_run_date"] = str(date.today())
spark.createDataFrame(diag).write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("main.pricing.conformal_coverage_log")
```

Keeping a history of coverage diagnostics is important. If coverage in the top decile degrades from 89% to 78% between two runs, that is a signal the model needs recalibration or retraining. An append-mode Delta table with a run date column gives you a monitoring time series with no additional infrastructure.

### Scheduling recalibration

Conformal prediction separates the training step from the calibration step. You can update calibration without retraining the model. This matters in practice: a GBM that takes three hours to train on a full portfolio can have its calibration refreshed in minutes by running `cp.calibrate()` on the most recent quarter's data.

Set a monthly or quarterly calibration refresh. Set an annual (or biannual) full retrain. The decision rule:

- If coverage-by-decile degrades but marginal coverage is stable: calibrate only
- If marginal coverage degrades: retrain, then recalibrate
- If the feature set or portfolio mix changes materially: retrain, then recalibrate

---

## Step 10: Using the CoverageDiagnostics tool independently

The coverage diagnostic tools work independently of the predictor. If you have intervals from another source - a GLM with bootstrap intervals, a Bayesian posterior predictive, or intervals from MAPIE - you can apply the same framework:

```python
from insurance_conformal import CoverageDiagnostics

# external_lower, external_upper: arrays of interval bounds from any source
diag = CoverageDiagnostics(
    y_test=y_test,
    lower=external_lower,
    upper=external_upper,
    point=external_point,
    alpha=0.10,
)

diag.coverage_by_decile()
diag.plot()
diag.summary()
```

This is useful for comparing interval quality across methods. The coverage-by-decile diagnostic is the right framework for evaluating any set of prediction intervals for insurance data, regardless of how they were constructed. If your current method uses parametric Tweedie confidence intervals, running CoverageDiagnostics on them will reveal whether they achieve correct coverage in the tails - and almost certainly, they will not.

---

## When conformal prediction breaks down

The finite-sample coverage guarantee is unconditional but not unconditional. It depends on one assumption: exchangeability.

**Temporal distribution shift.** The most common failure in insurance. If claim frequency increases materially between the calibration period and the deployment period - due to economic conditions, weather, fraud trends, or a change in your underwriting appetite - the calibration scores no longer represent the test score distribution. The intervals will be systematically narrow. The coverage-by-decile diagnostic will show this as uniformly low coverage across all deciles, not a decile-specific pattern.

Detection: track marginal coverage monthly on a held-out slice of recent business. If it drops below target by more than 3-4pp, recalibrate. If recalibration on recent data still gives poor coverage, the model itself needs retraining.

**Portfolio composition change.** If you change your underwriting guidelines between the calibration period and deployment - adding a new distribution channel, entering a new geographic area, changing vehicle eligibility rules - the calibration scores may not represent the new book. Recalibrate on data from the new composition.

**Tiny calibration sets.** Below 200-300 observations, the coverage is much more variable than the theoretical guarantee suggests. The guarantee still holds on average, but individual runs can be materially off target. Do not use conformal prediction on books with fewer than 300 observations in the calibration set without significant caveats.

**Prediction without calibration.** The `InsuranceConformalPredictor` will raise an error if you call `predict_interval()` before `calibrate()`. This is intentional. There is no meaningful default for the calibration quantile - it must be estimated from data.

**Stochastic claims processes with a lot of zeros.** For lines of business where most policies have zero claims in any year - commercial property, professional liability, some lines of motor - the Tweedie distribution handles the zero mass, but the non-conformity scores can behave oddly in very sparse data regimes. Run the coverage diagnostic carefully. If coverage in the bottom two or three deciles is substantially above target (intervals too wide for the smallest predicted risks), this is likely a sparse-zero effect. The `anscombe` or `deviance` score may perform better for those specific lines.

**Regression to the mean in thin cells.** If your model has regularisation that pulls thin cells towards the mean (which all GBMs do via min_data_in_leaf and max_depth), the intervals for thin-cell risks will reflect that regularisation. The model predicts closer to the mean than the true risk; the residuals are correspondingly smaller than they would be without regularisation; the calibration quantile underestimates true variance for those cells. This is not really a conformal prediction failure - it is a model failure. Conformal prediction is honest about what the model says; if the model is regularised, the intervals reflect the regularised predictions.

**Heterogeneous calibration data.** If your calibration set mixes data from different claim years with very different inflation or frequency trends, the calibration quantile is an average over a non-stationary distribution. The temporal split helps here, but it does not eliminate the problem if the calibration period itself spans multiple years with different loss environments.

---

## Limitations

Be upfront about these when presenting intervals to a pricing committee, reserving team, or regulator.

**Marginal coverage only, not conditional coverage.** The formal guarantee is that the average coverage across all test observations is at least 1 - α. It does not guarantee 90% coverage for every subgroup. In practice, with `pearson_weighted`, coverage-by-decile is flat, which gives practical conditional coverage. But there is no mathematical guarantee for any specific subgroup. If the regulator asks "do you guarantee 90% coverage for elderly drivers specifically?", the correct answer is: "Not formally. The guarantee is marginal. Here is our coverage-by-decile diagnostic showing coverage is uniform across risk levels."

**Intervals are not Bayesian credible intervals.** The coverage guarantee is frequentist: across many repetitions of the data-generating process, the interval will contain the true value at least 90% of the time. It does not say "given this specific observation, the true value is in the interval with 90% probability." Conformal prediction intervals and Bayesian credible intervals answer different questions. For most insurance applications, the frequentist guarantee is what you want.

**The model's predictions are fixed.** Conformal prediction calibrates around whatever the model produces. If the model is systematically biased - underpredicting large losses, for example - the intervals will be centred on biased predictions. They will be wide enough to achieve marginal coverage, but the centre of the interval will be wrong. Coverage-by-decile will reveal this: you will see coverage above target on one side of the distribution and below on the other. The fix is model improvement, not conformal recalibration.

**No claim-count decomposition.** These intervals are for pure premium (expected loss cost per policy). You cannot directly decompose them into a frequency interval and a severity interval. For use cases that require separate frequency and severity uncertainty - some reinsurance pricing analyses, motor injury reserve estimates - you need to build separate conformal predictors for each component and accept that the composition is not straightforward.

**Not a replacement for stochastic reserving.** The reserve range estimates in Step 8 are prediction intervals for future outcomes, not stochastic reserve distributions. Stochastic reserving involves a different set of assumptions about claims development, inflation, and correlation across development periods. Conformal intervals for pure premium are an input to a reserving conversation, not a complete reserving framework.

---

## Summary

The conformal prediction workflow in four steps:

1. **Temporal split**: `temporal_split(X, y, calibration_frac=0.20, date_col="accident_year")`
2. **Train**: standard CatBoost Tweedie model on training split
3. **Calibrate**: `cp = InsuranceConformalPredictor(model, nonconformity="pearson_weighted", tweedie_power=1.5)` then `cp.calibrate(X_cal, y_cal)`
4. **Validate and use**: `cp.coverage_by_decile(X_test, y_test, alpha=0.10)` before trusting any downstream application

The coverage-by-decile diagnostic is the gate. Do not use the intervals for business decisions without running it and confirming coverage is flat across deciles. With `pearson_weighted`, it will be. With `raw`, it will not be for insurance data.

The three practical applications:
- **Uncertain risk flagging**: relative interval width identifies risks in thin regions of the feature space; flag the top decile for underwriting referral
- **Minimum premium floors**: upper bound of a conservative interval (90-95%) provides a data-driven, risk-specific floor that scales with the model's uncertainty rather than a flat multiplier
- **Reserve ranges**: sum of individual intervals gives naive portfolio bounds; CLT-based aggregation under independence gives the central scenario; these are inputs to a reserving conversation, not a complete framework

The library's key design decision is `pearson_weighted` as the default score. Raw residuals are wrong for insurance data. The 30% interval width reduction from the variance-weighted score is not a marginal improvement - it is the difference between intervals that are useful for underwriting decisions and intervals that are uniformly too wide for low-risk policies and uniformly too narrow where it matters most.

The guarantee is distribution-free: it holds for any model, any data distribution, any number of trees, any feature set, as long as the calibration data is exchangeable with the test data. In insurance, that means a temporal split and recalibration when the loss environment changes. Both are manageable. The intervals are not free, but they are honest.
