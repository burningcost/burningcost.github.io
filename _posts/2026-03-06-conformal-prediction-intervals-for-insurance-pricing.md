---
layout: post
title: "Conformal Prediction Intervals for Insurance Pricing Models"
date: 2026-03-06
categories: [techniques]
tags: [conformal-prediction, gbm, catboost, uncertainty, tweedie, pricing, python]
---

Your Tweedie GBM gives point estimates. That is a problem.

A point estimate tells you the model's best guess for expected loss cost. It tells you nothing about how confident the model is in that guess, which varies enormously across the portfolio. A straightforward risk in a dense area of the feature space with thousands of similar policies behind it is not the same as an unusual commercial fleet risk that sits in a sparse corner. The model gives you a number in both cases. Without uncertainty quantification, you cannot tell them apart.

The standard approach to this is parametric confidence intervals - bootstrap the GLM coefficients, or propagate variance through the Tweedie dispersion parameter. Both approaches depend on distributional assumptions. If the model is misspecified (and it always is), the intervals are wrong in a way that is difficult to characterise.

Conformal prediction offers a different kind of guarantee. It does not need to assume anything about the error distribution. It produces intervals that contain the true value at least 90% of the time - not in expectation conditional on the model being correct, but unconditionally, as a finite-sample guarantee.

We built [`insurance-conformal`](https://github.com/burningcost/insurance-conformal) to apply conformal prediction to insurance pricing models. The key contribution is handling the heteroscedasticity that standard conformal implementations ignore.

---

## What conformal prediction guarantees

Split conformal prediction works as follows. You have a fitted model. You hold out a calibration set - data the model has never seen. For each calibration observation, you compute a non-conformity score: a number that measures how badly the model was wrong for that observation. Then, for a target miscoverage rate α, you find the (1 - α) quantile of those calibration scores. When predicting on new observations, you construct an interval around each point prediction using that quantile.

The coverage guarantee is:

```
P(y_test ∈ [lower, upper]) ≥ 1 - α
```

This holds for any data distribution and any model, as long as the calibration set was genuinely held out from training and the calibration and test data are exchangeable. That last word - exchangeable - roughly means "drawn from the same distribution in the same order". In insurance, it means you should calibrate on recent experience and test on more recent experience. It does not mean you can calibrate on 2023 and test on 2019.

The split conformal algorithm itself is twenty lines of code. The hard part is choosing the right non-conformity score.

---

## Why raw residuals are wrong for insurance data

Most conformal prediction implementations - and all general-purpose libraries - use the absolute residual as the non-conformity score:

```
score(y, ŷ) = |y - ŷ|
```

For Gaussian data this is fine. For insurance data it is wrong, and the error is systematic.

Consider a portfolio with a mix of standard motor risks (expected loss £400/year) and high-value properties (expected loss £8,000/year). The Tweedie distribution has variance proportional to the mean raised to the power p: Var(Y) ∝ μ^p. For a compound Poisson-Gamma model with p = 1.5, variance scales as μ^1.5. A £400 risk has about 90× less variance than an £8,000 risk.

The raw residual ignores this entirely. It treats a £300 miss on an £8,000 risk identically to a £300 miss on a £400 risk. The single calibration quantile is then wrong for both: it produces intervals that are too wide for low-risk policies and too narrow for large risks, exactly where miscoverage matters most.

The correct score is the locally-weighted Pearson residual:

```
score(y, ŷ) = |y - ŷ| / ŷ^(p/2)
```

Dividing by ŷ^(p/2) normalises by the model's expected standard deviation at each risk level. The calibration quantile is then on a variance-stabilised scale, and the inversion back to interval bounds is score-specific:

```
interval = ŷ ± q × ŷ^(p/2)
```

where q is the calibration quantile. This produces intervals that automatically widen proportionally with risk size, which is what the data's variance structure requires.

The practical effect is approximately 30% narrower intervals with identical coverage guarantees. This is not free - you are paying with a slightly more involved score function - but it is an improvement with no downside. The result is based on Manna et al. (2025), arXiv:2507.06921.

---

## Using the library

Install:

```bash
uv add insurance-conformal

# CatBoost support:
uv add "insurance-conformal[catboost]"
```

The workflow mirrors what you already do. Fit the model, then wrap it:

```python
from catboost import CatBoostRegressor, Pool
from insurance_conformal import InsuranceConformalPredictor

train_pool = Pool(data=X_train, label=y_train)
model = CatBoostRegressor(loss_function="Tweedie:variance_power=1.5", verbose=0)
model.fit(train_pool)

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",  # default for Tweedie models
    distribution="tweedie",
    tweedie_power=1.5,
)
```

Calibrate on a held-out set. This must be data the model has never seen:

```python
cp.calibrate(X_cal, y_cal)
```

For insurance, the calibration set should be temporally recent. A random 20% holdout mixes years, which means you are calibrating on data from all periods and testing on all periods simultaneously. That is not wrong in a strict technical sense, but it obscures any temporal trend in model residuals. Use a temporal split:

```python
from insurance_conformal.utils import temporal_split

X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
    X, y,
    calibration_frac=0.20,
    date_col="accident_year",
)

train_pool = Pool(data=X_train, label=y_train)
model.fit(train_pool)
cp.calibrate(X_cal, y_cal)
```

Then generate intervals on the test set:

```python
intervals = cp.predict_interval(X_test, alpha=0.10)
# DataFrame: lower, point, upper

print(intervals.head())
#       lower   point    upper
# 0    0.0121  0.0845   0.3291
# 1    0.0034  0.0231   0.0901
# 2    0.1820  1.2742   4.9621
```

`alpha=0.10` gives 90% prediction intervals. The lower bound is clipped at zero unconditionally - insurance losses are non-negative, and intervals with negative lower bounds are not useful.

---

## Coverage-by-decile: the diagnostic that matters

The marginal coverage guarantee is a floor averaged across all test observations. In practice, that average can conceal a serious problem: correct overall coverage with badly miscovered tails.

Consider a model with 90.1% marginal coverage. Sounds fine. If coverage in the top decile of predicted risk is 65%, you have a material problem - the portfolio's large risks have intervals that miss roughly one claim in three. An insurer relying on these intervals for reserving or capacity planning would be materially misled.

The coverage-by-decile diagnostic exposes this:

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

This is what well-calibrated intervals look like with `pearson_weighted`. Coverage is flat across deciles - the score is correctly accounting for variance structure.

For comparison, here is the same data with the `raw` absolute residual score:

```
   decile  mean_predicted  n_obs  coverage  target_coverage
0       1          0.0234    400     0.973             0.90
1       2          0.0512    400     0.961             0.90
3       4          0.1534    400     0.942             0.90
...
7       8          1.4210    400     0.871             0.90
8       9          2.5871    400     0.814             0.90
9      10          5.8340    400     0.723             0.90
```

Overall marginal coverage: 90.3%. Top decile coverage: 72.3%. The raw score overtreats low-risk policies - their intervals are much wider than necessary - and undertreats large risks, which are under-covered by 17 percentage points. The aggregate number hides both failures simultaneously.

Run the full diagnostic and plot with:

```python
cp.summary(X_test, y_test, alpha=0.10)

fig = cp.coverage_plot(X_test, y_test, alpha=0.10)
fig.savefig("coverage_by_decile.png", dpi=150)
```

`coverage_plot()` draws the decile coverage series with Wilson score confidence bands, which correctly propagate finite-sample uncertainty in the coverage estimate. With 400 observations per decile, those bands are roughly ±4pp - enough to distinguish a 65% coverage reading from a 90% one unambiguously.

---

## Non-conformity scores available

| Score | Formula | When to use |
|---|---|---|
| `pearson_weighted` | `\|y - ŷ\| / ŷ^(p/2)` | Default. Tweedie pure premium models. |
| `pearson` | `\|y - ŷ\| / sqrt(ŷ)` | Pure Poisson frequency models (p=1). |
| `deviance` | Deviance residual | Exact statistical optimality; slower to invert numerically. |
| `anscombe` | Anscombe transform | Variance-stabilising alternative to deviance. |
| `raw` | `\|y - ŷ\|` | Baseline comparison only. |

Ranked by interval width (narrowest first, coverage identical):
`pearson_weighted ≥ deviance ≥ anscombe > pearson > raw`

Use `pearson_weighted` unless you have a specific reason not to. The Tweedie power is read from CatBoost's `loss_function` parameter or passed explicitly. If auto-detection fails, the library warns and defaults to p=1.5. Pass `tweedie_power=` explicitly if your model is something else:

```python
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    tweedie_power=1.8,  # explicit, skips auto-detection
)
```

---

## Design choices

**Split conformal, not cross-conformal.** Cross-conformal is more statistically efficient - it uses the full dataset for both training and calibration by refitting the model on each fold. For a CatBoost model that takes three hours to train, this is not practical. Split conformal trains once, calibrates once, and gives the same finite-sample guarantee with a smaller calibration set as the only cost.

**No MAPIE dependency.** MAPIE is a good library, but it does not expose the insurance-specific scores here. The split conformal algorithm itself is simple enough to own: `conformal_quantile()` is 20 lines, plus the score functions. We have no interest in adding a dependency for functionality we can write in an afternoon.

**Lower bound clipped at zero.** Insurance losses are non-negative. An interval with a negative lower bound is mathematically coherent but practically useless. We clip unconditionally.

---

## What conformal prediction is not

It is not a way to get narrower intervals than your model's prediction accuracy warrants. If the model has large residuals, the calibration quantile will be large, and the intervals will be wide. The coverage guarantee is a floor, not a ceiling - conformal prediction cannot make a bad model look good.

It is also not a replacement for model calibration. A model that is systematically biased - for example, one that consistently underpredicts large losses - will produce intervals that are wide enough to achieve marginal coverage but shifted in a way that is diagnostically revealing. The coverage-by-decile diagnostic will show this as a pattern of low coverage on one side of the decile distribution, which points back to the model, not the conformal wrapper.

---

## Getting started

```bash
uv add insurance-conformal
```

Source and issue tracker on [GitHub](https://github.com/burningcost/insurance-conformal). The library is built around a single entry point - `InsuranceConformalPredictor` - and wraps any sklearn-compatible model. The coverage diagnostics work independently of the predictor via `CoverageDiagnostics` if you have intervals from another source and want to apply the same framework.

The first thing to check after calibrating is always `coverage_by_decile()`. If the top decile is more than 5 percentage points below target, switch from `raw` to `pearson_weighted`. If it is still off, try `deviance`. If coverage is non-monotone across deciles - high in the middle, low at both ends - your calibration data is not representative of the test distribution, and the temporal split is the first place to investigate.
