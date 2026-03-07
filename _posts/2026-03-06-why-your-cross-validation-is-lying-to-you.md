---
layout: post
title: "Why Your Cross-Validation is Lying to You"
date: 2026-03-06
categories: [pricing, python, libraries]
tags: [cross-validation, temporal-leakage, ibnr, walk-forward, insurance-cv, sklearn, catboost]
description: "Standard k-fold cross-validation is wrong for insurance pricing models. How temporal leakage and IBNR contamination inflate CV scores, and how walk-forward validation fixes both problems."
---

Your GBM is tuned. CV loss looks good across five folds. You commit the hyperparameters, run prospective monitoring, and six months into the rating year the modelled-to-actual ratio is drifting. The loss ratio is worse than the CV results suggested it would be. You open a ticket. Someone says "overfitting." Everyone nods.

It is not overfitting. It is leakage - baked directly into the cross-validation methodology from the first day of the project. The CV results were never honest. The model looked better than it was because the evaluation methodology let future information into the past.

This post explains exactly how that happens with standard k-fold on insurance data, what the consequences are, and how `insurance-cv` fixes it.

---

## Three ways k-fold breaks on insurance data

### 1. Temporal leakage

Insurance claims develop over time. A motor claim reported 18 months after the accident may still be open, with reserves moving as liability is established and repair costs are agreed. A casualty claim may take years.

K-fold randomly allocates policies to folds. A policy that incepted in March 2020 might land in your test fold. The claims development data used to construct your target variable - incurred losses, ultimate estimates - was snapshotted at your data extraction date, say January 2024. That snapshot includes 46 months of development. If that policy appears in a training fold alongside one from March 2022, your model is implicitly learning from a world where March 2022 claims already have 22 months of development. That is fine for training.

The problem is the test fold. If fold 3 contains mostly 2020 policies and fold 1 contains mostly 2022 policies, and you train fold 3 while testing on fold 1, you have trained on data with substantially more development than the test set you are evaluating against. The test set's targets are understated because they are less developed. The model, trained on more mature data, appears to predict them well - but the reason it appears accurate is partly that the claim development pattern in the training data contains forward-looking information about how the test claims will eventually develop.

This is temporal leakage. K-fold does it routinely. Not always dramatically, but systematically.

### 2. IBNR contamination

Incurred But Not Reported claims are the more direct version of the same problem. For any accident date near your training cutoff, some claims will not yet have been reported at all. If the accident date is in your training set, the claim count for that policy might be zero not because nothing happened, but because the claim hasn't been reported yet.

Training on these policies teaches the model that the risk was lower than it actually was. Evaluating on a test fold drawn randomly from the same period means some test policies have the same problem. Both train and test losses are understated, and the model appears to generalise better than it will prospectively - because prospective evaluation will use fully-developed claims.

The fix is a development buffer: exclude a window of time before each test period from both training and test sets. For motor own damage, three to six months. For casualty and professional indemnity, potentially 24 to 48 months. K-fold has no concept of this buffer. It cannot have one, because its folds are not temporally ordered.

### 3. Seasonal confounding

Motor claims peak in winter. Property claims follow weather events. If a randomly-assembled test fold happens to contain a disproportionate share of December policies, it will have higher raw claim rates than a fold that over-represents July. This is not a signal about model generalisation - it is random seasonal noise in fold composition.

Prospective deployment does not have this randomness. Your model will price a future rating year, which has its own seasonal structure. The CV metric should reflect how well the model generalises to a contiguous future period, not to a randomly shuffled slice of the past.

Walk-forward validation gives you this. Each test fold is a contiguous future window. The seasonal mix in the test set is representative of a real prospective period, not an artefact of random sampling.

---

## What goes wrong in practice

Here is the concrete version. Suppose you have four years of UK motor data - 2020 through 2023 - and you run five-fold CV with a CatBoost Poisson frequency model.

K-fold will assign, say, 40,000 policies to each fold at random. Some folds will be January-heavy. Some will be December-heavy. The training window for each fold will contain policies from all four years mixed with test policies from all four years. Some training claims will have 48 months of development. Some test claims will have 6 months. The model scores reasonably well because the random mixing means it has seen development patterns similar to whatever appears in the test fold.

Now deploy that model. Prospectively, every policy you price has inception dates in 2026. Your claims development data for the rating year 2026 will have at most 12 months of development when you next evaluate performance. There is no random mixing with 2020 data. The evaluation is clean and temporal. And the model, trained on a leaky CV metric, turns out to have been tuned partly on phantom signal.

The gap between CV loss and prospective monitored loss is the cost of this leakage. We have seen it run to 5-8 percentage points of Poisson deviance on real motor portfolios - not catastrophic, but enough to make hyperparameter selection meaningfully wrong and model comparison unreliable.

---

## How insurance-cv fixes it

```bash
uv add insurance-cv
```

The library provides walk-forward splits that enforce a strict temporal boundary between training and test data, with a configurable IBNR buffer.

```python
import polars as pl
from insurance_cv import walk_forward_split
from insurance_cv.diagnostics import temporal_leakage_check, split_summary

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=18,   # at least 1.5 years to cover a full seasonal cycle
    test_months=6,
    step_months=6,
    ibnr_buffer_months=3,  # standard for motor own damage
)
```

The training window expands with each fold - all historical data is always in training. The test window advances by `step_months` each time. The IBNR buffer excludes policies with inception dates in the three months before each test window from both training and test sets.

Before fitting anything, run the diagnostic:

```python
check = temporal_leakage_check(splits, df, date_col="inception_date")
if check["errors"]:
    raise RuntimeError("\n".join(check["errors"]))
```

Then inspect the fold structure:

```python
print(split_summary(splits, df, date_col="inception_date"))
```

```
fold  train_n  test_n  train_end   test_start  gap_days  ibnr_buffer_months
   1     2841     957  2019-12-31  2020-04-01        91                   3
   2     4189    1002  2020-06-30  2020-10-01        93                   3
   3     5612     988  2020-12-31  2021-04-01        90                   3
   4     7044    1031  2021-06-30  2021-10-01        93                   3
   5     8501     975  2021-12-31  2022-04-01        91                   3
   6     9987    1008  2022-06-30  2022-10-01        93                   3
```

Train end is always before test start. The gap is always at least 91 days - the IBNR buffer enforced in calendar time. This is what an honest evaluation looks like.

---

## sklearn compatibility

`InsuranceCV` wraps the splits as a sklearn-compatible CV splitter. Pass it directly to `cross_val_score` or `GridSearchCV` without any changes to your existing model code:

```python
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import cross_val_score
from insurance_cv.splits import InsuranceCV

features = ["vehicle_age", "driver_age", "ncd_years", "area_code"]
cat_features = ["area_code"]

# insurance-cv works with numpy arrays; convert from Polars first
X = df.select(features).to_numpy()
y = df["claim_count"].to_numpy()

model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=200,
    learning_rate=0.05,
    depth=6,
    verbose=0,
)

cv = InsuranceCV(splits, df)
scores = cross_val_score(
    model, X, y,
    cv=cv,
    scoring="neg_mean_poisson_deviance",
)

print(f"Mean Poisson deviance: {-scores.mean():.4f} (+/- {scores.std():.4f})")
```

The `InsuranceCV` object implements `split()` and `get_n_splits()` per the sklearn `BaseCrossValidator` interface. Any sklearn utility that accepts a CV splitter will work.

---

## Line-specific split types

`walk_forward_split` is the right default for most short-to-medium tail lines. The library also provides two specialised splitters.

**Policy year splits** align fold boundaries to 1 January - 31 December policy year boundaries. Use these when your rating structure changes annually - a post-rate-change test period trained on pre-rate-change data without a clean year boundary will mix the old and new rate structures in ways that flatter apparent lift.

```python
from insurance_cv import policy_year_split

splits = policy_year_split(
    df,
    date_col="inception_date",
    n_years_train=3,
    n_years_test=1,
    step_years=1,
)
# Produces: PY2018-2020 -> PY2021, PY2019-2021 -> PY2022, PY2020-2022 -> PY2023
```

**Accident year splits** are for long-tail lines - liability, professional indemnity - where accident year development matters. The splitter filters out accident years with insufficient median development (configurable; default 12 months) so that immature years never appear as test targets.

```python
from insurance_cv import accident_year_split

splits = accident_year_split(
    df,
    date_col="accident_date",
    development_col="development_months",
    min_development_months=12,
)
```

For employers' liability or PI, where IBNR development can run for a decade, you would typically set `min_development_months=36` or higher. The splitter will silently exclude accident years that do not meet this threshold from test folds. Check the output of `split_summary` to confirm which years were excluded.

---

## Choosing the IBNR buffer

The buffer is the most consequential parameter, and the default of three months is only right for the shortest-tail motor lines. Here is a working guide:

| Line | Typical buffer |
|---|---|
| Motor own damage | 3-6 months |
| Motor third party property | 6-12 months |
| Motor third party bodily injury | 12-24 months |
| Home buildings | 6-12 months |
| Employers' liability | 24-36 months |
| Professional indemnity | 24-48 months |

These are starting points, not rules. The right value depends on your claims handling speed, the proportion of large and complex claims, and whether your target is paid losses, incurred, or an ultimate estimate. If you are modelling ultimates from a reserving triangle, your IBNR problem is partly handled by the triangulation process - but the buffer still applies to the period between your latest valuation date and the test period.

When in doubt, set the buffer longer. A longer buffer shrinks your usable test window, which costs you folds and statistical power. That is a real cost. But it is smaller than the cost of evaluating your model on partially-developed claims and believing the result.

---

## What this won't fix

Walk-forward CV with an IBNR buffer removes temporal leakage. It does not fix other problems common in insurance model evaluation.

**Exposure changes.** If the portfolio grew materially during your training period - new distribution channels, changed appetite - the exposure mix in later test folds is different from earlier ones. Walk-forward handles this correctly (the model trained on the growing portfolio tests on the subsequent period), but it will not alert you to the fact that your most recent folds are more representative than your earlier ones.

**Rate changes.** If you changed rates significantly between training and test periods, your claim frequency in the test period partly reflects those rate changes (because they shifted the mix of risks you wrote). The CV metric conflates model quality with the appropriateness of rate changes. Keep a log of rate change dates and use `policy_year_split` at year boundaries where major changes were made.

**Target definition.** Walk-forward CV with a sound IBNR buffer gives you an honest evaluation of whatever target you defined. If the target is wrong - incurred losses without an ultimate development factor, for example - the evaluation will be honest about the wrong thing.

---

## Getting started

```bash
uv add insurance-cv
```

Source and issue tracker at [github.com/burningcost/insurance-cv](https://github.com/burningcost/insurance-cv).

The API is stable. The three split generators cover the main use cases in UK personal and commercial lines. What is not yet implemented: rolling-window splits (where old data drops out of training as the window advances, rather than expanding), which some teams prefer for very long portfolios where 2015 data is genuinely stale.

The core argument is simple: a model that performs well in k-fold CV on insurance data has proved it can interpolate within a temporally-shuffled dataset. It has not proved it can generalise prospectively. Those are different tests, and the second one is the one that matters.

Run `split_summary` before you tune anything. If the `gap_days` column contains zeros, you have a problem.
