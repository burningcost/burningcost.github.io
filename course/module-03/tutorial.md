# Module 3: GBMs for Insurance Pricing

**Modern Insurance Pricing with Python and Databricks**

---

## What GBMs capture that GLMs cannot

The honest version of this conversation starts with what GLMs are good at, because the case for GBMs is only coherent if you know what you are adding.

A GLM with a log link and multiplicative rating factors is a strong baseline. It is interpretable by construction: each factor level produces a relativity, the premium is the product of those relativities, and any pricing actuary can reconstruct the calculation from a factor table. It generalises well under the assumption that the true rate structure is multiplicative. On clean insurance data with the main rating factors present, a well-specified GLM captures 60-75% of the predictable variation in claims.

GBMs capture what the GLM misses. Specifically:

**Interactions.** A GLM with a log link assumes that the effect of vehicle group and driver age combine multiplicatively and independently. The truth in motor pricing is that young drivers in high-performance vehicles are worse than the multiplicative combination suggests - the interaction is superadditive. GLMs can model interactions, but you have to specify them explicitly. A GBM discovers them without being told.

**Non-linearities.** NCD is an ordered categorical variable. Its effect on frequency is roughly linear at low NCD years and then flattens out. A GLM treats NCD as either a set of dummies (no assumed shape) or a continuous variable (assumed linear). A GBM fits the actual curve without specification. The difference is largest at the tails.

**Three-way and higher-order effects.** These are almost impossible to specify correctly in a GLM. A GBM finds them in the data. Whether they are real effects or noise is a validation question, not a modelling question - which is why cross-validation done correctly matters so much.

The reason GBMs have not replaced GLMs in production UK pricing is not performance. The Actuary magazine ran a piece on this in November 2023 ('Price wars: the rise of gradient boosting machines') and the BofE/FCA joint survey in 2024 confirmed GBMs are in production at the major UK motor and home writers. The reason they coexist with GLMs rather than replacing them is:

- **Rating engine compatibility.** Radar and Emblem work with factor tables. A GBM does not produce a factor table directly. The SHAP relativity extraction in Module 4 solves this, but it is a non-trivial step.
- **Regulatory expectations.** The FCA's AI and machine learning supervisory guidance (FS 23/3, 2023) and the broader Consumer Duty requirements expect firms to be able to explain how prices are set. A GBM that cannot be reduced to a reviewable rating structure creates governance friction.
- **Monitoring and drift.** A GLM relativity table drifts in ways that are easy to detect with standard A/E monitoring. A GBM's internal representation drifts in ways that require dedicated monitoring pipelines.

The practical conclusion is this: GBMs are signal detectors. They find what is in the data. GLMs are the production format, because the industry's tooling is built around multiplicative factor tables. The workflow at sophisticated UK shops is: GBM for development, SHAP relativities (Module 4) to extract the signals, GLM fitted on those signals for production. We are building the first part of that pipeline.

---

## Why CatBoost specifically

Three reasons, all practical.

**Native categorical handling.** CatBoost accepts categorical features directly via the `cat_features` parameter. You pass column names; the library handles the encoding internally using ordered target statistics. You do not need to maintain a LabelEncoder or OneHotEncoder alongside the model. For insurance data with features like vehicle group (20-50 levels), area (5-10 levels), and NCD (6 levels), this removes an entire class of preprocessing bug.

**Ordered boosting.** Standard gradient boosting uses all training observations to compute the target statistics for each feature level, including the observation being fitted. This causes a subtle overfitting problem on small-to-medium datasets: the model fits some of the noise in the training set as though it were signal, because it has seen the training observations' targets when computing the statistics used to fit them. CatBoost's ordered boosting algorithm processes observations in a random permutation and uses only previous observations in that permutation for statistics computation, which eliminates this source of overfitting. On a 200,000-policy book this makes a measurable difference in hold-out deviance. On a 2-million-policy book the difference is smaller.

**SHAP integration.** CatBoost has built-in SHAP computation that is faster than the generic `shap` library for tree models, and it returns exact SHAP values (not approximate). This matters in Module 4, where we are extracting relativities from SHAP values. Symmetric trees - CatBoost's default tree structure - also make the SHAP computation analytically tractable rather than requiring the TreeExplainer approximation.

---

## Setup

Install the required packages. In Databricks notebooks, use `%sh uv pip install` followed by a Python kernel restart. The full list:

```python
# In a Databricks notebook, run in a shell cell:
# %sh uv pip install catboost polars insurance-cv optuna mlflow
# dbutils.library.restartPython()

import polars as pl
import numpy as np
import catboost
from catboost import CatBoostRegressor, Pool
import insurance_cv
from insurance_cv import WalkForwardCV
import optuna
import mlflow
import mlflow.catboost
from mlflow import MlflowClient
import matplotlib.pyplot as plt
```

We load the same synthetic motor dataset from Module 2. If you have already completed Module 2, your Delta table `pricing.motor.policies` has 100,000 policies. If not, run the data generation cell from Module 2's notebook first.

```python
CATALOG = "pricing"
SCHEMA  = "motor"

# Load from Delta via Spark, convert to Polars
spark_df = spark.table(f"{CATALOG}.{SCHEMA}.policies")
df = pl.from_pandas(spark_df.toPandas())

print(df.shape)
print(df.dtypes)
```

---

## Feature engineering

The feature set is the same as Module 2. We add one extra step: explicitly declare which columns are categorical so CatBoost can handle them natively.

```python
# Continuous features
CONTINUOUS_FEATURES = [
    "driver_age",
    "vehicle_group",   # ABI group 1-50, treat as continuous
    "ncd_years",
]

# Categorical features - passed to CatBoost as cat_features
CAT_FEATURES = [
    "area",
    "conviction_flag",   # binary, but CatBoost handles it as categorical
]

FEATURES = CONTINUOUS_FEATURES + CAT_FEATURES

# Target variables
FREQ_TARGET   = "claim_count"
SEV_TARGET    = "avg_severity"   # we will derive this below
EXPOSURE_COL  = "exposure"

# Derive average severity for the severity model
# We only use policies with at least one claim
df = df.with_columns([
    (pl.col("claim_amount") / pl.col("claim_count"))
    .alias("avg_severity")
])
```

One note on `conviction_flag`: it is 0/1 binary. You can pass it to CatBoost as either a continuous or categorical feature. As categorical, CatBoost fits a separate statistic for level "0" and level "1". As continuous, it assumes a linear effect. For a binary variable with no order, categorical is the right choice even though the values are numeric.

---

## Temporal cross-validation with `insurance-cv`

This is the section that pricing actuaries get wrong most often, so we spend time on it.

### Why random splits are wrong for insurance data

Random 80/20 splits mix policy years. A model trained on 80% of a 2020-2023 dataset validated on the remaining 20% of 2020-2023 data looks better than it performs on 2024 data. The reason:

1. **Leakage from IBNR.** Claims that have not yet fully developed as of the validation date are present in the training set. The model learns to predict reported claims on a portfolio where development is still in progress.
2. **Temporal autocorrelation.** Adjacent policy years share external factors (inflation, weather, supply chain effects) that make a model trained on 2021 data look good on randomly-held-out 2021 data, but not on 2024 data where the inflation environment has changed.
3. **Rating cycle effects.** A book that was underpriced in 2021 and re-rated in 2022 contains a pricing inflection that a random split treats as random noise. A temporal split reveals it.

The correct approach is walk-forward cross-validation: train on years 2020-2021, validate on 2022; train on 2020-2022, validate on 2023; train on 2020-2023, validate on 2024. Each fold uses earlier data to predict later data, replicating the actual deployment scenario.

### `insurance-cv` setup

`insurance-cv` implements walk-forward splits for insurance data with two insurance-specific additions: policy year boundaries, and an IBNR development buffer.

```python
from insurance_cv import WalkForwardCV

# The dataset needs a policy year column
df = df.with_columns([
    pl.col("inception_date").dt.year().alias("policy_year")
])

# Walk-forward CV: train on each year, validate on the next
# ibnr_buffer_years: exclude the most recent N years from training
# because their claims are not yet fully developed
cv = WalkForwardCV(
    year_col="policy_year",
    min_train_years=2,       # need at least 2 years of history to train
    ibnr_buffer_years=1,     # exclude the most recent year from training
    n_splits=3,              # produce 3 train/validation fold pairs
)

# cv.split() returns (train_idx, val_idx) pairs
folds = list(cv.split(df))

print(f"Number of folds: {len(folds)}")
for i, (train_idx, val_idx) in enumerate(folds):
    train_years = df[train_idx]["policy_year"].unique().sort().to_list()
    val_years   = df[val_idx]["policy_year"].unique().sort().to_list()
    print(f"Fold {i+1}: train years {train_years}, val years {val_years}")
```

The `ibnr_buffer_years=1` setting excludes the most recent training year from the training set. If we are validating on 2023, we train on data through 2021 only, because 2022 claims will not be fully reported. For a long-tail class of business (employers' liability, motor TP injury), you would increase this to 2-3 years. For motor AXD, 1 year is usually sufficient.

One limitation worth knowing: `insurance-cv` works on policy inception year, not accident year. If your claims data is structured by accident year and you have multi-year policies, you need to align the exposure cut to the policy inception structure before using it. The library's documentation covers this case.

### Running the frequency model with walk-forward CV

```python
from catboost import CatBoostRegressor, Pool
import numpy as np

def poisson_deviance(y_true, y_pred, exposure):
    """Scaled Poisson deviance per unit exposure."""
    freq_true = y_true / exposure
    freq_pred = y_pred / exposure
    # When freq_true = 0, the log term is zero by convention (0 * log(0) = 0).
    # Handle explicitly rather than fudging the log argument.
    freq_pred = np.clip(freq_pred, 1e-10, None)
    deviance = 2 * exposure * (
        np.where(freq_true > 0, freq_true * np.log(freq_true / freq_pred), 0.0)
        - (freq_true - freq_pred)
    )
    return deviance.sum() / exposure.sum()

cv_deviances = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    df_train = df[train_idx]
    df_val   = df[val_idx]

    X_train = df_train[FEATURES].to_pandas()
    y_train = df_train[FREQ_TARGET].to_numpy()
    w_train = df_train[EXPOSURE_COL].to_numpy()

    X_val   = df_val[FEATURES].to_pandas()
    y_val   = df_val[FREQ_TARGET].to_numpy()
    w_val   = df_val[EXPOSURE_COL].to_numpy()

    # CatBoost Pool: exposure enters as baseline (log-offset), not sample_weight
    train_pool = Pool(
        data=X_train,
        label=y_train,
        baseline=np.log(w_train),    # log(exposure) as offset
        cat_features=CAT_FEATURES,
    )
    val_pool = Pool(
        data=X_val,
        label=y_val,
        baseline=np.log(w_val),
        cat_features=CAT_FEATURES,
    )

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Poisson",
        eval_metric="Poisson",
        random_seed=42,
        verbose=0,
    )
    model.fit(train_pool, eval_set=val_pool)

    # Predictions are on the count scale (model outputs lambda * exposure)
    y_pred = model.predict(val_pool)
    fold_deviance = poisson_deviance(y_val, y_pred, w_val)
    cv_deviances.append(fold_deviance)
    print(f"Fold {fold_idx+1}: Poisson deviance = {fold_deviance:.4f}")

print(f"\nMean CV Poisson deviance: {np.mean(cv_deviances):.4f} (+/- {np.std(cv_deviances):.4f})")
```

One common mistake: the GLM from Module 2 uses `statsmodels`' offset parameter, which takes `log(exposure)` directly. CatBoost's `baseline` parameter also takes `log(exposure)`. Both are correct. What is wrong is passing `exposure` (not log-transformed) as the baseline - CatBoost interprets the baseline as already log-transformed and adds it to the linear predictor, so passing untransformed exposure produces a wildly incorrect offset.

---

## Hyperparameter tuning with Optuna

We tune three parameters. Understanding what they do on insurance data matters more than the tuning mechanism itself.

**`depth`**: the maximum depth of each tree. For insurance data with 5-10 main rating factors, depth 4-6 is usually right. Deeper trees fit more complex interactions but overfit faster, especially on smaller datasets. CatBoost's symmetric trees are full binary trees of the given depth (every leaf is at depth D), which limits complexity relative to asymmetric boosting.

**`learning_rate`**: the shrinkage applied to each tree. Lower learning rates generalise better but require more iterations (more compute). For a dataset of 100,000 policies, 0.03-0.10 with 500-1000 iterations is a sensible search space.

**`l2_leaf_reg`**: L2 regularisation on the leaf values. Increase this if the model is overfitting (train deviance much lower than CV deviance). Values of 1-10 are typical; the default is 3.

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Use the last fold only for the tuning objective
# (tuning on all folds is more rigorous but slower)
train_idx_tune, val_idx_tune = folds[-1]

df_train_tune = df[train_idx_tune]
df_val_tune   = df[val_idx_tune]

X_train_t = df_train_tune[FEATURES].to_pandas()
y_train_t = df_train_tune[FREQ_TARGET].to_numpy()
w_train_t = df_train_tune[EXPOSURE_COL].to_numpy()

X_val_t   = df_val_tune[FEATURES].to_pandas()
y_val_t   = df_val_tune[FREQ_TARGET].to_numpy()
w_val_t   = df_val_tune[EXPOSURE_COL].to_numpy()

train_pool_t = Pool(X_train_t, y_train_t, baseline=np.log(w_train_t), cat_features=CAT_FEATURES)
val_pool_t   = Pool(X_val_t,   y_val_t,   baseline=np.log(w_val_t),   cat_features=CAT_FEATURES)

def objective(trial):
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 1000),
        "depth":         trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Poisson",
        "eval_metric":   "Poisson",
        "random_seed":   42,
        "verbose":       0,
    }
    model = CatBoostRegressor(**params)
    model.fit(train_pool_t, eval_set=val_pool_t)
    y_pred = model.predict(val_pool_t)
    return poisson_deviance(y_val_t, y_pred, w_val_t)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=True)

best_params = study.best_params
print("Best parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"Best CV deviance: {study.best_value:.4f}")
```

40 trials takes 10-15 minutes on a Standard_DS3_v2 cluster. If you are on Free Edition and time is a constraint, reduce to 20 trials. The best parameters from 20 trials are usually within 2-3% of the best from 40.

---

## Training the final models and logging to MLflow

With the best hyperparameters found, we train the final frequency and severity models on all available training data and log everything to MLflow.

### Frequency model

```python
mlflow.set_experiment("/Users/you@insurer.com/motor-gbm-module03")

# Full training set (all years except the held-out test year)
train_all_idx = df.filter(pl.col("policy_year") < df["policy_year"].max())
# Held-out test set: most recent year
test_idx      = df.filter(pl.col("policy_year") == df["policy_year"].max())

df_train_final = df.filter(pl.col("policy_year") < df["policy_year"].max())
df_test        = df.filter(pl.col("policy_year") == df["policy_year"].max())

X_train_f = df_train_final[FEATURES].to_pandas()
y_train_f = df_train_final[FREQ_TARGET].to_numpy()
w_train_f = df_train_final[EXPOSURE_COL].to_numpy()

X_test_f  = df_test[FEATURES].to_pandas()
y_test_f  = df_test[FREQ_TARGET].to_numpy()
w_test_f  = df_test[EXPOSURE_COL].to_numpy()

final_train_pool = Pool(X_train_f, y_train_f, baseline=np.log(w_train_f), cat_features=CAT_FEATURES)
final_test_pool  = Pool(X_test_f,  y_test_f,  baseline=np.log(w_test_f),  cat_features=CAT_FEATURES)

freq_params = {
    **best_params,
    "loss_function": "Poisson",
    "eval_metric":   "Poisson",
    "random_seed":   42,
    "verbose":       100,
}

with mlflow.start_run(run_name="freq_catboost_tuned") as run_freq:
    mlflow.log_params(freq_params)
    mlflow.log_param("model_type", "catboost_freq")
    mlflow.log_param("cv_strategy", "walk_forward_ibnr1")
    mlflow.log_param("n_cv_folds", len(folds))
    mlflow.log_param("feature_set", FEATURES)
    mlflow.log_param("cat_features", CAT_FEATURES)

    freq_model = CatBoostRegressor(**freq_params)
    freq_model.fit(final_train_pool, eval_set=final_test_pool)

    # Test set predictions
    y_pred_freq = freq_model.predict(final_test_pool)
    test_deviance_freq = poisson_deviance(y_test_f, y_pred_freq, w_test_f)

    mlflow.log_metric("test_poisson_deviance", test_deviance_freq)
    mlflow.log_metric("mean_cv_deviance", np.mean(cv_deviances))
    mlflow.log_metric("cv_deviance_std", np.std(cv_deviances))

    # Log the model
    mlflow.catboost.log_model(freq_model, "freq_model")
    freq_run_id = run_freq.info.run_id

print(f"Frequency model run ID: {freq_run_id}")
print(f"Test Poisson deviance: {test_deviance_freq:.4f}")
```

### Severity model

For the severity model, we restrict to policies with at least one claim and use a Gamma loss function. Exposure does not enter the severity model as an offset - severity is the average cost per claim, which is independent of exposure. We use no baseline parameter.

```python
df_train_sev = df_train_final.filter(pl.col("claim_count") > 0)
df_test_sev  = df_test.filter(pl.col("claim_count") > 0)

X_train_s = df_train_sev[FEATURES].to_pandas()
y_train_s = df_train_sev[SEV_TARGET].to_numpy()

X_test_s  = df_test_sev[FEATURES].to_pandas()
y_test_s  = df_test_sev[SEV_TARGET].to_numpy()

sev_train_pool = Pool(X_train_s, y_train_s, cat_features=CAT_FEATURES)
sev_test_pool  = Pool(X_test_s,  y_test_s,  cat_features=CAT_FEATURES)

sev_params = {
    **best_params,   # use same tuned depth/learning_rate/l2_leaf_reg
    "loss_function": "Tweedie:variance_power=2",   # Gamma equivalent
    "eval_metric":   "RMSE",
    "random_seed":   42,
    "verbose":       100,
}

with mlflow.start_run(run_name="sev_catboost_tuned") as run_sev:
    mlflow.log_params(sev_params)
    mlflow.log_param("model_type", "catboost_sev")
    mlflow.log_param("n_claims_train", len(df_train_sev))
    mlflow.log_param("n_claims_test",  len(df_test_sev))

    sev_model = CatBoostRegressor(**sev_params)
    sev_model.fit(sev_train_pool, eval_set=sev_test_pool)

    y_pred_sev = sev_model.predict(sev_test_pool)
    rmse_sev = np.sqrt(np.mean((y_test_s - y_pred_sev)**2))
    mae_sev  = np.mean(np.abs(y_test_s - y_pred_sev))

    mlflow.log_metric("test_rmse_severity", rmse_sev)
    mlflow.log_metric("test_mae_severity", mae_sev)
    mlflow.catboost.log_model(sev_model, "sev_model")
    sev_run_id = run_sev.info.run_id

print(f"Severity model run ID: {sev_run_id}")
```

A note on the Gamma objective in CatBoost: CatBoost does not have a dedicated `Gamma` loss function by that name. `Tweedie:variance_power=2` is mathematically equivalent to the Gamma log-link model (the Tweedie family at power=2 is the Gamma distribution). Power=1 is Poisson; power between 1 and 2 is the compound Poisson-Gamma used for aggregate losses. Power=2 is correct for the severity model.

---

## Model registry

Register the frequency model as the primary GBM artefact for this product and development cycle.

```python
client = MlflowClient()

# Register
freq_uri = f"runs:/{freq_run_id}/freq_model"
registered_freq = mlflow.register_model(
    model_uri=freq_uri,
    name="motor_freq_catboost_m03",
)

# Set alias instead of stage (stages deprecated in MLflow 2.9+)
client.set_registered_model_alias(
    name="motor_freq_catboost_m03",
    alias="challenger",     # "challenger" because it competes against the GLM
    version=registered_freq.version,
)

# Tag with context
client.set_model_version_tag(
    name="motor_freq_catboost_m03",
    version=registered_freq.version,
    key="module",
    value="module_03",
)
client.set_model_version_tag(
    name="motor_freq_catboost_m03",
    version=registered_freq.version,
    key="cv_strategy",
    value="walk_forward_ibnr1",
)

print(f"Registered: motor_freq_catboost_m03 version {registered_freq.version} as 'challenger'")
```

We use the "challenger" alias rather than "production" because the GBM has not yet been approved for production. In Module 4, when we have SHAP relativities that a pricing committee can review, the model that survives governance review gets promoted to "production".

Loading the challenger model later:

```python
model = mlflow.catboost.load_model("models:/motor_freq_catboost_m03@challenger")
```

---

## Comparing GBM to the GLM from Module 2

This is the section that determines what you do with the GBM.

### Loading the GLM predictions

The Module 2 GLM is logged in MLflow. Load its predictions on the test set:

```python
# Load the GLM from Module 2's registry entry
glm_model = mlflow.statsmodels.load_model("models:/motor_freq_glm@production")

# Generate GLM predictions on the test set
# The GLM model requires a pandas DataFrame with the same features and an exposure column
import pandas as pd
X_test_glm = df_test[FEATURES + [EXPOSURE_COL]].to_pandas()

# statsmodels GLM predictions: pass the test data and exposure offset
glm_pred = glm_model.predict(X_test_glm)  # returns predicted claim counts
```

If you have not completed Module 2 or do not have the GLM registered, you can approximate by fitting a simple Poisson GLM using statsmodels directly in this notebook. The comparison structure is the same either way.

### Gini coefficient

The Gini coefficient measures how well the model separates high-risk from low-risk policies. A Gini of 0 means the model has no discriminatory power; 1 means perfect discrimination. In practice, UK motor frequency models produce Ginis of 0.30-0.50; a well-specified GLM is typically at the lower end of that range.

```python
from sklearn.metrics import roc_auc_score

def gini(y_true_counts, y_pred_counts, exposure):
    """Gini coefficient for a Poisson frequency model."""
    # Convert to binary: did the policy have a claim?
    y_binary = (y_true_counts > 0).astype(int)
    # Predicted frequency as the score
    y_score  = y_pred_counts / exposure
    auc = roc_auc_score(y_binary, y_score)
    return 2 * auc - 1

gini_gbm = gini(y_test_f, y_pred_freq, w_test_f)
gini_glm = gini(y_test_f, glm_pred,    w_test_f)

print(f"Frequency Gini - GBM: {gini_gbm:.3f}, GLM: {gini_glm:.3f}")
print(f"Gini lift: {gini_gbm - gini_glm:.3f} ({(gini_gbm / gini_glm - 1)*100:.1f}%)")
```

A Gini lift of 0.03-0.07 (3-7 Gini points) is typical when adding a GBM to a GLM on the same feature set. The lift comes from captured interactions and non-linearities. If the lift is larger than 0.10, check whether you have a data leakage problem.

### Calibration

A model can discriminate well and still be wrong about absolute levels. Calibration checks whether the predicted rates match actual observed rates across the predicted score distribution.

```python
def calibration_plot(y_true, y_pred, exposure, n_bins=10, label=""):
    """Actual vs predicted frequency by decile of predicted frequency."""
    freq_pred = y_pred / exposure
    freq_true = y_true / exposure

    # Bin by predicted frequency decile
    bins = np.quantile(freq_pred, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.digitize(freq_pred, bins[1:-1])

    actuals, predicteds = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        actuals.append(freq_true[mask].mean())
        predicteds.append(freq_pred[mask].mean())

    plt.plot(predicteds, actuals, "o-", label=label)

fig, ax = plt.subplots(figsize=(8, 6))
calibration_plot(y_test_f, y_pred_freq, w_test_f, label="GBM")
calibration_plot(y_test_f, glm_pred,    w_test_f, label="GLM")
ax.plot([0, ax.get_xlim()[1]], [0, ax.get_ylim()[1]], "k--", label="Perfect")
ax.set_xlabel("Mean predicted frequency")
ax.set_ylabel("Mean actual frequency")
ax.set_title("Calibration: GBM vs GLM")
ax.legend()
plt.tight_layout()
mlflow.log_figure(fig, "calibration_comparison.png")
plt.show()
```

A well-calibrated model has points on or near the diagonal. Points above the diagonal mean the model under-predicts in that decile (worse risks than the model thinks). Points below mean over-prediction. For both GLM and GBM, the overall level should be well-calibrated; the GBM should show better calibration at the tails where GLM interactions are missed.

### Double lift chart

The double lift chart is the standard insurance diagnostic for comparing two predictive models. It answers: of the policies that the GBM predicts as high-risk that the GLM does not, are they actually high-risk?

```python
def double_lift(y_true, pred_a, pred_b, exposure, n_bins=10, label_a="GBM", label_b="GLM"):
    """Double lift: actual frequency by decile of ratio pred_a/pred_b."""
    ratio = (pred_a / exposure) / (pred_b / exposure)
    freq_true = y_true / exposure

    bins = np.quantile(ratio, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.digitize(ratio, bins[1:-1])

    ratios_mid, actuals = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ratios_mid.append(ratio[mask].mean())
        actuals.append(freq_true[mask].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios_mid, actuals, "o-")
    ax.axhline(freq_true.mean(), linestyle="--", color="grey", label="Portfolio mean")
    ax.set_xlabel(f"Ratio: {label_a} predicted / {label_b} predicted")
    ax.set_ylabel("Actual observed frequency")
    ax.set_title(f"Double Lift: {label_a} vs {label_b}")
    ax.legend()
    plt.tight_layout()
    return fig

fig_dl = double_lift(y_test_f, y_pred_freq, glm_pred, w_test_f)
mlflow.log_figure(fig_dl, "double_lift_gbm_vs_glm.png")
plt.show()
```

What you are looking for: a positively sloping double lift chart. The left side of the chart (policies where GBM predicts lower risk than GLM) should have actual frequencies below the portfolio mean; the right side (policies where GBM predicts higher risk than GLM) should have actual frequencies above the mean. A flat chart means the GBM is not finding anything the GLM cannot. An upward-sloping chart confirms the GBM is identifying genuine additional risk signals.

---

## When to use GBM vs GLM vs both

The decision is not binary, and it is not purely a performance decision.

**Use the GLM when:**
- The model output needs to feed directly into Radar or Emblem as a factor table, and you cannot or will not run the SHAP relativity extraction in Module 4
- Regulatory pressure requires a model that a non-technical pricing committee can sign off by reading a factor table
- Your dataset has fewer than 30,000 policies and ordered boosting does not help enough to justify the interpretability cost
- The rating factors are clean, categorical, and have known multiplicative structure - the GLM is already capturing the signal

**Use the GBM when:**
- You are in model development mode and want to know what is in the data
- You suspect interactions that the GLM is not capturing, and the double lift chart from a previous GBM run confirmed this
- You have a feature set that includes continuous variables with non-linear effects (telematics score, vehicle age, sums insured) where the GLM's binning is a crude approximation
- You are building a hybrid: GBM for signal detection, SHAP relativities fed back into a GLM for production

**Use both in tandem when:**
- You are at a firm that has adopted the GBM-to-GLM pipeline: GBM discovers the interactions, SHAP extracts the relativities, GLM is fitted with those interaction features for production. This is the current state of practice at tier-1 UK motor writers.
- You are running champion-challenger: the GLM is in production, the GBM is the challenger, and you are accumulating evidence on a holdout before deciding whether to promote

**The honest answer on lift:** A Gini lift of 3-5 points on a synthetic dataset like ours is real but may not be worth the operational overhead if the firm's infrastructure is built around Radar factor tables. A Gini lift of 3-5 points on a real book where the GLM has been in production for three years without a structural review is worth acting on.

---

## Feature importance

Before finishing, extract the feature importances from the frequency model. These are useful for understanding what the model is doing, even though SHAP values (Module 4) give a more complete picture.

```python
importances = freq_model.get_feature_importance(type="FeatureImportance")
feature_names = FEATURES

imp_df = (
    pl.DataFrame({
        "feature": feature_names,
        "importance": importances.tolist(),
    })
    .sort("importance", descending=True)
)

print(imp_df)

# Log as MLflow artifact
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(imp_df["feature"].to_list(), imp_df["importance"].to_list())
ax.set_xlabel("Feature importance (PredictionValuesChange)")
ax.set_title("CatBoost frequency model: feature importances")
plt.tight_layout()

with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_figure(fig, "feature_importance.png")

plt.show()
```

Feature importance in CatBoost's default mode is PredictionValuesChange: the average change in model output when a feature is used in a split, normalised to sum to 100. It tells you which features the model relies on, but not how they influence predictions for individual policies. That is what Module 4 covers. Mean absolute SHAP importance requires type="ShapValues" and can give different feature rankings for correlated inputs.

---

## Summary

What we built:

- A CatBoost Poisson frequency model and Tweedie severity model on 100,000 synthetic motor policies
- Temporal cross-validation using `insurance-cv` with a 1-year IBNR buffer, avoiding the optimistic metrics that come from random splits
- Hyperparameter tuning with Optuna across 40 trials
- Full MLflow tracking: parameters, metrics, calibration plots, double lift charts, feature importances
- Model registration in the Databricks model registry with a "challenger" alias

What we found:

- The GBM shows a genuine Gini lift over the GLM, driven by captured interactions (particularly young driver / high vehicle group combinations that the multiplicative GLM under-prices)
- Calibration is good for both models at the portfolio level; the GBM shows better calibration at the tails
- The double lift chart confirms the GBM is finding risk signals the GLM misses

What comes next:

- Module 4 extracts SHAP relativities from the frequency model - converting the GBM's internal representation into a factor table that a pricing committee can review and a rating engine can import
- That factor table is what takes the GBM from "challenger in development" to "production model"
