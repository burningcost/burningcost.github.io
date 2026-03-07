# Databricks notebook source
# MAGIC %md
# MAGIC # Module 3: GBMs for Insurance Pricing
# MAGIC ## Modern Insurance Pricing with Python and Databricks
# MAGIC
# MAGIC This notebook covers the full Module 3 workflow:
# MAGIC
# MAGIC 1. Install packages and load the motor policy data from Delta
# MAGIC 2. Feature engineering
# MAGIC 3. Temporal cross-validation with `insurance-cv`
# MAGIC 4. Hyperparameter tuning with Optuna
# MAGIC 5. Train final frequency (Poisson) and severity (Gamma/Tweedie) models
# MAGIC 6. MLflow experiment tracking: params, metrics, artefacts
# MAGIC 7. Model registry registration
# MAGIC 8. Comparison with the GLM from Module 2: Gini, calibration, double lift
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Databricks Runtime 14.3 LTS or later
# MAGIC - Unity Catalog enabled (Free Edition includes this)
# MAGIC - Module 2 completed: the `pricing.motor.policies` Delta table must exist
# MAGIC   with 100,000 synthetic motor policies and an `inception_date` column
# MAGIC
# MAGIC **Free Edition note:** All cells in this notebook run on Databricks Free Edition.
# MAGIC Hyperparameter tuning with 40 trials takes approximately 20-30 minutes on a
# MAGIC Standard_DS3_v2 single-node cluster. Reduce `N_TRIALS` to 15 if time is short.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install packages
# MAGIC
# MAGIC CatBoost, Polars, insurance-cv, and Optuna are not pre-installed on the
# MAGIC standard Databricks Runtime. Install them here, then restart the Python kernel
# MAGIC so subsequent cells can import them.
# MAGIC
# MAGIC `insurance-cv` provides walk-forward cross-validation splits that respect
# MAGIC policy year boundaries and IBNR development buffers. Install it via `uv add insurance-cv`
# MAGIC in a local environment; in Databricks, `uv pip install` is the right mechanism.

# COMMAND ----------

# MAGIC %sh uv pip install catboost polars insurance-cv optuna scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC Set your catalog and schema names here. These should match whatever you
# MAGIC set in Module 1 and Module 2. Change MLFLOW_EXPERIMENT_PATH to your
# MAGIC Databricks user email.

# COMMAND ----------

CATALOG  = "pricing"
SCHEMA   = "motor"

# Change this to your Databricks user path
MLFLOW_EXPERIMENT_PATH = "/Users/you@insurer.com/motor-gbm-module03"

# Reduce this if you are on Free Edition and want faster tuning
N_TRIALS = 40

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports

# COMMAND ----------

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
from insurance_cv import WalkForwardCV
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import mlflow
import mlflow.catboost
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

print(f"CatBoost version:    {__import__('catboost').__version__}")
print(f"Polars version:      {pl.__version__}")
print(f"MLflow version:      {mlflow.__version__}")
print(f"insurance-cv version: {__import__('insurance_cv').__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load data
# MAGIC
# MAGIC We load the 100,000-policy synthetic motor dataset from the Delta table
# MAGIC written in Module 2. The table contains:
# MAGIC - `policy_id`, `inception_date`, `exposure`
# MAGIC - Rating factors: `area`, `vehicle_group`, `ncd_years`, `driver_age`, `conviction_flag`
# MAGIC - Targets: `claim_count`, `claim_amount`

# COMMAND ----------

spark_df = spark.table(f"{CATALOG}.{SCHEMA}.policies")
df = pl.from_pandas(spark_df.toPandas())

print(f"Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")
print(df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature engineering
# MAGIC
# MAGIC We add two derived columns:
# MAGIC - `policy_year`: extracted from inception_date for temporal cross-validation
# MAGIC - `avg_severity`: claim amount divided by claim count (for the severity model)
# MAGIC
# MAGIC We also declare which features are categorical. CatBoost accepts categorical
# MAGIC features directly without any manual encoding.

# COMMAND ----------

df = df.with_columns([
    pl.col("inception_date").dt.year().alias("policy_year"),
    pl.when(pl.col("claim_count") > 0)
      .then(pl.col("claim_amount") / pl.col("claim_count"))
      .otherwise(None)
      .alias("avg_severity"),
])

# Features for both frequency and severity models
CONTINUOUS_FEATURES = ["driver_age", "vehicle_group", "ncd_years"]
CAT_FEATURES        = ["area", "conviction_flag"]
FEATURES            = CONTINUOUS_FEATURES + CAT_FEATURES

# Targets and exposure
FREQ_TARGET  = "claim_count"
SEV_TARGET   = "avg_severity"
EXPOSURE_COL = "exposure"

print("Policy year distribution:")
print(df.group_by("policy_year").len().sort("policy_year"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Temporal cross-validation with insurance-cv
# MAGIC
# MAGIC We use WalkForwardCV from insurance-cv to create splits that respect policy
# MAGIC year ordering. This is more appropriate than random splits for insurance data
# MAGIC because:
# MAGIC
# MAGIC 1. Random splits mix policy years, producing optimistic metrics that do not
# MAGIC    reflect real out-of-time performance
# MAGIC 2. The ibnr_buffer_years parameter excludes the most recent training year,
# MAGIC    because its claims are not yet fully reported
# MAGIC
# MAGIC For motor AXD, 1 year is a sufficient IBNR buffer.
# MAGIC For long-tail classes, increase to 2-3 years.

# COMMAND ----------

cv = WalkForwardCV(
    year_col="policy_year",
    min_train_years=2,
    ibnr_buffer_years=1,
    n_splits=3,
)

folds = list(cv.split(df))

print(f"Number of CV folds: {len(folds)}")
for i, (train_idx, val_idx) in enumerate(folds):
    train_years = df[train_idx]["policy_year"].unique().sort().to_list()
    val_years   = df[val_idx]["policy_year"].unique().sort().to_list()
    n_train     = len(train_idx)
    n_val       = len(val_idx)
    print(f"  Fold {i+1}: train={train_years} (n={n_train:,}), val={val_years} (n={n_val:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Helper functions
# MAGIC
# MAGIC Poisson deviance and Gini coefficient used throughout.

# COMMAND ----------

def poisson_deviance(y_true, y_pred, exposure):
    """Scaled Poisson deviance per unit exposure."""
    freq_true = y_true / exposure
    freq_pred = y_pred / exposure
    freq_pred = np.clip(freq_pred, 1e-10, None)
    freq_true_safe = np.where(freq_true > 0, freq_true, 1e-10)
    deviance = 2 * exposure * (
        np.where(freq_true > 0, freq_true * np.log(freq_true_safe / freq_pred), 0)
        - (freq_true - freq_pred)
    )
    return deviance.sum() / exposure.sum()

def gini(y_true_counts, y_pred_counts, exposure):
    """Gini coefficient for a Poisson frequency model."""
    y_binary = (y_true_counts > 0).astype(int)
    y_score  = y_pred_counts / exposure
    auc = roc_auc_score(y_binary, y_score)
    return 2 * auc - 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cross-validation baseline
# MAGIC
# MAGIC Run the frequency model across all folds with default parameters to establish
# MAGIC a CV baseline before tuning.

# COMMAND ----------

cv_deviances = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    df_train_cv = df[train_idx]
    df_val_cv   = df[val_idx]

    X_train = df_train_cv[FEATURES].to_pandas()
    y_train = df_train_cv[FREQ_TARGET].to_numpy()
    w_train = df_train_cv[EXPOSURE_COL].to_numpy()

    X_val   = df_val_cv[FEATURES].to_pandas()
    y_val   = df_val_cv[FREQ_TARGET].to_numpy()
    w_val   = df_val_cv[EXPOSURE_COL].to_numpy()

    # Exposure as log-offset (baseline), not sample weight
    train_pool = Pool(X_train, y_train, baseline=np.log(w_train), cat_features=CAT_FEATURES)
    val_pool   = Pool(X_val,   y_val,   baseline=np.log(w_val),   cat_features=CAT_FEATURES)

    baseline_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Poisson",
        eval_metric="Poisson",
        random_seed=42,
        verbose=0,
    )
    baseline_model.fit(train_pool, eval_set=val_pool)

    y_pred_cv = baseline_model.predict(val_pool)
    fold_dev  = poisson_deviance(y_val, y_pred_cv, w_val)
    cv_deviances.append(fold_dev)
    print(f"Fold {fold_idx+1}: Poisson deviance = {fold_dev:.4f}")

baseline_cv_deviance = np.mean(cv_deviances)
print(f"\nBaseline mean CV Poisson deviance: {baseline_cv_deviance:.4f} (+/- {np.std(cv_deviances):.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Hyperparameter tuning with Optuna
# MAGIC
# MAGIC We tune three parameters that matter most for insurance data:
# MAGIC
# MAGIC - `depth` (4-7): symmetric tree depth. For 5-10 rating factors, depth 4-6 is
# MAGIC   typically right. Deeper trees fit more interactions but overfit faster on
# MAGIC   smaller datasets.
# MAGIC - `learning_rate` (0.02-0.15): shrinkage per tree. Lower values generalise better
# MAGIC   but require more iterations.
# MAGIC - `l2_leaf_reg` (1-10): L2 regularisation on leaf values. Increase if train
# MAGIC   deviance is substantially better than CV deviance.
# MAGIC
# MAGIC We tune on the last fold only (most recent history, most realistic validation set).

# COMMAND ----------

# Use last fold for tuning
train_idx_t, val_idx_t = folds[-1]

df_train_t = df[train_idx_t]
df_val_t   = df[val_idx_t]

X_tt = df_train_t[FEATURES].to_pandas()
y_tt = df_train_t[FREQ_TARGET].to_numpy()
w_tt = df_train_t[EXPOSURE_COL].to_numpy()

X_vt = df_val_t[FEATURES].to_pandas()
y_vt = df_val_t[FREQ_TARGET].to_numpy()
w_vt = df_val_t[EXPOSURE_COL].to_numpy()

tune_train_pool = Pool(X_tt, y_tt, baseline=np.log(w_tt), cat_features=CAT_FEATURES)
tune_val_pool   = Pool(X_vt, y_vt, baseline=np.log(w_vt), cat_features=CAT_FEATURES)

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
    m = CatBoostRegressor(**params)
    m.fit(tune_train_pool, eval_set=tune_val_pool)
    y_pred = m.predict(tune_val_pool)
    return poisson_deviance(y_vt, y_pred, w_vt)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
print("\nBest parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"\nBest tune deviance:    {study.best_value:.4f}")
print(f"Baseline CV deviance:  {baseline_cv_deviance:.4f}")
print(f"Improvement:           {(baseline_cv_deviance - study.best_value):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Train final models on full training data
# MAGIC
# MAGIC The most recent policy year is held out as the test set (unseen data).
# MAGIC Everything before it is training data.
# MAGIC
# MAGIC We train two models:
# MAGIC - Frequency: CatBoostRegressor with Poisson loss, exposure as log-offset
# MAGIC - Severity: CatBoostRegressor with Tweedie(power=2) loss, restricted to claims > 0

# COMMAND ----------

max_year     = df["policy_year"].max()
df_train_all = df.filter(pl.col("policy_year") < max_year)
df_test      = df.filter(pl.col("policy_year") == max_year)

print(f"Training set: {len(df_train_all):,} policies (years < {max_year})")
print(f"Test set:     {len(df_test):,} policies (year = {max_year})")

# Frequency pools
X_train_f = df_train_all[FEATURES].to_pandas()
y_train_f = df_train_all[FREQ_TARGET].to_numpy()
w_train_f = df_train_all[EXPOSURE_COL].to_numpy()

X_test_f  = df_test[FEATURES].to_pandas()
y_test_f  = df_test[FREQ_TARGET].to_numpy()
w_test_f  = df_test[EXPOSURE_COL].to_numpy()

freq_train_pool = Pool(X_train_f, y_train_f, baseline=np.log(w_train_f), cat_features=CAT_FEATURES)
freq_test_pool  = Pool(X_test_f,  y_test_f,  baseline=np.log(w_test_f),  cat_features=CAT_FEATURES)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9a. Frequency model with MLflow tracking

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

freq_final_params = {
    **best_params,
    "loss_function": "Poisson",
    "eval_metric":   "Poisson",
    "random_seed":   42,
    "verbose":       100,
}

with mlflow.start_run(run_name="freq_catboost_tuned") as run_freq:
    mlflow.log_params(freq_final_params)
    mlflow.log_param("model_type",      "catboost_frequency")
    mlflow.log_param("cv_strategy",     "walk_forward_ibnr1")
    mlflow.log_param("n_cv_folds",      len(folds))
    mlflow.log_param("features",        FEATURES)
    mlflow.log_param("cat_features",    CAT_FEATURES)
    mlflow.log_param("n_train",         len(df_train_all))
    mlflow.log_param("n_test",          len(df_test))
    mlflow.log_param("train_years",     df_train_all["policy_year"].unique().sort().to_list())
    mlflow.log_param("test_year",       int(max_year))

    freq_model = CatBoostRegressor(**freq_final_params)
    freq_model.fit(freq_train_pool, eval_set=freq_test_pool)

    y_pred_freq = freq_model.predict(freq_test_pool)

    test_dev_freq  = poisson_deviance(y_test_f, y_pred_freq, w_test_f)
    gini_freq      = gini(y_test_f, y_pred_freq, w_test_f)

    mlflow.log_metric("test_poisson_deviance",    test_dev_freq)
    mlflow.log_metric("test_gini",                gini_freq)
    mlflow.log_metric("baseline_cv_deviance",     baseline_cv_deviance)
    mlflow.log_metric("tuned_cv_deviance",        study.best_value)

    # Feature importance plot
    importances = freq_model.get_feature_importance(type="FeatureImportance")
    imp_df = (
        pl.DataFrame({"feature": FEATURES, "importance": importances.tolist()})
        .sort("importance", descending=True)
    )
    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    ax_imp.barh(imp_df["feature"].to_list(), imp_df["importance"].to_list())
    ax_imp.set_xlabel("Feature importance")
    ax_imp.set_title("CatBoost frequency model: feature importances")
    plt.tight_layout()
    mlflow.log_figure(fig_imp, "feature_importance.png")
    plt.show()

    # Log model artefact
    mlflow.catboost.log_model(freq_model, "freq_model")
    freq_run_id = run_freq.info.run_id

print(f"\nFrequency model run ID: {freq_run_id}")
print(f"Test Poisson deviance:  {test_dev_freq:.4f}")
print(f"Test Gini:              {gini_freq:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9b. Severity model
# MAGIC
# MAGIC We restrict to policies with at least one claim.
# MAGIC Exposure does not enter the severity model - severity is average cost per
# MAGIC claim, which is exposure-independent.
# MAGIC
# MAGIC CatBoost does not have a named "Gamma" loss function. Tweedie with
# MAGIC variance_power=2 is mathematically equivalent to the Gamma log-link model.

# COMMAND ----------

df_train_sev = df_train_all.filter(pl.col("claim_count") > 0)
df_test_sev  = df_test.filter(pl.col("claim_count") > 0)

print(f"Severity training set: {len(df_train_sev):,} claims")
print(f"Severity test set:     {len(df_test_sev):,} claims")

X_train_s = df_train_sev[FEATURES].to_pandas()
y_train_s = df_train_sev[SEV_TARGET].to_numpy()

X_test_s  = df_test_sev[FEATURES].to_pandas()
y_test_s  = df_test_sev[SEV_TARGET].to_numpy()

sev_train_pool = Pool(X_train_s, y_train_s, cat_features=CAT_FEATURES)
sev_test_pool  = Pool(X_test_s,  y_test_s,  cat_features=CAT_FEATURES)

sev_params = {
    **best_params,
    "loss_function": "Tweedie:variance_power=2",
    "eval_metric":   "RMSE",
    "random_seed":   42,
    "verbose":       100,
}

with mlflow.start_run(run_name="sev_catboost_tuned") as run_sev:
    mlflow.log_params(sev_params)
    mlflow.log_param("model_type",     "catboost_severity")
    mlflow.log_param("n_claims_train", len(df_train_sev))
    mlflow.log_param("n_claims_test",  len(df_test_sev))

    sev_model = CatBoostRegressor(**sev_params)
    sev_model.fit(sev_train_pool, eval_set=sev_test_pool)

    y_pred_sev = sev_model.predict(sev_test_pool)
    rmse_sev   = float(np.sqrt(np.mean((y_test_s - y_pred_sev)**2)))
    mae_sev    = float(np.mean(np.abs(y_test_s - y_pred_sev)))

    mlflow.log_metric("test_rmse_severity", rmse_sev)
    mlflow.log_metric("test_mae_severity",  mae_sev)
    mlflow.catboost.log_model(sev_model, "sev_model")
    sev_run_id = run_sev.info.run_id

print(f"\nSeverity model run ID: {sev_run_id}")
print(f"Test RMSE severity:    {rmse_sev:.0f}")
print(f"Test MAE severity:     {mae_sev:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Model registry
# MAGIC
# MAGIC Register the frequency model. We use the "challenger" alias because the GBM
# MAGIC has not yet been through committee review. In Module 4, when SHAP relativities
# MAGIC have been produced and reviewed, the model can be promoted to "production".
# MAGIC
# MAGIC Note: transition_model_version_stage() is deprecated in MLflow 2.9+.
# MAGIC Use set_registered_model_alias() instead.

# COMMAND ----------

client = MlflowClient()

freq_model_name = "motor_freq_catboost_m03"
freq_uri        = f"runs:/{freq_run_id}/freq_model"

registered_freq = mlflow.register_model(
    model_uri=freq_uri,
    name=freq_model_name,
)

client.set_registered_model_alias(
    name=freq_model_name,
    alias="challenger",
    version=registered_freq.version,
)

for key, val in {
    "module":       "module_03",
    "cv_strategy":  "walk_forward_ibnr1",
    "test_gini":    f"{gini_freq:.3f}",
    "test_deviance": f"{test_dev_freq:.4f}",
}.items():
    client.set_model_version_tag(
        name=freq_model_name,
        version=registered_freq.version,
        key=key,
        value=val,
    )

print(f"Registered: {freq_model_name} v{registered_freq.version} as 'challenger'")
print(f"Load with: mlflow.catboost.load_model('models:/{freq_model_name}@challenger')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Comparison with the Module 2 GLM
# MAGIC
# MAGIC We compare the GBM against the GLM from Module 2 using:
# MAGIC - Gini coefficient
# MAGIC - Calibration curve
# MAGIC - Double lift chart
# MAGIC
# MAGIC If you have not completed Module 2, this section generates approximate GLM
# MAGIC predictions using a simple multiplicative model for comparison purposes.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11a. Load GLM predictions
# MAGIC
# MAGIC If you completed Module 2 and have the GLM registered, uncomment the first block.
# MAGIC Otherwise, the second block generates approximate predictions from a Poisson GLM
# MAGIC fitted here.

# COMMAND ----------

# Option A: Load from Module 2 registry (uncomment if available)
# glm_model = mlflow.statsmodels.load_model("models:/motor_freq_glm@production")
# X_test_glm = df_test[FEATURES + [EXPOSURE_COL]].to_pandas()
# y_pred_glm = glm_model.predict(X_test_glm)

# Option B: Fit a quick Poisson GLM for comparison
import statsmodels.api as sm
import statsmodels.formula.api as smf

df_train_glm = df_train_all.to_pandas()
df_test_glm  = df_test.to_pandas()

# Treat categoricals as strings for formula interface
for col in CAT_FEATURES:
    df_train_glm[col] = df_train_glm[col].astype(str)
    df_test_glm[col]  = df_test_glm[col].astype(str)

formula = f"{FREQ_TARGET} ~ {' + '.join(CONTINUOUS_FEATURES)} + {' + '.join([f'C({c})' for c in CAT_FEATURES])}"

glm_fit = smf.glm(
    formula=formula,
    data=df_train_glm,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(df_train_glm[EXPOSURE_COL]),
).fit()

y_pred_glm_raw = glm_fit.predict(
    exog=df_test_glm,
    offset=np.log(df_test_glm[EXPOSURE_COL]),
)
y_pred_glm = y_pred_glm_raw.to_numpy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11b. Gini comparison

# COMMAND ----------

gini_gbm = gini(y_test_f, y_pred_freq, w_test_f)
gini_glm_val = gini(y_test_f, y_pred_glm,  w_test_f)

print(f"Frequency Gini - GBM: {gini_gbm:.3f}")
print(f"Frequency Gini - GLM: {gini_glm_val:.3f}")
print(f"Gini lift:            {gini_gbm - gini_glm_val:+.3f} ({(gini_gbm / gini_glm_val - 1) * 100:+.1f}%)")

# Log comparison metrics to the frequency run
with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_metric("glm_gini",        gini_glm_val)
    mlflow.log_metric("gini_lift_vs_glm", gini_gbm - gini_glm_val)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11c. Calibration curve

# COMMAND ----------

def calibration_data(y_true, y_pred, exposure, n_bins=10):
    freq_pred = y_pred / exposure
    freq_true = y_true / exposure
    bins      = np.quantile(freq_pred, np.linspace(0, 1, n_bins + 1))
    bin_idx   = np.digitize(freq_pred, bins[1:-1])
    actuals, predicteds = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 10:
            continue
        actuals.append(freq_true[mask].mean())
        predicteds.append(freq_pred[mask].mean())
    return np.array(predicteds), np.array(actuals)

pred_gbm, act_gbm = calibration_data(y_test_f, y_pred_freq, w_test_f)
pred_glm, act_glm = calibration_data(y_test_f, y_pred_glm,  w_test_f)

fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
ax_cal.plot(pred_gbm, act_gbm, "o-", label="GBM")
ax_cal.plot(pred_glm, act_glm, "s--", label="GLM")
lim = max(pred_gbm.max(), pred_glm.max()) * 1.05
ax_cal.plot([0, lim], [0, lim], "k:", label="Perfect calibration")
ax_cal.set_xlabel("Mean predicted frequency")
ax_cal.set_ylabel("Mean actual frequency")
ax_cal.set_title("Calibration: GBM vs GLM")
ax_cal.legend()
plt.tight_layout()

with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_figure(fig_cal, "calibration_gbm_vs_glm.png")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11d. Double lift chart
# MAGIC
# MAGIC The double lift chart shows actual frequency by decile of the ratio
# MAGIC (GBM prediction / GLM prediction). An upward slope confirms the GBM is
# MAGIC finding risk signals that the GLM misses. A flat chart means the GBM
# MAGIC provides no additional discrimination beyond the GLM on this dataset.

# COMMAND ----------

freq_pred_gbm = y_pred_freq / w_test_f
freq_pred_glm = y_pred_glm  / w_test_f
ratio         = freq_pred_gbm / freq_pred_glm
freq_actual   = y_test_f / w_test_f

n_bins     = 10
bins       = np.quantile(ratio, np.linspace(0, 1, n_bins + 1))
bin_idx    = np.digitize(ratio, bins[1:-1])

ratio_mids, actuals_dl = [], []
for b in range(n_bins):
    mask = bin_idx == b
    if mask.sum() < 20:
        continue
    ratio_mids.append(ratio[mask].mean())
    actuals_dl.append(freq_actual[mask].mean())

fig_dl, ax_dl = plt.subplots(figsize=(8, 5))
ax_dl.plot(ratio_mids, actuals_dl, "o-")
ax_dl.axhline(freq_actual.mean(), linestyle="--", color="grey", label="Portfolio mean frequency")
ax_dl.set_xlabel("GBM predicted frequency / GLM predicted frequency")
ax_dl.set_ylabel("Actual observed frequency")
ax_dl.set_title("Double lift: GBM vs GLM")
ax_dl.legend()
plt.tight_layout()

with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_figure(fig_dl, "double_lift_gbm_vs_glm.png")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary
# MAGIC
# MAGIC What we built in this notebook:
# MAGIC
# MAGIC - CatBoost Poisson frequency model with exposure as log-offset (not sample weight)
# MAGIC - CatBoost Tweedie(power=2) severity model restricted to claims > 0
# MAGIC - Temporal cross-validation using insurance-cv: walk-forward splits with 1-year IBNR buffer
# MAGIC - Optuna hyperparameter tuning across depth, learning_rate, l2_leaf_reg
# MAGIC - Full MLflow tracking: parameters, metrics, calibration/double-lift figures, model artefacts
# MAGIC - Model registry registration as "challenger" (pending Module 4 committee review)
# MAGIC - Comparison with Module 2 GLM: Gini, calibration, double lift
# MAGIC
# MAGIC The models are registered in the Databricks model registry.
# MAGIC In Module 4, we extract SHAP relativities from the frequency model to produce
# MAGIC a factor table that a pricing committee can review.

# COMMAND ----------

# Final output
print("=" * 60)
print("Module 3 complete")
print("=" * 60)
print(f"Frequency model: {freq_model_name} v{registered_freq.version} @challenger")
print(f"  Test Gini:     {gini_gbm:.3f}  (GLM: {gini_glm_val:.3f}, lift: {gini_gbm - gini_glm_val:+.3f})")
print(f"  Test deviance: {test_dev_freq:.4f}")
print()
print(f"Severity model run ID: {sev_run_id}")
print(f"  Test RMSE:     {rmse_sev:.0f}")
print()
print("Next: Module 4 - SHAP Relativities")
print("  Extract multiplicative factor tables from the GBM frequency model")
