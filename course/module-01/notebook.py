# Databricks notebook source
# MAGIC %md
# MAGIC # Module 1: Databricks for Pricing Teams
# MAGIC ## Modern Insurance Pricing with Python and Databricks
# MAGIC
# MAGIC This notebook walks through the full Module 1 workflow on synthetic motor data:
# MAGIC
# MAGIC 1. Workspace and catalog setup
# MAGIC 2. Generating and loading synthetic motor policy data to Delta
# MAGIC 3. EDA: one-way frequency, severity distribution, exposure distribution
# MAGIC 4. MLflow experiment tracking
# MAGIC 5. A CatBoost frequency model as a smoke test
# MAGIC 6. Audit trail entry
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Databricks Runtime 14.3 LTS or later
# MAGIC - Unity Catalog enabled on your workspace (Free Edition includes this)
# MAGIC - The `pricing` catalog must exist: if you are on Free Edition, use your personal sandbox catalog
# MAGIC
# MAGIC **Free Edition users:** All steps in this notebook work on Free Edition. Section 7 (Workflows)
# MAGIC in the tutorial requires a paid workspace; the notebook covers what you can run interactively.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install packages
# MAGIC
# MAGIC CatBoost and Polars are not pre-installed on the standard Databricks Runtime.
# MAGIC Install them here. The kernel restart ensures the new packages are importable
# MAGIC in subsequent cells.
# MAGIC
# MAGIC If you are using `uv` for package management in a local development environment,
# MAGIC the equivalent is `uv add catboost polars`. In Databricks notebooks, `uv pip install`
# MAGIC is the right mechanism.

# COMMAND ----------

# MAGIC %sh uv pip install catboost polars

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Workspace setup
# MAGIC
# MAGIC Set your catalog and schema names here. If you are on Free Edition and do not
# MAGIC have a `pricing` catalog, change CATALOG to your user sandbox catalog
# MAGIC (usually your username or `main`).
# MAGIC
# MAGIC **Important:** If you are adopting Databricks into an existing environment,
# MAGIC your platform team has probably already defined the catalog structure and naming
# MAGIC conventions. Use what they have set up rather than creating new catalogs -
# MAGIC agree on the structure first. The names below are targets for a greenfield setup.

# COMMAND ----------

# Constants: set once, used throughout the notebook
# Do not interpolate user-supplied strings into SQL via f-strings.
# These are hardcoded constants and are safe. Application code reading
# table names from user input requires parameterised queries instead.
CATALOG = "pricing"      # Change to your catalog name if needed
SCHEMA  = "motor"
VOLUME  = "raw"

# COMMAND ----------

# Verify Unity Catalog is available and you can access the catalog
try:
    spark.sql(f"DESCRIBE CATALOG {CATALOG}")
    print(f"Catalog '{CATALOG}' is accessible.")
except Exception as e:
    print(f"Cannot access catalog '{CATALOG}': {e}")
    print("If you are on Free Edition, try CATALOG = 'main' or your username.")
    raise

# COMMAND ----------

# Create the schema and volume if they do not exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA} COMMENT 'Motor pricing models and data'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

print(f"Schema {CATALOG}.{SCHEMA} is ready.")
print(f"Volume {CATALOG}.{SCHEMA}.{VOLUME} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic motor policy data
# MAGIC
# MAGIC We generate 10,000 synthetic motor policies with realistic structure.
# MAGIC This is not a toy dataset: the claim frequency follows a Poisson process
# MAGIC with multiplicative rating factors, and the NCD distribution is skewed
# MAGIC towards higher values as it is in a real UK motor book.
# MAGIC
# MAGIC The true data-generating process (DGP) parameters are logged explicitly
# MAGIC so you can compare them against what the model recovers.

# COMMAND ----------

import polars as pl
import numpy as np
from datetime import date, timedelta

rng = np.random.default_rng(seed=42)
n = 10_000

# COMMAND ----------

# Rating factor levels
age_bands      = ["17-25", "26-35", "36-50", "51-65", "66+"]
vehicle_groups = ["A", "B", "C", "D", "E"]
regions        = ["London", "South East", "Midlands", "North", "Scotland", "Wales"]

# Age band distribution: young drivers are a small but important segment
age_probs = [0.08, 0.22, 0.35, 0.25, 0.10]

# Vehicle group distribution
vg_probs = [0.15, 0.25, 0.30, 0.20, 0.10]

# NCD distribution: UK motor books are heavily skewed towards higher NCD
# experienced policyholders dominate. Uniform 0-5 is not realistic.
# Approximate real-book distribution:
ncd_probs = [0.05, 0.10, 0.15, 0.20, 0.20, 0.30]  # NCD 0 through NCD 5

ncd_years    = rng.choice(range(6), size=n, p=ncd_probs)
age_band_arr = rng.choice(age_bands, size=n, p=age_probs)
vg_arr       = rng.choice(vehicle_groups, size=n, p=vg_probs)
region_arr   = rng.choice(regions, size=n)  # uniform across regions

# Exposure in years: uniform between 0.25 and 1.0
# Represents mix of annual, 9-month, 6-month, and 3-month policies
exposure_arr = rng.uniform(0.25, 1.0, size=n)

df = pl.DataFrame({
    "policy_id":     [f"POL{i:06d}" for i in range(n)],
    "age_band":      age_band_arr.tolist(),
    "vehicle_group": vg_arr.tolist(),
    "region":        region_arr.tolist(),
    "ncd_years":     ncd_years.tolist(),
    "exposure":      exposure_arr.tolist(),
})

# COMMAND ----------

# True DGP parameters - log these so learners can compare model output against them
# Base rate is per unit exposure per year
DGP_BASE_RATE = 0.08  # 8% claim frequency at base level (exp(-2.526) ≈ 0.08)

DGP_AGE_FACTORS = {"17-25": 2.5, "26-35": 1.4, "36-50": 1.0, "51-65": 0.9, "66+": 1.1}
DGP_VG_FACTORS  = {"A": 0.7, "B": 0.9, "C": 1.0, "D": 1.2, "E": 1.5}
DGP_NCD_FACTORS = {0: 2.0, 1: 1.6, 2: 1.3, 3: 1.1, 4: 1.0, 5: 0.85}

print("True DGP parameters:")
print(f"  Base frequency (per policy-year): {DGP_BASE_RATE:.3f}")
print(f"  Age factors: {DGP_AGE_FACTORS}")
print(f"  Vehicle group factors: {DGP_VG_FACTORS}")
print(f"  NCD factors: {DGP_NCD_FACTORS}")
print(f"\n  Note: factors are multiplicative on the base rate, weighted by exposure.")
print(f"  A 17-25 driver in vehicle group E with NCD 0 has expected frequency:")
print(f"  {DGP_BASE_RATE * 2.5 * 1.5 * 2.0:.3f} per year of exposure.")

# COMMAND ----------

# Generate claim counts and amounts
expected_freq = np.array([
    DGP_BASE_RATE
    * DGP_AGE_FACTORS[row["age_band"]]
    * DGP_VG_FACTORS[row["vehicle_group"]]
    * DGP_NCD_FACTORS[row["ncd_years"]]
    * row["exposure"]
    for row in df.iter_rows(named=True)
])

# Poisson frequency
claim_counts = rng.poisson(expected_freq)

# Gamma severity: mean £3,000 per claim, shape 2 (CV = 1/sqrt(2) ≈ 0.71)
# Severity is independent of frequency factors in this DGP
claim_amounts = np.where(
    claim_counts > 0,
    rng.gamma(shape=2.0, scale=1500.0, size=n) * claim_counts,
    0.0
)

df = df.with_columns([
    pl.Series("claim_count",  claim_counts.tolist()).cast(pl.Int32),
    pl.Series("claim_amount", claim_amounts.tolist()),
])

print(f"Generated {n:,} policies")
print(f"Overall claim frequency: {claim_counts.sum() / exposure_arr.sum():.4f} claims per policy-year")
print(f"Mean claim amount (claimants only): £{claim_amounts[claim_amounts > 0].mean():,.0f}")
print(f"Total claim count: {claim_counts.sum():,}")
print(f"\nFirst five rows:")
print(df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load to Delta and set table properties

# COMMAND ----------

# PySpark does not yet natively accept Polars DataFrames.
# Convert to pandas at the Spark boundary only.
spark_df = spark.createDataFrame(df.to_pandas())

(spark_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.policies"))

print(f"Saved {n:,} rows to {CATALOG}.{SCHEMA}.policies")

# COMMAND ----------

# Set table properties
# delta.autoOptimize.optimizeWrite was deprecated in DBR 11.3 - do not use it.
# Use delta.autoCompact.enabled instead.
#
# Retention: 7 years for production tables matches FCA Consumer Duty evidence guidance.
# For development/training tables, 90 days is sufficient.
#
# GDPR note: Delta version history retains all historical data including PII.
# If your book contains policyholders who may exercise right to erasure, discuss
# the tension between FCA data retention and GDPR right to erasure with your DPO
# before setting retention policies on tables containing personal data.
spark.sql(f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.policies
    SET TBLPROPERTIES (
        'delta.autoCompact.enabled' = 'true',
        'delta.logRetentionDuration' = 'interval 7 years',
        'delta.deletedFileRetentionDuration' = 'interval 7 years'
    )
""")

print("Table properties set.")
print("  autoCompact.enabled: true  - keeps file sizes sensible over time")
print("  logRetentionDuration: 7 years  - for FCA Consumer Duty evidence")

# COMMAND ----------

# Verify the table exists and inspect
display(spark.sql(f"DESCRIBE EXTENDED {CATALOG}.{SCHEMA}.policies"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Exploratory data analysis
# MAGIC
# MAGIC We work in Polars for all data manipulation. PySpark is used only to read from
# MAGIC and write to Delta.

# COMMAND ----------

# Read back from Delta (confirms the write worked)
df = pl.from_pandas(spark.table(f"{CATALOG}.{SCHEMA}.policies").toPandas())

print(f"Loaded {len(df):,} rows from {CATALOG}.{SCHEMA}.policies")
print(f"Schema: {df.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Exposure distribution

# COMMAND ----------

exposure_summary = df.select("exposure").describe()
print(exposure_summary)

# Distribution by rounded exposure band
exposure_dist = (
    df
    .with_columns(
        pl.col("exposure").round(1).alias("exposure_band")
    )
    .group_by("exposure_band")
    .agg(pl.len().alias("policy_count"))
    .sort("exposure_band")
)
print("\nPolicy count by exposure band (rounded to 0.1 years):")
print(exposure_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Frequency one-way by rating factor

# COMMAND ----------

def frequency_oneway(df: pl.DataFrame, factor: str) -> pl.DataFrame:
    """
    One-way frequency analysis by a single rating factor.

    Returns a DataFrame with exposure, claim count, and frequency per policy-year
    for each level of the factor.
    """
    return (
        df
        .group_by(factor)
        .agg([
            pl.col("exposure").sum().alias("exposure"),
            pl.col("claim_count").sum().alias("claims"),
        ])
        .with_columns(
            (pl.col("claims") / pl.col("exposure")).alias("freq_pa")
        )
        .sort(factor)
    )

# COMMAND ----------

print("Frequency by age band:")
print(frequency_oneway(df, "age_band"))

# COMMAND ----------

print("\nFrequency by vehicle group:")
print(frequency_oneway(df, "vehicle_group"))

# COMMAND ----------

print("\nFrequency by NCD years:")
ncd_ow = frequency_oneway(df, "ncd_years")
print(ncd_ow)

# COMMAND ----------

print("\nFrequency by region:")
print(frequency_oneway(df, "region"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Severity distribution (claimants only)

# COMMAND ----------

# Severity: claim amount per claim, for policies with at least one claim
claimants = df.filter(pl.col("claim_count") > 0).with_columns(
    (pl.col("claim_amount") / pl.col("claim_count")).alias("cost_per_claim")
)

severity_summary = claimants.select("cost_per_claim").describe()
print("Severity distribution (cost per claim):")
print(severity_summary)

# COMMAND ----------

# Severity by vehicle group (higher groups tend to have higher repair costs)
severity_by_vg = (
    claimants
    .group_by("vehicle_group")
    .agg([
        pl.col("claim_count").sum().alias("claim_count"),
        pl.col("claim_amount").sum().alias("total_loss"),
        pl.col("cost_per_claim").mean().alias("mean_cost_per_claim"),
    ])
    .with_columns(
        (pl.col("total_loss") / pl.col("claim_count")).alias("weighted_mean_severity")
    )
    .sort("vehicle_group")
)

print("\nSeverity by vehicle group:")
print(severity_by_vg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 NCD distribution check
# MAGIC
# MAGIC UK motor books are skewed towards higher NCD values. Uniform distributions
# MAGIC in synthetic data produce unrealistic one-way analyses. Our DGP uses the
# MAGIC approximate real-book distribution. Compare the output here against the
# MAGIC true DGP proportions.

# COMMAND ----------

ncd_dist = (
    df
    .group_by("ncd_years")
    .agg([
        pl.len().alias("policy_count"),
        pl.col("exposure").sum().alias("exposure"),
    ])
    .with_columns(
        (pl.col("policy_count") / pl.col("policy_count").sum()).alias("pct_policies")
    )
    .sort("ncd_years")
)

print("NCD distribution (compare against DGP probs: [0.05, 0.10, 0.15, 0.20, 0.20, 0.30]):")
print(ncd_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature engineering for modelling

# COMMAND ----------

# Create a modelling dataset
# In a real project this would be a separate Delta table with its own version history
df_model = df.with_columns([
    # NCD as a string for CatBoost categorical treatment
    pl.col("ncd_years").cast(pl.Utf8).alias("ncd_group"),
    # Log exposure as offset for Poisson model
    pl.col("exposure").log().alias("log_exposure"),
])

cat_features = ["age_band", "vehicle_group", "region", "ncd_group"]
num_features = ["exposure"]

feature_cols = cat_features + num_features
target_col   = "claim_count"

print(f"Feature columns: {feature_cols}")
print(f"Target: {target_col} (Poisson with exposure offset)")
print(f"\nModelling dataset: {len(df_model):,} rows")

# COMMAND ----------

# Train/validation split: 80/20 by row index
# In a real project use walk-forward CV with an IBNR buffer (see Module 3)
train_size = int(0.8 * len(df_model))
df_train   = df_model[:train_size]
df_val     = df_model[train_size:]

print(f"Train: {len(df_train):,} rows")
print(f"Val:   {len(df_val):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. CatBoost frequency model with MLflow tracking
# MAGIC
# MAGIC This is the smoke test: a simple CatBoost Poisson model to confirm the
# MAGIC environment is set up correctly and MLflow logging works. Module 3 covers
# MAGIC CatBoost hyperparameter choices for insurance data in detail.
# MAGIC
# MAGIC CatBoost handles categorical features natively - no ordinal encoding needed.
# MAGIC Pass the categorical column names to `cat_features` and CatBoost handles the rest.

# COMMAND ----------

import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor, Pool
from mlflow import MlflowClient

# COMMAND ----------

# Set the MLflow experiment
# In a shared workspace, use your username in the path to avoid collisions
experiment_path = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/motor-frequency-module1"
mlflow.set_experiment(experiment_path)
print(f"MLflow experiment: {experiment_path}")

# COMMAND ----------

# Prepare arrays for CatBoost
# CatBoost accepts categorical features by index or name
X_train = df_train.select(feature_cols).to_pandas()
y_train = df_train[target_col].to_numpy()
w_train = df_train["exposure"].to_numpy()  # exposure as sample weight for Poisson

X_val   = df_val.select(feature_cols).to_pandas()
y_val   = df_val[target_col].to_numpy()
w_val   = df_val["exposure"].to_numpy()

# CatBoost Pool: specify categorical feature names
cat_feature_indices = [X_train.columns.tolist().index(c) for c in cat_features]

train_pool = Pool(X_train, label=y_train, weight=w_train, cat_features=cat_feature_indices)
val_pool   = Pool(X_val,   label=y_val,   weight=w_val,   cat_features=cat_feature_indices)

# COMMAND ----------

# Model parameters
# These are conservative starter parameters for an infrastructure smoke test.
# Module 3 covers the tuning choices that matter for insurance data.
params = {
    "iterations":    300,
    "learning_rate": 0.05,
    "depth":         4,
    "loss_function": "Poisson",   # Poisson deviance for frequency
    "eval_metric":   "Poisson",
    "random_seed":   42,
    "verbose":       100,
}

# COMMAND ----------

with mlflow.start_run(run_name="freq_catboost_module1_smoke_test") as run:
    # Log parameters
    mlflow.log_params(params)
    mlflow.log_params({
        "n_train":       len(df_train),
        "n_val":         len(df_val),
        "cat_features":  str(cat_features),
        "feature_set":   "v1_base_rating_factors",
        "offset":        "log(exposure) via sample_weight",
    })

    # Train
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Predictions: model output is log(frequency), exponentiate for frequency
    pred_train = model.predict(train_pool)
    pred_val   = model.predict(val_pool)

    # Poisson deviance: 2 * sum(y*log(y/mu) - (y-mu))
    # Only defined where y > 0 for the log term; use where clause
    def poisson_deviance(y_true, y_pred, weights):
        """Exposure-weighted Poisson deviance."""
        y_pred = np.maximum(y_pred, 1e-10)  # avoid log(0)
        dev = 2.0 * (
            np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
            - (y_true - y_pred)
        )
        return np.average(dev, weights=weights)

    train_dev = poisson_deviance(y_train, pred_train, w_train)
    val_dev   = poisson_deviance(y_val,   pred_val,   w_val)

    mlflow.log_metrics({
        "train_poisson_deviance": train_dev,
        "val_poisson_deviance":   val_dev,
        "best_iteration":         model.best_iteration_,
    })

    # Log the model artefact
    mlflow.catboost.log_model(model, "freq_model")

    run_id = run.info.run_id

print(f"\nResults:")
print(f"  Train Poisson deviance: {train_dev:.6f}")
print(f"  Val Poisson deviance:   {val_dev:.6f}")
print(f"  Best iteration:         {model.best_iteration_}")
print(f"\nRun ID: {run_id}")
print(f"Open the MLflow UI (Experiments tab) to see the full run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Feature importance
# MAGIC
# MAGIC CatBoost produces feature importance natively. These are the PredictionValuesChange
# MAGIC importances - roughly, how much the model output changes when a feature is permuted.
# MAGIC Module 4 covers SHAP-based importances which are more interpretable for pricing.

# COMMAND ----------

importances = pl.DataFrame({
    "feature":    feature_cols,
    "importance": model.get_feature_importance().tolist(),
}).sort("importance", descending=True)

print("Feature importances (PredictionValuesChange):")
print(importances)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 True DGP vs model comparison
# MAGIC
# MAGIC The synthetic data has known true parameters. Compare the model's predictions
# MAGIC against the true DGP as a sanity check.

# COMMAND ----------

# Mean predicted frequency vs mean actual frequency by age band
comparison = (
    df_val
    .with_columns(
        pl.Series("predicted_freq", pred_val / df_val["exposure"].to_numpy())
    )
    .group_by("age_band")
    .agg([
        pl.col("exposure").sum().alias("exposure"),
        pl.col("claim_count").sum().alias("actual_claims"),
        pl.col("predicted_freq").mean().alias("mean_predicted_freq"),
    ])
    .with_columns(
        (pl.col("actual_claims") / pl.col("exposure")).alias("actual_freq_pa")
    )
    .sort("age_band")
)

print("Age band: actual vs predicted frequency (validation set):")
print(comparison.select(["age_band", "actual_freq_pa", "mean_predicted_freq"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model registry and aliases
# MAGIC
# MAGIC Register the model so it can be loaded by name rather than by run ID.
# MAGIC Uses aliases (not stages) - stages were deprecated in MLflow 2.9+.

# COMMAND ----------

client = MlflowClient()

# Register the model from the run we just completed
model_uri = f"runs:/{run_id}/freq_model"
model_name = "motor_freq_catboost_module1"

registered = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Registered model '{model_name}' version {registered.version}")

# Set an alias - use this instead of stage transitions (deprecated in MLflow 2.9+)
client.set_registered_model_alias(
    name=model_name,
    alias="development",
    version=registered.version,
)

# Tag with metadata
client.set_model_version_tag(
    name=model_name,
    version=registered.version,
    key="module",
    value="module1_smoke_test",
)

print(f"Alias 'development' -> version {registered.version}")
print(f"\nTo load this model:")
print(f"  model = mlflow.catboost.load_model('models:/{model_name}@development')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. FCA audit trail
# MAGIC
# MAGIC Every model run should produce an audit record that ties together:
# MAGIC - The MLflow run ID (parameters and metrics)
# MAGIC - The data version (which Delta table version was used)
# MAGIC - Model metadata (what the model is, who is responsible)
# MAGIC
# MAGIC This table is append-only. Grant INSERT but not UPDATE or DELETE to
# MAGIC the pricing team group.

# COMMAND ----------

from datetime import datetime

# Get the current Delta table version (the version used for training)
table_version = (
    spark.sql(f"DESCRIBE HISTORY {CATALOG}.{SCHEMA}.policies LIMIT 1")
    .collect()[0]["version"]
)

audit_record = {
    "run_id":            run_id,
    "model_name":        "freq_catboost_motor_v1",
    "model_version":     1,
    "registered_name":   model_name,
    "registered_version": int(registered.version),
    "training_date":     datetime.utcnow().isoformat(),
    "data_table":        f"{CATALOG}.{SCHEMA}.policies",
    "data_version":      int(table_version),
    "train_deviance":    float(train_dev),
    "val_deviance":      float(val_dev),
    "n_train":           len(df_train),
    "n_val":             len(df_val),
    "feature_set":       "v1_base_rating_factors",
    "approved_by":       None,
    "deployed_to_prod":  False,
    "notes":             "Module 1 smoke test - synthetic data only",
}

# Create the governance schema if needed
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.governance COMMENT 'Audit and governance tables'")

# Append to the audit log
audit_df = spark.createDataFrame([audit_record])
(audit_df
    .write
    .format("delta")
    .mode("append")
    .saveAsTable(f"{CATALOG}.governance.model_run_log"))

print(f"Audit record written to {CATALOG}.governance.model_run_log")
print(f"  run_id: {run_id}")
print(f"  data_version: {table_version}")
print(f"  val_deviance: {val_dev:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Delta table time travel
# MAGIC
# MAGIC Check the version history of the policies table. In a real project you would use
# MAGIC this to reproduce a training run: query the data at exactly the version that
# MAGIC was used when the model was trained.

# COMMAND ----------

# View the history
display(spark.sql(f"DESCRIBE HISTORY {CATALOG}.{SCHEMA}.policies"))

# COMMAND ----------

# Reading a specific version by number
# Useful for reproducing a training run
df_v0 = pl.from_pandas(
    spark.read.format("delta")
    .option("versionAsOf", 0)
    .table(f"{CATALOG}.{SCHEMA}.policies")
    .toPandas()
)

print(f"Version 0 of policies table: {len(df_v0):,} rows")
print("This is the exact data that was current when the audit record was written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Z-ordering for query performance
# MAGIC
# MAGIC For tables you query repeatedly with the same filters, Z-ordering improves
# MAGIC performance. Run this after loading the data, not on every write.
# MAGIC
# MAGIC This rewrites the table, which takes time. Only run when query performance
# MAGIC is a problem.

# COMMAND ----------

# Z-order by the factors most commonly used in WHERE clauses
# This is optional for the training dataset but included for completeness
spark.sql(f"""
    OPTIMIZE {CATALOG}.{SCHEMA}.policies
    ZORDER BY (region, vehicle_group)
""")

print("OPTIMIZE complete. Query performance for region and vehicle_group filters improved.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC You have:
# MAGIC
# MAGIC 1. Set up a Unity Catalog schema and volume for motor pricing data
# MAGIC 2. Generated 10,000 synthetic motor policies with a realistic DGP
# MAGIC 3. Loaded the data to a Delta table with correct retention properties
# MAGIC 4. Done one-way EDA by age band, vehicle group, NCD, and region
# MAGIC 5. Trained a CatBoost Poisson frequency model and logged it to MLflow
# MAGIC 6. Registered the model with an alias
# MAGIC 7. Written an audit record that ties the run to the data version
# MAGIC
# MAGIC The MLflow experiment is in your Experiments tab. Click it to see the
# MAGIC run parameters, metrics, and model artefact. This is the record you
# MAGIC point to when asked to demonstrate reproducibility.
# MAGIC
# MAGIC **Next:** Module 2 covers GLMs in Python - replicating what Emblem does
# MAGIC with statsmodels, including offset terms and one-way/two-way analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: MERGE pattern for incremental data loads
# MAGIC
# MAGIC When new claims data arrives (weekly or monthly bordereaux), use MERGE
# MAGIC rather than overwriting the entire table. MERGE is atomic and idempotent
# MAGIC on a unique key - retrying a MERGE that failed halfway does not produce
# MAGIC duplicate records.
# MAGIC
# MAGIC The pattern below shows how to update existing claims and insert new ones.

# COMMAND ----------

# Generate some "update" data - in production this would be a new bordereaux extract
update_data = df.sample(n=100, seed=99).with_columns(
    (pl.col("claim_amount") * 1.05).alias("claim_amount")  # simulate a 5% case reserve increase
)

# Step 1: register as temp view (separate statement - createOrReplaceTempView returns None)
update_spark = spark.createDataFrame(update_data.to_pandas())
update_spark.createOrReplaceTempView("bordereaux_updates")

# Step 2: MERGE using the temp view
spark.sql(f"""
    MERGE INTO {CATALOG}.{SCHEMA}.policies AS target
    USING bordereaux_updates AS source
    ON target.policy_id = source.policy_id
    WHEN MATCHED THEN UPDATE SET
        target.claim_amount = source.claim_amount,
        target.claim_count  = source.claim_count
    WHEN NOT MATCHED THEN INSERT *
""")

print("MERGE complete.")
print("Updated 100 rows with revised claim amounts.")
print(f"New table version: {spark.sql(f'DESCRIBE HISTORY {CATALOG}.{SCHEMA}.policies LIMIT 1').collect()[0]['version']}")
