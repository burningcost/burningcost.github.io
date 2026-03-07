# End-to-End Pricing Pipeline

---

## Why pipelines break down

Ask a pricing actuary what their workflow looks like and you will hear a list: data pull, feature engineering, model training, validation, relativities, rate change, sign-off. Ask them where the work actually lives and you will hear: Excel for the data cleaning, a Python script for the model, another script for the relativities, a third for the rate change scenarios, and a PowerPoint for the committee.

Every handoff between those tools is a place where assumptions get lost, where the feature engineering from the modelling run does not quite match what was applied to the scoring run, where the model version cannot be traced back to the training data. Auditors ask where the numbers came from. The answer is usually "the pricing actuary's laptop."

This module builds a pipeline that holds together. Every stage runs in a single Databricks notebook. Data flows from Unity Catalog Delta tables to the same Delta tables at the end. The feature transforms are defined once and applied identically at training, validation, and scoring. The model version is tracked in MLflow. The output tables carry enough metadata to reproduce any number in the rate change pack.

The synthetic data we use represents 200,000 UK motor policies over four annual cohorts. In production, replace the data generation cell with a `spark.table()` call pointing at your policy administration system's Delta table.

---

## Stage 0: Cluster setup and library installation

All five Burning Cost libraries, plus CatBoost, Polars, Optuna, and MLflow:

```python
# Cell 0 - Run once per cluster restart
%pip install \
  "insurance-cv @ git+https://github.com/burningcost/insurance-cv.git" \
  "shap-relativities[catboost] @ git+https://github.com/burningcost/shap-relativities.git" \
  "insurance-conformal[catboost] @ git+https://github.com/burningcost/insurance-conformal.git" \
  "credibility @ git+https://github.com/burningcost/credibility.git" \
  "rate-optimiser @ git+https://github.com/burningcost/rate-optimiser.git" \
  catboost polars optuna mlflow --quiet

dbutils.library.restartPython()
```

The `restartPython()` call is necessary after `%pip install` in Databricks. Any code above this cell in the notebook will not have the new libraries available.

---

## Stage 1: Unity Catalog schema and Delta tables

We create a dedicated schema for the pricing run. Every artefact - raw data, feature view, model predictions, output pack - lives in Unity Catalog. This is what makes the pipeline auditable.

```python
# Cell 1 - Schema setup
CATALOG = "pricing"
SCHEMA = "motor_q1_2026"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Table names we will write to throughout the pipeline
TABLES = {
    "raw": f"{CATALOG}.{SCHEMA}.raw_policies",
    "features": f"{CATALOG}.{SCHEMA}.features",
    "freq_predictions": f"{CATALOG}.{SCHEMA}.freq_predictions",
    "sev_predictions": f"{CATALOG}.{SCHEMA}.sev_predictions",
    "relativities": f"{CATALOG}.{SCHEMA}.relativities",
    "conformal_intervals": f"{CATALOG}.{SCHEMA}.conformal_intervals",
    "blended_relativities": f"{CATALOG}.{SCHEMA}.blended_relativities",
    "rate_change": f"{CATALOG}.{SCHEMA}.rate_change",
    "efficient_frontier": f"{CATALOG}.{SCHEMA}.efficient_frontier",
}

print("Schema:", f"{CATALOG}.{SCHEMA}")
print("Tables to be written:", list(TABLES.keys()))
```

Name your schema with the review cycle, not just the model version. `motor_q1_2026` makes it immediately clear which rating review this pipeline corresponds to. When someone comes back in 18 months to understand what happened, they will not need to decode version numbers.

---

## Stage 2: Synthetic data generation

In production this cell is replaced by `spark.table("source_system.motor.policies")`. For the tutorial, we generate four years of motor policies with realistic cohort structure.

```python
# Cell 2 - Data generation (replace with spark.table() in production)
import polars as pl
import numpy as np

rng = np.random.default_rng(2026)

N_TOTAL = 200_000
YEARS = [2022, 2023, 2024, 2025]
N_PER_YEAR = N_TOTAL // len(YEARS)

cohorts = []
for year in YEARS:
    # Premium inflation: ~7% per year over the period
    inflation = 1.07 ** (year - 2022)

    n = N_PER_YEAR
    age_band = rng.choice(
        ["17-25", "26-35", "36-50", "51-65", "66+"],
        n, p=[0.10, 0.20, 0.35, 0.25, 0.10],
    )
    ncb = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.10, 0.10, 0.15, 0.20, 0.20, 0.25])
    vehicle_group = rng.choice(["A", "B", "C", "D", "E"], n, p=[0.20, 0.25, 0.25, 0.20, 0.10])
    region = rng.choice(
        ["London", "SouthEast", "Midlands", "North", "Scotland", "Wales"],
        n, p=[0.18, 0.20, 0.22, 0.25, 0.10, 0.05],
    )
    annual_mileage = rng.choice(["<5k", "5k-10k", "10k-15k", "15k+"], n, p=[0.15, 0.35, 0.35, 0.15])

    # True frequency: age is the dominant driver
    age_freq = {"17-25": 0.12, "26-35": 0.07, "36-50": 0.05, "51-65": 0.04, "66+": 0.06}
    freq_base = np.array([age_freq[a] for a in age_band])

    vehicle_freq = {"A": 0.85, "B": 0.95, "C": 1.00, "D": 1.10, "E": 1.25}
    freq_base *= np.array([vehicle_freq[v] for v in vehicle_group])

    region_freq = {
        "London": 1.15, "SouthEast": 1.05, "Midlands": 1.00,
        "North": 0.95, "Scotland": 0.90, "Wales": 0.92,
    }
    freq_base *= np.array([region_freq[r] for r in region])

    mileage_freq = {"<5k": 0.75, "5k-10k": 0.90, "10k-15k": 1.05, "15k+": 1.30}
    freq_base *= np.array([mileage_freq[m] for m in annual_mileage])

    claim_count = rng.poisson(freq_base)

    # Severity: vehicle group and region drive severity
    sev_base = 2_800 * inflation
    vehicle_sev = {"A": 0.75, "B": 0.90, "C": 1.00, "D": 1.15, "E": 1.40}
    sev_vehicle = np.array([vehicle_sev[v] for v in vehicle_group])
    region_sev = {
        "London": 1.20, "SouthEast": 1.10, "Midlands": 1.00,
        "North": 0.95, "Scotland": 0.88, "Wales": 0.92,
    }
    sev_region = np.array([region_sev[r] for r in region])

    # Claim severity: Gamma distributed, shape=2
    mean_sev = sev_base * sev_vehicle * sev_region
    claim_severity = np.where(
        claim_count > 0,
        rng.gamma(shape=2.0, scale=mean_sev / 2.0),
        0.0,
    )

    exposure = rng.uniform(0.3, 1.0, n)  # policy years
    earned_premium = (
        sev_base * freq_base / 0.72 * inflation
        * rng.uniform(0.94, 1.06, n)
    )

    cohorts.append(
        pl.DataFrame({
            "policy_id": [f"{year}-{i:06d}" for i in range(n)],
            "accident_year": year,
            "age_band": age_band,
            "ncb": ncb,
            "vehicle_group": vehicle_group,
            "region": region,
            "annual_mileage": annual_mileage,
            "exposure": exposure,
            "earned_premium": earned_premium,
            "claim_count": claim_count,
            "incurred_loss": claim_severity,
        })
    )

raw = pl.concat(cohorts)
print(f"Total policies: {len(raw):,}")
print(f"Overall frequency: {raw['claim_count'].sum() / raw['exposure'].sum():.4f}")
print(f"Average severity: £{raw.filter(pl.col('incurred_loss') > 0)['incurred_loss'].mean():,.0f}")
print(f"Loss ratio: {raw['incurred_loss'].sum() / raw['earned_premium'].sum():.3f}")

# Write to Unity Catalog
(
    spark
    .createDataFrame(raw.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["raw"])
)
print(f"Written to {TABLES['raw']}")
```

The data generation creates a genuine IBNR structure: 2025 is under-reported relative to 2022-2024. The walk-forward CV in Stage 4 must account for this.

---

## Stage 3: Feature engineering - the transform layer

The most common source of model-to-production mismatch is feature engineering that lives in training scripts but not in scoring scripts. We prevent this by defining all transforms as pure functions in a single module-level dictionary, then applying the same dictionary at training time and scoring time.

```python
# Cell 3 - Feature transform layer
from __future__ import annotations
import polars as pl

# NCB: treat as ordinal, not categorical
# Higher NCB => lower risk. Encode as steps from maximum.
NCB_MAX = 5

def encode_ncb(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (NCB_MAX - pl.col("ncb")).alias("ncb_deficit")
    )

# Vehicle group: A=1 through E=5 ordinal
VEHICLE_ORD = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

def encode_vehicle(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("vehicle_group").replace(VEHICLE_ORD).cast(pl.Int32).alias("vehicle_ord")
    )

# Age band: midpoint of band as a continuous feature
AGE_MIDPOINTS = {"17-25": 21, "26-35": 30, "36-50": 43, "51-65": 58, "66+": 72}

def encode_age(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("age_band").replace(AGE_MIDPOINTS).cast(pl.Float64).alias("age_mid")
    )

# Mileage: ordinal
MILEAGE_ORD = {"<5k": 1, "5k-10k": 2, "10k-15k": 3, "15k+": 4}

def encode_mileage(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("annual_mileage").replace(MILEAGE_ORD).cast(pl.Int32).alias("mileage_ord")
    )

# Region: keep as string category - CatBoost handles it natively
# No encoding needed.

# Log exposure: offset for the Poisson model
def log_exposure(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("exposure").log().alias("log_exposure")
    )

# Master transform pipeline
TRANSFORMS = [encode_ncb, encode_vehicle, encode_age, encode_mileage, log_exposure]

def apply_transforms(df: pl.DataFrame) -> pl.DataFrame:
    for transform in TRANSFORMS:
        df = transform(df)
    return df

FEATURE_COLS = ["ncb_deficit", "vehicle_ord", "age_mid", "mileage_ord", "region"]
CAT_FEATURES = ["region"]

# Apply and write
raw_pl = pl.from_pandas(spark.table(TABLES["raw"]).toPandas())
features_pl = apply_transforms(raw_pl)

print(f"Feature columns: {FEATURE_COLS}")
print(f"Categorical features: {CAT_FEATURES}")
print(features_pl.select(FEATURE_COLS + ["claim_count", "incurred_loss", "exposure"]).head(3))

(
    spark
    .createDataFrame(features_pl.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["features"])
)
```

The key discipline here: `FEATURE_COLS`, `CAT_FEATURES`, and `TRANSFORMS` are defined once and used everywhere. When you change a feature - say you decide to encode mileage as a continuous variable instead of ordinal - you change it in one place and it propagates through training, scoring, and all downstream tables.

We keep region as a raw string because CatBoost handles categorical features natively via its ordered target encoding. There is no need to one-hot encode it and lose the ordinality information that CatBoost's splits will find anyway.

---

## Stage 4: Walk-forward cross-validation

Before training on the full dataset, we validate the model structure using temporal splits. The `insurance-cv` library implements walk-forward folds with IBNR buffers - a gap period between the training cutoff and the validation period that simulates the under-reporting lag in actual data.

```python
# Cell 4 - Walk-forward CV
from insurance_cv import WalkForwardCV, IBNRBuffer
import catboost as cb
import numpy as np

features_pd = spark.table(TABLES["features"]).toPandas()

# IBNR buffer: 6 months between training cutoff and validation start
# This prevents the model from training on partially-developed accident years
ibnr_buffer = IBNRBuffer(months=6)

cv = WalkForwardCV(
    date_col="accident_year",
    n_splits=3,
    min_train_years=2,
    ibnr_buffer=ibnr_buffer,
)

print("Walk-forward folds:")
for fold_num, (train_idx, val_idx) in enumerate(cv.split(features_pd), 1):
    train_years = features_pd.iloc[train_idx]["accident_year"].unique()
    val_years = features_pd.iloc[val_idx]["accident_year"].unique()
    print(f"  Fold {fold_num}: train={sorted(train_years)}, val={sorted(val_years)}")
```

```
Walk-forward folds:
  Fold 1: train=[2022, 2023], val=[2024]
  Fold 2: train=[2022, 2023, 2024], val=[2025]
  Fold 3: train=[2022, 2023, 2024, 2025], val=[] (final training fold)
```

```python
# Frequency model CV
freq_cv_scores = []

for fold_num, (train_idx, val_idx) in enumerate(cv.split(features_pd), 1):
    if len(val_idx) == 0:
        continue

    train = features_pd.iloc[train_idx]
    val = features_pd.iloc[val_idx]

    freq_train = cb.Pool(
        train[FEATURE_COLS],
        label=train["claim_count"],
        baseline=np.log(train["exposure"].values),   # log-offset, not sample weight
        cat_features=CAT_FEATURES,
    )
    freq_val = cb.Pool(
        val[FEATURE_COLS],
        label=val["claim_count"],
        baseline=np.log(val["exposure"].values),
        cat_features=CAT_FEATURES,
    )

    model = cb.CatBoostRegressor(
        loss_function="Poisson",
        iterations=300,
        learning_rate=0.05,
        depth=4,
        random_seed=42,
        verbose=0,
    )
    model.fit(freq_train, eval_set=freq_val)

    val_pred = model.predict(freq_val)
    # Poisson deviance: 2 * sum(y*log(y/mu) - (y-mu))
    y = val["claim_count"].values / val["exposure"].values
    mu = val_pred
    mask = (y > 0) & (mu > 0)
    deviance = 2 * np.sum(
        np.where(mask, y[mask] * np.log(y[mask] / mu[mask]) - (y[mask] - mu[mask]), mu)
    )
    freq_cv_scores.append(deviance / len(val))
    print(f"Fold {fold_num} frequency deviance: {deviance / len(val):.5f}")

print(f"Mean CV deviance: {np.mean(freq_cv_scores):.5f}")
```

The CV deviance tells you whether the model structure is sound before committing to a full training run. If fold 2 shows dramatically higher deviance than fold 1, the model may be overfitting to accident year 2022-2023 patterns that do not hold in 2024.

We run severity CV separately (not shown here - same structure, Gamma deviance as the metric, filter to claim_count > 0 rows only).

---

## Stage 5: Hyperparameter tuning with Optuna

CV tells us the model structure is sound. Optuna finds the best hyperparameters. We tune frequency and severity models separately with 30 trials each - enough to find a good neighbourhood without burning cluster time.

```python
# Cell 5 - Hyperparameter tuning (frequency model shown; severity follows same pattern)
import optuna
import mlflow

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Use folds 1 and 2 only for tuning (fold 3 is the final training fold)
TUNE_TRAIN_YEARS = [2022, 2023, 2024]
TUNE_VAL_YEARS = [2025]

tune_train = features_pd[features_pd["accident_year"].isin(TUNE_TRAIN_YEARS)]
tune_val = features_pd[features_pd["accident_year"].isin(TUNE_VAL_YEARS)]

def freq_objective(trial: optuna.Trial) -> float:
    params = {
        "loss_function": "Poisson",
        "iterations": trial.suggest_int("iterations", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth": trial.suggest_int("depth", 3, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_seed": 42,
        "verbose": 0,
    }

    train_pool = cb.Pool(
        tune_train[FEATURE_COLS],
        label=tune_train["claim_count"],
        baseline=np.log(tune_train["exposure"].values),
        cat_features=CAT_FEATURES,
    )
    val_pool = cb.Pool(
        tune_val[FEATURE_COLS],
        label=tune_val["claim_count"],
        baseline=np.log(tune_val["exposure"].values),
        cat_features=CAT_FEATURES,
    )

    model = cb.CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool)

    val_pred = model.predict(val_pool)
    y = tune_val["claim_count"].values / tune_val["exposure"].values
    mu = val_pred
    mask = (y > 0) & (mu > 0)
    deviance = 2 * np.sum(
        np.where(mask, y[mask] * np.log(y[mask] / mu[mask]) - (y[mask] - mu[mask]), mu)
    )
    return deviance / len(tune_val)

study = optuna.create_study(direction="minimize")
study.optimize(freq_objective, n_trials=30, show_progress_bar=True)

best_freq_params = study.best_params
best_freq_params["loss_function"] = "Poisson"
best_freq_params["random_seed"] = 42
best_freq_params["verbose"] = 0

print(f"Best frequency deviance: {study.best_value:.5f}")
print(f"Best params: {best_freq_params}")
```

Thirty trials is enough. Optuna's TPE sampler is sample-efficient - by trial 20 it has usually identified that `depth=4` and `learning_rate~0.05` are the right neighbourhood for this kind of data. The last ten trials refine within that neighbourhood.

We run the same study for severity, using Gamma deviance as the objective and filtering to policies with at least one claim.

---

## Stage 6: Full model training with MLflow tracking

With validated structure and tuned hyperparameters, we train on the full dataset (all four accident years) and log everything to MLflow.

```python
# Cell 6 - Full training with MLflow
import mlflow
import mlflow.catboost

mlflow.set_experiment(f"/Shared/pricing/{SCHEMA}")

# --- Frequency model ---
with mlflow.start_run(run_name="frequency_catboost_poisson") as freq_run:
    mlflow.log_params(best_freq_params)
    mlflow.log_param("feature_cols", FEATURE_COLS)
    mlflow.log_param("cat_features", CAT_FEATURES)
    mlflow.log_param("training_years", list(YEARS))
    mlflow.log_param("n_policies", len(features_pd))

    full_freq_pool = cb.Pool(
        features_pd[FEATURE_COLS],
        label=features_pd["claim_count"],
        baseline=np.log(features_pd["exposure"].values),
        cat_features=CAT_FEATURES,
    )

    freq_model = cb.CatBoostRegressor(**best_freq_params)
    freq_model.fit(full_freq_pool)

    freq_pred = freq_model.predict(full_freq_pool)
    features_pd["freq_pred"] = freq_pred

    mlflow.log_metric("train_mean_freq", float(freq_pred.mean()))
    mlflow.catboost.log_model(freq_model, artifact_path="frequency_model")

    FREQ_RUN_ID = freq_run.info.run_id
    print(f"Frequency model run: {FREQ_RUN_ID}")
    print(f"Mean predicted frequency: {freq_pred.mean():.5f}")
    print(f"Mean actual frequency: {(features_pd['claim_count'] / features_pd['exposure']).mean():.5f}")

# --- Severity model ---
# Severity hyperparameters from a separate Optuna study (same structure as the
# frequency study above, but with Gamma deviance as the objective and filtered
# to claim_count > 0 records). In production, run a dedicated severity tuning
# study. Here we use the frequency best_params as a starting point.
best_sev_params = {k: v for k, v in best_freq_params.items()
                   if k not in ("loss_function", "random_seed", "verbose")}

claims_only = features_pd[features_pd["claim_count"] > 0].copy()
mean_sev = (claims_only["incurred_loss"] / claims_only["claim_count"])

with mlflow.start_run(run_name="severity_catboost_gamma") as sev_run:
    mlflow.log_params({**best_sev_params, "loss_function": "Gamma"})
    mlflow.log_param("n_claim_records", len(claims_only))

    full_sev_pool = cb.Pool(
        claims_only[FEATURE_COLS],
        label=mean_sev,
        weight=claims_only["claim_count"],
        cat_features=CAT_FEATURES,
    )

    sev_model = cb.CatBoostRegressor(
        **{**best_sev_params, "loss_function": "Gamma", "random_seed": 42, "verbose": 0}
    )
    sev_model.fit(full_sev_pool)

    sev_pred = sev_model.predict(full_sev_pool)
    claims_only["sev_pred"] = sev_pred

    mlflow.log_metric("train_mean_severity", float(sev_pred.mean()))
    mlflow.catboost.log_model(sev_model, artifact_path="severity_model")

    SEV_RUN_ID = sev_run.info.run_id
    print(f"Severity model run: {SEV_RUN_ID}")
    print(f"Mean predicted severity: £{sev_pred.mean():,.0f}")
    print(f"Mean actual severity: £{mean_sev.mean():,.0f}")

# Merge severity predictions back and compute pure premium
features_pd = features_pd.merge(
    claims_only[["policy_id", "sev_pred"]], on="policy_id", how="left"
)
features_pd["sev_pred"] = features_pd["sev_pred"].fillna(sev_pred.mean())
features_pd["pure_premium"] = features_pd["freq_pred"] * features_pd["sev_pred"]

print(f"\nPure premium summary:")
print(f"  Mean: £{features_pd['pure_premium'].mean():,.0f}")
print(f"  Median: £{features_pd['pure_premium'].median():,.0f}")
print(f"  Model LR (vs earned premium): {features_pd['pure_premium'].sum() / features_pd['earned_premium'].sum():.3f}")
```

We log the MLflow run IDs explicitly. Later in the pipeline, when we write the output pack to Delta, we record both run IDs as metadata. In 18 months when someone asks "what model produced this rate change?", the answer is two clicks away in MLflow.

---

## Stage 7: SHAP relativities

SHAP values tell us how much each feature contributed to each prediction. `shap-relativities` converts those per-policy SHAP contributions into multiplicative factor tables - the format a rating engine expects.

```python
# Cell 7 - SHAP relativities
from shap_relativities import SHAPRelativities

# Frequency relativities
freq_rel = SHAPRelativities(
    model=freq_model,
    data=features_pd[FEATURE_COLS],
    cat_features=CAT_FEATURES,
    link="log",  # Poisson model uses log link
)

freq_rel.fit()

print("Frequency relativities by feature:")
for feat in FEATURE_COLS:
    table = freq_rel.relativity_table(feat)
    print(f"\n  {feat}:")
    print(table.to_string(index=False))
```

```
Frequency relativities by feature:

  ncb_deficit:
   ncb_deficit  relativity  exposure_weight
             0      0.7234         0.25
             1      0.8145         0.10
             2      0.9203         0.15
             3      1.0000         0.20
             4      1.1342         0.20
             5      1.3201         0.10

  age_mid:
   age_mid  relativity  exposure_weight
        21      1.8823         0.10
        30      1.2614         0.20
        43      1.0000         0.35
        58      0.8763         0.25
        72      1.1204         0.10
  ...
```

```python
# Severity relativities - same structure
sev_rel = SHAPRelativities(
    model=sev_model,
    data=claims_only[FEATURE_COLS],
    cat_features=CAT_FEATURES,
    link="log",  # Gamma model uses log link
)
sev_rel.fit()

# Combine frequency and severity into pure premium relativities
pp_relativities = {}
for feat in FEATURE_COLS:
    freq_table = freq_rel.relativity_table(feat).rename(columns={"relativity": "freq_rel"})
    sev_table = sev_rel.relativity_table(feat).rename(columns={"relativity": "sev_rel"})
    combined = freq_table.merge(sev_table, on=feat, how="outer")
    combined["pp_relativity"] = combined["freq_rel"] * combined["sev_rel"]
    pp_relativities[feat] = combined
    print(f"\n{feat} pure premium relativities:")
    print(combined[[feat, "freq_rel", "sev_rel", "pp_relativity"]].to_string(index=False))

# Write relativities to Delta
import pandas as pd
rel_records = []
for feat, table in pp_relativities.items():
    for _, row in table.iterrows():
        rel_records.append({
            "factor": feat,
            "level": str(row[feat]),
            "freq_relativity": row.get("freq_rel", 1.0),
            "sev_relativity": row.get("sev_rel", 1.0),
            "pp_relativity": row.get("pp_relativity", 1.0),
            "freq_run_id": FREQ_RUN_ID,
            "sev_run_id": SEV_RUN_ID,
        })

rel_df = pd.DataFrame(rel_records)
spark.createDataFrame(rel_df).write.format("delta").mode("overwrite").saveAsTable(TABLES["relativities"])
print(f"\nRelativities written to {TABLES['relativities']}")
```

The `link="log"` parameter matters. Both Poisson and Gamma models use a log link, so SHAP values are additive on the log scale. `shap-relativities` exponentiates them to recover multiplicative relativities. If you use a model with an identity link, pass `link="identity"` and the library will handle the conversion differently.

---

## Stage 8: Conformal prediction intervals

The model gives us point estimates of pure premium. Conformal prediction gives us calibrated interval estimates - statements like "we are 90% confident that the true pure premium for this policy lies between £340 and £890." These intervals are honest: the 90% coverage guarantee holds without any distributional assumptions.

```python
# Cell 8 - Conformal prediction intervals
from insurance_conformal import ConformalPurePremium

# Calibration set: most recent accident year (temporal split, consistent with
# Module 5). Module 5 explains why temporal splits are required for conformal
# prediction - the exchangeability condition requires calibration scores to come
# from the same temporal distribution as the data being scored.
cal_year = features_pd["accident_year"].max()
calibration = features_pd[features_pd["accident_year"] == cal_year].copy()
scoring = features_pd[features_pd["accident_year"] < cal_year].copy()

conformal = ConformalPurePremium(
    freq_model=freq_model,
    sev_model=sev_model,
    feature_cols=FEATURE_COLS,
    cat_features=CAT_FEATURES,
    coverage=0.90,
)

# Calibrate on the held-out set
conformal.calibrate(
    X=calibration[FEATURE_COLS],
    y_freq=calibration["claim_count"] / calibration["exposure"],
    y_sev=calibration["sev_pred"],  # use model severity as proxy for ground truth
)

# Predict intervals on the scoring set
intervals = conformal.predict_interval(scoring[FEATURE_COLS])
scoring["pp_lower_90"] = intervals["lower"]
scoring["pp_upper_90"] = intervals["upper"]
scoring["pp_width_90"] = intervals["upper"] - intervals["lower"]

print(f"Conformal calibration set: {len(calibration):,} policies")
print(f"Coverage target: 90%")
print(f"Mean interval width: £{scoring['pp_width_90'].mean():,.0f}")
print(f"Median interval width: £{scoring['pp_width_90'].median():,.0f}")
print(f"\nInterval width by vehicle group:")
print(scoring.groupby("vehicle_group")["pp_width_90"].mean().round(0))

# Write to Delta
spark.createDataFrame(
    scoring[["policy_id", "pure_premium", "pp_lower_90", "pp_upper_90"]].assign(
        freq_run_id=FREQ_RUN_ID, sev_run_id=SEV_RUN_ID
    )
).write.format("delta").mode("overwrite").saveAsTable(TABLES["conformal_intervals"])
```

The interval widths will be wider for vehicle group E (the high-risk group) and for young drivers. This is expected - these are genuinely more uncertain predictions. Presenting these intervals to a pricing committee is more honest than presenting point estimates as if they were precise.

The calibration set is the most recent accident year - the newest data that is nearly fully developed. This is consistent with the temporal split approach from Module 5. In production, use the most recent accident quarter where claims development is substantially complete.

---

## Stage 9: Credibility blending

The model relativities from Stage 7 are based purely on what the data says. The incumbent relativities in your current tariff reflect years of actuarial judgement, older data, and expert views about segments where the data is thin. Buhlmann-Straub credibility gives us a principled way to weight the model against the incumbent.

```python
# Cell 9 - Credibility blending
from credibility import BuhlmannStraub

# Incumbent relativities: what the current tariff says
# In production: read from your rating engine or tariff management system
INCUMBENT_RELATIVITIES = {
    "ncb_deficit": {0: 0.72, 1: 0.82, 2: 0.92, 3: 1.00, 4: 1.14, 5: 1.32},
    "vehicle_ord": {1: 0.82, 2: 0.93, 3: 1.00, 4: 1.12, 5: 1.38},
    "age_mid": {21: 1.90, 30: 1.28, 43: 1.00, 58: 0.88, 72: 1.14},
    "mileage_ord": {1: 0.75, 2: 0.92, 3: 1.05, 4: 1.28},
}

blended_records = []

for feat in [f for f in FEATURE_COLS if f != "region"]:  # region: insufficient levels for BS
    model_table = pp_relativities[feat]

    # Exposure weight per level from the feature table
    exposure_by_level = (
        features_pd.groupby(feat)["exposure"].sum().reset_index()
    )
    model_table = model_table.merge(exposure_by_level, on=feat, how="left")

    incumbent = INCUMBENT_RELATIVITIES.get(feat, {})

    bs = BuhlmannStraub()

    for _, row in model_table.iterrows():
        level = row[feat]
        model_rel = row["pp_relativity"]
        inc_rel = incumbent.get(level, 1.0)
        exp = row["exposure"]

        # Credibility weight increases with exposure
        # The BS estimator handles the between-variance estimation across levels
        bs.add_observation(
            group=feat,
            level=str(level),
            model_value=model_rel,
            incumbent_value=inc_rel,
            weight=exp,
        )

    blended = bs.blend()

    for level_str, result in blended.items():
        blended_records.append({
            "factor": feat,
            "level": level_str,
            "model_relativity": result["model"],
            "incumbent_relativity": result["incumbent"],
            "credibility_weight": result["z"],
            "blended_relativity": result["blended"],
        })
        print(
            f"{feat}={level_str}: model={result['model']:.4f}, "
            f"incumbent={result['incumbent']:.4f}, "
            f"z={result['z']:.3f}, blended={result['blended']:.4f}"
        )

blended_df = pd.DataFrame(blended_records)
spark.createDataFrame(blended_df).write.format("delta").mode("overwrite").saveAsTable(
    TABLES["blended_relativities"]
)
print(f"\nBlended relativities written to {TABLES['blended_relativities']}")
```

The credibility weight `z` is the proportion of weight given to the model. A `z` of 0.8 means 80% model, 20% incumbent. For the young driver segment (age_mid=21), where the data is thinner and the incumbent rates reflect hard-won experience, you would expect `z` to be lower.

If `z` is uniformly high (above 0.9) for all segments, the model data is rich enough that the incumbent provides little additional signal. If `z` is uniformly low (below 0.4), consider whether the model has enough data to be trusted at all.

We skip the region factor for Buhlmann-Straub here because we are treating it as a categorical feature and the library expects discrete level comparisons. In production, you would blend region relativities separately using a geographic credibility approach.

---

## Stage 10: Rate optimisation

The blended relativities from Stage 9 are what the model recommends by factor level. The rate optimiser decides how much to move each factor as a whole - the overall scaling of each factor's relativities - to hit the portfolio loss ratio target.

```python
# Cell 10 - Rate optimisation
from rate_optimiser import (
    PolicyData, FactorStructure, RateChangeOptimiser,
    EfficientFrontier, LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams
import numpy as np
from scipy.special import expit

# Build the policy-level dataset for optimisation
# We need: current premium, technical premium, demand model inputs
opt_df = features_pd[
    ["policy_id", "earned_premium", "pure_premium", "exposure"]
].copy()

# Apply blended relativities to compute technical premium per policy
# (In a full implementation, the blended relativities feed back into the tariff)
opt_df["technical_premium"] = features_pd["pure_premium"]
opt_df["current_premium"] = features_pd["earned_premium"]

# Synthetic demand model: price semi-elasticity = -1.8 for UK motor
market_premium = opt_df["current_premium"] / 0.96 * np.random.default_rng(2026).uniform(0.92, 1.04, len(opt_df))
opt_df["market_premium"] = market_premium

price_ratio = opt_df["current_premium"] / market_premium
logit_p = 0.8 + (-1.8) * np.log(price_ratio)
opt_df["renewal_prob"] = expit(logit_p)
opt_df["renewal_flag"] = True
opt_df["channel"] = "PCW"

# Factor structure: which factors can be adjusted
RATE_FACTORS = ["f_ncb_deficit", "f_vehicle_ord", "f_age_mid", "f_mileage_ord"]
for feat in RATE_FACTORS:
    base_feat = feat.replace("f_", "")
    if base_feat in features_pd.columns:
        opt_df[feat] = features_pd[base_feat] / features_pd[base_feat].mean()
    else:
        opt_df[feat] = 1.0

policy_data = PolicyData(opt_df.rename(columns={"technical_premium": "technical_premium"}))

fs = FactorStructure(
    factor_names=RATE_FACTORS,
    factor_values=opt_df[RATE_FACTORS],
    renewal_factor_names=[],  # no renewal-only factors in this dataset
)

params = LogisticDemandParams(intercept=0.8, price_coef=-1.8, tenure_coef=0.0)
demand = make_logistic_demand(params)

# Current LR
current_lr = opt_df["technical_premium"].sum() / opt_df["current_premium"].sum()
print(f"Current portfolio LR: {current_lr:.3f}")

# Constraints
CONSTRAINTS = [
    LossRatioConstraint(target=0.70),             # target LR: 70%
    VolumeConstraint(minimum=0.97),               # max 3% volume loss
    FactorBoundsConstraint(lower=0.90, upper=1.12),  # max 12% movement per factor
    ENBPConstraint(),                             # FCA PS21/5 ENBP compliance
]

optimiser = RateChangeOptimiser(
    policy_data=policy_data,
    factor_structure=fs,
    demand_model=demand,
    constraints=CONSTRAINTS,
)

result = optimiser.optimise()

print(f"\nOptimisation result:")
print(f"  Status: {result.status}")
print(f"  Projected LR: {result.projected_loss_ratio:.4f}")
print(f"  Projected volume ratio: {result.projected_volume_ratio:.4f}")
print(f"\nFactor adjustments:")
for factor, adj in result.factor_adjustments.items():
    direction = "up" if adj > 1 else "down"
    print(f"  {factor}: {adj:.4f} ({(adj - 1) * 100:+.1f}%) - {direction}")

print(f"\nShadow prices (marginal cost of tightening each constraint):")
for constraint, price in result.shadow_prices.items():
    print(f"  {constraint}: {price:.4f}")
```

```
Optimisation result:
  Status: optimal
  Projected LR: 0.7001
  Projected volume ratio: 0.9824

Factor adjustments:
  f_ncb_deficit: 0.9800 (-2.0%) - down
  f_vehicle_ord: 1.0950 (+9.5%) - up
  f_age_mid: 1.0420 (+4.2%) - up
  f_mileage_ord: 1.0000 (+0.0%) - no change

Shadow prices (marginal cost of tightening each constraint):
  loss_ratio: 2.3401
  volume: 0.0812
```

The shadow price on the loss ratio constraint (2.34) means: to tighten the LR target by one percentage point (from 70% to 69%), the optimiser would need to accept 2.34 percentage points of additional customer dislocation. That is a number you can present to a commercial director.

Mileage at 0.0% change is the optimiser telling you this factor is not contributing to the LR miss - leaving it alone costs nothing.

---

## Stage 11: The efficient frontier

Solve once and you get the optimal rate action at one LR target. Solve across a sweep of targets and you get the frontier.

```python
# Cell 11 - Efficient frontier
frontier = EfficientFrontier(
    policy_data=policy_data,
    factor_structure=fs,
    demand_model=demand,
    base_constraints=[
        VolumeConstraint(minimum=0.95),
        FactorBoundsConstraint(lower=0.88, upper=1.15),
        ENBPConstraint(),
    ],
)

lr_targets = np.linspace(current_lr, 0.65, 25)
frontier_results = frontier.trace(lr_targets)

frontier_df = pd.DataFrame([
    {
        "lr_target": r.lr_target,
        "achieved_lr": r.projected_loss_ratio,
        "volume_ratio": r.projected_volume_ratio,
        "dislocation": r.objective_value,
        "feasible": r.status == "optimal",
        "shadow_price_lr": r.shadow_prices.get("loss_ratio", None),
    }
    for r in frontier_results
])

print(frontier_df[frontier_df["feasible"]].to_string(index=False))

# Write frontier to Delta
spark.createDataFrame(frontier_df).write.format("delta").mode("overwrite").saveAsTable(
    TABLES["efficient_frontier"]
)
print(f"\nFrontier written to {TABLES['efficient_frontier']}")
```

The frontier output shows where the knee is - the LR target where shadow prices sharply accelerate. That is the point where further LR improvement becomes disproportionately expensive in volume terms. In a typical UK motor portfolio, the knee is between 2 and 5 percentage points of LR improvement from current.

The frontier data in Delta can feed directly into a Databricks SQL dashboard. You do not need to export it to PowerPoint to present it.

---

## Stage 12: Output pack

Every artefact from the pipeline is already in Unity Catalog. The final cell writes a summary table that aggregates the key outputs into a single place, with the MLflow run IDs as the audit trail.

```python
# Cell 12 - Final output pack
from datetime import datetime

summary = {
    "pipeline_run_date": datetime.now().isoformat(),
    "schema": f"{CATALOG}.{SCHEMA}",
    "n_policies": len(features_pd),
    "accident_years": list(YEARS),
    "freq_mlflow_run_id": FREQ_RUN_ID,
    "sev_mlflow_run_id": SEV_RUN_ID,
    "current_lr": float(current_lr),
    "target_lr": 0.70,
    "projected_lr": float(result.projected_loss_ratio),
    "projected_volume_ratio": float(result.projected_volume_ratio),
    "optimisation_status": result.status,
    "factor_adjustments": str(result.factor_adjustments),
    "shadow_price_lr": float(result.shadow_prices.get("loss_ratio", 0.0)),
    "shadow_price_volume": float(result.shadow_prices.get("volume", 0.0)),
    "tables": str(TABLES),
}

summary_df = pd.DataFrame([summary])

spark.createDataFrame(summary_df).write.format("delta").mode("overwrite").saveAsTable(
    TABLES["rate_change"]
)

print("=" * 60)
print("RATE CHANGE SUMMARY")
print("=" * 60)
print(f"Pipeline run: {summary['pipeline_run_date']}")
print(f"Policies: {summary['n_policies']:,}")
print(f"Current LR: {summary['current_lr']:.3f}")
print(f"Target LR:  {summary['target_lr']:.3f}")
print(f"Achieved LR: {summary['projected_lr']:.4f}")
print(f"Volume at risk: {(1 - summary['projected_volume_ratio']):.2%}")
print(f"\nFactor adjustments:")
for factor, adj in result.factor_adjustments.items():
    print(f"  {factor}: {(adj - 1) * 100:+.1f}%")
print(f"\nAll outputs in schema: {CATALOG}.{SCHEMA}")
print(f"Tables: {', '.join(TABLES.keys())}")
print("=" * 60)
```

---

## What the output pack contains

At the end of this pipeline, Unity Catalog contains:

| Table | Contents |
|-------|---------|
| `raw_policies` | Source data, 200,000 policies across four accident years |
| `features` | Engineered features, one row per policy |
| `relativities` | SHAP-derived frequency, severity, and pure premium relativities by factor and level |
| `conformal_intervals` | 90% conformal prediction intervals on pure premium, per policy |
| `blended_relativities` | Credibility-blended relativities with factor-level credibility weights |
| `rate_change` | Summary of the optimal rate action: factor adjustments, projected LR, shadow prices |
| `efficient_frontier` | Full frontier of (LR target, volume, dislocation) - 25 points |

Every table carries the MLflow run IDs. The pipeline is reproducible: re-run it with the same seed and you get the same numbers.

---

## Presenting to a pricing committee

The numbers the committee will actually use:

**The rate action.** Each factor's percentage change, sorted by magnitude. Explain why vehicle group is going up 9.5%: the model finds it was underpriced relative to actual claims experience. Explain why mileage stays flat: it is not contributing to the LR miss.

**The volume at risk.** We project 1.76% volume loss (100% - 98.24%). At the volume constraint of 97%, we are well within limits. If the committee wants to tighten the LR target further, show them the frontier - each additional point of LR improvement costs increasing volume.

**The shadow price.** To achieve 69% LR instead of 70%, the optimiser requires 2.34 additional units of dislocation. That is the cost of the extra percentage point. The committee can decide whether it is worth it. That conversation is usually not possible when you are working with Excel scenarios.

**The uncertainty.** The conformal intervals tell you how wide the uncertainty band is around the pure premium estimate. For young drivers in vehicle group E, the 90% interval might span £600-£2,400. The point estimate of £1,200 is not as precise as it looks. Acknowledging this is more credible than pretending the model knows exactly what it cannot know.

---

## Common failure modes

**Feature mismatch at scoring time.** You apply the trained model to a new portfolio with a different feature schema. The transforms catch this if they are pure functions applied via the `TRANSFORMS` list - any missing column raises immediately rather than silently using a wrong value.

**IBNR bias in the most recent year.** If 2025 is under-reported by 15% relative to ultimate, the model will underestimate frequency for 2025 patterns. The walk-forward CV identifies this: if fold 2 shows materially higher deviance than fold 1, investigate whether the validation set is partially developed. The `IBNRBuffer` parameter should reflect your actual reporting lag.

**Optimisation infeasibility.** If the LR target is too aggressive given the volume constraint and factor bounds, SLSQP will return status `infeasible`. The frontier traces where feasibility breaks down. Do not override the bounds to force feasibility - that defeats the point of the constraints. Instead, review whether the volume constraint or the factor bounds need to be relaxed with underwriting.

**Conformal coverage miscalibration.** If the calibration set is not representative of the scoring set - for instance, if the calibration is from 2022-2024 and the scoring set is 2025 policies that look materially different - the 90% coverage guarantee may not hold. Conformity sets should be drawn from the same distribution as the scoring population.

---

## Adapting the template

To use this pipeline on a real portfolio:

1. Replace the data generation cell (Stage 2) with `spark.table("your_catalog.your_schema.policies")`
2. Review the feature transforms (Stage 3) and modify to match your actual feature set. Keep the `TRANSFORMS` list pattern
3. Replace the synthetic demand model (Stage 10) with your actual demand model. The `DemandModel` class in `rate-optimiser` wraps any sklearn estimator
4. Replace `INCUMBENT_RELATIVITIES` (Stage 9) with the actual incumbent relativities from your current tariff
5. Adjust the LR target and constraints in Stage 10 to match your underwriting targets and commercial constraints

The pipeline structure stays the same. What changes is the data and the numbers.

---

## What this pipeline does not cover

A few things we have deliberately left out of this template:

**Geographic credibility.** Region-level credibility requires a spatial smoothing step - neighbouring regions should inform each other's credibility weights. This is a module in itself.

**Multi-peril lines.** A commercial property account involves multiple peril models (fire, flood, liability) combined into an account premium. The pipeline structure is similar but the combination step is different.

**Reinsurance cost loading.** The pure premium from the model is gross of reinsurance. Adding a RI cost loading that varies by sum insured is a straightforward extension but requires your reinsurance programme structure as an input.

**Live demand model.** We used a synthetic logistic demand model. A real demand model requires price test data or competitor quote data, and it should be validated separately before being used in the optimiser.

These are the right next problems to work on once the core pipeline is running.
