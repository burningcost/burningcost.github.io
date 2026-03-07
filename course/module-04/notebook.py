# Databricks notebook source
# MAGIC %md
# MAGIC # Module 4: SHAP Relativities — From GBM to Rating Factor Tables
# MAGIC
# MAGIC Full workflow on synthetic UK motor data. Runs end-to-end on a single-node
# MAGIC Databricks cluster (DBR 14.x, ML runtime recommended).
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Generates a 50,000-policy synthetic UK motor portfolio with known true parameters
# MAGIC 2. Trains a Poisson frequency GBM and a Gamma severity GBM (CatBoost)
# MAGIC 3. Extracts multiplicative relativities from both models using SHAP
# MAGIC 4. Validates extraction numerically
# MAGIC 5. Compares to a benchmark Poisson GLM
# MAGIC 6. Produces factor tables and continuous curves
# MAGIC 7. Exports in Radar-compatible CSV format
# MAGIC 8. Writes results to Delta tables (Unity Catalog)
# MAGIC
# MAGIC Runtime: ~10 minutes on a small cluster (4 cores).
# MAGIC
# MAGIC Databricks Free Edition is sufficient for running the exercises.

# COMMAND ----------

# MAGIC %pip install "shap-relativities[all]" catboost polars statsmodels --quiet

# COMMAND ----------

# Restart the Python interpreter after pip install so imports work cleanly
dbutils.library.restartPython()

# COMMAND ----------

import warnings
import json
from datetime import date

import numpy as np
import polars as pl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import catboost as cb
import statsmodels.api as sm
import statsmodels.formula.api as smf

from shap_relativities import SHAPRelativities
from shap_relativities.datasets.motor import load_motor, TRUE_FREQ_PARAMS, TRUE_SEV_PARAMS

print("shap-relativities imported successfully")
print(f"CatBoost version: {cb.__version__}")
print(f"Polars version: {pl.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic data

# COMMAND ----------

# load_motor returns a Polars DataFrame
df = load_motor(n_policies=50_000, seed=42)

print(f"Portfolio: {df.height:,} policies")
print(f"Exposure: {df['exposure'].sum():.0f} earned years")
print(f"Claims: {df['claim_count'].sum():,} ({df['claim_count'].sum() / df['exposure'].sum():.3f} per earned year)")
print(f"Incurred: £{df['incurred'].sum() / 1e6:.1f}m")
print()
print("Area distribution:")
print(
    df.group_by("area")
    .agg(pl.col("exposure").sum().alias("earned_years"))
    .sort("area")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### True DGP parameters
# MAGIC
# MAGIC These are the parameters used to generate the data. A correctly specified model
# MAGIC should recover these.

# COMMAND ----------

print("True frequency parameters:")
for k, v in TRUE_FREQ_PARAMS.items():
    if "area" in k:
        print(f"  {k}: log-relativity = {v:.3f}, relativity = {np.exp(v):.3f}")
    elif "ncd" in k:
        print(f"  {k}: {v:.3f} per year → NCD=5 relativity = {np.exp(5*v):.3f}")
    else:
        print(f"  {k}: {v:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature engineering

# COMMAND ----------

# All feature engineering in Polars
df = df.with_columns([
    (pl.col("conviction_points") > 0).cast(pl.Int8).alias("has_convictions"),
    pl.col("annual_mileage").log().alias("log_mileage"),
])

# Age bands for the banded analysis (Step 7) — stored for grouping later
age_bins = [17, 22, 25, 30, 40, 55, 70, 86]
age_labels = ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"]

df = df.with_columns(
    pl.col("driver_age").cut(
        breaks=age_bins[1:-1],
        labels=age_labels,
    ).alias("age_band")
)

print("Feature matrix preview (Polars):")
print(
    df.select([
        "area", "ncd_years", "has_convictions", "vehicle_group",
        "driver_age", "log_mileage", "claim_count", "exposure",
    ]).head(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train Poisson frequency model (CatBoost)
# MAGIC
# MAGIC CatBoost advantages for this workflow:
# MAGIC - **Native categoricals**: `area` is passed as a string. No ordinal encoding.
# MAGIC - **Native SHAP**: `get_feature_importance(type='ShapValues')` — no external shap library call needed.
# MAGIC - **Exposure offset**: use `baseline=log(exposure)` in the Pool. Do NOT also set
# MAGIC   `weight=exposure` — that would double-count exposure. Choose one approach.

# COMMAND ----------

# Frequency features
freq_features = [
    "area",           # string categorical — declared in cat_features below
    "ncd_years",
    "has_convictions",
    "vehicle_group",
    "driver_age",
    "log_mileage",
]

# Bridge to pandas at the CatBoost boundary
X_freq_pd = df.select(freq_features).to_pandas()
y_freq_pd = df["claim_count"].to_pandas()
exposure_pd = df["exposure"].to_pandas()

# Log-exposure offset: sets the initial log-prediction to log(exposure).
# The model then learns the frequency (rate) contribution net of exposure.
log_exposure = np.log(exposure_pd.clip(lower=1e-6))

train_pool = cb.Pool(
    data=X_freq_pd,
    label=y_freq_pd,
    baseline=log_exposure,     # exposure offset — correct approach, no weight=exposure
    cat_features=["area"],
)

freq_params = {
    "loss_function": "Poisson",
    "learning_rate": 0.05,
    "depth": 5,
    "min_data_in_leaf": 50,
    "iterations": 300,
    "random_seed": 42,
    "verbose": 0,
}

print("Training frequency model (CatBoost Poisson)...")
freq_model = cb.CatBoostRegressor(**freq_params)
freq_model.fit(train_pool)

# Quick in-sample check: predicted vs actual claim rate
preds_freq = freq_model.predict(train_pool) * exposure_pd
actual_rate = y_freq_pd.sum() / exposure_pd.sum()
pred_rate = preds_freq.sum() / exposure_pd.sum()
print(f"Actual claim rate: {actual_rate:.4f}")
print(f"Predicted claim rate (in-sample): {pred_rate:.4f}")
print("(In-sample rates match because Poisson mean is preserved)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train Gamma severity model (CatBoost)
# MAGIC
# MAGIC Severity model notes:
# MAGIC - Trained on claim records only (policies with claim_count > 0)
# MAGIC - Weighted by claim count — policies with multiple claims contribute proportionally
# MAGIC - Consider truncating large losses at a threshold before fitting to avoid
# MAGIC   individual large claims dominating the SHAP attribution for unrelated features

# COMMAND ----------

# Severity model: claims only, Gamma distribution, log link
claims_mask = df["claim_count"] > 0
claims_df = df.filter(claims_mask).with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
)

sev_features = [
    "area",
    "vehicle_group",
    "driver_age",
    "has_convictions",
]

X_sev_pd = claims_df.select(sev_features).to_pandas()
y_sev_pd = claims_df["avg_severity"].to_pandas()
claim_count_pd = claims_df["claim_count"].to_pandas()

train_pool_sev = cb.Pool(
    data=X_sev_pd,
    label=y_sev_pd,
    weight=claim_count_pd,    # weight by number of claims, not exposure
    cat_features=["area"],
)

sev_params = {
    "loss_function": "Tweedie:variance_power=2",  # Gamma equivalent in CatBoost
    "learning_rate": 0.05,
    "depth": 4,
    "min_data_in_leaf": 20,
    "iterations": 200,
    "random_seed": 42,
    "verbose": 0,
}

print("Training severity model (CatBoost Gamma)...")
sev_model = cb.CatBoostRegressor(**sev_params)
sev_model.fit(train_pool_sev)

print(f"Severity model: {claims_df.height:,} claims")
print(f"Mean observed severity: £{y_sev_pd.mean():,.0f}")
print(f"Mean predicted severity (in-sample): £{sev_model.predict(train_pool_sev).mean():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Extract frequency relativities

# COMMAND ----------

sr_freq = SHAPRelativities(
    model=freq_model,
    X=X_freq_pd,
    exposure=exposure_pd,
    categorical_features=["area", "ncd_years", "has_convictions"],
    continuous_features=["vehicle_group", "driver_age", "log_mileage"],
    feature_perturbation="tree_path_dependent",
)

print("Computing SHAP values for frequency model (CatBoost native)...")
sr_freq.fit()
print("Done.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5a. Validate extraction

# COMMAND ----------

checks = sr_freq.validate()

all_passed = True
for name, result in checks.items():
    status = "PASS" if result.passed else "FAIL"
    if not result.passed:
        all_passed = False
    print(f"[{status}] {name}: {result.message}")

if not all_passed:
    raise RuntimeError(
        "Validation failed. Do not use these relativities until the failure is investigated."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b. Extract categorical relativities and compare to true DGP

# COMMAND ----------

rels_freq = sr_freq.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "area": "A",
        "ncd_years": 0,
        "has_convictions": 0,
    },
)

# Show all categorical features
for feature in ["area", "ncd_years", "has_convictions"]:
    feat_rels = rels_freq[rels_freq["feature"] == feature].copy()
    print(f"\n--- {feature} ---")
    print(feat_rels[["level", "relativity", "lower_ci", "upper_ci", "n_obs"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5c. Compare to true DGP

# COMMAND ----------

print("NCD relativities: GBM vs true DGP")
print(f"{'NCD years':<12} {'GBM':>8} {'True DGP':>10} {'Ratio':>8}")
print("-" * 42)

ncd_rels = rels_freq[rels_freq["feature"] == "ncd_years"].set_index("level")["relativity"]
for k in range(6):
    true_rel = np.exp(TRUE_FREQ_PARAMS["ncd_years"] * k)
    gbm_rel = ncd_rels.get(k, np.nan)
    ratio = gbm_rel / true_rel if not np.isnan(gbm_rel) else np.nan
    print(f"NCD = {k:<6} {gbm_rel:>8.3f} {true_rel:>10.3f} {ratio:>8.3f}")

print()
print("Area relativities: GBM vs true DGP")
print(f"{'Area':<8} {'GBM':>8} {'True DGP':>10} {'Ratio':>8}")
print("-" * 38)

area_rels = rels_freq[rels_freq["feature"] == "area"].set_index("level")["relativity"]
for band in ["A", "B", "C", "D", "E", "F"]:
    true_log_rel = TRUE_FREQ_PARAMS.get(f"area_{band}", 0.0)
    true_rel = np.exp(true_log_rel)
    gbm_rel = area_rels.get(band, np.nan)
    ratio = gbm_rel / true_rel if not np.isnan(gbm_rel) else np.nan
    print(f"Area {band}  {gbm_rel:>8.3f} {true_rel:>10.3f} {ratio:>8.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Continuous feature curves

# COMMAND ----------

# Driver age: U-shaped, use LOESS smoothing
age_curve = sr_freq.extract_continuous_curve(
    feature="driver_age",
    n_points=150,
    smooth_method="loess",
)

# Vehicle group: broadly monotone, LOESS is fine
vg_curve = sr_freq.extract_continuous_curve(
    feature="vehicle_group",
    n_points=100,
    smooth_method="loess",
)

# Annual mileage (log scale): enforce monotonicity with isotonic regression
mileage_curve = sr_freq.extract_continuous_curve(
    feature="log_mileage",
    n_points=100,
    smooth_method="isotonic",
)

# Convert log_mileage back to miles for display
mileage_curve["mileage"] = np.exp(mileage_curve["feature_value"])

print("Continuous curve shapes:")
print(f"  driver_age: {len(age_curve)} points, relativity range [{age_curve['relativity'].min():.2f}, {age_curve['relativity'].max():.2f}]")
print(f"  vehicle_group: {len(vg_curve)} points, relativity range [{vg_curve['relativity'].min():.2f}, {vg_curve['relativity'].max():.2f}]")
print(f"  log_mileage: {len(mileage_curve)} points, relativity range [{mileage_curve['relativity'].min():.2f}, {mileage_curve['relativity'].max():.2f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a. Plot continuous curves

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Driver age
ax = axes[0]
ax.plot(age_curve["feature_value"], age_curve["relativity"], color="steelblue", lw=2)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
ax.set_xlabel("Driver age")
ax.set_ylabel("Relativity (mean = 1.0)")
ax.set_title("Frequency relativity by driver age\n(LOESS smoothed)")
ax.set_ylim(0.4, 3.0)

# Vehicle group
ax = axes[1]
ax.plot(vg_curve["feature_value"], vg_curve["relativity"], color="darkorange", lw=2)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
ax.set_xlabel("ABI vehicle group")
ax.set_ylabel("Relativity (mean = 1.0)")
ax.set_title("Frequency relativity by vehicle group\n(LOESS smoothed)")

# Annual mileage
ax = axes[2]
ax.plot(mileage_curve["mileage"] / 1000, mileage_curve["relativity"], color="seagreen", lw=2)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6)
ax.set_xlabel("Annual mileage (thousands)")
ax.set_ylabel("Relativity (mean = 1.0)")
ax.set_title("Frequency relativity by annual mileage\n(isotonic regression)")

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Banded age factor table
# MAGIC
# MAGIC The correct approach: compute SHAP on the original feature matrix (which the model
# MAGIC was trained on), then aggregate the continuous `driver_age` SHAP values by band
# MAGIC using exposure-weighted means.
# MAGIC
# MAGIC Do NOT pass `age_band` as a feature to the explainer — the model was trained on
# MAGIC `driver_age` and TreeSHAP needs the same feature matrix.

# COMMAND ----------

# sr_freq was fit on the original features including continuous driver_age.
# We already have the SHAP values — no need to refit.
shap_vals = sr_freq.shap_values()          # numpy array, shape (n_obs, n_features)
feature_names = sr_freq.feature_names_    # list of feature names in SHAP order

age_idx = feature_names.index("driver_age")
age_shap = shap_vals[:, age_idx]          # driver_age SHAP per observation

# Build a Polars frame with age_band (computed in Step 2) and SHAP values
shap_frame = pl.DataFrame({
    "age_band": df["age_band"].to_list(),
    "age_shap": age_shap.tolist(),
    "exposure": df["exposure"].to_list(),
})

# Exposure-weighted mean SHAP per band
band_stats = shap_frame.group_by("age_band").agg([
    (pl.col("age_shap") * pl.col("exposure")).sum().alias("weighted_shap_sum"),
    pl.col("exposure").sum().alias("total_exposure"),
    pl.col("age_shap").std().alias("shap_std"),
    pl.col("exposure").count().alias("n_obs"),
]).with_columns(
    (pl.col("weighted_shap_sum") / pl.col("total_exposure")).alias("mean_shap")
)

# Base level: 30-39 band
base_shap = band_stats.filter(pl.col("age_band") == "30-39")["mean_shap"][0]
base_n = band_stats.filter(pl.col("age_band") == "30-39")["n_obs"][0]
base_std = band_stats.filter(pl.col("age_band") == "30-39")["shap_std"][0]

# Relativity and 95% CI using full variance formula: SE(k)^2 + SE(0)^2
z = 1.96
band_rels = band_stats.with_columns([
    (pl.col("mean_shap") - base_shap).exp().alias("relativity"),
    # SE for each band
    (pl.col("shap_std") / pl.col("n_obs").cast(pl.Float64).sqrt()).alias("se_k"),
]).with_columns([
    # Base level SE (constant across rows)
    pl.lit(base_std / (base_n ** 0.5)).alias("se_base"),
]).with_columns([
    # Full variance: sqrt(SE(k)^2 + SE(base)^2)
    ((pl.col("se_k") ** 2 + pl.col("se_base") ** 2).sqrt()).alias("se_total"),
]).with_columns([
    # Log-relativity = mean_shap(k) - mean_shap(base)
    (pl.col("mean_shap") - base_shap).alias("log_rel"),
]).with_columns([
    ((pl.col("log_rel") - z * pl.col("se_total")).exp()).alias("lower_ci"),
    ((pl.col("log_rel") + z * pl.col("se_total")).exp()).alias("upper_ci"),
])

# Sort by natural age band order
band_order = {b: i for i, b in enumerate(age_labels)}
band_rels_sorted = (
    band_rels
    .with_columns(
        pl.col("age_band").replace(band_order).cast(pl.Int32).alias("sort_key")
    )
    .sort("sort_key")
)

print("Age band relativities (aggregated from continuous driver_age SHAP):")
print(
    band_rels_sorted.select([
        "age_band", "relativity", "lower_ci", "upper_ci", "n_obs", "total_exposure"
    ])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Severity relativities

# COMMAND ----------

sr_sev = SHAPRelativities(
    model=sev_model,
    X=X_sev_pd,
    exposure=None,   # severity model: weight by claim count, not exposure
    categorical_features=["area", "has_convictions"],
    continuous_features=["vehicle_group", "driver_age"],
)

print("Computing SHAP values for severity model (CatBoost native)...")
sr_sev.fit()

checks_sev = sr_sev.validate()
print(f"Reconstruction check: {'PASS' if checks_sev['reconstruction'].passed else 'FAIL'}")

rels_sev = sr_sev.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "area": "A",
        "has_convictions": 0,
    },
)

print("\nSeverity area relativities:")
print(rels_sev[rels_sev["feature"] == "area"][["level", "relativity", "lower_ci", "upper_ci"]].to_string(index=False))

print("\nSeverity conviction relativity:")
print(rels_sev[rels_sev["feature"] == "has_convictions"][["level", "relativity", "lower_ci", "upper_ci"]].to_string(index=False))

print("\nTrue severity parameters for comparison:")
print(f"  Vehicle group: {TRUE_SEV_PARAMS['vehicle_group']:.3f} per group → group 50 vs 1: {np.exp(49 * TRUE_SEV_PARAMS['vehicle_group']):.2f}x")
print(f"  Driver age young: exp({TRUE_SEV_PARAMS['driver_age_young']:.2f}) = {np.exp(TRUE_SEV_PARAMS['driver_age_young']):.2f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Benchmark GLM comparison

# COMMAND ----------

# Bridge to pandas for statsmodels
df_pd = df.to_pandas()

# Fit a Poisson GLM for comparison
# Treat area and ncd_years as categorical in the GLM (same as GBM)
df_pd["ncd_factor"] = df_pd["ncd_years"].astype(str)

glm_formula = (
    "claim_count ~ C(area, Treatment('A')) "
    "+ C(ncd_factor, Treatment('0')) "
    "+ has_convictions "
    "+ vehicle_group "
    "+ driver_age"
)

print("Fitting Poisson GLM...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm = smf.glm(
        formula=glm_formula,
        data=df_pd,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=np.log(df_pd["exposure"].clip(1e-6)),
        var_weights=np.ones(len(df_pd)),
    ).fit(maxiter=50)

print(f"GLM converged: {glm.converged}")
print(f"GLM AIC: {glm.aic:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9a. Compare NCD relativities: GBM vs GLM

# COMMAND ----------

# GLM NCD relativities
glm_ncd = {}
for k in range(6):
    if k == 0:
        glm_ncd[k] = 1.0
    else:
        param_name = f"C(ncd_factor, Treatment('0'))[T.{k}]"
        glm_ncd[k] = np.exp(glm.params.get(param_name, 0.0))

# GBM NCD relativities
gbm_ncd = rels_freq[rels_freq["feature"] == "ncd_years"].set_index("level")["relativity"]

print(f"{'NCD':<6} {'GBM':>8} {'GLM':>8} {'True':>8} {'GBM/GLM':>9}")
print("-" * 44)
for k in range(6):
    true_r = np.exp(TRUE_FREQ_PARAMS["ncd_years"] * k)
    gbm_r = gbm_ncd.get(k, np.nan)
    glm_r = glm_ncd.get(k, 1.0)
    ratio = gbm_r / glm_r if not np.isnan(gbm_r) else np.nan
    print(f"{k:<6} {gbm_r:>8.3f} {glm_r:>8.3f} {true_r:>8.3f} {ratio:>9.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9b. Plot GBM vs GLM driver age effect

# COMMAND ----------

# GLM driver age: linear coefficient (misses the U-shape)
glm_age_beta = glm.params.get("driver_age", 0.0)
glm_age_mean = df_pd["driver_age"].mean()

age_range = np.arange(17, 86)
glm_age_rel = np.exp(glm_age_beta * (age_range - glm_age_mean))
glm_age_rel = glm_age_rel / glm_age_rel.mean()  # mean normalise for comparison

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(age_curve["feature_value"], age_curve["relativity"],
        color="steelblue", lw=2.5, label="GBM (LOESS)")
ax.plot(age_range, glm_age_rel,
        color="crimson", lw=2, linestyle="--", label="GLM (linear)")
ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5)
ax.set_xlabel("Driver age")
ax.set_ylabel("Relativity (mean = 1.0)")
ax.set_title("Frequency relativity by driver age: GBM vs GLM\nGBM captures U-shape; GLM is constrained to linear")
ax.legend()
ax.set_ylim(0.3, 3.5)
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Factor table visualisation

# COMMAND ----------

sr_freq.plot_relativities(
    features=["area", "ncd_years", "has_convictions"],
    show_ci=True,
    figsize=(14, 5),
)
display(plt.gcf())
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Export for Radar

# COMMAND ----------

def to_radar_format(
    rels: pd.DataFrame,
    feature_name_map: dict | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Format relativities for WTW Radar import.

    Parameters
    ----------
    rels : pd.DataFrame
        Output from extract_relativities(). Categorical features only —
        Radar cannot directly import continuous curves.
    feature_name_map : dict | None
        Mapping from Python column names to Radar variable names.
        e.g. {"area": "AreaBand", "ncd_years": "NCDYears"}
    output_path : str | None
        If provided, write to CSV at this path.

    Returns
    -------
    pd.DataFrame with columns Factor, Level, Relativity.
    """
    out = rels[["feature", "level", "relativity"]].copy()
    out.columns = ["Factor", "Level", "Relativity"]
    out["Relativity"] = out["Relativity"].round(4)
    out["Level"] = out["Level"].astype(str)

    if feature_name_map:
        out["Factor"] = out["Factor"].map(feature_name_map).fillna(out["Factor"])

    if output_path:
        out.to_csv(output_path, index=False)
        print(f"Written {len(out)} rows to {output_path}")

    return out


# Map Python names to Radar variable names
radar_name_map = {
    "area": "AreaBand",
    "ncd_years": "NCDYears",
    "has_convictions": "ConvictionFlag",
}

# Categorical frequency relativities
cat_rels = rels_freq[rels_freq["feature"].isin(["area", "ncd_years", "has_convictions"])]
radar_df = to_radar_format(cat_rels, feature_name_map=radar_name_map)

# Also include age band relativities from the Polars frame (convert for Radar)
age_radar = (
    band_rels_sorted
    .select(["age_band", "relativity"])
    .to_pandas()
    .rename(columns={"age_band": "Level", "relativity": "Relativity"})
)
age_radar["Factor"] = "DriverAgeBand"
age_radar["Relativity"] = age_radar["Relativity"].round(4)
age_radar["Level"] = age_radar["Level"].astype(str)

radar_full = pd.concat([radar_df, age_radar[["Factor", "Level", "Relativity"]]], ignore_index=True)

print("\nRadar export preview:")
print(radar_full.head(20).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Write results to Unity Catalog

# COMMAND ----------

RUN_DATE = str(date.today())
MODEL_NAME = "freq_catboost_v1_module04"

# Frequency relativities
rels_freq_out = rels_freq.copy()
rels_freq_out["model_name"] = MODEL_NAME
rels_freq_out["model_type"] = "frequency"
rels_freq_out["run_date"] = RUN_DATE
rels_freq_out["feature_set"] = "continuous"

rels_sev_out = rels_sev.copy()
rels_sev_out["model_name"] = MODEL_NAME.replace("freq", "sev")
rels_sev_out["model_type"] = "severity"
rels_sev_out["run_date"] = RUN_DATE
rels_sev_out["feature_set"] = "categorical"

all_rels = pd.concat([rels_freq_out, rels_sev_out], ignore_index=True)

# Write to Delta table
(
    spark.createDataFrame(all_rels)
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module04_shap_relativities")
)

print(f"Written {len(all_rels)} relativity rows to main.pricing.module04_shap_relativities")

# COMMAND ----------

# Write age band relativities from Polars — convert to Spark via pandas bridge
age_rels_pd = band_rels_sorted.select([
    "age_band", "relativity", "lower_ci", "upper_ci", "n_obs", "total_exposure"
]).to_pandas()
age_rels_pd["model_name"] = MODEL_NAME
age_rels_pd["model_type"] = "frequency"
age_rels_pd["feature"] = "driver_age_band"
age_rels_pd["run_date"] = RUN_DATE

(
    spark.createDataFrame(age_rels_pd)
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module04_shap_relativities_banded")
)

print(f"Written banded age relativities to main.pricing.module04_shap_relativities_banded")

# COMMAND ----------

# Validation log
val_records = []
for model_label, model_checks in [("frequency", checks), ("severity", checks_sev)]:
    for check_name, result in model_checks.items():
        val_records.append({
            "model_name": MODEL_NAME,
            "model_type": model_label,
            "check_name": check_name,
            "passed": result.passed,
            "value": float(result.value),
            "message": result.message,
            "run_date": RUN_DATE,
        })

(
    spark.createDataFrame(val_records)
    .write
    .format("delta")
    .mode("append")
    .saveAsTable("main.pricing.module04_validation_log")
)

print(f"Written {len(val_records)} validation records to main.pricing.module04_validation_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 12a. Serialise SHAP values for later use

# COMMAND ----------

# Store SHAP values as a Delta table — more queryable than a JSON blob.
# This lets you re-band, re-normalise, or re-validate without refitting.
shap_vals_freq = sr_freq.shap_values()  # numpy array, shape (n_obs, n_features)

shap_df = pd.DataFrame(
    shap_vals_freq,
    columns=[f"shap_{c}" for c in freq_features]
)
shap_df["expected_value"] = sr_freq._expected_value
shap_df["model_name"] = MODEL_NAME
shap_df["run_date"] = RUN_DATE

(
    spark.createDataFrame(shap_df)
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module04_shap_values")
)

print(f"Written {len(shap_df):,} rows of SHAP values to main.pricing.module04_shap_values")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Summary: what we extracted

# COMMAND ----------

print("=" * 60)
print("SHAP RELATIVITY EXTRACTION — SUMMARY")
print("=" * 60)
print()
print(f"Portfolio: {df.height:,} policies, {df['exposure'].sum():.0f} earned years")
print(f"Frequency model: CatBoost Poisson, {freq_params['iterations']} trees, depth {freq_params['depth']}")
print(f"Severity model: CatBoost Gamma (Tweedie p=2), {sev_params['iterations']} trees")
print()
print("Validation:")
print(f"  Reconstruction error (freq): {checks['reconstruction'].value:.2e}")
print(f"  Reconstruction error (sev):  {checks_sev['reconstruction'].value:.2e}")
print()
print("Key relativities (frequency, base level normalisation):")
print()
print("  Area band (vs A):")
for _, row in rels_freq[rels_freq["feature"] == "area"].iterrows():
    band = row["level"]
    print(f"    Area {band}: {row['relativity']:.3f} [{row['lower_ci']:.3f}, {row['upper_ci']:.3f}]")
print()
print("  NCD years (vs NCD=0):")
for _, row in rels_freq[rels_freq["feature"] == "ncd_years"].iterrows():
    print(f"    NCD={int(row['level'])}: {row['relativity']:.3f} [{row['lower_ci']:.3f}, {row['upper_ci']:.3f}]")
print()
print("  Conviction flag:")
for _, row in rels_freq[rels_freq["feature"] == "has_convictions"].iterrows():
    label = "Clean" if int(row["level"]) == 0 else "Convictions"
    print(f"    {label}: {row['relativity']:.3f}")
print()
print("Delta tables written:")
print("  main.pricing.module04_shap_relativities")
print("  main.pricing.module04_shap_relativities_banded")
print("  main.pricing.module04_validation_log")
print("  main.pricing.module04_shap_values")
