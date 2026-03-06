---
layout: post
title: "From GBM to Radar: A Complete Databricks Workflow for Pricing Actuaries"
date: 2026-03-06
categories: [tutorials]
tags: [databricks, lightgbm, shap, radar, pricing, motor, unity-catalog, mlflow, python]
---

Most UK pricing teams we speak to are running Databricks. Almost none of them are running it well.

The workflow we see most often: load a claims extract into a notebook, fit a LightGBM in a single cell, pickle the model, hand-extract some PDPs, and email a CSV to whoever has Radar access. Unity Catalog sits unused. MLflow has one run in it from a proof-of-concept someone did in 2023. The Jobs scheduler has never been touched.

The reason this persists is not laziness. It is the absence of a worked example that connects the actuarial output — a Radar-loadable factor table with auditable relativities — to the Databricks infrastructure that should be generating it. This post is that worked example.

What we are going to build: a scheduled Databricks Job that reads claims and exposure data from Delta tables in Unity Catalog, trains separate Poisson and Gamma LightGBM models, validates them using [`insurance-cv`](https://github.com/burningcost/insurance-cv) temporal splits, extracts SHAP relativities with [`shap-relativities`](https://github.com/burningcost/shap-relativities), benchmarks them against a GLM, and writes a Radar-format CSV to a versioned Delta table. The job runs on the 15th of every month. Nothing requires a human to press a button.

This is not a rehash of our [SHAP validation notebook](https://github.com/burningcost/databricks-shap-notebook). That notebook validates the extraction method against a known data-generating process on synthetic data. This is the thing you would actually build at a carrier.

---

## 1. Data on Databricks — Unity Catalog and Delta Tables

Before fitting anything, you need the data in the right shape and in the right place.

We use Unity Catalog with a three-level namespace. The canonical table for a UK personal lines motor book is `pricing.motor.claims_exposure`: one row per policy-period, partitioned by `accident_year`. This structure maps directly to how actuaries think about the data and how triangles are constructed — accident year is the primary index for every downstream analysis.

Two things consistently trip teams up at this stage.

First, the table structure. Exposure and claims belong at policy level, not claim level. `exposure_years` is a float representing earned policy years. `claim_count` and `incurred` are aggregated at the policy-period level. If your source data is a claim-level bordereaux, aggregate before loading into this table, not inside your modelling notebook. The modelling notebook should not be doing data engineering.

Second, partitioning. Partition by `accident_year`, not by `policy_inception_date`. The difference matters when your query filters by accident year — which every query in this pipeline does. Partitioning by inception date means every accident-year filter scans every partition. On a five-year book with 10 million policy-periods, that is the difference between a two-second query and a two-minute one.

The DDL:

```sql
CREATE TABLE IF NOT EXISTS pricing.motor.claims_exposure (
  policy_id        STRING NOT NULL,
  accident_year    INT NOT NULL,
  exposure_years   DOUBLE NOT NULL,
  claim_count      INT NOT NULL,
  incurred         DOUBLE NOT NULL,
  area_band        STRING,
  ncd_years        INT,
  vehicle_group    INT,
  driver_age       INT,
  annual_mileage   DOUBLE
)
USING DELTA
PARTITIONED BY (accident_year)
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true');
```

Reading into pandas for modelling is one line:

```python
df = (
    spark.table("pricing.motor.claims_exposure")
    .filter(F.col("accident_year").between(2019, 2024))
    .filter(F.col("status") == "active")
    .toPandas()
)
```

The schema is load-bearing. A wrong type here — `claim_count` as a float, `exposure_years` as an integer — breaks the model pipeline downstream in ways that produce wrong numbers rather than errors. Enforce it at the DDL stage, not with defensive code in the notebook.

---

## 2. Feature Engineering

Feature engineering for a motor book is mostly about knowing what not to do.

Three things actually matter in practice.

**Temporal features.** Accident year and policy year are separate signals and should be separate model inputs. Accident-year trend (claims inflation, legal environment changes) and exposure-year mix shift (portfolio growth or contraction by segment) are different quantities. Conflating them — using a single `year` column — biases both effects. Centre accident year around the median to avoid intercept instability:

```python
df["accident_year_centred"] = df["accident_year"] - df["accident_year"].median()
```

**Ordered categoricals.** ABI area band is an ordinal six-level variable. Vehicle group (ABI 1–50) is quasi-continuous with a non-linear risk profile. NCD years (0–5) is ordered ordinal. None of these should be one-hot encoded; none of them should be passed to LightGBM's `categorical_feature` parameter. LightGBM's native categorical handling is designed for unordered categories and makes split decisions that can arbitrarily reorder levels — which produces SHAP values that are difficult to interpret and relativities that do not respect the ordering the data clearly shows. Use integer encoding with explicit ordering and let the GBM learn the shape:

```python
# NCD as integer-encoded ordinal — NOT passed as categorical_feature
# Ordering matters for SHAP attribution; LightGBM's categorical handling loses it
df["ncd_years"] = df["ncd_years"].clip(0, 5).astype(int)
```

**Exposure handling.** The most frequently wrong thing in a frequency model. LightGBM's `weight` parameter is not a proper offset — it scales the contribution to the loss function rather than entering the linear predictor as `log(exposure)`. For a Poisson model this means the model does not correctly account for the relationship between exposure and claim count; it just weights larger exposures more heavily in the gradient calculation. This is standard practice because LightGBM does not support proper offsets, but it is an approximation, and the error is typically 2–5% at the mean and larger in the tails for policies with very short or very long exposure periods. Document the choice:

```python
# Exposure as weight — correct practice given LightGBM's API constraints
train_ds = lgb.Dataset(X_train, label=y_train, weight=df_train["exposure_years"])
```

---

## 3. Temporal Cross-Validation with `insurance-cv`

Standard k-fold cross-validation is wrong for insurance data, and not in a subtle way.

A random 80/20 split mixes accident years in training and validation. A 2022 policy ends up in the training set while a 2021 policy with an identical risk profile ends up in validation. The model has effectively seen the future. Your CV metrics are optimistic, by a margin that depends on how fast the book is changing. On a stable, mature motor book the bias might be 2–3 Gini points. On a book with rapid mix shift or large claims inflation, it can be large enough to change which model architecture you select.

[`insurance-cv`](https://github.com/burningcost/insurance-cv) implements walk-forward temporal splits that respect the insurance time structure: train on accident years 2019–2021, validate on 2022; train on 2019–2022, validate on 2023. The library also handles the IBNR buffer — by default it excludes the most recent 12 months from any training fold, because late-reported claims in the most recent period will be systematically under-counted, biasing severity models downward. Forget this and your severity model will look better in cross-validation than it will on unseen data:

```python
from insurance_cv import TemporalSplit

splitter = TemporalSplit(
    time_column="accident_year",
    train_years=3,          # rolling 3-year training window
    ibnr_buffer_months=12,  # exclude last 12 months from each training fold
)

for fold_n, (train_idx, val_idx) in enumerate(splitter.split(df)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    w_tr = df.iloc[train_idx]["exposure_years"]
    # ... fit, score, log to MLflow
```

Log every CV fold to MLflow. By the time you run the final model, you want a full record of every candidate hyperparameter set and its out-of-time performance — not just the winner:

```python
import mlflow

with mlflow.start_run(run_name=f"lgbm_freq_fold_{fold_n}"):
    mlflow.log_params(lgb_params)
    mlflow.log_metric("oot_poisson_deviance", val_deviance)
    mlflow.log_metric("oot_gini", val_gini)
```

On hyperparameter selection: we keep to manual tuning here. The three parameters that matter most for a UK motor book are `min_child_samples` (minimum leaf size — the single most important parameter for avoiding overfitting on sparse cells), `num_leaves` (model complexity; start at 31 and resist the urge to increase it), and `learning_rate` (lower is better; use early stopping against the held-out temporal fold to find the right tree count). Optuna and Hyperopt both work identically in Databricks notebooks, but they are not the bottleneck in this workflow.

---

## 4. Training Frequency and Severity Models

Two models, trained separately. A Poisson LightGBM for frequency (claims per earned year) and a Gamma LightGBM for severity (mean incurred per claim, on claims-only data).

**Frequency model.** `objective="poisson"` with `min_child_samples=200`. The case for 200 rather than the default 20 is straightforward: motor pricing cells are uneven. Area band D × NCD 0 × vehicle group 40+ might have 200 observations across five training years. With `min_child_samples=20` the model will overfit to that cell. With 200 it regularises, and the SHAP relativities for sparse cells will be more stable. The cost is a slight loss of lift in dense cells; the benefit is that the factor table does not need manual smoothing before it goes into Radar.

Use `num_leaves=31` and `max_depth=5` as starting points. An unconstrained deep tree will produce relativities that are noisier than your GLM for thin segments. The pricing committee will rightly ask why NCD 0 in area band C has a different relativity from NCD 0 in area band B when there is no theoretical reason for it. Constrain the depth to reduce spurious interactions.

```python
# Frequency model — final training run after CV
freq_model = lgb.train(
    {**lgb_params_freq, "objective": "poisson"},
    lgb.Dataset(X_train_all, label=y_freq, weight=exposure),
    num_boost_round=best_iter_freq,
    valid_sets=[lgb.Dataset(X_val_oot, label=y_freq_val, weight=exposure_val)],
    callbacks=[lgb.log_evaluation(100)],
)

# Severity model — claims only
claims_mask = df_train_all["claim_count"] > 0
sev_model = lgb.train(
    {**lgb_params_sev, "objective": "gamma"},
    lgb.Dataset(
        X_train_all[claims_mask],
        label=df_train_all.loc[claims_mask, "incurred"],
    ),
    num_boost_round=best_iter_sev,
)
```

**Severity model.** `objective="gamma"`, trained on claims-only rows. Two observations worth flagging. First: mileage and vehicle age tend to have stronger severity effects than frequency effects on most UK personal lines motor books. The feature importance rankings between the two models are often visually striking and are worth showing to a pricing committee — it builds confidence in the modelling approach and surfaces questions the GLM may never have raised. Second: the Gamma model uses a log link, so SHAP values are in log space exactly as with the Poisson model. The extraction is mathematically identical; the relativities represent a different quantity (severity rather than frequency), but the code is the same.

Log both models to MLflow immediately after training:

```python
with mlflow.start_run(run_name="freq_severity_final"):
    mlflow.lightgbm.log_model(freq_model, "freq_model")
    mlflow.lightgbm.log_model(sev_model, "sev_model")
    mlflow.log_metrics({"freq_gini": freq_gini, "sev_gini": sev_gini})
```

The model that generated a given Radar factor table must be traceable. MLflow makes this trivial; not doing it creates an audit problem that will surface at exactly the wrong time.

---

## 5. SHAP Relativity Extraction

This is where [`shap-relativities`](https://github.com/burningcost/shap-relativities) does its work.

The maths: for a Poisson GBM with log link, SHAP values are additive in log space. Every prediction decomposes exactly as `log(μ_i) = expected_value + SHAP_area_i + SHAP_ncd_i + ...`. Exposure-weighted means of those SHAP values give level contributions; exponentiation gives multiplicative relativities. This is directly analogous to `exp(β)` from a GLM — same interpretation, same format, usable in the same factor table structure.

We run `SHAPRelativities` on both models separately, applied to the full training population rather than a single CV fold. Using the full population matters because sparse factor levels may have fewer than 30 observations on any individual fold. Relativities estimated from fewer than 30 observations are statistically unreliable; the reconstruction check may pass while the individual level estimates are noise.

```python
from shap_relativities import SHAPRelativities

# Frequency SHAP
sr_freq = SHAPRelativities(
    model=freq_model,
    X=X_train_all,
    exposure=df_train_all["exposure_years"],
    categorical_features=CATEGORICAL_FEATURES,
    continuous_features=CONTINUOUS_FEATURES,
)
sr_freq.fit()
rels_freq = sr_freq.extract_relativities(base_levels=BASE_LEVELS)

# Severity SHAP — claims-only population, weighted by claim count
sr_sev = SHAPRelativities(
    model=sev_model,
    X=X_train_all[claims_mask],
    exposure=df_train_all.loc[claims_mask, "claim_count"],
    categorical_features=CATEGORICAL_FEATURES,
    continuous_features=CONTINUOUS_FEATURES,
)
sr_sev.fit()
rels_sev = sr_sev.extract_relativities(base_levels=BASE_LEVELS)
```

Note the exposure argument for the severity model: claim count rather than policy years. The severity relativity for a factor level should reflect the average SHAP contribution across claims, weighted by how many claims came from each policy. Weighting by earned years would over-represent policies with long exposure periods that happen to have made few claims.

Run `validate()` on both models and treat a reconstruction failure as a hard stop:

```python
for label, sr in [("freq", sr_freq), ("sev", sr_sev)]:
    checks = sr.validate()
    if not checks["reconstruction"].passed:
        raise ValueError(
            f"{label} SHAP reconstruction check failed: {checks['reconstruction'].message}"
        )
```

The reconstruction check verifies that `exp(shap.sum(axis=1) + expected_value)` matches the model's predictions within 1e-4. If it fails, the SHAP values are not trustworthy and the factor table should not be produced. This is not a warning to log and continue past. The most common cause is a mismatch between the model's link function and the SHAP output type — easily fixed, but the fix must be made before the export runs.

The comparison between frequency and severity relativities is one of the most practically useful outputs of this workflow. On most UK motor books, NCD is a stronger frequency factor than a severity factor — claims from high-NCD policyholders tend to be attritional, and claim size is less correlated with the claim-attentive behaviour that NCD tracks. Area band often shows the opposite pattern: large claims (theft, serious accidents) are more geographically concentrated than small claims. These patterns are theoretically motivated, and when the SHAP relativities confirm them it builds appropriate confidence. When they do not, it warrants investigation.

---

## 6. Comparing GBM Relativities to the GLM Benchmark

You cannot take a GBM factor table to a pricing committee without a reference point. The reference is always the existing GLM.

This comparison serves two purposes. First, it gives the committee a familiar anchor — they know what the GLM says about NCD and area band, and they can assess whether the GBM's deviations are plausible. Second, it identifies factors where the two models genuinely disagree, which is where the interesting work is. An unexplained 20% difference in the area band C relativity is either a GBM overfitting problem or a real signal the GLM was missing. You need to know which.

We fit a Poisson GLM using `statsmodels` — the actuarially familiar tool that gives proper `exp(β)` output with standard errors, unlike sklearn's `PoissonRegressor` which does not expose coefficients in that form:

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

glm = smf.glm(
    formula="claim_count ~ C(area_band) + C(ncd_years) + driver_age + vehicle_group",
    data=df_train_all,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(df_train_all["exposure_years"]),
).fit()

glm_rels = np.exp(glm.params).rename("glm_relativity")
```

Note that the GLM uses a proper `offset` — log exposure enters the linear predictor correctly, not as a weight. This is one of the genuine model-form differences between the GLM benchmark and the LightGBM, and it contributes a small but systematic difference in relativities that should be understood rather than glossed over.

The honest observation about what this comparison typically shows: on a mature, well-maintained UK motor book with a GLM that has been tuned over several years, the SHAP relativities agree with the GLM relativities to within 5–10% on most categorical factors. The GBM's advantage is concentrated in the continuous features — driver age curves, mileage curves — and in interactions the GLM never modelled. The area-NCD-age combination that the GLM treated as additive is often genuinely interactive, and the GBM will find it where data supports it.

If the SHAP relativities diverge materially from the GLM on a factor where there is no theoretical reason for divergence — say, a 25% difference in the NCD 3 relativity with no known portfolio composition change — that is a signal to investigate. Not to trust the GBM, not to trust the GLM, but to understand what is driving the difference. Sometimes it is a data issue both models have handled differently. Sometimes it is an interaction confound. Occasionally it is a genuine improvement worth explaining and defending. The diagnostic plot to build is a scatter of SHAP relativity vs GLM `exp(β)` for every categorical level, with a 45-degree reference line and labels by factor and level.

---

## 7. Export to Radar Format

Radar (Willis Towers Watson) expects a factor table as `FactorName`, `Level`, `Relativity` — three columns, one row per factor level. The format is simple. The four things that go wrong are consistent enough to be worth naming.

**Column naming.** Radar is case-sensitive. `factorname` will not import. `FactorName`, `Level`, `Relativity` exactly.

**Continuous features.** Radar does not interpolate a continuous curve. Driver age and mileage need to be discretised into bands before export. The banding should align with the existing GLM banding — not because the GBM's continuous curve is wrong, but because any relativity movement that results from re-banding will be indistinguishable from genuine model signal in a pricing committee review. Keep the bands consistent and change one thing at a time.

**Base level.** The base level for each factor must appear in the export file with `Relativity = 1.000`. Radar will error if it is missing. `shap-relativities` produces it by construction; do not filter it out when building the export DataFrame.

**Versioning.** Every export should carry `model_version`, `export_date`, and `job_run_id`. Write to both a versioned DBFS path and a Delta table in Unity Catalog. The requirement is simple: given a Radar factor table that was live on a given date, you must be able to identify the model run that produced it, the training data it used, and the CV results that justified its deployment.

```python
radar_export = (
    rating_table[["feature", "level", "relativity"]]
    .rename(columns={"feature": "FactorName", "level": "Level", "relativity": "Relativity"})
    .assign(
        ModelVersion=model_version,
        ExportDate=pd.Timestamp.today().strftime("%Y-%m-%d"),
        JobRunId=dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId().get(),
    )
)

# Write to versioned DBFS path
export_path = f"/dbfs/pricing/radar_exports/{model_version}/rating_relativities_radar.csv"
radar_export.to_csv(export_path, index=False)

# Write to Unity Catalog for audit trail
spark.createDataFrame(radar_export).write.format("delta").mode("append").saveAsTable(
    "pricing.motor.radar_exports"
)
```

If you use Earnix rather than Radar, the same CSV structure works with column remapping. The `FactorName`/`Level`/`Relativity` convention maps directly to Earnix's expected input. We are not making that a separate section because the workflow is identical.

---

## 8. Scheduling as a Databricks Job

The notebook workflow becomes a pipeline by turning it into a Databricks Job with three tasks.

`data_prep` reads from Unity Catalog, runs feature engineering, and writes the prepared dataset to a scratch Delta table. `train_and_extract` reads that table, trains both models, extracts and validates SHAP relativities, and writes the rating table to Unity Catalog. `export` reads the rating table, produces the Radar CSV, writes to DBFS, and sends a completion notification. Tasks are linearly dependent: if the reconstruction check fails in `train_and_extract`, the `export` task does not run and the existing Radar factor table is not overwritten.

The schedule is the 15th of each month — after month-end bordereaux are available, before the next pricing committee meeting:

```json
{
  "name": "motor_pricing_monthly",
  "schedule": {
    "quartz_cron_expression": "0 0 6 15 * ?",
    "timezone_id": "Europe/London"
  },
  "tasks": [
    {
      "task_key": "data_prep",
      "notebook_task": {"notebook_path": "/pricing/01_data_prep"}
    },
    {
      "task_key": "train_and_extract",
      "depends_on": [{"task_key": "data_prep"}],
      "notebook_task": {"notebook_path": "/pricing/02_train_extract"}
    },
    {
      "task_key": "export",
      "depends_on": [{"task_key": "train_and_extract"}],
      "notebook_task": {"notebook_path": "/pricing/03_export"}
    }
  ]
}
```

Put this JSON in your repository alongside the notebooks. It is infrastructure as code. A manual Jobs UI configuration that exists only in Databricks is not reproducible and is not auditable — the same arguments that apply to the factor table apply to the pipeline that generates it.

This is the argument for Databricks that most teams never hear when the platform is sold to them. The case is not the collaborative notebooks. It is not the MLflow integration, useful as that is. It is the ability to run the entire pipeline — raw Delta tables to Radar export — on a schedule, with retries, failure notifications, and a full audit trail in Unity Catalog, without anyone having to be at their desk. A two-week manual retrain cycle, with all the associated version-control risk and key-person dependency, becomes an overnight job that fails loudly when something goes wrong.

---

## 9. What This Unlocks — and What It Does Not

When a team runs this workflow for the first time, three things change. The model retrain cycle drops from a two-week manual process to an overnight job. The audit trail for every factor table is automatic and does not depend on anyone remembering to save a version. The team can run experiments by forking the Job, changing hyperparameters, and comparing out-of-time performance in MLflow without touching the production pipeline.

Now the honest caveats, because they matter and they do not disappear by running this workflow.

**Correlated features.** SHAP attribution for correlated features — area band and any socioeconomic proxy, annual mileage and vehicle type — is not uniquely defined under the default `tree_path_dependent` method. On a UK motor book where multiple correlated factors are in the model simultaneously, the default attribution can give misleading marginal relativities. Switch to `feature_perturbation="interventional"` with a representative background dataset if you have correlated features you care about separately. It is slower. It is the correct approach.

**Pure premium combination.** We extract frequency and severity relativities separately and present them side by side. We do not combine them into a pure premium relativity table by multiplying level-by-level. That multiplication is not mathematically valid — the scales are different, the link functions interact, and the portfolio averages do not cancel cleanly. The correct approach is mSHAP (Lindstrom et al., 2022), which combines SHAP values in prediction space. It is on our roadmap. Until then: present frequency and severity separately and be explicit about what combining them does and does not mean.

**Actuarial judgement.** This workflow removes the mechanical work. It does not remove the judgement. The pricing actuary reviewing the factor table before it goes into Radar is still doing the important part: checking that relativities are directionally sensible, that thin cells have not driven unstable estimates, that the GLM comparison does not reveal anything that needs explaining. What the workflow does is give that actuary more time and better information to exercise that judgement, rather than spending two weeks generating the inputs manually.

---

## Libraries and Setup

| Library | Purpose | Install |
|---------|---------|---------|
| [`shap-relativities`](https://github.com/burningcost/shap-relativities) | SHAP extraction from LightGBM | `pip install git+https://github.com/burningcost/shap-relativities` |
| [`insurance-cv`](https://github.com/burningcost/insurance-cv) | Temporal cross-validation splits | `pip install git+https://github.com/burningcost/insurance-cv` |
| `lightgbm` | Model training | `pip install lightgbm` |
| `statsmodels` | GLM benchmark | `pip install statsmodels` |
| `mlflow` | Experiment tracking | Pre-installed on Databricks Runtime 14+ |

The full notebook set — `01_data_prep`, `02_train_extract`, `03_export` — and the Job JSON definition are available as a downloadable archive. The notebooks map 1:1 to the sections above and can be imported directly into a Databricks workspace.
