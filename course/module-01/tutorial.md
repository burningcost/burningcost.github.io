---
layout: default
title: "Module 1: Databricks for Pricing Teams"
---

# Module 1: Databricks for Pricing Teams

**Modern Insurance Pricing with Python and Databricks**

---

## Who this module is for

You price motor or home insurance for a living. Your team has probably outgrown whatever it is running now. The main modelling script is a 3,000-line notebook that nobody touches without anxiety. Model results live in a shared drive folder whose organisation is understood by one person. The model retrain involves three people, two of whom need to be available on the same day. When the FCA asks you to demonstrate that your pricing is consistent with Consumer Duty, the honest answer is that you have a spreadsheet and some emails.

This module teaches you to set Databricks up for a pricing project, specifically. Not the general-purpose data engineering setup from the official docs - the setup that a pricing actuary can actually work in.

---

## What Databricks actually is

Ignore the marketing for a moment. Databricks is:

1. **Managed Apache Spark**: distributed compute provisioned and managed for you. You write Python or SQL. Databricks handles the cluster infrastructure.
2. **Delta Lake**: a storage format for tabular data that adds transactions, versioning, and schema enforcement on top of Parquet files. Think of it as a proper database table stored in cloud object storage.
3. **MLflow**: an experiment tracking system for model training runs. It logs parameters, metrics, and artefacts so you can answer "which run produced this model?" in under a minute.
4. **Unity Catalog**: a data governance layer. Access control, data lineage, and a central metastore so that every table, view, and model in the organisation lives in one place.
5. **A workspace**: a web UI and API that ties it all together. Your IT team provisions it in Azure, AWS, or GCP and you get a URL to log into.

For pricing work, the stack you actually use day-to-day is: Spark for reading and writing large datasets, Polars for single-machine data manipulation, Delta tables instead of CSV data passes, and MLflow to track what you did and when. Unity Catalog sits underneath and provides the governance layer your compliance team needs.

Most large UK insurers are running Databricks on Azure. Some are on AWS. A few have on-premises deployments. The product works the same way regardless of cloud provider; the difference is in how your IT team has set it up.

---

## Free Edition vs paid

Databricks Free Edition gives you a workspace with:

- Single-user clusters (no shared compute)
- 15 GB of storage
- Unity Catalog enabled
- MLflow experiment tracking
- Delta Lake

What you cannot do on Free Edition:

- Databricks Workflows (scheduled jobs)
- SQL Warehouses (serverless SQL compute)
- Team collaboration features (multiple users on shared clusters)
- Cluster policies (cap maximum cluster sizes)
- Model Serving

For this module and the exercises, Free Edition is sufficient. When you move to a production setup at your employer, you will be on a paid workspace provisioned by your platform team.

---

## Your workspace: realistic expectations

The workspace structure shown in tutorials - a clean catalog called `pricing`, schemas named after product lines, neat folder hierarchies - is what a greenfield setup looks like. Real environments are different.

In most organisations, a data engineering team provisioned Databricks before the pricing team had any involvement. You will find a catalog called `prod_data_warehouse` or `insurance_datalake_v2`, schemas named after data engineering teams rather than product lines, and governance permissions held by a central platform team who take three weeks to respond to access requests.

The most important advice for the first month: work with your platform team to agree on a naming convention before your first model run. Renaming Unity Catalog objects after downstream notebooks have hardcoded references to them is painful. Agree on the catalog structure first, then build.

This module shows the clean setup you should aim for. When you encounter an existing environment, adapt it - the principles carry over even if the exact names do not.

---

## Cluster configuration

### What type of cluster to use

For pricing work, you almost always want one of two setups:

**Single-node cluster for development:**
```yaml
cluster_type: single_node
node_type_id: Standard_DS3_v2  # Azure; 4 cores, 14 GB RAM
runtime_version: 14.3.x-scala2.12  # Databricks Runtime 14.3 LTS
autotermination_minutes: 60
spark_conf:
  spark.databricks.cluster.profile: singleNode
  spark.master: local[*]
```

A single-node cluster runs Spark in local mode. This sounds like a compromise but it is the right choice for most pricing development work. A CatBoost model on 500,000 policies trains in 5-10 minutes on a single node. Polars runs faster on a single node than on a distributed cluster for datasets under a few hundred million rows because it avoids the network shuffle overhead entirely. You only need distributed compute when your data genuinely does not fit on one machine.

**Job cluster for batch runs:**
```yaml
cluster_type: job
node_type_id: Standard_DS4_v2  # 8 cores, 28 GB RAM
num_workers: 0  # single node for the job
runtime_version: 14.3.x-scala2.12
autotermination_minutes: 10
```

Job clusters start fresh for each job run and terminate when the job finishes. Use these for scheduled model retrains and data pipeline runs - not for interactive development. They cost less than leaving a cluster running.

### Cost management

This section exists because every UK insurer that has adopted Databricks has had at least one month where an unattended cluster ran over the weekend. An 8-core Standard_DS3_v2 cluster left running costs roughly £150-300 per week depending on your agreement.

**Autotermination:** Always set `autotermination_minutes`. Sixty minutes is reasonable for development clusters.

**Cluster policies:** If you have admin access, set cluster policies to cap the maximum cluster size junior analysts can launch. This is not about distrust - it is about preventing expensive mistakes.

**Cost attribution:** Unity Catalog tags can attribute compute spend to projects. If your platform team has set this up, use it. If not, ask them to.

**SQL Warehouses:** For read-only queries against Delta tables - running reports, checking data quality, exploring results - a Databricks SQL Warehouse is often cheaper than spinning up a cluster. Warehouses are serverless and charge only for the time queries are actually running.

---

## Unity Catalog for pricing data

### The structure

Unity Catalog organises data in three levels: Catalog > Schema > Table. The rough equivalent in traditional databases is: Database Server > Database > Table.

For a pricing team, a sensible structure is:

```
pricing                        (catalog - created by platform team)
├── motor                      (schema - one per product)
│   ├── policies               (raw exposure data)
│   ├── claims                 (raw claims data)
│   ├── model_relativities     (model output, feeds rating engine)
│   └── model_run_log          (audit trail, append-only)
├── home
│   └── ...
└── governance
    └── model_run_log          (cross-product audit table)
```

In practice, you may not get to create this structure from scratch. But this is the target shape to aim for.

### Creating a schema for pricing work

The catalog itself is created by your platform team. Assuming you have permission to create schemas within it:

```python
# Set these constants at the top of every notebook
CATALOG = "pricing"
SCHEMA  = "motor"
VOLUME  = "raw"

# Unity Catalog uses three-part names: catalog.schema.table
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA} COMMENT 'Motor pricing models and data'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
```

A note on f-strings and SQL: `CATALOG`, `SCHEMA`, and `VOLUME` are hardcoded constants in a notebook, so f-string interpolation here is fine. Do not use f-strings to interpolate user-supplied values into SQL - if application code reads a table name from user input and substitutes it into a query, use parameterised queries instead.

After a model run, open the Unity Catalog UI, navigate to your table, and click the Lineage tab. You will see which tables fed it and which notebook run produced it. You did not have to instrument this - it is a by-product of using Unity Catalog table references rather than DBFS paths.

One limitation worth knowing: when you call `.toPandas()` on a Spark DataFrame and then work with it as a Polars DataFrame (via `pl.from_pandas()`), the subsequent Python operations are invisible to Unity Catalog lineage tracking. The lineage graph shows the source table and the destination table, but not the intermediate transformations. For most pricing teams this is acceptable. For audit purposes it means the lineage is directionally correct but incomplete.

### Access control

Unity Catalog uses GRANT/REVOKE syntax that looks like SQL but applies at the object level:

```sql
-- Pricing team can read and write to the motor schema
GRANT CREATE TABLE, SELECT, MODIFY ON SCHEMA pricing.motor TO GROUP pricing_team;

-- Compliance team gets read access everywhere, writes nowhere
GRANT SELECT ON CATALOG pricing TO GROUP compliance_team;

-- Audit table is append-only from notebooks: grant INSERT but not UPDATE/DELETE
GRANT INSERT ON TABLE pricing.motor.model_run_log TO GROUP pricing_team;
```

One question the tutorial cannot answer for you: who can write to production tables? In a real environment, the distinction between who can model (write to development schemas) and who can promote to production is your pricing governance decision, not a Databricks configuration. Databricks can enforce the policy you design - it cannot design the policy for you.

A common pattern: modelling notebooks write to `pricing.motor_dev.model_relativities`. A separate promotion step, gated by a manual approval in your governance process, copies the table to `pricing.motor.model_relativities` which feeds the rating engine. Databricks Workflows can trigger on approval if you have the paid tier.

---

## Delta tables as a replacement for flat-file data passes

### Why flat files create problems

The typical pricing team data flow:

1. Data team exports claims data as a CSV or SAS file and drops it in a shared drive folder
2. Pricing analyst reads the file, does some manipulation, saves another file
3. That file is read by another analyst or a model script
4. Something changes upstream, the file gets re-exported, and nobody knows which model run used which version of the data

Delta tables solve this with versioning. Every write to a Delta table creates a new version. You can query any historical version, and every read is logged in the audit trail.

### Creating a Delta table for policy exposure

Here is the complete pattern for creating a Delta table from a Polars DataFrame:

```python
import polars as pl

# Polars for data manipulation
df = pl.read_parquet("/Volumes/pricing/motor/raw/policies_2024.parquet")

# Schema enforcement: define the schema explicitly rather than inferring it
df = df.with_columns([
    pl.col("policy_id").cast(pl.Utf8),
    pl.col("inception_date").cast(pl.Date),
    pl.col("exposure").cast(pl.Float64),
    pl.col("vehicle_group").cast(pl.Categorical),
    pl.col("region").cast(pl.Categorical),
    pl.col("claim_count").cast(pl.Int32),
    pl.col("claim_amount").cast(pl.Float64),
])

# PySpark does not yet natively accept Polars DataFrames
# Convert to pandas at the Spark boundary only
spark_df = spark.createDataFrame(df.to_pandas())

# Save as Delta table - three-part name goes to Unity Catalog
(spark_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.policies"))
```

### Table properties: what to set and why

```sql
ALTER TABLE pricing.motor.policies
SET TBLPROPERTIES (
    'delta.autoCompact.enabled' = 'true',
    'delta.logRetentionDuration' = 'interval 7 years',
    'delta.deletedFileRetentionDuration' = 'interval 7 years'
);
```

`delta.autoCompact.enabled` keeps file sizes sensible as the table accumulates writes. Without it, you end up with thousands of small Parquet files and query performance degrades.

`delta.logRetentionDuration` controls how long Delta keeps the transaction log and old file versions. For production pricing tables, 7 years matches the FCA data retention guidance for Consumer Duty evidence. For development tables, 90 days is usually sufficient.

**A note on GDPR:** Delta's version history retains all historical data, including PII from previous versions. If your book contains policyholders who exercise their GDPR right to erasure, you must remove their data from both the current table and the historical versions. The options are: (a) VACUUM the table to remove old versions (losing time travel history), (b) maintain a separate erasure log and filter at query time, or (c) discuss column-level masking with your DPO. None of these are simple. Before setting retention policies on tables containing PII, discuss the tension between data retention for FCA audit purposes and right to erasure with your DPO.

### Time travel: querying historical versions

```python
# The version at any point in time
df_yesterday = spark.read.format("delta").option("timestampAsOf", "2024-11-01").table(f"{CATALOG}.{SCHEMA}.policies")

# Or by version number
df_v3 = spark.read.format("delta").option("versionAsOf", 3).table(f"{CATALOG}.{SCHEMA}.policies")
```

This is useful for reproducibility. When a model retrain produces unexpected results, you can query the data as it was at the time of the previous run.

### Updating the bordereaux: MERGE operations

Rather than overwriting an entire table when new claims data arrives, MERGE lets you update existing rows and insert new ones in a single atomic operation:

```python
# Register the update data as a temp view first - separate statement
update_df.createOrReplaceTempView("bordereaux_updates")

# Then run the MERGE
spark.sql(f"""
    MERGE INTO {CATALOG}.{SCHEMA}.claims AS target
    USING bordereaux_updates AS source
    ON target.claim_ref = source.claim_ref
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
```

Note the order: `createOrReplaceTempView()` must be a separate statement before the `spark.sql()` call. `createOrReplaceTempView()` returns `None`, so any attempt to chain it inside a dict argument to `spark.sql()` will fail at runtime.

Each MERGE is atomic - it either completes fully or rolls back. The Delta transaction log records exactly what changed and when.

### Z-ordering for query performance

For tables you query repeatedly with the same filters, Z-ordering co-locates related data on disk:

```sql
OPTIMIZE pricing.motor.claims
ZORDER BY (region, vehicle_group)
```

This improves query performance for filters like `WHERE region = 'London' AND vehicle_group = 'C'`. Run OPTIMIZE when query performance degrades, not on a rigid schedule.

---

## Reading existing data sources

### From Unity Catalog Volumes

Unity Catalog Volumes are accessible via the `/Volumes/` path (not `/dbfs/Volumes/` - that is a legacy DBFS path that will fail or silently read from the wrong location):

```python
# Correct: Unity Catalog Volume path
df = pl.read_parquet("/Volumes/pricing/motor/raw/policies_2024.parquet")

# Reading a CSV with explicit schema
df = pl.read_csv(
    "/Volumes/pricing/motor/raw/policies_2024.csv",
    dtypes={"policy_id": pl.Utf8, "claim_amount": pl.Float64}
)
```

### From SAS files

```python
# Install pyreadstat - restart the Python kernel after installing
uv pip install pyreadstat
dbutils.library.restartPython()

import pyreadstat

# Unity Catalog Volume path - not /dbfs/Volumes/
df_pd, meta = pyreadstat.read_sas7bdat("/Volumes/pricing/motor/raw/claims_triangle.sas7bdat")

# Column name note: SAS 9+ allows names up to 32 characters.
# Older SAS 6 files truncate column names to 8 characters.
# Check meta.column_names if names look truncated.

df = pl.from_pandas(df_pd)
```

### From a SQL database via JDBC

```python
jdbc_df = (spark.read
    .format("jdbc")
    .option("url", "jdbc:sqlserver://your-server.database.windows.net:1433;database=PolicyAdmin")
    .option("dbtable", "dbo.motor_policies")
    .option("user", dbutils.secrets.get("pricing-scope", "db-user"))
    .option("password", dbutils.secrets.get("pricing-scope", "db-password"))
    .load())

df = pl.from_pandas(jdbc_df.toPandas())
```

One critical note: this assumes the policy admin system is network-accessible from your Databricks cluster. In most UK insurance environments, the policy admin system sits on-premises behind a corporate firewall, and Databricks runs in Azure or AWS. The cluster cannot reach the on-prem database without either a private endpoint, a VPN gateway, or an ETL pipeline that pulls data to cloud storage first.

If your policy admin system is on-premises, check with your platform team whether Databricks has network access to it. If not, data must be staged to a Volume first via your existing ETL pipeline. This is the most common integration problem pricing teams hit in the first month of a Databricks rollout.

---

## Working with data: Polars vs PySpark

The decision rule for a pricing team:

| Scenario | Use |
|----------|-----|
| Data fits on one machine (< ~50GB) | Polars |
| Data requires distributed processing | PySpark |
| Reading/writing Delta tables | PySpark (then convert to Polars) |
| CatBoost, statsmodels, scikit-learn training | Polars or pandas (these take arrays, not Spark DataFrames) |
| SQL queries against Delta tables | Spark SQL or Databricks SQL Warehouse |

For most pricing work, the pattern is: read from Delta with Spark, convert to Polars for feature engineering and EDA, train the model, log results, convert back to Spark to write outputs.

```python
# Read from Delta with Spark
spark_df = spark.table(f"{CATALOG}.{SCHEMA}.policies")

# Convert to Polars for manipulation
df = pl.from_pandas(spark_df.toPandas())

# Do your work in Polars
df_features = df.with_columns([
    (pl.col("claim_amount") / pl.col("exposure")).alias("severity"),
    pl.col("ncd_years").cast(pl.Utf8).alias("ncd_group"),
])

# Write back via Spark
spark.createDataFrame(df_features.to_pandas()).write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.features")
```

---

## MLflow experiment tracking from first principles

### What problem MLflow solves

Without experiment tracking, model development looks like this: you run a CatBoost model, it produces a result, you adjust the hyperparameters, run it again, and gradually lose track of which parameters produced which output. Three weeks later, a colleague asks why the latest model performs differently from last month's. You have a notebook with commented-out code and no clear record of what you actually ran.

MLflow solves this by logging parameters, metrics, and model artefacts in a structured way. Every training run gets a run ID that ties everything together.

### The basic pattern

```python
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier

# Set the experiment - creates it if it doesn't exist
mlflow.set_experiment("/Users/you@insurer.com/motor-frequency")

with mlflow.start_run(run_name="freq_catboost_v1") as run:
    # Log your parameters
    mlflow.log_params({
        "model_type":   "catboost",
        "objective":    "Poisson",
        "iterations":   500,
        "learning_rate": 0.05,
        "depth":        6,
        "dataset_version": 3,
        "feature_set":  "v2_with_telematics",
    })

    # Train
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Poisson",
        verbose=100,
    )
    model.fit(X_train, y_train, sample_weight=exposure_train)

    # Log metrics
    mlflow.log_metrics({
        "train_poisson_deviance": train_dev,
        "val_poisson_deviance":   val_dev,
        "val_gini":               val_gini,
    })

    # Log the model artefact
    mlflow.catboost.log_model(model, "freq_model")

    run_id = run.info.run_id

print(f"Run ID: {run_id}")
```

After this runs, open the MLflow UI (Experiments tab in the Databricks workspace), find your experiment, and you can see all runs in a table. Click any run for parameters, metrics, and the model artefact. Filter by metric value to find the best run.

### The audit trail entry

For Consumer Duty compliance, you need to be able to show what your model was doing at any point in time. MLflow gives you the model artefact and the metrics. Add an audit record that ties the MLflow run to the data version:

```python
from datetime import datetime

audit_record = {
    "run_id":          run_id,
    "model_name":      "freq_catboost_motor_v3",
    "model_version":   3,
    "training_date":   datetime.utcnow().isoformat(),
    "data_version":    3,
    "data_table":      f"{CATALOG}.{SCHEMA}.policies",
    "val_deviance":    val_dev,
    "approved_by":     None,
    "deployed_to_prod": False,
}

audit_df = spark.createDataFrame([audit_record])
audit_df.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.governance.model_run_log")
```

The audit log table should be append-only. Grant the pricing team INSERT access but not UPDATE or DELETE.

### Registering a model

Once a run has produced a model you want to keep:

```python
from mlflow import MlflowClient

client = MlflowClient()

# Register the model
model_uri = f"runs:/{run_id}/freq_model"
registered = mlflow.register_model(model_uri=model_uri, name="motor_freq_catboost")

# Use aliases rather than stages (stages are deprecated in MLflow 2.9+)
client.set_registered_model_alias(
    name="motor_freq_catboost",
    alias="production",
    version=registered.version,
)

# Tag with metadata
client.set_model_version_tag(
    name="motor_freq_catboost",
    version=registered.version,
    key="approved_by",
    value="pricing_lead",
)
```

Note: `transition_model_version_stage()` (which assigns models to "Staging", "Production", "Archived" stages) was deprecated in MLflow 2.9. Databricks Runtime 14.3 LTS ships with MLflow 2.11+. Use `set_registered_model_alias()` instead.

To load the production model:

```python
model = mlflow.catboost.load_model("models:/motor_freq_catboost@production")
```

---

## Local development with Databricks Connect

Some pricing teams find it more productive to develop locally in VS Code or PyCharm using Databricks Connect, then push to the workspace for scheduled runs. Databricks Connect lets you run Spark code locally against a remote Databricks cluster.

This module does not cover the setup in detail, but it is worth knowing the option exists. If your team does code review via Git and finds the web-based notebook editor limiting, Databricks Connect may be worth setting up. The official documentation covers the VS Code integration.

---

## A worked example: motor pricing EDA

This section shows the complete setup-to-EDA workflow on a synthetic motor dataset. The notebook in this module covers the same steps in runnable form.

### Step 1: Generate synthetic data and load to Delta

```python
import polars as pl
import numpy as np

rng = np.random.default_rng(seed=42)
n = 10_000

# Realistic NCD distribution: UK motor books are skewed towards higher NCD
# roughly: 30% at NCD 5, 20% NCD 4, 20% NCD 3, 15% NCD 2, 10% NCD 1, 5% NCD 0
ncd_probs = [0.05, 0.10, 0.15, 0.20, 0.20, 0.30]
ncd_years = rng.choice(range(6), size=n, p=ncd_probs)

df = pl.DataFrame({
    "policy_id":     [f"POL{i:06d}" for i in range(n)],
    "age_band":      rng.choice(["17-25", "26-35", "36-50", "51-65", "66+"], size=n, p=[0.08, 0.22, 0.35, 0.25, 0.10]),
    "vehicle_group": rng.choice(["A","B","C","D","E"], size=n, p=[0.15, 0.25, 0.30, 0.20, 0.10]),
    "region":        rng.choice(["London","South East","Midlands","North","Scotland","Wales"], size=n),
    "ncd_years":     ncd_years.tolist(),
    "exposure":      rng.uniform(0.25, 1.0, size=n).tolist(),
})

# Synthetic claims: Poisson frequency with multiplicative structure
base_rate   = 0.08
age_factor  = {"17-25": 2.5, "26-35": 1.4, "36-50": 1.0, "51-65": 0.9, "66+": 1.1}
vg_factor   = {"A": 0.7, "B": 0.9, "C": 1.0, "D": 1.2, "E": 1.5}
ncd_factor  = {0: 2.0, 1: 1.6, 2: 1.3, 3: 1.1, 4: 1.0, 5: 0.85}

expected_freq = np.array([
    base_rate
    * age_factor[row["age_band"]]
    * vg_factor[row["vehicle_group"]]
    * ncd_factor[row["ncd_years"]]
    * row["exposure"]
    for row in df.iter_rows(named=True)
])

claim_counts  = rng.poisson(expected_freq)
claim_amounts = np.where(
    claim_counts > 0,
    rng.gamma(shape=2.0, scale=1500.0, size=n) * claim_counts,
    0.0
)

df = df.with_columns([
    pl.Series("claim_count",  claim_counts.tolist()).cast(pl.Int32),
    pl.Series("claim_amount", claim_amounts.tolist()),
])

# Save to Delta via Spark
spark.createDataFrame(df.to_pandas()).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.policies")
```

### Step 2: Frequency one-way by rating factor

```python
# Read back from Delta
df = pl.from_pandas(spark.table(f"{CATALOG}.{SCHEMA}.policies").toPandas())

# Frequency one-way by age band
age_oneway = (
    df
    .group_by("age_band")
    .agg([
        pl.col("exposure").sum().alias("exposure"),
        pl.col("claim_count").sum().alias("claim_count"),
    ])
    .with_columns(
        (pl.col("claim_count") / pl.col("exposure")).alias("freq_pa")
    )
    .sort("age_band")
)

print(age_oneway)
```

The full notebook shows EDA for all rating factors and the distribution of claim amounts.

---

## What this module does not cover

Module 2 covers GLMs in Python - the Poisson frequency and gamma severity models that belong to this dataset. Module 3 covers CatBoost training in detail, including the hyperparameter choices that matter for insurance data. Module 8 assembles the full pipeline.

This module covers the workspace infrastructure. The goal is that by the end of it, you have a clean workspace, a Delta table with policy data, an MLflow experiment, and enough understanding of the setup to extend it when the next module introduces more modelling code.

---

## Summary

What we covered:

- Databricks is managed Spark + Delta Lake + MLflow + Unity Catalog, deployed as a workspace your IT team provisions
- Single-node clusters for development, job clusters for batch runs
- Unity Catalog gives you data governance, access control, and lineage for free once you use three-part table names
- Delta tables replace flat-file data passes with versioned, queryable, auditable tables
- MLflow tracks every model run so you can always answer "which run produced this result?"
- The FCA audit trail is an append-only Delta table that ties model metadata to data versions

The notebook for this module walks through the setup step by step on synthetic motor data. Run it in your Free Edition workspace before moving to Module 2.
