# Module 1: Databricks for Pricing Teams

Part of the **Modern Insurance Pricing with Python and Databricks** course.

---

## What this module covers

Most Databricks tutorials aimed at actuaries teach you the Databricks UI in the abstract. This module teaches you to set up a workspace for a specific purpose: pricing insurance.

The difference is in the details. Where to put your tables. How to set retention properties that satisfy an FCA audit. Why the flat-file data pass between pricing and MI teams is the thing Databricks actually fixes. How to stop losing track of which model run produced which output.

By the end of this module you have a working pricing workspace on Databricks Free Edition with:

- A Unity Catalog schema and volume for motor pricing data
- A Delta table with synthetic motor policies (10,000 rows, realistic structure)
- An MLflow experiment with a logged CatBoost frequency model
- An audit trail table that ties model runs to data versions

---

## Files

**`tutorial.md`** - The written guide. Read this first. ~4,000 words covering the full setup from cluster configuration to MLflow model registration. Written for someone who prices motor or home insurance for a living and is being asked to move to Databricks.

**`notebook.py`** - The Databricks notebook. Import this into your Databricks workspace and run it cell by cell. Covers everything in the tutorial in runnable form. All steps work on Databricks Free Edition.

---

## Requirements

- A Databricks workspace (Free Edition works for everything here)
- Python basics: you should know what a DataFrame is and be able to read a for loop
- No prior Databricks experience required - this module assumes none

---

## What you will not find here

- The vendor pitch for Databricks. We explain what it actually is, what it is good at, and what problems it does not solve.
- Pandas as the primary data manipulation library. We use Polars throughout. Pandas appears only at the Spark boundary where a conversion is unavoidable.
- Other GBM frameworks. We use CatBoost throughout.
- Simplified patterns that do not reflect real-world constraints. The tutorial covers the greenfield ideal and acknowledges that most teams adopt Databricks into an existing IT estate where the catalog is already named something else and a platform team controls access.

---

## Part of the MVP bundle

This module is included in the £295 MVP bundle (modules 1, 2, 4, 6). The full course is £495. Individual modules are £79.

See [burningcost.github.io/course](https://burningcost.github.io/course/) for the full curriculum.
