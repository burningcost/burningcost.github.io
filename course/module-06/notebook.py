# Databricks notebook source
# MAGIC %md
# MAGIC # Module 6: Credibility & Bayesian Pricing - The Thin-Cell Problem
# MAGIC
# MAGIC Full workflow on synthetic UK motor data. Runs end-to-end on a single-node
# MAGIC Databricks cluster (DBR 14.x ML runtime recommended, 4+ cores).
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Generates a synthetic UK motor portfolio with area-level variation and deliberately thin cells
# MAGIC 2. Computes Bühlmann-Straub credibility estimates using the `credibility` library
# MAGIC 3. Fits a hierarchical Bayesian frequency model using the `bayesian-pricing` library
# MAGIC 4. Produces the shrinkage plot - observed rate vs credibility-weighted estimate
# MAGIC 5. Compares classical Z to Bayesian credibility_factor per segment
# MAGIC 6. Checks convergence (R-hat, ESS, divergences)
# MAGIC 7. Stores results in Delta tables with MLflow tracking
# MAGIC
# MAGIC **Runtime:** Classical credibility: < 1 minute. Bayesian MCMC: 5–15 minutes on a 4-core cluster.

# COMMAND ----------

# MAGIC %pip install \
# MAGIC   "credibility[all] @ git+https://github.com/burningcost/credibility.git" \
# MAGIC   "bayesian-pricing[all] @ git+https://github.com/burningcost/bayesian-pricing.git" \
# MAGIC   arviz \
# MAGIC   --quiet
# MAGIC # Note: these libraries are available on GitHub. PyPI publication is planned.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import warnings
import multiprocessing
from datetime import date

import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import arviz as az
import mlflow

from credibility import BuhlmannStraub
from bayesian_pricing import HierarchicalFrequency, HierarchicalSeverity
from bayesian_pricing.datasets.motor import load_motor_with_thin_cells, TRUE_AREA_PARAMS

print("Libraries imported successfully.")
print(f"Available CPU cores: {multiprocessing.cpu_count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic data
# MAGIC
# MAGIC The `load_motor_with_thin_cells` function generates a UK motor portfolio where:
# MAGIC - Most postcode districts have moderate exposure (200–2,000 policy-years)
# MAGIC - 30% of districts are deliberately thin (< 50 policy-years)
# MAGIC - True area-level frequency relativities vary from 0.7x to 2.1x the portfolio mean
# MAGIC - Claim frequency follows a Poisson process at 6.8% per annum base rate
# MAGIC
# MAGIC This setup ensures we have a realistic mix of data-rich and data-poor segments,
# MAGIC so the shrinkage effect of credibility is visible.

# COMMAND ----------

np.random.seed(42)

# load_motor_with_thin_cells returns a pandas DataFrame; convert to Polars immediately
df = pl.from_pandas(load_motor_with_thin_cells(
    n_policies=80_000,
    n_districts=120,          # 120 postcode districts
    thin_fraction=0.30,       # 30% of districts are thin (< 50 earned years)
    seed=42,
))

print(f"Portfolio: {len(df):,} policies")
print(f"Exposure:  {df['earned_years'].sum():.0f} earned years")
print(f"Claims:    {df['claim_count'].sum():,}")
print(f"Raw frequency: {df['claim_count'].sum() / df['earned_years'].sum():.4f} per earned year")
print()

# District exposure distribution
dist_exp = df.group_by("postcode_district").agg(
    pl.col("earned_years").sum().alias("total_earned_years")
)
exp_vals = dist_exp["total_earned_years"]
print("District exposure distribution (quartiles):")
print(f"  count  {len(exp_vals)}")
print(f"  mean   {exp_vals.mean():.1f}")
print(f"  std    {exp_vals.std():.1f}")
print(f"  min    {exp_vals.min():.1f}")
print(f"  25%    {exp_vals.quantile(0.25):.1f}")
print(f"  50%    {exp_vals.quantile(0.50):.1f}")
print(f"  75%    {exp_vals.quantile(0.75):.1f}")
print(f"  max    {exp_vals.max():.1f}")
print()
print(f"Districts with < 50 earned years: {(exp_vals < 50).sum()} / {len(exp_vals)}")
print(f"Districts with < 10 earned years: {(exp_vals < 10).sum()} / {len(exp_vals)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### True area parameters
# MAGIC
# MAGIC Because we generated the data, we know the true frequency relativities.
# MAGIC After fitting, we can check whether our credibility estimates recover them better
# MAGIC than naive observed rates do.

# COMMAND ----------

print("True area-level frequency relativities (sample):")
print(f"{'District':<20} {'True log-rel':>14} {'True relativity':>16}")
print("-" * 52)
for district, log_rel in sorted(TRUE_AREA_PARAMS.items())[:15]:
    print(f"{district:<20} {log_rel:>14.3f} {np.exp(log_rel):>16.3f}")

print(f"\n... ({len(TRUE_AREA_PARAMS)} districts total)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Aggregate to district level
# MAGIC
# MAGIC Bühlmann-Straub and HierarchicalFrequency both work on segment-level sufficient statistics.
# MAGIC We aggregate the policy-level data to one row per postcode district per year.
# MAGIC This gives us the multi-period structure that Bühlmann-Straub requires.

# COMMAND ----------

# District × year sufficient statistics - needed for B-S (multi-period structure)
# Filter to earned_years > 0.5 to exclude near-zero exposure rows that would produce
# near-infinite frequencies. Clipping to a tiny value like 1e-6 masks bad data;
# explicit filtering removes it.
dist_year = (
    df.group_by(["postcode_district", "accident_year"])
    .agg([
        pl.col("claim_count").sum().alias("claims"),
        pl.col("earned_years").sum().alias("earned_years"),
    ])
    .filter(pl.col("earned_years") > 0.5)
    .with_columns([
        (pl.col("claims") / pl.col("earned_years")).alias("claim_frequency"),
    ])
    .sort(["postcode_district", "accident_year"])
)

print(f"District × year rows: {len(dist_year):,}")
print(f"Years in data: {sorted(dist_year['accident_year'].to_list())}")
print()
print("Sample:")
print(dist_year.head(12))

# COMMAND ----------

# District-level totals - needed for HierarchicalFrequency
dist_totals = (
    df.group_by("postcode_district")
    .agg([
        pl.col("claim_count").sum().alias("claims"),
        pl.col("earned_years").sum().alias("earned_years"),
    ])
    .filter(pl.col("earned_years") > 0.5)
    .with_columns([
        (pl.col("claims") / pl.col("earned_years")).alias("observed_rate"),
    ])
    .sort("postcode_district")
)

print(f"District-level summary: {len(dist_totals)} districts")
print(dist_totals.select(["postcode_district", "claims", "earned_years", "observed_rate"]).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bühlmann-Straub credibility
# MAGIC
# MAGIC Classical credibility weighting. Fast, transparent, no MCMC required.
# MAGIC
# MAGIC Key outputs:
# MAGIC - `grand_mean`: the portfolio collective mean (mu_hat)
# MAGIC - `v_hat_`: Expected Process Variance (EPV) - within-district year-to-year variation
# MAGIC - `a_hat_`: Variance of Hypothetical Means (VHM) - between-district heterogeneity
# MAGIC - `k_`: Bühlmann's K = v/a - how much exposure you need for Z = 0.5
# MAGIC - `results`: per-district credibility factors and blended estimates
# MAGIC
# MAGIC We use log_transform=True because we are working in a multiplicative (Poisson log-link)
# MAGIC framework. Applying B-S in rate space and then converting to relativities introduces
# MAGIC a Jensen's inequality bias. log_transform=True applies the blending in log-rate space.

# COMMAND ----------

# credibility library expects pandas input; bridge from Polars here
bs = BuhlmannStraub(log_transform=True)
bs.fit(
    data=dist_year.to_pandas(),
    group_col="postcode_district",
    value_col="claim_frequency",
    weight_col="earned_years",
)

print("Bühlmann-Straub structural parameters:")
print(f"  Grand mean (mu):  {bs.grand_mean:.5f}  ({bs.grand_mean * 100:.3f}% per year)")
print(f"  EPV (v):          {bs.v_hat_:.7f}  (within-district variance)")
print(f"  VHM (a):          {bs.a_hat_:.7f}  (between-district variance)")
print(f"  K = v/a:          {bs.k_:.1f}")
print(f"  Implied half-credibility exposure: {bs.k_:.0f} earned years")
print()
# Z = w/(w+K); solve for w at Z = 0.50, 0.67, 0.90
print("Interpretation of K (Z = w/(w+K)):")
print(f"  A district needs {bs.k_:.0f} earned years for Z = 0.50")
print(f"  A district needs {2 * bs.k_:.0f} earned years for Z = 0.67  [w = 2K/(2K+K) = 0.667]")
print(f"  A district needs {9 * bs.k_:.0f} earned years for Z = 0.90  [w = 9K/(9K+K) = 0.900]")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Credibility factor distribution

# COMMAND ----------

# Bridge bs.results (pandas) to Polars for all downstream manipulation
bs_results = pl.from_pandas(bs.results)

print(f"Credibility factor distribution across {len(bs_results)} districts:")
z_vals = bs_results["Z"]
print(f"  count  {len(z_vals)}")
print(f"  mean   {z_vals.mean():.3f}")
print(f"  std    {z_vals.std():.3f}")
print(f"  min    {z_vals.min():.3f}")
print(f"  25%    {z_vals.quantile(0.25):.3f}")
print(f"  50%    {z_vals.quantile(0.50):.3f}")
print(f"  75%    {z_vals.quantile(0.75):.3f}")
print(f"  max    {z_vals.max():.3f}")
print()
print("Districts with Z < 0.10 (nearly all prior):")
thin = bs_results.filter(pl.col("Z") < 0.10).sort("Z")
print(thin.select(["group", "exposure", "obs_mean", "Z", "credibility_estimate"]).head(10))
print()
print("Districts with Z > 0.80 (mostly own experience):")
thick = bs_results.filter(pl.col("Z") > 0.80).sort("Z", descending=True)
print(thick.select(["group", "exposure", "obs_mean", "Z", "credibility_estimate"]).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare B-S estimates to true relativities

# COMMAND ----------

bs_vs_true = bs_results.with_columns([
    pl.col("group").map_elements(lambda d: TRUE_AREA_PARAMS.get(d, 0.0), return_dtype=pl.Float64).alias("true_log_rel"),
])
bs_vs_true = bs_vs_true.with_columns([
    (bs.grand_mean * (pl.col("true_log_rel").exp())).alias("true_rate"),
])

mse_observed = ((bs_vs_true["obs_mean"] - bs_vs_true["true_rate"]) ** 2).mean()
mse_bs = ((bs_vs_true["credibility_estimate"] - bs_vs_true["true_rate"]) ** 2).mean()

mape_observed = ((bs_vs_true["obs_mean"] - bs_vs_true["true_rate"]).abs() / bs_vs_true["true_rate"]).mean()
mape_bs = ((bs_vs_true["credibility_estimate"] - bs_vs_true["true_rate"]).abs() / bs_vs_true["true_rate"]).mean()

print("Accuracy comparison (lower is better):")
print(f"  MSE  - Observed rate vs true rate:             {mse_observed:.8f}")
print(f"  MSE  - B-S credibility estimate vs true rate:  {mse_bs:.8f}")
print(f"  MSE reduction from B-S: {(1 - mse_bs / mse_observed) * 100:.1f}%")
print()
print(f"  MAPE - Observed rate vs true rate:             {mape_observed * 100:.2f}%")
print(f"  MAPE - B-S credibility estimate vs true rate:  {mape_bs * 100:.2f}%")
print(f"  MAPE reduction from B-S: {(1 - mape_bs / mape_observed) * 100:.1f}%")
print()
print("Note: MSE gives disproportionate weight to thin cells with high rate volatility.")
print("MAPE is more informative about typical accuracy across the portfolio.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Hierarchical Bayesian frequency model
# MAGIC
# MAGIC We now fit a Poisson hierarchical model using the `bayesian-pricing` library.
# MAGIC This gives us:
# MAGIC - Posterior distributions (not just point estimates) for each district's rate
# MAGIC - A Bayesian credibility_factor comparable to B-S Z
# MAGIC - Proper uncertainty quantification for thin cells
# MAGIC
# MAGIC The model uses non-centered parameterization by default to avoid the funnel
# MAGIC geometry that causes divergent transitions in standard hierarchical models.

# COMMAND ----------

n_chains = min(4, multiprocessing.cpu_count())
print(f"Fitting HierarchicalFrequency with {n_chains} chains...")

mlflow.set_experiment("/pricing/credibility-bayesian/module06")

with mlflow.start_run(run_name="hierarchical_frequency_v1"):

    hf = HierarchicalFrequency(
        group_cols=["postcode_district"],
        claims_col="claims",
        exposure_col="earned_years",
        n_chains=n_chains,
        n_samples=1000,
        n_warmup=1000,
        target_accept=0.90,
    )

    # HierarchicalFrequency expects pandas; bridge from Polars
    hf.fit(dist_totals.to_pandas())

    print("Fit complete.")

    # ---- Convergence diagnostics ----
    trace = hf.posteriors
    rhat = az.rhat(trace)
    ess_bulk = az.ess(trace, method="bulk")

    max_rhat = float(rhat.max().to_array().max())
    min_ess = float(ess_bulk.min().to_array().min())
    n_div = int(trace.sample_stats["diverging"].sum())

    # Variance components need higher ESS than the global minimum.
    # Underpowered sigma estimation is the most common way Bayesian credibility
    # factors are wrong without the model appearing to fail convergence.
    sigma_ess = float(ess_bulk["sigma_postcode_district"].min())

    print()
    print("Convergence diagnostics:")
    print(f"  Max R-hat:              {max_rhat:.4f}  ({'OK' if max_rhat < 1.01 else 'INVESTIGATE'})")
    print(f"  Min ESS (bulk):         {min_ess:.0f}  ({'OK' if min_ess > 400 else 'INVESTIGATE'})")
    print(f"  sigma ESS:              {sigma_ess:.0f}  ({'OK' if sigma_ess > 1000 else 'INVESTIGATE - increase n_samples'})")
    print(f"  Divergences:            {n_div}  ({'OK' if n_div == 0 else 'INVESTIGATE'})")

    # ---- Variance components ----
    sigma_district = float(trace.posterior["sigma_postcode_district"].mean())
    print()
    print("Variance components:")
    print(f"  sigma_postcode_district: {sigma_district:.4f}  (log-scale between-district SD)")
    print(f"  Implied between-district range (±1 SD): [{np.exp(-sigma_district):.3f}, {np.exp(sigma_district):.3f}]")

    # ---- MLflow logging ----
    mlflow.log_metric("max_rhat", max_rhat)
    mlflow.log_metric("min_ess_bulk", min_ess)
    mlflow.log_metric("sigma_ess", sigma_ess)
    mlflow.log_metric("n_divergences", n_div)
    mlflow.log_metric("n_segments", len(dist_totals))
    mlflow.log_metric("grand_mean", hf.grand_mean_)
    mlflow.log_metric("sigma_district", sigma_district)

    trace.to_netcdf("/tmp/posteriors_module06.nc")
    mlflow.log_artifact("/tmp/posteriors_module06.nc", "posteriors")

    results_path = "/tmp/credibility_results_module06.csv"
    hf.results.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path, "results")

    print()
    print("Run logged to MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bayesian results preview

# COMMAND ----------

# hf.results is a pandas DataFrame from the bayesian-pricing library
# Convert to Polars for display and all subsequent operations
results = pl.from_pandas(hf.results)

print("HierarchicalFrequency results (sample):")
print(results.select([
    "postcode_district", "claims", "earned_years",
    "observed_rate", "posterior_mean", "posterior_sd",
    "credibility_factor", "lower_90", "upper_90"
]).head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The shrinkage plot
# MAGIC
# MAGIC This is the key diagnostic. We plot observed rates against posterior means.
# MAGIC - Points near the 45° line: dense segments trusted by the model
# MAGIC - Points near the horizontal (grand mean) line: thin segments shrunk toward the portfolio average
# MAGIC - Colour = Bayesian credibility factor Z (green = high Z, red = low Z)
# MAGIC - Point size = log exposure

# COMMAND ----------

# Merge in true rates for overlay (bridge to numpy for plotting)
true_log_rels = np.array([TRUE_AREA_PARAMS.get(d, 0.0) for d in results["postcode_district"].to_list()])
results = results.with_columns([
    pl.Series("true_log_rel", true_log_rels),
    pl.Series("true_rate", hf.grand_mean_ * np.exp(true_log_rels)),
])

log_exposure = np.log1p(results["earned_years"].to_numpy())
sizes = 20 + 100 * (log_exposure - log_exposure.min()) / (log_exposure.max() - log_exposure.min())

fig, ax = plt.subplots(figsize=(10, 8))

sc = ax.scatter(
    results["observed_rate"].to_numpy(),
    results["posterior_mean"].to_numpy(),
    s=sizes,
    alpha=0.65,
    c=results["credibility_factor"].to_numpy(),
    cmap="RdYlGn",
    vmin=0, vmax=1,
    edgecolors="none",
    zorder=3,
)

# 45-degree line (no shrinkage)
obs_arr = results["observed_rate"].to_numpy()
rate_range = [obs_arr.min() * 0.9, obs_arr.max() * 1.1]
ax.plot(rate_range, rate_range, "k--", alpha=0.25, lw=1.5, label="Observed = estimate (no shrinkage)")

# Grand mean line
ax.axhline(hf.grand_mean_, color="steelblue", linestyle=":", alpha=0.6, lw=1.5,
           label=f"Grand mean = {hf.grand_mean_:.4f}")

plt.colorbar(sc, label="Credibility factor Z (Bayesian)")
ax.set_xlabel("Observed claim frequency")
ax.set_ylabel("Posterior mean claim frequency")
ax.set_title("Shrinkage plot: credibility effect on thin cells\n"
             "(point size = log exposure, colour = credibility factor Z)")
ax.legend(loc="upper left")
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare Bühlmann-Straub Z to Bayesian credibility_factor

# COMMAND ----------

# Join Polars DataFrames
bs_for_join = bs_results.rename({
    "group": "postcode_district",
    "Z": "Z_bs",
    "credibility_estimate": "bs_estimate",
}).select(["postcode_district", "Z_bs", "bs_estimate"])

comparison = results.join(bs_for_join, on="postcode_district", how="inner")

print("Credibility factor comparison: Bühlmann-Straub Z vs Bayesian Z")
print()
print(comparison.select([
    "postcode_district", "earned_years",
    "Z_bs", "credibility_factor",
    "bs_estimate", "posterior_mean",
]).head(20))

# COMMAND ----------

# Correlation between B-S Z and Bayesian Z (numpy for scalar stats)
z_bs_arr = comparison["Z_bs"].to_numpy()
z_bay_arr = comparison["credibility_factor"].to_numpy()
corr = np.corrcoef(z_bs_arr, z_bay_arr)[0, 1]
mad = np.abs(z_bs_arr - z_bay_arr).mean()

print(f"\nCorrelation between B-S Z and Bayesian credibility_factor: {corr:.4f}")
print(f"Mean absolute difference in Z values: {mad:.4f}")
print()
print("Note: the Bayesian credibility_factor is a numerical approximation from posterior moments,")
print("not an exact quantity for this non-conjugate Poisson-lognormal model.")
print("For very thin groups (Z < 0.10) the two may diverge beyond the Normal approximation alone.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot Z comparison

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Z vs Z scatter
ax = axes[0]
ax.scatter(comparison["Z_bs"].to_numpy(), comparison["credibility_factor"].to_numpy(),
           alpha=0.6, s=30, edgecolors="none", c="steelblue")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1.5)
ax.set_xlabel("Bühlmann-Straub Z")
ax.set_ylabel("Bayesian credibility factor")
ax.set_title("Credibility factor comparison\nB-S vs Bayesian")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Right: estimate vs estimate scatter
ax = axes[1]
bs_est_arr = comparison["bs_estimate"].to_numpy()
post_mean_arr = comparison["posterior_mean"].to_numpy()
ax.scatter(bs_est_arr, post_mean_arr,
           alpha=0.6, s=30, edgecolors="none", c="darkorange")
est_range = [min(bs_est_arr.min(), post_mean_arr.min()) * 0.9,
             max(bs_est_arr.max(), post_mean_arr.max()) * 1.1]
ax.plot(est_range, est_range, "k--", alpha=0.3, lw=1.5)
ax.set_xlabel("Bühlmann-Straub estimate")
ax.set_ylabel("Bayesian posterior mean")
ax.set_title("Credibility estimate comparison\nB-S vs Bayesian")

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Posterior predictive check
# MAGIC
# MAGIC Simulate datasets from the fitted Bayesian model and compare to observed data.
# MAGIC If the model is well-calibrated, the observed claim counts should fall within
# MAGIC the posterior predictive distribution.

# COMMAND ----------

ppc = hf.posterior_predictive_check(dist_totals.to_pandas())

# Compare observed total claims to PPC distribution
ppc_total_claims = ppc["total_claims"]
obs_total = dist_totals["claims"].sum()

print("Posterior predictive check - total claims:")
print(f"  Observed: {obs_total:,}")
print(f"  PPC mean: {ppc_total_claims.mean():.0f}")
print(f"  PPC 5th percentile:  {np.percentile(ppc_total_claims, 5):.0f}")
print(f"  PPC 95th percentile: {np.percentile(ppc_total_claims, 95):.0f}")
in_interval = np.percentile(ppc_total_claims, 5) <= obs_total <= np.percentile(ppc_total_claims, 95)
print(f"  Observed in 90% predictive interval: {in_interval}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### PPC: district-level calibration
# MAGIC
# MAGIC What fraction of districts have their observed claim count inside the 90% posterior
# MAGIC predictive interval? We expect roughly 90% - that is what "calibrated" means.
# MAGIC
# MAGIC If district-level 90% PPC coverage is materially below 90% (e.g. 75%), the model
# MAGIC is misspecified. Options: switch to Negative Binomial, revisit the random effect
# MAGIC prior, or revert to Bühlmann-Straub and document why the Bayesian model failed.

# COMMAND ----------

ppc_calibration = ppc["district_calibration"]  # DataFrame: district, obs_claims, lower_90, upper_90
covered = ((ppc_calibration["obs_claims"] >= ppc_calibration["lower_90"]) &
           (ppc_calibration["obs_claims"] <= ppc_calibration["upper_90"]))
coverage_rate = covered.mean()

print(f"District-level 90% PPC coverage: {coverage_rate * 100:.1f}%  (target: 90%)")

if abs(coverage_rate - 0.90) < 0.05:
    print("  Coverage is within 5pp of target - model is well calibrated.")
elif coverage_rate < 0.85:
    print("  Coverage is below 85% - model may be overconfident.")
    print("  Actions: (1) try likelihood='negative_binomial', (2) widen sigma prior,")
    print("  (3) if persistent, revert to Bühlmann-Straub and document why.")
else:
    print("  Coverage is above 95% - model may be underconfident (too much uncertainty).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Recovery check: how well do credibility estimates recover true rates?

# COMMAND ----------

true_rates = pl.Series(
    "true_rate",
    hf.grand_mean_ * np.exp(
        [TRUE_AREA_PARAMS.get(d, 0.0) for d in comparison["postcode_district"].to_list()]
    )
)
comparison = comparison.with_columns(true_rates)

obs_arr   = comparison["observed_rate"].to_numpy()
bs_arr    = comparison["bs_estimate"].to_numpy()
bayes_arr = comparison["posterior_mean"].to_numpy()
true_arr  = comparison["true_rate"].to_numpy()

mse_observed = ((obs_arr   - true_arr) ** 2).mean()
mse_bs       = ((bs_arr    - true_arr) ** 2).mean()
mse_bayes    = ((bayes_arr - true_arr) ** 2).mean()

mape_observed = (np.abs(obs_arr   - true_arr) / true_arr).mean()
mape_bs       = (np.abs(bs_arr    - true_arr) / true_arr).mean()
mape_bayes    = (np.abs(bayes_arr - true_arr) / true_arr).mean()

print("MSE vs true rate (lower is better):")
print(f"  Observed rate:              {mse_observed:.8f}  (no pooling)")
print(f"  Bühlmann-Straub estimate:   {mse_bs:.8f}  ({(1 - mse_bs/mse_observed)*100:.1f}% reduction)")
print(f"  Bayesian posterior mean:    {mse_bayes:.8f}  ({(1 - mse_bayes/mse_observed)*100:.1f}% reduction)")
print()
print("MAPE vs true rate (lower is better):")
print(f"  Observed rate:              {mape_observed * 100:.2f}%")
print(f"  Bühlmann-Straub estimate:   {mape_bs * 100:.2f}%  ({(1 - mape_bs/mape_observed)*100:.1f}% reduction)")
print(f"  Bayesian posterior mean:    {mape_bayes * 100:.2f}%  ({(1 - mape_bayes/mape_observed)*100:.1f}% reduction)")
print()
print("Both credibility methods substantially outperform naive observed rates.")
print("Bayesian and B-S are close - the Normal approximation in B-S works well here.")

# COMMAND ----------

# Split by exposure tier using Polars when/then/otherwise (no pd.cut)
comparison = comparison.with_columns(
    pl.when(pl.col("earned_years") < 50)
    .then(pl.lit("< 50 yr (thin)"))
    .when(pl.col("earned_years") < 200)
    .then(pl.lit("50-200 yr"))
    .when(pl.col("earned_years") < 1000)
    .then(pl.lit("200-1000 yr"))
    .otherwise(pl.lit("> 1000 yr (dense)"))
    .alias("exposure_tier")
)

print("\nMSE and MAPE vs true rate by exposure tier:")
print()
for tier in ["< 50 yr (thin)", "50-200 yr", "200-1000 yr", "> 1000 yr (dense)"]:
    grp = comparison.filter(pl.col("exposure_tier") == tier)
    n = len(grp)
    if n == 0:
        continue
    o = grp["observed_rate"].to_numpy()
    b = grp["bs_estimate"].to_numpy()
    y = grp["posterior_mean"].to_numpy()
    t = grp["true_rate"].to_numpy()
    mse_o = ((o - t) ** 2).mean()
    mse_b = ((b - t) ** 2).mean()
    mse_y = ((y - t) ** 2).mean()
    mape_o = (np.abs(o - t) / t).mean() * 100
    mape_b = (np.abs(b - t) / t).mean() * 100
    mape_y = (np.abs(y - t) / t).mean() * 100
    print(f"  {tier} ({n} districts):")
    print(f"    MSE  - Observed: {mse_o:.8f}  |  B-S: {mse_b:.8f}  |  Bayes: {mse_y:.8f}")
    print(f"    MAPE - Observed: {mape_o:.1f}%  |  B-S: {mape_b:.1f}%  |  Bayes: {mape_y:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Format credibility estimates as a factor table
# MAGIC
# MAGIC Convert the posterior means into multiplicative relativities vs the grand mean,
# MAGIC with 90% credible intervals. This is the format your pricing committee expects.

# COMMAND ----------

factor_table = comparison.select([
    "postcode_district", "earned_years", "claims",
    "observed_rate", "posterior_mean", "lower_90", "upper_90",
    "credibility_factor",
])

# Convert to relativities vs grand mean
factor_table = factor_table.with_columns([
    (pl.col("posterior_mean") / hf.grand_mean_).alias("relativity"),
    (pl.col("lower_90") / hf.grand_mean_).alias("ci_lower"),
    (pl.col("upper_90") / hf.grand_mean_).alias("ci_upper"),
    (pl.col("observed_rate") / hf.grand_mean_).alias("observed_relativity"),
])

# Round for presentation
factor_table = factor_table.with_columns([
    pl.col("relativity").round(3),
    pl.col("ci_lower").round(3),
    pl.col("ci_upper").round(3),
    pl.col("credibility_factor").round(3),
    pl.col("observed_relativity").round(3),
])

factor_table_display = factor_table.sort("relativity", descending=True)

print("Credibility-weighted factor table (sorted by relativity, top 15):")
print()
print(factor_table_display.select([
    "postcode_district", "earned_years", "observed_relativity",
    "relativity", "ci_lower", "ci_upper", "credibility_factor"
]).head(15))

print()
print(f"\nGrand mean (base rate): {hf.grand_mean_:.5f} ({hf.grand_mean_ * 100:.3f}% per year)")
print("Relativities are multiplicative vs the grand mean.")
print("90% credible intervals widen substantially for thin cells - this is correct, not a bug.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Write results to Unity Catalog
# MAGIC
# MAGIC Hard gate on convergence: if the model has not converged or has divergences,
# MAGIC we raise an error rather than silently write potentially biased posteriors.
# MAGIC A model that writes bad estimates to Delta is worse than one that fails loudly.

# COMMAND ----------

if max_rhat > 1.01 or n_div > 0:
    raise ValueError(
        f"Model failed convergence: max_rhat={max_rhat:.4f}, divergences={n_div}. "
        "Results not written to Unity Catalog."
    )

RUN_DATE = str(date.today())
MODEL_NAME_BAYES = "hierarchical_freq_v1_module06"
MODEL_NAME_BS = "buhlmann_straub_v1_module06"

# COMMAND ----------

# Bayesian results
bayes_out = factor_table.with_columns([
    pl.lit(MODEL_NAME_BAYES).alias("model_name"),
    pl.lit("hierarchical_poisson").alias("model_type"),
    pl.lit(RUN_DATE).alias("run_date"),
    pl.lit(max_rhat).alias("max_rhat"),
    pl.lit(n_div).alias("n_divergences"),
])

(
    spark.createDataFrame(bayes_out.to_pandas())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module06_credibility_estimates")
)

print(f"Written {len(bayes_out)} rows to main.pricing.module06_credibility_estimates")

# COMMAND ----------

# Bühlmann-Straub results
bs_out = bs_results.rename({
    "group": "postcode_district",
    "obs_mean": "observed_rate",
    "credibility_estimate": "bs_estimate",
}).with_columns([
    (pl.col("bs_estimate") / bs.grand_mean).alias("bs_relativity"),
    pl.lit(MODEL_NAME_BS).alias("model_name"),
    pl.lit("buhlmann_straub").alias("model_type"),
    pl.lit(RUN_DATE).alias("run_date"),
    pl.lit(bs.grand_mean).alias("grand_mean"),
    pl.lit(bs.v_hat_).alias("v_hat"),
    pl.lit(bs.a_hat_).alias("a_hat"),
    pl.lit(bs.k_).alias("K"),
])

(
    spark.createDataFrame(bs_out.to_pandas())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module06_bs_estimates")
)

print(f"Written {len(bs_out)} rows to main.pricing.module06_bs_estimates")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary

# COMMAND ----------

print("=" * 65)
print("MODULE 6 - CREDIBILITY & BAYESIAN PRICING SUMMARY")
print("=" * 65)
print()
print("Portfolio:")
print(f"  {len(df):,} policies, {df['earned_years'].sum():.0f} earned years")
print(f"  {len(dist_totals)} postcode districts")
thin_count = dist_totals.filter(pl.col("earned_years") < 50).height
print(f"  {thin_count} thin districts (< 50 earned years)")
print()
print("Bühlmann-Straub:")
print(f"  Grand mean:  {bs.grand_mean:.5f}")
print(f"  K = v/a:     {bs.k_:.1f}  (half-credibility at {bs.k_:.0f} earned years)")
z_min = bs_results["Z"].min()
z_max = bs_results["Z"].max()
print(f"  Z range:     [{z_min:.3f}, {z_max:.3f}]")
print(f"  MSE vs true: {mse_bs:.8f}  ({(1 - mse_bs/mse_observed)*100:.1f}% vs observed)")
print(f"  MAPE vs true:{mape_bs * 100:.2f}%  ({(1 - mape_bs/mape_observed)*100:.1f}% vs observed)")
print()
print("Bayesian (HierarchicalFrequency):")
print(f"  Grand mean:             {hf.grand_mean_:.5f}")
print(f"  sigma_district:         {sigma_district:.4f}  (log-scale between-district SD)")
print(f"  Max R-hat:              {max_rhat:.4f}")
print(f"  sigma ESS:              {sigma_ess:.0f}")
print(f"  Divergences:            {n_div}")
z_bay_min = results["credibility_factor"].min()
z_bay_max = results["credibility_factor"].max()
print(f"  Z range:                [{z_bay_min:.3f}, {z_bay_max:.3f}]")
print(f"  MSE vs true:            {mse_bayes:.8f}  ({(1 - mse_bayes/mse_observed)*100:.1f}% vs observed)")
print(f"  MAPE vs true:           {mape_bayes * 100:.2f}%  ({(1 - mape_bayes/mape_observed)*100:.1f}% vs observed)")
print(f"  PPC district coverage:  {coverage_rate * 100:.1f}%  (target 90%)")
print()
print("Delta tables written:")
print("  main.pricing.module06_credibility_estimates")
print("  main.pricing.module06_bs_estimates")
