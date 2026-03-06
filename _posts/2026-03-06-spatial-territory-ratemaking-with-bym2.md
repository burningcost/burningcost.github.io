---
layout: post
title: "Your Territory Banding Is Wrong"
date: 2026-03-06
categories: [techniques]
tags: [spatial, territory, BYM2, ICAR, pymc, credibility, ratemaking, motor, python, bayesian]
---

Every UK motor insurer runs some version of the same process: take postcode sector loss ratios, run k-means, assign sectors to bands, file the bands with your actuarial peer reviewer, move on. The territory model is a solved problem. You have 12 bands. Your Gini is fine.

Except the territory model is not a solved problem. It is a problem that most teams have given up trying to solve properly.

Here is what k-means territory banding actually does. You have 11,200 postcode sectors. Most of them have thin data - median claim count per sector per year for a mid-sized UK motor book is probably 20–30 claims. You compute a loss ratio per sector from that noisy data. You run k-means on the noisy loss ratios. You get bands that reflect the noise as much as the signal. Adjacent sectors end up in different bands not because the underlying risk differs but because one of them happened to have a bad year. You then apply those bands as if they were the truth.

The classical fix for thin data is credibility theory. Bühlmann-Straub says: blend the segment's own experience with the portfolio prior, with the blend weight proportional to the exposure and inversely proportional to between-group variance. We [use this elsewhere]({{ site.baseurl }}{% post_url 2026-03-06-buhlmann-straub-credibility-in-python %}) in the rating model. But standard credibility is one-dimensional: it borrows from the portfolio mean, not from neighbours. It does not know that SW12 is adjacent to SW11 and should probably have a similar risk profile.

Geographic credibility should borrow from neighbours. The BYM2 model does exactly this, and it is what `insurance-spatial` implements.

---

## The problem in numbers

UK motor pricing at postcode sector level means 11,200 parameters. The exposure distribution across those sectors is heavily skewed. Dense urban sectors - inner London, central Manchester - have thousands of policies. Rural sectors in mid-Wales or the Scottish Highlands may have 50. A GLM with postcode sector as a categorical predictor will fit 11,200 separate territory parameters with no pooling and no regularisation.

Adjacent sectors can diverge by 30–40% on sparse data, purely because of sampling noise. Standard practice is to smooth this by banding - grouping sectors into 6 to 20 clusters. The smoothing is real but the method is ad hoc: k-means on historical loss ratios discards the spatial structure entirely. SW11 gets grouped with N1 because both had a bad year, not because they are geographically similar. The resulting bands have hard boundaries - a sector on one side of a band threshold gets a 20% loading; a sector 200 metres away on the other side gets nothing.

GBMs handle territory differently but not better. The postcode sector as a high-cardinality categorical gets target-encoded or hashed. Lat/lon as continuous features produces rectangular splits. The GBM will learn spatial structure implicitly if it exists, but you cannot extract it. There is no "territory relativity" you can put in a factor table, sign off on, or file with the regulator. When the FCA asks how you derived your geographic factors, "the GBM did it" is not a satisfying answer.

Vendor tools - Emblem, Akur8 - have some form of spatial treatment. Emblem's territory grouping is k-means with a credibility adjustment. Akur8's geomodelling, per NAIC 2022 materials, uses something like a penalised GLM with spatial regularisation, but the specific penalty is not disclosed. When you use a vendor's proprietary spatial smoothing, you are producing factors you cannot audit.

---

## Geographic credibility is spatial smoothing

The insight connecting credibility theory to spatial models is worth making explicit, because the actuarial and statistical literature got here by different routes and uses different vocabulary for the same thing.

Bühlmann-Straub credibility gives each segment a blended estimate:

```
P_i = Z_i × x̄_i + (1 − Z_i) × μ
```

where `Z_i = w_i / (w_i + K)` and the portfolio mean `μ` is the thing you borrow toward. For a territory model, borrowing toward the portfolio mean is conservative but crude. The prior should be: "risk similar to my neighbours, absent strong evidence otherwise." That is a spatial prior.

The ICAR (Intrinsic Conditional Autoregressive) model formalises this. The spatial effect `φ` for each area is defined through its pairwise differences with neighbours:

```
p(φ) ∝ exp(−½ · Σ_{i~j} (φᵢ − φⱼ)²)
```

This penalises differences between adjacent areas. Areas with no data get pulled toward their neighbours. Areas with abundant data diverge from their neighbours only when the evidence is strong. The prior is local, not global - each area borrows from its geographic context, not from the abstract portfolio mean.

The ICAR model alone has an identifiability problem: you cannot separately estimate a global intercept and the spatial mean. More practically for insurance, not all local deviation from the mean is spatial. Some areas are genuinely unusual for non-spatial reasons - a new housing estate, a stretch of road with a cluster of intersections. Pure spatial smoothing would drag those back toward their neighbours and miss the genuine signal.

The BYM2 model (Besag-York-Mollié, Riebler et al. 2016) solves both problems. It combines the ICAR spatial component with an independent IID component:

```
b_i = σ · (√(ρ/s) · φᵢ + √(1−ρ) · θᵢ)

φ ~ ICAR(W)        # spatially smooth
θ ~ Normal(0, 1)   # area-specific noise
σ ~ HalfNormal(1)  # total geographic SD
ρ ~ Beta(0.5, 0.5) # proportion that is spatial
```

The `s` is the BYM2 scaling factor - the geometric mean of the ICAR marginal variances for the given adjacency graph. It normalises `φ` to unit variance so that `ρ` is interpretable regardless of graph topology.

The `ρ` parameter is the key diagnostic. It tells you, directly from the data, how much of the residual geographic variation is spatially structured. If `ρ → 1`, nearby sectors genuinely have similar risk - spatial smoothing is adding real information. If `ρ → 0`, territory variation is area-specific noise with no geographic pattern - spatial smoothing is not warranted and you are better off with simple credibility weighting. You get this answer from the posterior rather than having to decide in advance.

---

## Introducing insurance-spatial

`insurance-spatial` wraps BYM2 for UK personal lines territory pricing. It handles the adjacency construction, model fitting via PyMC v5's `pm.ICAR`, convergence diagnostics, and territory relativity extraction.

```bash
uv pip install insurance-spatial
uv pip install "insurance-spatial[geo]"      # adds geopandas + libpysal
uv pip install "insurance-spatial[nutpie]"   # adds Rust-based NUTS sampler
```

The full pipeline in four steps.

### Step 1: Build the adjacency

For synthetic data or testing, use the built-in grid builder:

```python
from insurance_spatial import build_grid_adjacency

adj = build_grid_adjacency(10, 10, connectivity="queen")
print(f"Areas: {adj.n}")             # 100
print(f"Scaling factor: {adj.scaling_factor:.3f}")
```

For real UK data, pass a GeoJSON of postcode sector polygons:

```python
from insurance_spatial.adjacency import from_geojson

adj = from_geojson(
    "postcode_sectors.geojson",
    area_col="PC_SECTOR",
    connectivity="queen",
    fix_islands=True,
)
```

The `fix_islands=True` argument connects disconnected components (Orkney, Shetland, Isles of Scilly) to their nearest mainland sector by Euclidean centroid distance. The ICAR model requires a connected graph; this is a library-level policy choice and should be documented in your model documentation.

Queen contiguity means two sectors are adjacent if their polygons share any boundary or vertex. For UK postcode sectors this gives roughly 6 neighbours per sector on average, and ~67,000 edges across the full 11,200-sector graph.

### Step 2: Test for spatial autocorrelation before fitting

You should confirm that spatial structure exists in your data before fitting a spatial model. Moran's I is the standard test:

```python
from insurance_spatial.diagnostics import moran_i
import numpy as np

# log(observed claims / expected claims) per sector - from your base GLM
log_oe = np.log(sector_observed / sector_expected)

test = moran_i(log_oe, adj, n_permutations=999)
print(test.interpretation)
# "Significant positive spatial autocorrelation (I=0.342, p=0.001).
#  Nearby areas have similar values. Spatial smoothing is warranted."
```

A significant positive Moran's I means nearby sectors have similar residuals - the spatial model is earning its complexity. If Moran's I is not significant, you do not have sufficient evidence of spatial structure in your data, and BYM2 will fit with `ρ → 0` anyway (i.e., fall back to independent credibility weighting). It is still informative to fit it, but you should not expect large differences from a non-spatial approach.

### Step 3: Fit the model

```python
from insurance_spatial import BYM2Model

model = BYM2Model(adjacency=adj, draws=1000, chains=4)
result = model.fit(
    claims=sector_observed_claims,   # np.ndarray, shape (N,)
    exposure=sector_expected_claims, # from base GLM - the two-stage approach
)
```

The exposure argument here is the expected claims from your base GLM, not raw policy-years. This is the two-stage approach: fit a standard GLM or GBM without territory first, extract sector-level expected claims, then pass the observed and expected counts to BYM2. The spatial model captures residual geographic variation after your other rating factors. The territory factor is auditable independently of the main model.

If nutpie is installed, the library uses it automatically. For N≈2,500 postcode districts, expect 8–12 minutes on a modern machine with 4 chains. For the full 11,200 postcode sectors, plan for 20–30 minutes with nutpie on 8 cores.

### Step 4: Check convergence and extract relativities

```python
diag = result.diagnostics()

# Convergence diagnostics
print(f"Max R-hat: {diag.convergence.max_rhat:.3f}")     # want < 1.01
print(f"Min ESS:   {diag.convergence.min_ess_bulk:.0f}") # want > 400
print(f"Divergences: {diag.convergence.n_divergences}")  # want 0

# rho tells you whether spatial smoothing found anything
print(diag.rho_summary)
# parameter  mean    sd   q025  q975
# rho        0.73   0.11  0.51  0.92

# Extract territory relativities
rels = result.territory_relativities(credibility_interval=0.95)
# area    | relativity | lower | upper | b_mean | b_sd
# SW12 3  |     1.182  | 1.041 | 1.340 |  0.167 | 0.063
# SW11 6  |     1.154  | 1.019 | 1.308 |  0.143 | 0.061
# ...
```

The `relativity` column is `exp(b_i)` normalised to geometric mean 1.0. The `lower` and `upper` columns are the 95% credibility interval. These go directly into your production GLM as a fixed offset - `b_mean` is the `ln_offset` column you import into Emblem or Radar.

Do not skip the convergence check. MCMC without diagnostics is not production-ready. R-hat above 1.01 on any parameter means the chains did not mix; the result cannot be used. If you see divergences after the default settings, increase `target_accept` to 0.95 in `BYM2Model`.

---

## The rho diagnostic and what it tells you

The posterior of `ρ` is more informative than most spatial model outputs because it directly answers the question you actually care about: is the geographic variation in my data spatially structured, or is it noise?

A posterior mean `ρ` of 0.73 (with 95% CI of 0.51–0.92, as in the example above) means roughly 73% of the residual territory variance is spatially smooth. BYM2 is borrowing meaningfully from neighbours. The territory factors it produces are substantially better-calibrated than raw sector loss ratios.

A posterior mean `ρ` of 0.15 means the residual territory variation is mostly idiosyncratic. Perhaps your base GLM already captures most of the geographic signal through covariates like IMD score, urban density, or flood risk. In that case BYM2 degrades gracefully: the spatial component has little weight, and the territory factors approximate simple credibility weighting toward the mean.

This is the correct behaviour. The model adapts to what the data support rather than imposing spatial smoothing regardless of evidence.

---

## The regulatory case

The FCA's pricing regulations under PS21/5 and Consumer Duty (July 2023) do not prohibit postcode rating. Postcode is not a protected characteristic under the Equality Act 2010. But they do require that your pricing is defensible, auditable, and demonstrably linked to genuine risk differentials.

A k-means territory banding based on volatile loss ratios is difficult to defend. If challenged, your best argument is "this is what the data shows" - but the data for most sectors shows very little reliably.

BYM2 strengthens the regulatory position in three specific ways.

First, the Moran's I test before fitting demonstrates that spatial structure exists in your data. You are not fitting a spatial model on a whim - you are responding to evidence.

Second, the posterior uncertainty quantifies exactly how much confidence you have in each territory factor. A sector with a wide credibility interval - say, 0.80 to 1.45 - should not drive a large rate change. The width of the interval is the quantitative evidence for caution. This is the FCA Consumer Duty argument: thin-data territory factors with wide posteriors should be treated conservatively.

Third, the methodology is fully transparent. The adjacency structure is documented (Queen contiguity from ONS boundary data, islands connected to nearest mainland). The model specification is explicit. The prior choices are justified. You can hand the technical documentation to your actuarial peer reviewer and they can evaluate it. You cannot do this with a vendor's proprietary spatial smoothing.

---

## UK data sources

You need boundary polygons to build the adjacency. ONS does not publish official postcode sector boundaries, but there are workable routes.

The free option: download OS CodePoint Open (unit postcode centroids), compute a Voronoi tessellation, dissolve by sector, clip to the UK coastline. The resulting polygons are approximate but adequate for adjacency construction - the exact boundary matters less than getting the neighbour graph right. `insurance-spatial[geo]` includes `from_geojson()` which consumes the output of this process.

The paid option: Ordnance Survey sells CodePoint Polygons, which are more accurate. If your organisation already licenses OS data (many large insurers do), this is the cleaner route.

For covariates, the freely available datasets that matter most for UK motor and home:

| Dataset | Source | Relevance |
|---|---|---|
| Index of Multiple Deprivation 2025 | MHCLG (gov.uk) | Income, crime, employment by LSOA |
| Vehicle crime by LSOA | data.police.uk | Direct motor theft signal |
| Flood risk postcodes | Environment Agency (data.gov.uk) | Essential for home buildings |
| ONSPD November 2025 | ONS Open Geography Portal | Postcode → sector/LSOA lookup |

Pass covariates to `BYM2Model.fit()` as an `(N, P)` array - they enter as fixed effects in the model. The spatial component then captures residual geographic variation after controlling for them, which avoids the spatial effect absorbing deprivation or crime as a proxy.

---

## Where this fits in the pipeline

BYM2 territory factors are designed to slot into an existing GLM as a fixed offset. The two-stage workflow:

1. Fit your main GLM or GBM without territory (or with crude district-level territory).
2. Aggregate to sector level: compute observed claim counts and expected claims from the base model per sector.
3. Fit `BYM2Model` on the sector-level O/E ratios.
4. Extract `b_mean` per sector from `territory_relativities()`.
5. Add `b_mean` as a `ln_offset` column in your Emblem or Radar model import.

This keeps the spatial model decoupled from the main model. You can update the territory factors annually without refitting the base model. The territory factor table is a standard actuarial artefact - one row per sector, one log-relativity, signed off independently.

For teams using `bayesian-pricing` for [thin-cell credibility]({{ site.baseurl }}{% post_url 2026-03-06-bayesian-hierarchical-models-for-thin-data-pricing %}), the two approaches are complementary. `bayesian-pricing` handles thin vehicle-group × driver-age intersections. `insurance-spatial` handles geographic smoothing. Both produce outputs in the same format - log-relativities with posterior intervals - and both use PyMC under the hood.

---

## Computational considerations

For district-level work (N≈3,000 postcode districts), a full BYM2 run with 4 chains × 1,000 draws takes under 10 minutes with nutpie on modern hardware. This is interactive-session fast - you can iterate on priors and covariate specifications in a working session.

For sector-level work (N=11,200), the scaling factor computation is the first bottleneck. The library uses eigendecomposition of the precision matrix, which is O(N³) and feasible up to N≈3,000. For full UK postcode sectors, compute the scaling factor once and cache it:

```python
# Run this once per adjacency structure, cache the result
sf = adj.scaling_factor  # may take several minutes for N=11,200
# Next time: AdjacencyMatrix(W=W, areas=areas, _scaling_factor=sf)
```

The MCMC itself scales linearly with N for the ICAR model - the pairwise difference formulation is O(N·K) where K≈6 mean neighbours, not O(N²). Published benchmarks on comparable models (NYC, N=2,095, 21 minutes on a 2019 dual-core machine) suggest 20–30 minutes on a modern 8-core machine for the full UK sector graph. Territory factors are updated annually in practice; a 30-minute overnight run is not a problem.

Install nutpie for the fastest sampling:

```bash
uv pip install "insurance-spatial[nutpie]"
```

nutpie uses a Rust NUTS implementation and is typically 2–5x faster than PyMC's default sampler for models of this type.

---

## Get the library

```bash
uv pip install insurance-spatial
uv pip install "insurance-spatial[geo]"     # geopandas + libpysal for real boundaries
uv pip install "insurance-spatial[nutpie]"  # Rust sampler
```

Source is on [GitHub](https://github.com/burningcost/insurance-spatial). The library is at v0.1 - the core adjacency, BYM2 model, and diagnostics are stable. The covariate enrichment pipeline (IMD, police.uk crime, Environment Agency flood risk joins) and Stan backend are next.

---

## References

- Riebler, A., Sørbye, S.H., Simpson, D., & Rue, H. (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145–1165.
- Gschlössl, S., Schelldorfer, J., & Schnaus, M. (2019). Spatial statistical modelling of insurance risk. *Scandinavian Actuarial Journal*.
- Brockman, M.J., & Wright, T.S. (1992). Statistical motor rating: making effective use of your data. *Journal of the Institute of Actuaries*, 119, 457–543.
- Besag, J., York, J., & Mollié, A. (1991). Bayesian image restoration, with two applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1–59.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.C. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.
