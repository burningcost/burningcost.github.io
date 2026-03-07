---
layout: post
title: "Finding the Interactions Your GLM Missed"
date: 2026-03-07
categories: [techniques]
tags: [GLM, interactions, CANN, NID, shap, catboost, polars, pricing, python, motor]
published: false
---

Your motor frequency GLM has 12 rating factors. That means 66 possible pairwise interactions. You have tested, at most, a handful. The ones that felt obvious to whoever built the model five years ago, or the ones that showed up in a 2D actual-to-expected plot during a rate review, or the ones suggested by a GBM someone ran in a notebook and never fully documented.

Many interactions in a UK personal lines GLM are chosen by intuition or heuristic search. That is an imperfect way to choose them - not because actuaries have bad intuitions, but because the search space is too large for intuition to be systematic, the signal is buried under correlation structure, and "this pair felt worth testing" is not an auditable process.

There are 66 pairs for a 12-factor model. For a 20-factor model there are 190. Testing each one properly - fitting the GLM, computing the likelihood-ratio statistic, checking a 2D A/E plot, deciding whether the parameter count is worth the deviance gain - takes the better part of a day per pair if you do it carefully. You will not test all of them. You will miss interactions that are in the data.

`insurance-interactions` automates the search.

---

## What the problem actually looks like

Consider a standard UK motor GLM with age band, vehicle group, NCD, area, and annual mileage. The age-by-vehicle-group interaction is one of the most reliably non-additive in the data: young drivers in high-performance cars are not merely the product of the young-driver relativity and the sports-car relativity. They are materially worse. A main-effects GLM will systematically underprice this intersection and cross-subsidise it from the rest of the book.

Here is the concrete failure mode. Suppose the young-driver relativity is 2.3× and the high-performance vehicle group relativity is 1.8×. A multiplicative GLM prices the intersection at 2.3 × 1.8 = 4.14×. The observed frequency at that intersection might be 5.2× the base. You are underpricing by 25%.

The interaction exists in the data. The GBM you ran last year found it - your CatBoost model has a leaf somewhere in its tree structure that captures exactly this. The GLM did not find it because nobody thought to test it systematically, or they tested it and the LR test came back marginal, or the parameter count (10 age bands × 20 vehicle groups = 171 new parameters) looked too expensive without knowing how much deviance it would actually save.

Automated interaction detection solves this problem by giving you a ranked shortlist - the pairs most likely to improve your GLM, ordered by statistical evidence, before you have tested any of them.

---

## Three methods, each doing something different

`insurance-interactions` runs three distinct detection methods in sequence. Understanding what each one is actually doing matters for knowing when to trust the output.

### Stage 1: CANN - learning what the GLM missed

The first stage fits a Combined Actuarial Neural Network (Schelldorfer and Wüthrich, SSRN 3320525, 2019) on your GLM's residuals. The architecture uses a skip connection:

```
μ_CANN(x) = μ_GLM(x) × exp(NN(x; θ))
```

The GLM prediction enters as a fixed log-space offset. The neural network is zero-initialised at the output layer, so it starts exactly at the GLM prediction and learns only the residual structure the GLM cannot express. If your GLM is correctly specified - if the main effects are right and no interactions are missing - the CANN learns nothing and stays near zero everywhere. Any deviation from zero is structure the GLM is missing.

This is the right framing for the problem. The CANN is not a replacement for the GLM. It is a diagnostic: a flexible model trained to capture only what the GLM cannot. In Holvoet, Antonio and Henckaerts (2023, arXiv:2310.12671), the CANN consistently outperformed the base GLM on holdout Poisson deviance while remaining within noise of a well-tuned GBM - which tells you the signal it captures is genuine, not overfitting.

The output of Stage 1 is a trained neural network. The interaction detection happens in Stage 2.

### Stage 2: NID - reading interactions from the weights

Neural Interaction Detection (Tsang, Cheng and Liu, ICLR 2018) extracts the interaction structure from the trained CANN weights directly, without any additional model evaluation.

The insight is architectural. In a feedforward MLP, features can only interact at the first hidden layer: two features x_i and x_j can interact only if they both feed into the same first-layer hidden unit h_s through non-zero weights. The strength of their interaction through unit s depends on how much that unit influences the output and how strongly both features connect to it:

```
d(i,j) = Σ_s  z_s × min(|W1[s,i]|, |W1[s,j]|)
```

where z_s is the cumulative importance of unit s to the output (computed as the product of absolute weight matrices from layer 2 to the output), and the min aggregation forces both features to have meaningful input weights for the pair to score highly. This is computed in milliseconds after training, for all 66 pairs simultaneously.

The result is a ranked list of candidate interaction pairs. It is not a statistical test - it is a ranking. The statistical rigour comes in Stage 3.

### Stage 3: GLM testing - LR tests on the shortlist

For each top-K pair from the NID ranking, Stage 3 refits the GLM with the interaction added and computes a likelihood-ratio test statistic. This is the step actuaries already do manually; `insurance-interactions` does it automatically for the top 20 pairs rather than the 2 or 3 anyone bothers to test by hand.

The output table has one row per candidate pair. The `recommended` column is `True` when the interaction is significant after Bonferroni correction for multiple testing - standard practice when running 20 simultaneous LR tests.

---

## Using it

```bash
uv add insurance-interactions
```

For SHAP interaction validation (requires CatBoost):

```bash
uv add "insurance-interactions[shap]"
```

The input is your existing GLM's fitted values plus the training data:

```python
import polars as pl
from insurance_interactions import InteractionDetector, build_glm_with_interactions

# X: Polars DataFrame of rating factors
# y: claim counts
# exposure: earned policy years
# mu_glm: fitted values from your existing Poisson GLM

detector = InteractionDetector(family="poisson")
detector.fit(
    X=X_train,
    y=y_train,
    glm_predictions=mu_glm_train,
    exposure=exposure_train,
)

print(detector.interaction_table())
```

The interaction table:

| feature_1 | feature_2 | nid_score | delta_deviance_pct | lr_p | n_cells | recommended |
|---|---|---|---|---|---|---|
| age_band | vehicle_group | 0.847 | 1.23% | 0.0001 | 171 | True |
| age_band | ncd | 0.731 | 0.89% | 0.0003 | 54 | True |
| area | vehicle_group | 0.412 | 0.31% | 0.041 | 247 | False |
| ncd | annual_mileage | 0.289 | 0.18% | 0.092 | 30 | False |

Read the `n_cells` column alongside `delta_deviance_pct`. The age × vehicle group interaction saves 1.23% of base deviance at a cost of 171 new parameters. The age × NCD interaction saves 0.89% for only 54 parameters. On a credibility argument, that second interaction may be more useful than the first: you are getting three-quarters of the deviance gain at one-third of the parameter cost.

Once you have the shortlist:

```python
suggested = detector.suggest_interactions(top_k=5)
# [("age_band", "vehicle_group"), ("age_band", "ncd"), ...]

final_model, comparison = build_glm_with_interactions(
    X=X_train,
    y=y_train,
    exposure=exposure_train,
    interaction_pairs=suggested,
    family="poisson",
)
print(comparison)
```

`comparison` gives you the deviance, AIC and BIC of the base GLM versus the model with interactions added. The decision to add any given interaction remains with the actuary - the library provides the evidence, not the answer.

---

## Configuration that matters

Training is controlled via `DetectorConfig`:

```python
from insurance_interactions import DetectorConfig, InteractionDetector

cfg = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=300,
    cann_n_ensemble=5,      # average NID scores over 5 training runs
    cann_patience=30,
    top_k_nid=20,           # forward this many pairs to LR testing
    mlp_m=True,             # MLP-M variant: reduces false positives
    alpha_bonferroni=0.05,
)
detector = InteractionDetector(family="poisson", config=cfg)
```

Two configuration decisions that have real consequences:

**`cann_n_ensemble`**. CANN training is stochastic - different random seeds produce different weight matrices, which produce different NID rankings. A single run may rank a spurious pair highly because of a particular random initialisation. Averaging over 3-5 runs gives you stable rankings. We use 5 for any result we would take to a pricing committee.

**`mlp_m=True`**. The MLP-M architecture (also from Tsang et al. 2018) adds a separate small network for each feature's main effect, forcing the main MLP to model only interactions. This is important for UK motor data, where age and NCD are correlated (experienced drivers have higher NCD almost by definition). Without MLP-M, the NID algorithm can misattribute NCD's main effect signal as an age × NCD interaction. With MLP-M, the main effect is absorbed into the per-feature network and the MLP is clean. Set `mlp_m=True` for any portfolio with structurally correlated factors.

---

## Frequency and severity behave differently

Run the detector separately for each model component:

```python
freq_detector = InteractionDetector(family="poisson")
freq_detector.fit(X=X, y=claim_counts, glm_predictions=mu_freq_glm, exposure=exposure)

sev_detector = InteractionDetector(family="gamma")
sev_detector.fit(X=X_claims, y=claim_amounts, glm_predictions=mu_sev_glm, exposure=claim_counts)
```

The interactions that matter tend to differ. Age × vehicle group is typically a frequency interaction - young drivers in high-performance cars have higher claim frequency because of the combination of inexperience and vehicle capability. Severity is noisier: claim amounts are higher-variance than claim counts, which makes the CANN's signal harder to distinguish from noise in the residuals. Expect severity interactions to have wider confidence intervals and lower deviance gains for the same number of parameters.

In practice: add frequency interactions first. If a severity interaction is flagged by the NID and is also significant in the LR test, and the A/E plot for that pair looks non-additive, add it. Do not add severity interactions just because the NID score is high.

---

## SHAP interaction values as a second opinion

With the `[shap]` extra installed, a CatBoost model provides a second interaction ranking via TreeSHAP interaction values. This is computed differently from NID - it works on the CatBoost model rather than the CANN, and it directly computes the pairwise interaction contribution to each prediction rather than reading network weights.

The SHAP interaction value for a pair (i, j) across the portfolio:

```
Φ_{ij} = (1/n) Σ_k |φ_{ij}(x_k)|
```

The key word is "different." SHAP and NID are measuring interaction strength in different models (CatBoost vs. CANN) via different mathematical mechanisms. When they agree - when a pair ranks high in both - the evidence is strong. When they disagree, it is worth looking at the A/E plot for that pair before deciding.

The consensus ranking weights both:

```python
table = detector.interaction_table()
# Columns include: nid_score_normalised, shap_score_normalised, consensus_rank
```

Filter on `consensus_rank` when you want pairs that both methods flagged. Filter on `nid_score_normalised` alone when you do not have a CatBoost model available.

---

## The regulatory framing

UK actuaries working under FCA Consumer Duty pricing rules and internal model risk governance frameworks need interaction decisions to be auditable. A rate table with interactions added on the basis of intuition is harder to defend in a model review than one where the shortlist came from an automated detection procedure, each candidate was tested with a likelihood-ratio statistic, and the final decisions are documented alongside the test results.

This library is designed to support that process. It produces a ranked table with test statistics, Bonferroni-corrected significance, parameter costs, and deviance gains. The actuary decides which interactions to add. The library provides the shortlist and the evidence.

The output of `interaction_table()` is the document you take to the model review. Every row has a `recommended` flag, a p-value, and an `n_cells` cost. That is the basis for a defensible decision.

---

## Limitations to document

**Dataset size.** The CANN needs enough data to learn stable residual structure. Below roughly 5,000 policies, the NID rankings become noisy because the CANN itself is uncertain. The LR tests in Stage 3 still work below this threshold - they are standard GLM statistics - but the NID shortlist may miss genuine interactions or surface spurious ones. Use `n_ensemble=5` and check that the top-ranked pairs are stable across ensemble runs.

**Correlated features.** NID can spread interaction signal across correlated pairs. On UK motor data with a dataset where age and NCD are correlated above 0.4, activating `mlp_m=True` is not optional - it is required to reduce false positives. Even with MLP-M, treat pairs involving both age and NCD with some scepticism until the A/E plot confirms a genuine non-additive pattern.

**NID is a ranking, not a test.** The p-value that determines `recommended` comes from the LR test in Stage 3, not from NID. A high NID score means the CANN found evidence of this pair's interaction. A high NID score with a non-significant LR test means the interaction exists in the CANN but not at a level that would survive adding it to the GLM. Trust the LR test for significance. Use NID for prioritisation.

---

## Getting started

```bash
uv add insurance-interactions
```

Source and issue tracker on [GitHub](https://github.com/burningcost/insurance-interactions).

The minimum viable workflow is two calls: `detector.fit()` and `detector.interaction_table()`. Everything else - SHAP validation, ensemble averaging, MLP-M - is configuration that improves reliability. The core output is the ranked table and the set of interactions marked `recommended=True`.

For most UK personal lines teams, the interesting result will be that the interactions flagged by the automated search are not the ones that have been in the GLM for years. The ones in the GLM are the ones that looked obvious. The interesting ones are the ones that are not obvious but are in the data - the area × NCD interaction where high-NCD drivers in urban postcodes behave differently than you would expect from either factor alone, or the vehicle age × annual mileage interaction where high-mileage drivers in old vehicles represent a meaningfully different risk profile.

Those are the interactions your GLM missed. The library finds them.
