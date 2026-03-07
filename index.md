---
layout: home
---

<div class="bc-hero">
<h1>Open-source tools for UK insurance pricing teams</h1>
<p>13 Python libraries covering the full pricing workflow - from walk-forward cross-validation to constrained rate optimisation. Built by practitioners, for teams that already know what a GLM is.</p>
<div class="bc-cta-row">
  <a class="bc-btn bc-btn-primary" href="/course/">Training course</a>
  <a class="bc-btn" href="https://github.com/burningcost" target="_blank">GitHub</a>
</div>
</div>

<div class="bc-problem">
<p>Most UK pricing teams have adopted GBMs but are still taking GLM outputs to production. The GBM sits on a server outperforming the production model, but the outputs are not in a form that a rating engine, regulator, or pricing committee can work with. The missing piece is not technical skill - it is the tooling that bridges the two.</p>
<p>That is what we build here. Each library solves one specific problem in the pricing workflow, ships with actuarial tests, and produces outputs in formats that pricing teams already recognise.</p>
</div>

<div class="bc-repos">
<p class="bc-section-title">Libraries</p>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Model interpretation</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/shap-relativities" target="_blank">shap-relativities</a>
    <p>Extract multiplicative rating factor tables from CatBoost models using SHAP values. Same output format as exp(beta) from a GLM - factor tables, confidence intervals, exposure weighting, reconstruction validation.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Validation</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-cv" target="_blank">insurance-cv</a>
    <p>Temporally-correct cross-validation for insurance pricing models. Walk-forward splits with configurable IBNR buffers, Poisson and Gamma deviance scorers, sklearn-compatible API.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-conformal" target="_blank">insurance-conformal</a>
    <p>Distribution-free prediction intervals for insurance GBMs. Implements the variance-weighted non-conformity score from Manna et al. (2025) - roughly 30% narrower intervals than the naive approach with identical coverage guarantees.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Techniques</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/credibility" target="_blank">credibility</a>
    <p>Buhlmann-Straub credibility in Python, with mixed-model equivalence checks. Practical for capping thin segments, stabilising NCD factors, and blending a new model with an incumbent rate.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/bayesian-pricing" target="_blank">bayesian-pricing</a>
    <p>Hierarchical Bayesian models for thin-data pricing segments. Partial pooling across risk groups, with credibility factor output in a format that maps back to traditional actuarial review.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-interactions" target="_blank">insurance-interactions</a>
    <p>Tools for detecting, quantifying, and presenting interaction effects in insurance pricing models - the effects a main-effects-only GLM cannot see.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-causal" target="_blank">insurance-causal</a>
    <p>Causal inference methods for insurance pricing. Separating genuine risk signal from confounded association - relevant wherever rating factors are correlated with distribution channel or policyholder behaviour.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-spatial" target="_blank">insurance-spatial</a>
    <p>Spatial territory ratemaking using BYM2 models. Geographically smoothed relativities that borrow strength across adjacent areas - particularly useful for postcode-level home and motor models with thin data.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Commercial</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/rate-optimiser" target="_blank">rate-optimiser</a>
    <p>Constrained rate change optimisation for UK personal lines. Formulates the efficient frontier between loss ratio improvement and movement cap constraints as a linear programme.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-demand" target="_blank">insurance-demand</a>
    <p>Demand and conversion modelling for insurance pricing. Price elasticity curves, own-price and cross-price effects, integration with rate optimisation.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Compliance</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-fairness" target="_blank">insurance-fairness</a>
    <p>Proxy discrimination detection for insurance pricing models. Measures of disparate impact, fairness-accuracy trade-off analysis, FCA Consumer Duty documentation support.</p>
  </li>
</ul>
</div>

</div>

<div class="bc-course-strip">
<h3>Training course: Modern Insurance Pricing with Python and Databricks</h3>
<p>Eight modules written for pricing actuaries and analysts at UK personal lines insurers. Every module covers a real pricing problem - not a generic data science tutorial adapted to insurance.</p>
<div class="bc-course-modules">Modules: Databricks for pricing teams &middot; GLMs in Python (the bridge from Emblem) &middot; GBMs for insurance &middot; SHAP relativities &middot; Conformal prediction intervals &middot; Credibility and Bayesian pricing &middot; Constrained rate optimisation &middot; End-to-end pipeline capstone</div>
<a class="bc-btn" href="/course/">See the course</a>
</div>

<div class="bc-posts">
<p class="bc-section-title">Recent posts</p>
<ul class="bc-post-list">
{% for post in site.posts limit:8 %}
  <li class="bc-post-item">
    <span class="bc-post-date">{{ post.date | date: "%d %b %Y" }}</span>
    <a href="{{ post.url }}">{{ post.title }}</a>
  </li>
{% endfor %}
</ul>
</div>
