---
layout: course
title: "Modern Insurance Pricing with Python and Databricks"
description: "A practitioner-written course for UK personal lines pricing teams. Eight modules covering GLMs, GBMs, SHAP relativities, conformal prediction intervals, credibility, and constrained rate optimisation on Databricks."
permalink: /course/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Course",
  "name": "Modern Insurance Pricing with Python and Databricks",
  "description": "A practitioner-written course for UK personal lines pricing teams. Eight modules covering GLMs, GBMs, SHAP relativities, conformal prediction intervals, credibility, and constrained rate optimisation on Databricks.",
  "url": "https://burningcost.github.io/course/",
  "provider": {
    "@type": "Organization",
    "name": "Burning Cost",
    "url": "https://burningcost.github.io"
  },
  "author": {
    "@type": "Organization",
    "name": "Burning Cost",
    "url": "https://burningcost.github.io"
  },
  "educationalLevel": "Professional",
  "teaches": "Insurance pricing using Python, Databricks, GLMs, GBMs, SHAP, credibility theory, and constrained rate optimisation",
  "inLanguage": "en-GB",
  "offers": [
    {
      "@type": "Offer",
      "name": "Full course - all modules + all Burning Cost tools",
      "price": "295",
      "priceCurrency": "GBP",
      "availability": "https://schema.org/PreOrder"
    }
  ]
}
</script>

<!-- HERO -->
<section class="hero">
  <div class="hero-eyebrow">Practitioner course &middot; UK personal lines</div>
  <h1>Modern Insurance Pricing<br>with <em>Python and Databricks</em></h1>
  <p class="hero-sub">
    Eight modules. GLMs, GBMs, SHAP relativities, conformal intervals, credibility, and constrained rate optimisation.
    Every line of code written for insurance pricing specifically.
  </p>
  <p class="hero-detail">8 modules &nbsp;&middot;&nbsp; Databricks notebooks included &nbsp;&middot;&nbsp; Synthetic UK motor data throughout</p>
  <div class="hero-cta">
    <a href="#waitlist" class="btn btn-primary btn-lg">Join the waitlist</a>
    <a href="#modules" class="btn btn-outline btn-lg">View the modules</a>
  </div>
  <div class="hero-trust">
    <span class="trust-item"><span class="trust-check"></span>One-time payment, no subscription</span>
    <span class="trust-item"><span class="trust-check"></span>All future updates included</span>
    <span class="trust-item"><span class="trust-check"></span>Access to all Burning Cost tools</span>
  </div>
</section>

<!-- PROBLEM -->
<section class="problem-strip">
  <div class="problem-strip-inner">
    <span class="section-label">The problem</span>
    <h2>Generic Databricks tutorials will not teach you to price insurance</h2>
    <p>
      Most actuaries and analysts learn Databricks from the same place: tutorials aimed at software engineers doing retail churn or ad-click models.
      Those tutorials cover Delta Lake and MLflow in the abstract.
      They do not cover Poisson deviance as a loss function, IBNR buffers in cross-validation, or how to get SHAP relativities into a format that a pricing committee will actually accept.
    </p>
    <p>
      You can piece it together. People do. But it takes six months of wasted effort, and you end up with notebooks that work but that nobody else on the team can maintain, because they were written by someone learning two things at once.
    </p>
    <p>
      This course teaches Databricks for insurance pricing specifically. Every module starts with a real personal lines problem, uses realistic UK motor data, and ends with output that a pricing team can use.
    </p>
  </div>
</section>

<!-- WHO THIS IS FOR -->
<section class="audience">
  <div class="audience-inner">
    <div>
      <span class="section-label">Who this is for</span>
      <h2 class="section-h2">Pricing teams making the move to Python</h2>
      <p class="section-lead">
        This is not an introductory course to Python or to insurance pricing. It assumes you price things for a living and want to do it properly on Databricks.
      </p>
      <ul class="check-list">
        <li>Pricing actuaries and analysts at UK personal lines insurers</li>
        <li>Using Databricks now, or will be within the next twelve months</li>
        <li>Already know GLMs: you understand what a log link is and why it is there</li>
        <li>Can write basic Python: loops, functions, DataFrames</li>
        <li>Tired of adapting generic tutorials to insurance problems and hoping for the best</li>
      </ul>
    </div>
    <div>
      <span class="section-label">Who this is not for</span>
      <h2 class="section-h2">Not every course is the right course</h2>
      <p class="section-lead">
        Better to be clear about this upfront than to have you work through Module 2 wondering why you do not have the background.
      </p>
      <ul class="cross-list">
        <li>Complete beginners to Python (you need to know what a function is)</li>
        <li>People new to insurance pricing who need the fundamentals first</li>
        <li>Data scientists looking for a general ML course (this is for insurance)</li>
        <li>Teams on a platform other than Databricks (all exercises use Databricks specifically)</li>
      </ul>
    </div>
  </div>
</section>

<!-- WHAT YOU'LL BUILD -->
<section class="builds">
  <div class="builds-inner">
    <div class="builds-header">
      <span class="section-label">What you will build</span>
      <h2 class="section-h2">Tangible outputs from every module</h2>
    </div>
    <div class="builds-grid">
      <div class="build-card">
        <span class="build-card-num">Module 01</span>
        <div class="build-card-title">A clean, reproducible pricing workspace on Databricks</div>
        <div class="build-card-desc">Unity Catalog schema, Delta tables with synthetic motor data, MLflow experiment tracking, and an audit trail that ties model runs to data versions.</div>
      </div>
      <div class="build-card">
        <span class="build-card-num">Module 02</span>
        <div class="build-card-title">A GLM frequency-severity model a traditional reviewer can follow</div>
        <div class="build-card-desc">statsmodels GLM replicating Emblem's output, including offset terms, deviance statistics, one-way analysis, and a Radar-compatible factor table export.</div>
      </div>
      <div class="build-card">
        <span class="build-card-num">Module 03</span>
        <div class="build-card-title">CatBoost models with proper insurance cross-validation</div>
        <div class="build-card-desc">Poisson frequency and Gamma severity models with walk-forward CV, IBNR buffers, Optuna hyperparameter tuning, and a proper GBM-vs-GLM comparison.</div>
      </div>
      <div class="build-card">
        <span class="build-card-num">Module 04</span>
        <div class="build-card-title">A SHAP relativity table formatted for a pricing committee</div>
        <div class="build-card-desc">Multiplicative relativities extracted from a CatBoost GBM, in Excel format for review and CSV format for Radar, with proxy discrimination detection for Consumer Duty.</div>
      </div>
      <div class="build-card">
        <span class="build-card-num">Module 05</span>
        <div class="build-card-title">Calibrated prediction intervals on individual risk estimates</div>
        <div class="build-card-desc">Conformal prediction intervals with a finite-sample coverage guarantee, calibrated to your holdout data, roughly 30% narrower than the naive approach.</div>
      </div>
      <div class="build-card">
        <span class="build-card-num">Module 06</span>
        <div class="build-card-title">Credibility-weighted relativities for thin-cell segments</div>
        <div class="build-card-desc">Buhlmann-Straub credibility in Python, including NCD factor stabilisation and blending a new model with incumbent rates where the data is thin.</div>
      </div>
      <div class="build-card">
        <span class="build-card-num">Module 07</span>
        <div class="build-card-title">A constrained rate optimisation with an FCA compliance check</div>
        <div class="build-card-desc">Linear programming that hits a target loss ratio, respects movement caps, and has Consumer Duty PS21/5 constraints built in. Includes efficient frontier analysis and shadow price reporting.</div>
      </div>
      <div class="build-card">
        <span class="build-card-num">Module 08</span>
        <div class="build-card-title">A complete motor pricing pipeline, ready to use as a template</div>
        <div class="build-card-desc">From Delta ingestion to rate change recommendation, every component connected. Structured so you can swap in your own data and run it with minimal modification.</div>
      </div>
    </div>
  </div>
</section>

<!-- MODULES -->
<section class="modules" id="modules">
  <div class="modules-inner">
    <div class="modules-header">
      <span class="section-label" style="color:var(--accent)">The curriculum</span>
      <h2 class="section-h2 section-h2-light">Eight modules. No filler.</h2>
    </div>

    <!-- Module 01 -->
    <div class="module-card available">
      <div class="module-card-header">
        <span class="module-num-badge">01</span>
        <div class="module-card-content">
          <div class="module-card-title">Databricks for Pricing Teams</div>
          <div class="module-card-desc">What Databricks actually is, not the marketing version. Set up a workspace for pricing, not for a generic data pipeline.</div>
        </div>
        <div class="module-status"><span class="badge badge-available">Available</span></div>
      </div>
      <div class="module-card-detail">
        <ul class="module-covers">
          <li>Unity Catalog for pricing data: where to put tables, how to set retention for FCA audit</li>
          <li>Cluster configuration that does not cost a fortune to leave running</li>
          <li>Delta tables as a replacement for flat-file data passes between pricing and MI</li>
          <li>MLflow experiment tracking from first principles: log parameters, metrics, and artefacts</li>
          <li>An audit trail table that ties model runs to data versions</li>
        </ul>
        <div class="module-links">
          <a href="/course/module-01/" class="module-link">Module overview</a>
          <a href="/course/module-01/tutorial.md" class="module-link">Preview tutorial</a>
        </div>
      </div>
    </div>

    <!-- Module 02 -->
    <div class="module-card available">
      <div class="module-card-header">
        <span class="module-num-badge">02</span>
        <div class="module-card-content">
          <div class="module-card-title">GLMs in Python: The Bridge from Emblem</div>
          <div class="module-card-desc">How to replicate what Emblem does in Python, transparently. The same deviance statistics. The same factor tables. A workflow a traditional reviewer can follow.</div>
        </div>
        <div class="module-status"><span class="badge badge-available">Available</span></div>
      </div>
      <div class="module-card-detail">
        <ul class="module-covers">
          <li>statsmodels GLMs with offset terms, variance functions, and IRLS - the same algorithm Emblem uses</li>
          <li>One-way and two-way analysis, aliasing detection, and model comparison by deviance</li>
          <li>The difference between statsmodels and sklearn's GLM implementation, and when it matters</li>
          <li>Exporting factor tables to a format Radar can import directly</li>
          <li>On clean data, output matches Emblem to four decimal places given identical encodings</li>
        </ul>
        <div class="module-links">
          <a href="/course/module-02/" class="module-link">Module overview</a>
          <a href="/course/module-02/tutorial.md" class="module-link">Preview tutorial</a>
        </div>
      </div>
    </div>

    <!-- Module 03 -->
    <div class="module-card">
      <div class="module-card-header">
        <span class="module-num-badge">03</span>
        <div class="module-card-content">
          <div class="module-card-title">GBMs for Insurance Pricing</div>
          <div class="module-card-desc">CatBoost with Poisson, Gamma, and Tweedie objectives. Walk-forward cross-validation with IBNR buffers so you are not lying to yourself about out-of-sample performance.</div>
        </div>
        <div class="module-status"><span class="badge badge-soon">Coming soon</span></div>
      </div>
      <div class="module-card-detail">
        <ul class="module-covers">
          <li>CatBoost Poisson frequency and Gamma severity models with correct exposure handling</li>
          <li>Why default hyperparameters from generic tutorials are wrong for insurance data</li>
          <li>Walk-forward temporal cross-validation using <code>insurance-cv</code>, with IBNR buffer support</li>
          <li>Optuna hyperparameter tuning and MLflow experiment tracking</li>
          <li>Proper GBM-vs-GLM comparison: Gini, calibration curves, and double-lift charts</li>
        </ul>
      </div>
    </div>

    <!-- Module 04 -->
    <div class="module-card available">
      <div class="module-card-header">
        <span class="module-num-badge">04</span>
        <div class="module-card-content">
          <div class="module-card-title">SHAP Relativities</div>
          <div class="module-card-desc">How to get a factor table out of a GBM. Mathematically sound, reviewable by a pricing actuary, submittable to the FCA, importable into Radar.</div>
        </div>
        <div class="module-status"><span class="badge badge-available">Available</span></div>
      </div>
      <div class="module-card-detail">
        <ul class="module-covers">
          <li>SHAP values as a principled replacement for GLM relativities - not a heuristic</li>
          <li>Aggregating raw SHAP values into multiplicative factor tables with confidence intervals</li>
          <li>When SHAP relativities are honest and when they are misleading: interactions, correlated features</li>
          <li>Proxy discrimination detection using SHAP for FCA Consumer Duty compliance</li>
          <li>Excel and Radar-CSV exports using the open-source <code>shap-relativities</code> library</li>
        </ul>
        <div class="module-links">
          <a href="/course/module-04/" class="module-link">Module overview</a>
          <a href="/course/module-04/tutorial.md" class="module-link">Preview tutorial</a>
        </div>
      </div>
    </div>

    <!-- Module 05 -->
    <div class="module-card">
      <div class="module-card-header">
        <span class="module-num-badge">05</span>
        <div class="module-card-content">
          <div class="module-card-title">Conformal Prediction Intervals</div>
          <div class="module-card-desc">Prediction intervals with a finite-sample coverage guarantee that does not depend on distributional assumptions. Calibrated to your holdout data.</div>
        </div>
        <div class="module-status"><span class="badge badge-soon">Coming soon</span></div>
      </div>
      <div class="module-card-detail">
        <ul class="module-covers">
          <li>Why point estimates are not enough and why standard confidence intervals are the wrong answer</li>
          <li>Conformal prediction: the theory and why the coverage guarantee is unconditional</li>
          <li>Variance-weighted non-conformity scores for heteroscedastic insurance data (Manna et al., 2025)</li>
          <li>Intervals roughly 30% narrower than the naive approach, with identical coverage</li>
          <li>Using intervals to flag uncertain risks and set minimum premium floors</li>
        </ul>
      </div>
    </div>

    <!-- Module 06 -->
    <div class="module-card available">
      <div class="module-card-header">
        <span class="module-num-badge">06</span>
        <div class="module-card-content">
          <div class="module-card-title">Credibility and Bayesian Pricing</div>
          <div class="module-card-desc">You have 200 policies in a postcode area and 3 claims. Is 1.5% the true frequency, or noise? Buhlmann-Straub credibility and hierarchical Bayesian models for thin-cell segments.</div>
        </div>
        <div class="module-status"><span class="badge badge-available">Available</span></div>
      </div>
      <div class="module-card-detail">
        <ul class="module-covers">
          <li>Buhlmann-Straub credibility in Python using the open-source <code>credibility</code> library</li>
          <li>Its relationship to mixed models and partial pooling - when they agree and when they diverge</li>
          <li>Practical applications: NCD factor stabilisation, blending a new model with incumbent rates</li>
          <li>Full Bayesian hierarchical models via <code>bayesian-pricing</code> for segments where credibility is not enough</li>
          <li>What credibility does not protect you from when exposure mix is shifting</li>
        </ul>
        <div class="module-links">
          <a href="/course/module-06/" class="module-link">Module overview</a>
          <a href="/course/module-06/tutorial.md" class="module-link">Preview tutorial</a>
        </div>
      </div>
    </div>

    <!-- Module 07 -->
    <div class="module-card">
      <div class="module-card-header">
        <span class="module-num-badge">07</span>
        <div class="module-card-content">
          <div class="module-card-title">Constrained Rate Optimisation</div>
          <div class="module-card-desc">The module most courses do not have. Linear programming for rate changes that hit a target loss ratio, respect movement caps, and minimise cross-subsidy simultaneously.</div>
        </div>
        <div class="module-status"><span class="badge badge-soon">Coming soon</span></div>
      </div>
      <div class="module-card-detail">
        <ul class="module-covers">
          <li>The formal problem of finding which factors to move, by how much, subject to constraints</li>
          <li>Linear programming formulation using <code>rate-optimiser</code> and <code>scipy.optimize</code></li>
          <li>The efficient frontier of achievable (loss ratio, volume) outcomes for a rate review cycle</li>
          <li>Shadow price analysis: the marginal cost of tightening the LR target, quantified</li>
          <li>FCA PS21/5 Consumer Duty compliance constraints built into the optimisation</li>
        </ul>
      </div>
    </div>

    <!-- Module 08 -->
    <div class="module-card">
      <div class="module-card-header">
        <span class="module-num-badge">08</span>
        <div class="module-card-content">
          <div class="module-card-title">End-to-End Pipeline (Capstone)</div>
          <div class="module-card-desc">Every component from Modules 1-7 connected into a working motor pricing pipeline. Not a demonstration: a template for a real project.</div>
        </div>
        <div class="module-status"><span class="badge badge-soon">Coming soon</span></div>
      </div>
      <div class="module-card-detail">
        <ul class="module-covers">
          <li>Raw data from Unity Catalog through to a rate change pack in a single reproducible pipeline</li>
          <li>A feature transform layer defined as pure functions, versioned alongside the model</li>
          <li>CatBoost frequency and severity models, SHAP relativities, conformal intervals, credibility blending, rate optimisation</li>
          <li>Output tables written to Delta: relativities, rate change summary, efficient frontier data, model diagnostics</li>
          <li>Designed to work with your own motor portfolio data with minimal modification</li>
        </ul>
      </div>
    </div>

  </div>
</section>

<!-- SOCIAL PROOF -->
<section class="proof">
  <div class="proof-inner">
    <span class="section-label">Why trust us on this</span>
    <h2 class="section-h2">Built by practitioners who have done this at UK insurers</h2>

    <div class="proof-grid">
      <div class="proof-card">
        <p>14 open-source libraries, each solving one well-defined pricing problem. Over 600 tests. Every library used in the course was built by us.</p>
        <div class="proof-card-author">Open source</div>
        <div class="proof-card-role">github.com/burningcost</div>
      </div>
      <div class="proof-card">
        <p>Written for people who already know what a GLM is. No generic data science padding. Every module covers a real pricing workflow problem.</p>
        <div class="proof-card-author">Practitioner focus</div>
        <div class="proof-card-role">UK personal lines</div>
      </div>
      <div class="proof-card">
        <p>Executable Databricks notebooks with synthetic data that behaves like the real thing. Run the code, see the outputs, adapt to your own book.</p>
        <div class="proof-card-author">Hands-on</div>
        <div class="proof-card-role">Databricks notebooks</div>
      </div>
    </div>

    <div class="proof-libs">
      <div class="proof-libs-title">The course teaches you to use these open-source libraries. We built all of them.</div>
      <div class="proof-libs-grid">
        <div class="proof-lib-item">
          <div>
            <div class="proof-lib-name"><a href="https://github.com/burningcost/insurance-cv" target="_blank">insurance-cv</a></div>
            <div class="proof-lib-desc">Temporal CV with IBNR buffers</div>
          </div>
        </div>
        <div class="proof-lib-item">
          <div>
            <div class="proof-lib-name"><a href="https://github.com/burningcost/shap-relativities" target="_blank">shap-relativities</a></div>
            <div class="proof-lib-desc">SHAP values as multiplicative factor tables</div>
          </div>
        </div>
        <div class="proof-lib-item">
          <div>
            <div class="proof-lib-name"><a href="https://github.com/burningcost/credibility" target="_blank">credibility</a></div>
            <div class="proof-lib-desc">Buhlmann-Straub credibility in Python</div>
          </div>
        </div>
        <div class="proof-lib-item">
          <div>
            <div class="proof-lib-name"><a href="https://github.com/burningcost/insurance-conformal" target="_blank">insurance-conformal</a></div>
            <div class="proof-lib-desc">Distribution-free intervals for GBMs</div>
          </div>
        </div>
        <div class="proof-lib-item">
          <div>
            <div class="proof-lib-name"><a href="https://github.com/burningcost/rate-optimiser" target="_blank">rate-optimiser</a></div>
            <div class="proof-lib-desc">Constrained rate change optimisation</div>
          </div>
        </div>
        <div class="proof-lib-item">
          <div>
            <div class="proof-lib-name"><a href="https://github.com/burningcost/bayesian-pricing" target="_blank">bayesian-pricing</a></div>
            <div class="proof-lib-desc">Hierarchical Bayesian models for thin segments</div>
          </div>
        </div>
        <div class="proof-lib-item">
          <div>
            <div class="proof-lib-name"><a href="https://github.com/burningcost/insurance-distill" target="_blank">insurance-distill</a></div>
            <div class="proof-lib-desc">GBM-to-GLM distillation for production</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- PRICING -->
<section class="pricing" id="pricing">
  <div class="pricing-inner">
    <div class="pricing-header">
      <span class="section-label" style="color:var(--accent)">Pricing</span>
      <h2 class="section-h2 section-h2-light">One product. One price. Everything included.</h2>
      <p>Buy once. Get the full course, every update, and access to all Burning Cost tools as they ship.</p>
    </div>

    <div class="pricing-single">
      <div class="pricing-card featured pricing-card-single">
        <div class="pricing-card-label">Full course + all tools</div>
        <div class="pricing-card-price"><span>&pound;</span>295</div>
        <div class="pricing-card-sub">One-time payment &mdash; no subscription, no expiry</div>
        <ul class="pricing-card-includes">
          <li>All 8 modules &mdash; immediate access to published modules</li>
          <li>New modules as they publish, at no extra cost</li>
          <li>All future updates and curriculum additions included</li>
          <li>Access to every Burning Cost tool and product as it ships</li>
          <li>Written tutorials (~4,000 words per module)</li>
          <li>Databricks notebooks, runnable end-to-end</li>
          <li>Synthetic UK motor dataset throughout</li>
          <li>Databricks Free Edition compatible</li>
        </ul>
        <a href="#waitlist" class="btn btn-primary" style="width:100%;text-align:center;display:block;font-size:1.1rem;padding:0.85rem 1.5rem;">Join the waitlist &rarr;</a>
      </div>
    </div>

    <p class="pricing-cta-note">
      Payment processing is nearly ready. Join the waitlist and we will email you when it is live &mdash; waitlist members get first access at the launch price.
    </p>
  </div>
</section>

<!-- WAITLIST -->
<section class="waitlist-band" id="waitlist">
  <h2>Get notified when it launches</h2>
  <p>Payment processing is nearly ready. Leave your email and we will send you a link the day it goes live. Waitlist members get first access.</p>
  <div class="waitlist-form">
    <form action="https://formspree.io/f/pricing.frontier@gmail.com" method="POST" class="waitlist-formspree">
      <input type="email" name="email" placeholder="your@email.com" required class="waitlist-input">
      <button type="submit" class="btn btn-primary btn-lg">Join the waitlist</button>
    </form>
  </div>
</section>

<!-- FAQ -->
<section class="faq">
  <div class="faq-inner">
    <span class="section-label">FAQ</span>
    <h2 class="section-h2">Common questions</h2>
    <div class="faq-list">

      <div class="faq-item">
        <div class="faq-q">Do I need a paid Databricks account?</div>
        <div class="faq-a">No. All exercises are compatible with <a href="https://www.databricks.com/product/pricing/databricks-free-edition" target="_blank">Databricks Free Edition</a>. You do not need company approval or a budget to start.</div>
      </div>

      <div class="faq-item">
        <div class="faq-q">How much Python do I need?</div>
        <div class="faq-a">You should be able to read a function, understand a list comprehension, and follow a data pipeline. You do not need to be a software engineer. If you can write a basic script to load a CSV and filter rows, you have enough Python. The course introduces every library we use as we go.</div>
      </div>

      <div class="faq-item">
        <div class="faq-q">Do I need to know GLMs before starting?</div>
        <div class="faq-a">For most of the course, yes. You should know what a log link is, what the deviance statistic measures, and have built at least one frequency or severity model, even if it was in Emblem. Module 2 covers the Python implementation in detail but does not re-teach GLM theory from scratch.</div>
      </div>

      <div class="faq-item">
        <div class="faq-q">What data does the course use?</div>
        <div class="faq-a">Synthetic UK motor data throughout. 10,000 policies with realistic exposure, claim counts, development patterns, and rating factor structures. Not a Kaggle dataset. The data is generated to mirror the statistical properties of a real personal lines motor portfolio without using real customer data.</div>
      </div>

      <div class="faq-item">
        <div class="faq-q">Why Polars instead of Pandas?</div>
        <div class="faq-a">Polars is faster, has better null handling, and encourages a more explicit data pipeline style. On Databricks, it runs well alongside PySpark and does not require the Pandas-on-Spark shim. Pandas appears only at the boundary where <code>statsmodels</code> requires it. If you know Pandas, you will be able to follow Polars without difficulty.</div>
      </div>

      <div class="faq-item">
        <div class="faq-q">Why CatBoost and not XGBoost or other GBM libraries?</div>
        <div class="faq-a">CatBoost handles categorical features natively without ordinal encoding, which matters for insurance rating factors. It also has more stable default behaviour on small datasets. The Poisson, Gamma, and Tweedie objectives are well-tested and documented. That said, the concepts transfer directly to XGBoost or other gradient boosting libraries if your team is committed to a different tool.</div>
      </div>

      <div class="faq-item">
        <div class="faq-q">What does "access to all Burning Cost tools" mean?</div>
        <div class="faq-a">As we build new products &mdash; additional tools, dashboards, templates, or workflow utilities &mdash; course purchasers get access as part of the same one-time payment. We are building a suite of things for UK pricing teams. One payment gets you into all of it.</div>
      </div>

      <div class="faq-item">
        <div class="faq-q">When will the remaining modules be published?</div>
        <div class="faq-a">Modules 3, 5, 7, and 8 are in progress. We publish them as they are ready rather than waiting until the full course is complete. When you buy, you get immediate access to published modules and new ones as they land.</div>
      </div>

      <div class="faq-item">
        <div class="faq-q">What if the content does not meet my expectations?</div>
        <div class="faq-a">You can preview the tutorials for Modules 1, 2, 4, and 6 before buying. If you read those and feel the depth and style are not what you need, we would rather you saved the money. Email us and we will discuss it.</div>
      </div>

    </div>
  </div>
</section>

<!-- FINAL CTA -->
<section class="final-cta">
  <h2>Stop spending months adapting the wrong tutorials</h2>
  <p>Eight modules written specifically for UK personal lines pricing teams. GLMs, GBMs, SHAP relativities, credibility, rate optimisation. The full workflow, done properly. One price, everything included.</p>
  <div class="final-cta-actions">
    <a href="#waitlist" class="btn btn-primary btn-lg">Join the waitlist</a>
    <a href="#modules" class="btn btn-outline btn-lg">Review the modules</a>
  </div>
</section>
