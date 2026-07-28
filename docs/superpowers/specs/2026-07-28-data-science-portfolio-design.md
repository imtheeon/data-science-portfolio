# Data Science Portfolio — Design Spec

Date: 2026-07-28
Author: Leanthel Colon Cuevas (with Claude Code)
Status: Approved for planning

## Purpose

Leanthel is targeting entry-level data analyst roles with the Google Data
Analytics Professional Certificate. His existing portfolio
(`github.com/imtheeon/data-analytics-portfolio`) is 5 BI-dashboard-style
Chart.js projects; `text-to-sql-agent` demonstrates AI-agent work. This
initiative adds 5 new projects that demonstrate data science depth — stats,
ML, deployment — beyond BI-dashboard work, published as a new repo:
`github.com/imtheeon/data-science-portfolio`.

## Hard requirement: no fabricated data or metrics

Every stat, chart, and model result in every project must come from code
that actually ran on real data. No invented numbers, accuracy figures, or
findings. This is a direct reaction to a finding in the existing
`data-analytics-portfolio` repo: its dashboards are Chart.js frontends with
hand-typed numbers in `<script>` arrays, with no real analysis code behind
them. This new repo must not repeat that pattern. Where a project uses
simulated/synthetic data, it must be clearly labeled as simulated in the
README, consistent with the synthetic-data disclosure already used in
`text-to-sql-agent`.

## Repo structure

Single repo, one git-initialized directory (`~/data-science-portfolio`,
remote `github.com/imtheeon/data-science-portfolio`), one self-contained
subfolder per project:

```
data-science-portfolio/
├── README.md                          # root index linking all 5 projects
├── 01-ab-testing-experimentation/
├── 02-customer-segmentation/
├── 03-nlp-review-analysis/
├── 04-movie-recommender/
└── 05-fraud-detection-api/
```

Each subfolder has its own `README.md`, `requirements.txt`, `data/`
(clearly labeled real vs. simulated), and source (notebook and/or scripts
and/or app).

## Shared conventions

- **No-fabrication protocol**: every number in every README is produced by
  a script or notebook committed to the repo, actually executed before the
  number is written down. Real data is fetched via the Kaggle API (already
  configured locally as user `leanthelcolon`) or downloaded directly from
  grouplens.org for MovieLens. Synthetic data is seeded-RNG generated and
  explicitly labeled "Simulated data" in the README.
- **README style**: matches the existing dashboard repo's tone — stat-led
  hero numbers, "what this tells you about X" framing, dataset source and
  real/simulated label up top, method, findings, a business/action
  takeaway, how to run it locally, and a live demo link where feasible.
  Written so a reader understands what the project does and why it matters
  within about 30 seconds.
- **Live demo hosting**: Streamlit Community Cloud (free, connects
  directly to a GitHub repo, no card required) for interactive projects;
  GitHub Pages for static notebook-based output; Render (free web service
  tier) for project 5's live API.
- **Publishing gate**: nothing is pushed to the GitHub remote and nothing
  is deployed to Streamlit Cloud or Render until Leanthel has reviewed the
  full repo contents (code, READMEs, actual computed numbers) and
  explicitly approves. His resume and LinkedIn are not touched by this
  work.

## Project 1 — A/B Testing & Experimentation (`01-ab-testing-experimentation`)

**Base**: reuse the existing `~/portfolio/ab-analyzer` statistics engine
as-is — it already has real, tested, from-first-principles implementations
of the two-proportion z-test, Welch's t-test, chi-square test,
Bayesian analysis, and power/sequential analysis, plus 5 example scenarios
built on seeded-RNG synthetic data that is honestly documented as
synthetic in code comments.

**What's added**:
- A real-data headline case study using Kaggle's **Cookie Cats** mobile
  game A/B test dataset (genuine, not simulated).
- The 5 existing labeled-synthetic scenarios remain as teaching examples
  covering different verdict types (borderline significant, clear win,
  inconclusive, harmful variant, multi-variant with correction).
- A missing `app.py` (Streamlit) front end, since the engine currently has
  no entrypoint.
- A `README.md` (currently missing).

**Output**: a clear ship/no-ship recommendation with the statistical
reasoning (p-value, confidence interval, effect size, power) shown, not
just asserted.

**Demo**: Streamlit Community Cloud.

## Project 2 — Customer Segmentation via Clustering (`02-customer-segmentation`)

**Dataset**: **Online Retail II** (Kaggle) — real UK e-commerce
transaction data.

**Method**: compute RFM (Recency, Frequency, Monetary) features per
customer; K-means clustering, with elbow and silhouette analysis to
justify the chosen k; PCA for 2D cluster visualization.

**Output**: each real, computed cluster is translated into a concrete
marketing action (e.g. "Champions: X% of customers, Y% of revenue → loyalty
program"; "At-risk: ... → win-back campaign"), where X and Y are actual
numbers from the clustering, not invented.

**Format**: Jupyter notebook + README with embedded charts. Not a
classification project — explicitly unsupervised segmentation, distinct
from the 3 classification-style projects already in the BI dashboard repo.

**Demo**: static, GitHub Pages.

## Project 3 — NLP Sentiment / Topic Analysis (`03-nlp-review-analysis`)

**Dataset**: a real public review dataset (Amazon or Yelp reviews, via
Kaggle), sampled to a size that runs locally in reasonable time — the
sample size is stated honestly in the README, not hidden.

**Method**: classic/local NLP — VADER or a local HuggingFace sentiment
model for sentiment classification; LDA or NMF over TF-IDF for topic
extraction. Chosen over the Claude API specifically to keep this project
free to run, fully reproducible by anyone who clones the repo, and free of
API key management — decided explicitly in favor of the local approach
during design review.

**Validation**: where the dataset includes star ratings, predicted
sentiment is compared against them as an honesty check on the model, with
real computed agreement numbers.

**Format**: notebook + README. Optional small Streamlit demo (enter review
text, see predicted sentiment and nearest topic).

**Demo**: Streamlit Community Cloud, if the optional demo is built;
otherwise static GitHub Pages for the notebook output.

## Project 4 — Recommendation System (`04-movie-recommender`)

**Dataset**: **MovieLens 100k** (grouplens.org, small version, no auth
required).

**Method**: collaborative filtering — SVD via the `surprise` library, or a
manual cosine-similarity-based approach — evaluated with RMSE and
Precision@K on a real held-out split.

**Output**: concrete example recommendations ("for user X, top-5
recommended movies are...") generated from actually computed similarity
scores, alongside the honest evaluation metrics.

**Format**: notebook + README. Optional small Streamlit demo (pick a user
ID, see live recommendations).

**Demo**: Streamlit Community Cloud, if the optional demo is built;
otherwise static GitHub Pages.

## Project 5 — Fraud Detection Live API (`05-fraud-detection-api`)

**Decision**: retrains a real fraud model rather than reusing anything
from `data-analytics-portfolio`, since that repo's fraud dashboard has no
real model behind its numbers — this project directly closes that gap.

**Dataset**: **ULB Credit Card Fraud** (Kaggle) — real anonymized
transactions, ~0.17% fraud rate.

**Method**: train a classifier that properly handles the severe class
imbalance (e.g. class-weighted RandomForest or XGBoost, or SMOTE);
`churn-predictor`'s pipeline structure (`ColumnTransformer` +
classifier, saved via `joblib`) is used as a code-pattern template, though
the dataset, features, and target are entirely different. Metrics reported
honestly as precision/recall/PR-AUC — plain accuracy is misleading on data
this imbalanced and will not be the headline number.

**API**: FastAPI app with a `POST /predict` endpoint accepting transaction
features and returning a fraud probability and flag. Auto-generated
Swagger UI at `/docs` serves as the live interactive demo.

**Guardrails** (per hard requirement — no runaway cost):
- In-memory rate limiting (e.g. 10 requests/minute per IP).
- No external paid API dependency — the model is a local scikit-learn /
  xgboost artifact, nothing to leak and no per-call cost.
- Request validation/size limits, CORS restricted.
- Dockerfile for reproducible deployment.

**Deploy**: Render free web-service tier. Render sleeps the service after
inactivity — the README notes this so a cold-start delay on first request
isn't a surprise to a reviewer.

## Build execution

Once this spec is approved, a short implementation plan is written
(`writing-plans` skill), then the 5 projects are built via parallel
background agents — the same pattern used for the earlier stalled-apps
rebuild — each agent fetching real data, running real code, and writing
its README from actual output. Nothing is verified as "done" without the
underlying script/notebook actually having been run.

## Out of scope

- Leanthel's resume and LinkedIn — he updates those himself once projects
  are live and verified.
- The `~/portfolio` stalled-apps-rebuild consolidation (financial-analyzer,
  sql-analytics, pl-generator) — unrelated initiative, not touched here.
- Modifying the existing `data-analytics-portfolio` or `text-to-sql-agent`
  repos.
