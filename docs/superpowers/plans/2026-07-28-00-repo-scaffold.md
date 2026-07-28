# Repo Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the root-level structure and shared scaffolding for the `data-science-portfolio` repo so the 5 project plans each have a consistent home to build into.

**Architecture:** A single git repo with a root README (index page linking all 5 projects, updated with a stub now and finalized after all 5 projects are built), a root `.gitignore` covering Python/venv/data artifacts, and 5 empty project directories with per-directory `.gitkeep` placeholders removed once real content lands.

**Tech Stack:** Git, Markdown.

## Global Constraints

- No fabricated data or metrics anywhere in this repo — every number in every README must come from code that was actually run. (Spec: "Hard requirement: no fabricated data or metrics")
- Nothing is pushed to `github.com/imtheeon/data-science-portfolio` until Leanthel reviews and approves. (Spec: "Publishing gate")
- Repo root lives at `~/data-science-portfolio`, already git-initialized with `user.name = imtheeon`, `user.email = 128980799+imtheeon@users.noreply.github.com`.

---

### Task 1: Root README stub and .gitignore

**Files:**
- Create: `README.md` (repo root)
- Create: `.gitignore` (repo root)

**Interfaces:**
- Consumes: nothing.
- Produces: root `README.md` with 5 placeholder links (`01-ab-testing-experimentation/`, `02-customer-segmentation/`, `03-nlp-review-analysis/`, `04-movie-recommender/`, `05-fraud-detection-api/`) that each project plan will later fill in with a one-line summary once its own README exists.

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
.pytest_cache/
venv/
.venv/
env/

# Jupyter
.ipynb_checkpoints/

# Data (large downloaded datasets are not committed; scripts re-fetch them)
*.csv
!**/data/example_tests.py

# Env / secrets
.env

# OS
.DS_Store

# Subagent-driven-development scratch workspace (ledgers, briefs, review packages)
.superpowers/
```

- [ ] **Step 2: Write root `README.md`**

```markdown
# Data Science Portfolio

Leanthel Colon Cuevas — data analytics/data science portfolio, Google Data
Analytics Professional Certificate. Companion to
[data-analytics-portfolio](https://github.com/imtheeon/data-analytics-portfolio)
(BI dashboards) and [text-to-sql-agent](https://github.com/imtheeon/text-to-sql-agent)
(AI agent). These 5 projects demonstrate statistics, machine learning, and
deployment depth on real public datasets.

Every number shown in these projects is computed by code in this repo — no
hand-typed or invented figures. Where a project uses simulated data, that is
labeled explicitly in its README.

## Projects

| # | Project | What it demonstrates | Demo |
|---|---------|----------------------|------|
| 1 | [A/B Testing & Experimentation](01-ab-testing-experimentation/) | Hypothesis testing, power analysis, ship/no-ship decisions | _pending_ |
| 2 | [Customer Segmentation](02-customer-segmentation/) | RFM analysis, K-means clustering, marketing action mapping | _pending_ |
| 3 | [NLP Review Analysis](03-nlp-review-analysis/) | Sentiment classification, topic modeling | _pending_ |
| 4 | [Movie Recommender](04-movie-recommender/) | Collaborative filtering, recommendation evaluation | _pending_ |
| 5 | [Fraud Detection API](05-fraud-detection-api/) | Imbalanced classification, live FastAPI deployment | _pending_ |

## Note on data

Datasets are fetched by each project's own scripts (Kaggle API or direct
download) rather than committed to the repo — see each project's README for
the exact source and a real-vs-simulated label.
```

- [ ] **Step 3: Create empty project directories**

```bash
mkdir -p 01-ab-testing-experimentation 02-customer-segmentation 03-nlp-review-analysis 04-movie-recommender 05-fraud-detection-api
touch 01-ab-testing-experimentation/.gitkeep 02-customer-segmentation/.gitkeep 03-nlp-review-analysis/.gitkeep 04-movie-recommender/.gitkeep 05-fraud-detection-api/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add README.md .gitignore 01-ab-testing-experimentation/.gitkeep 02-customer-segmentation/.gitkeep 03-nlp-review-analysis/.gitkeep 04-movie-recommender/.gitkeep 05-fraud-detection-api/.gitkeep
git commit -m "Scaffold data-science-portfolio repo structure"
```

**Note for whoever runs project plans 01-05:** each project plan's final task removes that project's `.gitkeep` (once real files exist) and updates its row in the root `README.md` table (replacing `_pending_` with the actual demo link, or "local only" if no live demo was built), then commits that specific change.
