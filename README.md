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

| # | Project | Headline result (from the project's own README) |
|---|---------|----------------------|
| 1 | [A/B Testing & Experimentation](01-ab-testing-experimentation/) | Cookie Cats (90,189 real players): moving the gate from level 30 to 40 lowers 7-day retention from 19.02% to 18.20% (p = 0.0016), so no-ship. 1-day retention is inconclusive. |
| 2 | [Customer Segmentation](02-customer-segmentation/) | Online Retail II, 5,881 customers: two segments. The 45.5% of customers in "Champions" bring 90.2% of revenue. |
| 3 | [NLP Review Analysis](03-nlp-review-analysis/) | Amazon Fine Food Reviews, 20,000-review sample: VADER agrees with the star rating 79.94% of the time, but catches only about 41% of negative reviews. |
| 4 | [Movie Recommender](04-movie-recommender/) | MovieLens 100k: RMSE 1.0424 and Precision@5 of 0.032, about 1.76x a random baseline. |
| 5 | [Fraud Detection API](05-fraud-detection-api/) | Credit-card fraud (0.17% fraud rate): PR-AUC 0.83, catching 79 of 98 frauds in the test set at 0.84 precision. FastAPI and Docker, configured for Render but not deployed yet. |

None of these is deployed. Each project has run instructions and tests in its folder.

## Also in my portfolio

- [ml-pipeline-pricing-promo](https://github.com/imtheeon/ml-pipeline-pricing-promo) and [ml-pipeline-fremtpl2-claims](https://github.com/imtheeon/ml-pipeline-fremtpl2-claims): full gated machine-learning pipelines with a locked final exam
- [claim-denial-analysis](https://github.com/imtheeon/claim-denial-analysis), [review-insights-ai](https://github.com/imtheeon/review-insights-ai) and [pricing-promo-analysis](https://github.com/imtheeon/pricing-promo-analysis): SQL and dashboard analyses that end in ranked recommendations

## Note on data

Datasets are fetched by each project's own scripts (Kaggle API or direct
download) rather than committed to the repo — see each project's README for
the exact source and a real-vs-simulated label.
