# Movie Recommender

Item-based collaborative filtering on the real MovieLens 100k dataset,
evaluated on the official held-out test split.

**Real dataset**: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
— 100,000 real ratings from 943 real users on 1,682 real movies. Not
simulated.

## Results

Evaluated on the official `u1.base`/`u1.test` 80/20 split:

- **RMSE**: 1.0424 on 19,968 real held-out test ratings
- **Precision@5**: 0.0320 averaged over 50 real users with relevant held-out items
- **Random baseline Precision@5**: 0.0182 — the analytically-computed
  expected Precision@5 of picking 5 items at random from each of those
  same 50 users' own candidate pool (unrated items in the training
  matrix), averaged the same way. See `run_analysis.py` for the exact
  R/N computation.

**What this tells you**: on a 1-5 rating scale, an RMSE of ~1.04 means
predictions are typically off by about one star — usable for ranking
but not precise. A Precision@5 of 0.032 means roughly 1 in 30 top-5
recommendations lands on an item the user actually rated highly in the
held-out set — about **1.76x the random baseline of 0.0182**, computed
the same way over the same 50 users. That's a real, modest edge over
chance, not a dramatic one; a reminder that top-N hit-rate is a hard,
sparse metric on this dataset.

## Example recommendations (real output)

| User | Top 5 recommended movies |
|------|---------------------------|
| 1 | Convent, The (Convento, O) (1995); Coldblooded (1995); Farewell My Concubine (1993); Wild Reeds (1994); Brassed Off (1996) |
| 2 | Hearts and Minds (1996); Coldblooded (1995); Mamma Roma (1962); Outlaw, The (1943); Jupiter's Wife (1994) |
| 3 | Fille seule, La (A Single Girl) (1995); Girl in the Cadillac (1995); Mostro, Il (1994); Enfer, L' (1994); Grosse Fatigue (1994) |

## Method

- Item-based collaborative filtering: cosine similarity between movies
  over real user-rating vectors (`analysis/collaborative_filtering.py`).
- **Shrinkage regularization on similarity**: raw cosine similarity
  gives item pairs with only 1-2 shared raters a similarity score at or
  near 1.0 — mathematically "perfect" off a single data point, which is
  statistically unreliable, not a genuinely strong signal. To correct
  this, pairwise similarity is shrunk toward zero by co-rating support:
  `shrunk_sim = raw_sim * co_count / (co_count + 20)`, a standard
  neighborhood-CF regularization (Bell & Koren), applied with the
  literature's default shrinkage constant (`beta=20`) rather than tuned
  against a target score.
- Predicted rating = similarity-weighted average of the user's real
  ratings for the k=20 most similar movies they've actually rated.
- Evaluated with RMSE (rating-prediction accuracy) and Precision@5
  (top-N recommendation quality) on the official held-out test split.

### Why the metrics got slightly worse after the fix, and why that's not a red flag

Before shrinkage was added, this pipeline scored RMSE 0.9955 and
Precision@5 0.0360 on this same 80/20 split — both numbers slightly
*better* than the shrinkage-regularized results above (RMSE 1.0424,
Precision@5 0.0320). The unregularized version was removed anyway,
because those pre-fix numbers were propped up by a methodology flaw:
without shrinkage, the model routinely treated a single shared rater as
"the two movies are essentially identical," and 11 of 13 example
recommendations audited pre-fix turned out to have only 1-8 training
ratings against a catalog median around 22-24. That's overfitting to
noisy, low-support similarity scores, not a genuinely better model —
point metrics on one fixed 80/20 split can and do reward that kind of
overfitting.

Shrinkage trades a small amount of point-metric performance on this
specific split for a similarity measure that's actually trustworthy:
every pairwise similarity now reflects how much co-rating evidence
backs it up, not just its raw cosine value. Some individual
recommendations did visibly improve — e.g. User 1's list swapped a
single-rating title for `Farewell My Concubine (1993)` (31 training
ratings) and `Wild Reeds (1994)` (14 training ratings) — but the
aggregate RMSE/Precision@5 on this one split moved in the "worse"
direction. That's an honest, expected trade-off, not evidence the fix
was wrong.

**Residual limitation**: some recommended titles above still have very
few training ratings (e.g. several 1-rating items persist in User 2 and
User 3's lists). Shrinkage discounts unreliable similarity pairs, but
it can't manufacture better-supported neighbors where none exist — for
a user whose own rated catalog is itself made up of niche movies, their
entire local neighborhood can remain sparse even after regularization.

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_analysis.py     # real evaluation run
streamlit run app.py       # interactive demo
```

## Live demo

_pending deployment_

## Notes

- Dataset downloads automatically from grouplens.org on first run (no
  account/auth needed); not committed to this repo.
- The current real numbers above (RMSE, Precision@5, random baseline,
  example recommendations) come from `run_analysis.py` — see that file
  for the exact computation. The pre-fix comparison numbers in "Why the
  metrics got slightly worse after the fix" (RMSE 0.9955, Precision@5
  0.0360, "11 of 13" niche titles, and the training-rating counts like
  31/14) are not reproduced by `run_analysis.py` as it stands today —
  they were computed during the code-review process that led to the
  shrinkage fix (a one-off audit against the pre-fix, unshrunk version
  of `compute_item_similarity`), and are documented here as project
  history rather than a number you can regenerate by re-running the
  current script.
