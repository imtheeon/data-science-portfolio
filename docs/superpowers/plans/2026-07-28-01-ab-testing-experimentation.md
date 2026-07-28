# Project 1: A/B Testing & Experimentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `01-ab-testing-experimentation/` in the `data-science-portfolio` repo: a Streamlit app that runs real hypothesis testing, power analysis, and a ship/no-ship recommendation, on both a real public A/B dataset (Cookie Cats) and clearly-labeled synthetic teaching examples.

**Architecture:** Reuse the existing, already-tested `~/portfolio/ab-analyzer` statistics engine (frequentist, Bayesian, power, sequential, multivariate modules) unchanged. Add a new pure-function `verdict` module that turns a test result into a ship/hold/no-ship recommendation with stated reasoning, a real-data loader + analysis script for the Cookie Cats dataset, and a Streamlit `app.py` front end that ties it all together.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, statsmodels (test cross-checks only), Streamlit, Plotly, pytest, Kaggle API.

## Global Constraints

- No fabricated data or metrics: every number shown in the app/README is produced by running the code in this repo. (Spec: "Hard requirement")
- Any synthetic data must be labeled "Simulated data" in the README/UI. (Spec: "Shared conventions")
- Nothing pushed to GitHub or deployed to Streamlit Cloud until Leanthel reviews and approves. (Spec: "Publishing gate")
- Working directory for all tasks: `~/data-science-portfolio/01-ab-testing-experimentation/`.
- Source engine to copy from: `~/portfolio/ab-analyzer/` (read-only reference — do not modify the original).

---

### Task 1: Copy the statistics engine and verify it still passes

**Files:**
- Create: `01-ab-testing-experimentation/stats/__init__.py`, `frequentist.py`, `power.py`, `bayesian.py`, `multivariate.py`, `sequential.py` (copied from `~/portfolio/ab-analyzer/stats/`)
- Create: `01-ab-testing-experimentation/visualization/__init__.py`, `charts.py` (copied from `~/portfolio/ab-analyzer/visualization/`)
- Create: `01-ab-testing-experimentation/data/__init__.py`, `example_tests.py` (copied from `~/portfolio/ab-analyzer/data/`)
- Create: `01-ab-testing-experimentation/tests/__init__.py`, `test_frequentist.py`, `test_bayesian.py`, `test_power.py` (copied from `~/portfolio/ab-analyzer/tests/`)
- Create: `01-ab-testing-experimentation/requirements.txt`

**Interfaces:**
- Consumes: nothing (this is the foundation task).
- Produces: `stats.frequentist.{two_proportion_z_test, welch_t_test, chi_square_test, ZTestResult, TTestResult, ChiSquareResult}`, `stats.power.{required_sample_size, current_power, days_to_significance, PowerResult}`, `stats.bayesian.{bayesian_ab_test, beta_binomial_posterior, probability_each_variant_is_best}`, `stats.multivariate.{run_multivariate_test, VariantData, bonferroni_correction, benjamini_hochberg_correction}`, `stats.sequential.{run_sequential_monitoring, obf_boundary}`, `visualization.charts.{conversion_rate_comparison_chart, lift_confidence_interval_chart, bayesian_posterior_chart, expected_loss_chart, sample_size_curve_chart, sequential_monitoring_chart, multivariant_bar_chart, p_value_comparison_chart}`, `data.example_tests.{ALL_EXAMPLES, EXAMPLES_BY_KEY, ExampleTest, MultiVariantExample, SEARCH_ALGORITHM}` — all used by later tasks.

- [ ] **Step 1: Copy the source files**

```bash
cd ~/data-science-portfolio/01-ab-testing-experimentation
mkdir -p stats visualization data tests analysis
cp ~/portfolio/ab-analyzer/stats/{__init__.py,frequentist.py,power.py,bayesian.py,multivariate.py,sequential.py} stats/
cp ~/portfolio/ab-analyzer/visualization/{__init__.py,charts.py} visualization/
cp ~/portfolio/ab-analyzer/data/{__init__.py,example_tests.py} data/
cp ~/portfolio/ab-analyzer/tests/{__init__.py,test_frequentist.py,test_bayesian.py,test_power.py} tests/
touch analysis/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
streamlit==1.38.0
numpy==1.26.4
scipy==1.13.1
pandas==2.2.2
plotly==5.23.0
statsmodels==0.14.2
pytest==8.3.2
kaggle==1.6.17
```

- [ ] **Step 3: Create a virtualenv and install dependencies**

```bash
cd ~/data-science-portfolio/01-ab-testing-experimentation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 4: Run the copied test suite to confirm nothing broke in the copy**

Run: `cd ~/data-science-portfolio/01-ab-testing-experimentation && source venv/bin/activate && python -m pytest tests/ -v`

Expected: all tests PASS (same tests that already passed in `~/portfolio/ab-analyzer`, since the source files are unchanged).

- [ ] **Step 5: Commit**

```bash
cd ~/data-science-portfolio
git add 01-ab-testing-experimentation/stats 01-ab-testing-experimentation/visualization 01-ab-testing-experimentation/data 01-ab-testing-experimentation/tests 01-ab-testing-experimentation/analysis/__init__.py 01-ab-testing-experimentation/requirements.txt
git commit -m "Project 1: bring in the ab-analyzer statistics engine"
```

---

### Task 2: Ship/hold/no-ship verdict logic

**Files:**
- Create: `01-ab-testing-experimentation/analysis/verdict.py`
- Test: `01-ab-testing-experimentation/tests/test_verdict.py`

**Interfaces:**
- Consumes: `stats.frequentist.ZTestResult` (fields: `p_value`, `alpha`, `absolute_lift`, `ci_low`, `ci_high`, `is_significant` — from Task 1), `stats.power.PowerResult` (field: `power` — from Task 1).
- Produces: `analysis.verdict.ShipVerdict` (fields: `decision: str`, `reasons: list[str]`) and `analysis.verdict.recommend(z_result: ZTestResult, power_result: PowerResult, min_power: float = 0.8) -> ShipVerdict`, used by Task 4 and Task 5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_verdict.py
from stats.frequentist import two_proportion_z_test
from stats.power import current_power
from analysis.verdict import recommend


def test_significant_positive_lift_ships():
    z = two_proportion_z_test(100, 10000, 400, 10000)  # huge, obvious win
    p = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, p)
    assert verdict.decision == "ship"
    assert any("lifts conversion" in r for r in verdict.reasons)


def test_significant_negative_lift_does_not_ship():
    z = two_proportion_z_test(300, 2000, 255, 2000)  # variant significantly worse
    p = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, p)
    assert verdict.decision == "no-ship"
    assert any("WORSE" in r for r in verdict.reasons)


def test_not_significant_low_power_holds():
    z = two_proportion_z_test(45, 300, 51, 300)  # small sample, inconclusive
    p = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, p, min_power=0.95)  # force power below threshold
    assert verdict.decision == "hold"
    assert any("underpowered" in r for r in verdict.reasons)


def test_not_significant_adequate_power_no_ships():
    z = two_proportion_z_test(300, 3000, 300, 3000)  # identical rates, huge n
    p = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, p, min_power=0.5)
    assert verdict.decision == "no-ship"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_verdict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.verdict'`.

- [ ] **Step 3: Implement `analysis/verdict.py`**

```python
"""Turns a hypothesis-test result and a power result into a plain-language
ship / hold / no-ship recommendation with the reasoning spelled out.

Decision rules
--------------
- Significant, positive lift            -> "ship"
- Significant, negative lift            -> "no-ship" (variant is harmful)
- Not significant, power < min_power    -> "hold" (test is underpowered;
                                            more data needed before concluding
                                            "no effect")
- Not significant, power >= min_power   -> "no-ship" (adequately powered
                                            test found no real effect)
"""

from __future__ import annotations

from dataclasses import dataclass

from stats.frequentist import ZTestResult
from stats.power import PowerResult


@dataclass
class ShipVerdict:
    decision: str  # "ship" | "hold" | "no-ship"
    reasons: list[str]


def recommend(
    z_result: ZTestResult,
    power_result: PowerResult,
    min_power: float = 0.8,
) -> ShipVerdict:
    reasons: list[str] = []

    if not z_result.is_significant:
        reasons.append(
            f"p-value {z_result.p_value:.4f} is not below alpha "
            f"{z_result.alpha} — result is not statistically significant."
        )
        if power_result.power < min_power:
            reasons.append(
                f"Achieved power is only {power_result.power:.2f} (below "
                f"{min_power}) — the test may simply be underpowered to "
                "detect the observed effect; collect more data before "
                "concluding there is no effect."
            )
            return ShipVerdict(decision="hold", reasons=reasons)
        reasons.append(
            f"Achieved power is {power_result.power:.2f}, so the test was "
            "adequately powered and still found no significant effect."
        )
        return ShipVerdict(decision="no-ship", reasons=reasons)

    if z_result.absolute_lift > 0:
        reasons.append(
            f"p-value {z_result.p_value:.4f} < alpha {z_result.alpha}; "
            f"variant lifts the rate by {z_result.absolute_lift * 100:.2f} "
            f"points (95% CI [{z_result.ci_low * 100:.2f}, "
            f"{z_result.ci_high * 100:.2f}])."
        )
        reasons.append(f"Achieved power: {power_result.power:.2f}.")
        return ShipVerdict(decision="ship", reasons=reasons)

    reasons.append(
        f"p-value {z_result.p_value:.4f} < alpha {z_result.alpha}; variant "
        f"is significantly WORSE than control by "
        f"{abs(z_result.absolute_lift) * 100:.2f} points."
    )
    return ShipVerdict(decision="no-ship", reasons=reasons)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_verdict.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add 01-ab-testing-experimentation/analysis/verdict.py 01-ab-testing-experimentation/tests/test_verdict.py
git commit -m "Project 1: ship/hold/no-ship verdict logic"
```

---

### Task 3: Real-data loader for the Cookie Cats dataset

**Files:**
- Create: `01-ab-testing-experimentation/data/load_cookie_cats.py`

**Interfaces:**
- Consumes: Kaggle API credentials at `~/.kaggle/kaggle.json` (already configured on this machine, user `leanthelcolon`).
- Produces: `data.load_cookie_cats.download() -> pathlib.Path` (path to the downloaded `cookie_cats.csv`), used by Task 4.

- [ ] **Step 1: Write the loader**

```python
# data/load_cookie_cats.py
"""Downloads the real Cookie Cats mobile-game A/B test dataset from Kaggle.

Dataset: mursideyarkin/mobile-games-ab-testing-cookie-cats — real player
records (userid, experiment arm, rounds played, 1-day and 7-day retention).
This is genuine, not simulated, data; nothing in this file invents numbers,
it only fetches and locates the real CSV.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

from pathlib import Path

import kaggle

DATA_DIR = Path(__file__).parent
CSV_PATH = DATA_DIR / "cookie_cats.csv"
KAGGLE_DATASET = "mursideyarkin/mobile-games-ab-testing-cookie-cats"


def download() -> Path:
    if CSV_PATH.exists():
        return CSV_PATH

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET, path=str(DATA_DIR), unzip=True
    )

    if not CSV_PATH.exists():
        found = list(DATA_DIR.glob("*.csv"))
        if len(found) == 1:
            found[0].rename(CSV_PATH)
        else:
            raise FileNotFoundError(
                f"Expected cookie_cats.csv after Kaggle download, found: {found}"
            )
    return CSV_PATH


if __name__ == "__main__":
    path = download()
    print(f"Cookie Cats dataset ready at {path}")
```

- [ ] **Step 2: Run it to actually fetch the real dataset**

Run: `cd ~/data-science-portfolio/01-ab-testing-experimentation && source venv/bin/activate && python -m data.load_cookie_cats`
Expected: prints `Cookie Cats dataset ready at .../data/cookie_cats.csv`, and that file exists on disk. If this fails (dataset renamed/removed on Kaggle, or credential issue), stop and pick a substitute real Kaggle A/B dataset (e.g. "Marketing A/B Testing" by faviovaz) rather than fabricating data to fill the gap — update this task's dataset name/columns accordingly if that happens.

- [ ] **Step 3: Sanity-check the real data by hand**

Run: `python -c "import pandas as pd; df = pd.read_csv('data/cookie_cats.csv'); print(df.shape); print(df.columns.tolist()); print(df['version'].value_counts())"`
Expected: a real row count (tens of thousands), columns including `userid`, `version`, `sum_gamerounds`, `retention_1`, `retention_7`, and two version groups (`gate_30`, `gate_40`). Record the actual printed shape here in a comment at the top of Task 4's analysis script — do not guess it.

- [ ] **Step 4: Commit the loader (not the downloaded CSV — it's gitignored)**

```bash
git add 01-ab-testing-experimentation/data/load_cookie_cats.py
git commit -m "Project 1: real Cookie Cats dataset loader"
```

---

### Task 4: Real-data analysis script for Cookie Cats

**Files:**
- Create: `01-ab-testing-experimentation/analysis/cookie_cats_analysis.py`
- Test: `01-ab-testing-experimentation/tests/test_cookie_cats_analysis.py`

**Interfaces:**
- Consumes: `data.load_cookie_cats.download` (Task 3), `stats.frequentist.two_proportion_z_test`, `stats.power.current_power` (Task 1), `analysis.verdict.recommend` (Task 2).
- Produces: `analysis.cookie_cats_analysis.analyze_retention(df: pandas.DataFrame, retention_col: str, alpha: float = 0.05) -> tuple[ZTestResult, PowerResult, ShipVerdict]`, used by Task 5 (the Streamlit app) and to generate the real numbers for Task 6's README.

- [ ] **Step 1: Write the failing test (uses a small crafted DataFrame fixture to verify the wiring — not the real dataset, so the test is fast and offline; the real dataset is exercised in Step 5 below)**

```python
# tests/test_cookie_cats_analysis.py
import pandas as pd

from analysis.cookie_cats_analysis import analyze_retention


def test_analyze_retention_wires_stats_correctly():
    df = pd.DataFrame(
        {
            "version": ["gate_30"] * 100 + ["gate_40"] * 100,
            "retention_1": [1] * 40 + [0] * 60 + [1] * 55 + [0] * 45,
        }
    )
    z, power, verdict = analyze_retention(df, "retention_1")

    assert z.control_n == 100
    assert z.variant_n == 100
    assert z.control_conversions == 40
    assert z.variant_conversions == 55
    assert verdict.decision in {"ship", "hold", "no-ship"}
    assert 0.0 <= power.power <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_cookie_cats_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.cookie_cats_analysis'`.

- [ ] **Step 3: Implement the analysis script**

```python
# analysis/cookie_cats_analysis.py
"""Real-data A/B test analysis: Cookie Cats gate placement.

gate_30 (control) vs. gate_40 (variant) — does moving the level-30 paywall
gate to level 40 change 1-day and 7-day retention? Uses the real Kaggle
dataset loaded by data.load_cookie_cats, and the same from-scratch
two-proportion z-test / power analysis used throughout this project.
"""

from __future__ import annotations

import pandas as pd

from analysis.verdict import ShipVerdict, recommend
from data.load_cookie_cats import download
from stats.frequentist import ZTestResult, two_proportion_z_test
from stats.power import PowerResult, current_power


def load_data() -> pd.DataFrame:
    return pd.read_csv(download())


def analyze_retention(
    df: pd.DataFrame, retention_col: str, alpha: float = 0.05
) -> tuple[ZTestResult, PowerResult, ShipVerdict]:
    control = df[df["version"] == "gate_30"]
    variant = df[df["version"] == "gate_40"]

    control_n = len(control)
    variant_n = len(variant)
    control_conv = int(control[retention_col].sum())
    variant_conv = int(variant[retention_col].sum())

    z_result = two_proportion_z_test(
        control_conv, control_n, variant_conv, variant_n, alpha=alpha
    )
    power_result = current_power(
        z_result.control_rate, z_result.variant_rate, control_n, variant_n, alpha=alpha
    )
    verdict = recommend(z_result, power_result)
    return z_result, power_result, verdict


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} real players from the Cookie Cats dataset.")
    for col in ("retention_1", "retention_7"):
        z, p, v = analyze_retention(df, col)
        print(f"\n=== {col} ===")
        print(f"Control (gate_30) rate: {z.control_rate:.4f} (n={z.control_n})")
        print(f"Variant (gate_40) rate: {z.variant_rate:.4f} (n={z.variant_n})")
        print(f"p-value: {z.p_value:.4f}  |  95% CI on lift: [{z.ci_low:.4f}, {z.ci_high:.4f}]")
        print(f"Achieved power: {p.power:.4f}")
        print(f"Verdict: {v.decision.upper()}")
        for reason in v.reasons:
            print(f"  - {reason}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_cookie_cats_analysis.py -v`
Expected: PASS.

- [ ] **Step 5: Run the script against the real downloaded data and record the actual output**

Run: `python -m analysis.cookie_cats_analysis`
Expected: real printed control/variant rates, p-values, CIs, power, and verdicts for both `retention_1` and `retention_7`. Save this exact console output — it is the source of the real numbers used in Task 6's README (copy the actual printed values, not approximations).

- [ ] **Step 6: Commit**

```bash
git add 01-ab-testing-experimentation/analysis/cookie_cats_analysis.py 01-ab-testing-experimentation/tests/test_cookie_cats_analysis.py
git commit -m "Project 1: real Cookie Cats retention analysis"
```

---

### Task 5: Streamlit front end

**Files:**
- Create: `01-ab-testing-experimentation/app.py`
- Create: `01-ab-testing-experimentation/.streamlit/config.toml`

**Interfaces:**
- Consumes: `data.example_tests.{ALL_EXAMPLES, SEARCH_ALGORITHM}` (Task 1), `stats.frequentist.{two_proportion_z_test, welch_t_test}`, `stats.power.current_power`, `stats.multivariate.run_multivariate_test` (Task 1), `visualization.charts.*` (Task 1), `analysis.verdict.recommend` (Task 2), `analysis.cookie_cats_analysis.{load_data, analyze_retention}` (Task 4).
- Produces: a runnable Streamlit app; no other task depends on its internals.

- [ ] **Step 1: Write `.streamlit/config.toml`**

```toml
[theme]
base = "light"
primaryColor = "#2a78d6"
backgroundColor = "#fcfcfb"
secondaryBackgroundColor = "#f4f3ee"
textColor = "#0b0b0b"
font = "sans serif"

[server]
headless = true
```

- [ ] **Step 2: Write `app.py`**

```python
# app.py
"""Streamlit app: A/B Testing & Experimentation Analysis.

Two modes:
  - Real data: the Cookie Cats mobile-game retention experiment (Kaggle).
  - Simulated examples: 5 labeled-synthetic scenarios illustrating distinct
    verdict types (borderline, clear win, inconclusive, harmful, multi-variant).
"""

from __future__ import annotations

import streamlit as st

from analysis.cookie_cats_analysis import analyze_retention, load_data
from analysis.verdict import recommend
from data.example_tests import ALL_EXAMPLES, EXAMPLES_BY_KEY, SEARCH_ALGORITHM
from stats.frequentist import two_proportion_z_test, welch_t_test
from stats.multivariate import run_multivariate_test
from stats.power import current_power
from visualization.charts import (
    conversion_rate_comparison_chart,
    lift_confidence_interval_chart,
    multivariant_bar_chart,
    p_value_comparison_chart,
)

st.set_page_config(page_title="A/B Testing & Experimentation", layout="wide")
st.title("A/B Testing & Experimentation Analysis")
st.caption(
    "Real hypothesis testing, power analysis, and ship/no-ship recommendations "
    "— every number below comes from code in this repo, not hand-typed figures."
)

mode = st.sidebar.radio(
    "Data source",
    ["Real data: Cookie Cats (Kaggle)", "Simulated examples", "Multi-variant example (simulated)"],
)

if mode == "Real data: Cookie Cats (Kaggle)":
    st.header("Cookie Cats: Gate 30 vs. Gate 40")
    st.markdown(
        "**Real dataset** — [Cookie Cats mobile game A/B test]"
        "(https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats), "
        "not simulated. Does moving the level-30 paywall gate to level 40 "
        "change player retention?"
    )
    df = load_data()
    st.write(f"Loaded **{len(df):,}** real players.")

    metric = st.selectbox("Retention metric", ["retention_1", "retention_7"])
    z, power, verdict = analyze_retention(df, metric)

    col1, col2, col3 = st.columns(3)
    col1.metric("Control (gate_30) rate", f"{z.control_rate*100:.2f}%")
    col2.metric("Variant (gate_40) rate", f"{z.variant_rate*100:.2f}%")
    col3.metric("p-value", f"{z.p_value:.4f}")

    st.plotly_chart(
        conversion_rate_comparison_chart(
            "Gate 30 (Control)", "Gate 40 (Variant)",
            z.control_rate, z.variant_rate, z.control_n, z.variant_n,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        lift_confidence_interval_chart(z.absolute_lift, z.ci_low, z.ci_high, z.is_significant),
        use_container_width=True,
    )
    st.write(f"Achieved power: **{power.power:.2f}**")

    verdict_color = {"ship": "green", "hold": "orange", "no-ship": "red"}[verdict.decision]
    st.markdown(f"### Verdict: :{verdict_color}[{verdict.decision.upper()}]")
    for reason in verdict.reasons:
        st.markdown(f"- {reason}")

elif mode == "Simulated examples":
    st.header("Simulated Example Scenarios")
    st.info(
        "**Simulated data** — these 5 scenarios use seeded-RNG synthetic "
        "counts, chosen to illustrate 5 distinct real-world verdicts. "
        "Not real experiment data."
    )
    example_key = st.selectbox(
        "Scenario", options=[ex.key for ex in ALL_EXAMPLES],
        format_func=lambda k: EXAMPLES_BY_KEY[k].name,
    )
    ex = EXAMPLES_BY_KEY[example_key]
    st.subheader(ex.name)
    st.caption(ex.tagline)
    st.write(ex.narrative)

    z = two_proportion_z_test(ex.control_conversions, ex.control_n, ex.variant_conversions, ex.variant_n)
    power = current_power(z.control_rate, z.variant_rate, z.control_n, z.variant_n)
    verdict = recommend(z, power)

    col1, col2, col3 = st.columns(3)
    col1.metric(ex.control_label, f"{z.control_rate*100:.2f}%")
    col2.metric(ex.variant_label, f"{z.variant_rate*100:.2f}%")
    col3.metric("p-value", f"{z.p_value:.4f}")

    st.plotly_chart(
        conversion_rate_comparison_chart(ex.control_label, ex.variant_label, z.control_rate, z.variant_rate, z.control_n, z.variant_n),
        use_container_width=True,
    )
    st.plotly_chart(
        lift_confidence_interval_chart(z.absolute_lift, z.ci_low, z.ci_high, z.is_significant),
        use_container_width=True,
    )

    if ex.control_values is not None:
        t = welch_t_test(ex.control_values, ex.variant_values)
        st.write(f"**{ex.continuous_metric_name}**: control mean ${t.control_mean:.2f}, variant mean ${t.variant_mean:.2f}, p={t.p_value:.4f}")

    verdict_color = {"ship": "green", "hold": "orange", "no-ship": "red"}[verdict.decision]
    st.markdown(f"### Verdict: :{verdict_color}[{verdict.decision.upper()}]")
    for reason in verdict.reasons:
        st.markdown(f"- {reason}")

else:
    st.header(SEARCH_ALGORITHM.name)
    st.info("**Simulated data** — 4 candidate algorithms tested against control at once.")
    st.write(SEARCH_ALGORITHM.narrative)

    from stats.multivariate import VariantData

    control = VariantData(name=SEARCH_ALGORITHM.control_label, n=SEARCH_ALGORITHM.control_n, conversions=SEARCH_ALGORITHM.control_conversions)
    variants = [
        VariantData(name=name, n=n, conversions=conv)
        for name, n, conv in zip(SEARCH_ALGORITHM.variant_labels, SEARCH_ALGORITHM.variant_ns, SEARCH_ALGORITHM.variant_conversions)
    ]
    result = run_multivariate_test(control, variants)

    st.plotly_chart(
        multivariant_bar_chart(
            control.name, control.rate,
            [c.variant_name for c in result.comparisons],
            [c.variant_rate for c in result.comparisons],
            [c.significant_bh for c in result.comparisons],
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        p_value_comparison_chart(
            [c.variant_name for c in result.comparisons],
            [c.p_value_raw for c in result.comparisons],
            [c.p_value_bonferroni for c in result.comparisons],
            [c.p_value_bh for c in result.comparisons],
        ),
        use_container_width=True,
    )

    if result.winner:
        st.markdown(f"### Recommended winner: :green[{result.winner}]")
    else:
        st.markdown("### No variant reached significance after correction — :orange[hold]")
```

- [ ] **Step 3: Run the app locally and click through all three modes**

Run: `streamlit run app.py --server.headless true &` then check `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501` returns `200`; stop the background process afterward (`kill %1`).
Expected: app starts without exceptions; HTTP 200. If it errors, read the Streamlit traceback and fix the specific import/logic error before proceeding — do not skip this check.

- [ ] **Step 4: Commit**

```bash
git add 01-ab-testing-experimentation/app.py 01-ab-testing-experimentation/.streamlit
git commit -m "Project 1: Streamlit front end"
```

---

### Task 6: README with real numbers, root README update, deploy prep

**Files:**
- Create: `01-ab-testing-experimentation/README.md`
- Modify: `README.md` (repo root, row for project 1)
- Delete: `01-ab-testing-experimentation/.gitkeep`

**Interfaces:**
- Consumes: the actual console output captured in Task 4 Step 5 (real Cookie Cats numbers).
- Produces: nothing consumed by other tasks — this is the terminal task for project 1.

- [ ] **Step 1: Write `README.md`** using the exact numbers captured in Task 4 Step 5 (the template below has bracketed placeholders — replace every one with the real printed value before committing; do not commit any bracketed placeholder)

```markdown
# A/B Testing & Experimentation Analysis

Real hypothesis testing, power analysis, and a ship/no-ship recommendation
— on both a real public dataset and clearly-labeled simulated scenarios.

## Headline result: Cookie Cats gate placement

**Real dataset**: [Cookie Cats mobile game A/B test](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)
via Kaggle, [N] real players, not simulated.

Does moving the level-30 paywall gate to level 40 change player retention?

| Metric | Control (gate_30) | Variant (gate_40) | p-value | 95% CI on lift | Verdict |
|--------|--------------------|--------------------|---------|-----------------|---------|
| 1-day retention | [X]% | [Y]% | [p] | [[lo, hi]] | **[VERDICT]** |
| 7-day retention | [X]% | [Y]% | [p] | [[lo, hi]] | **[VERDICT]** |

**What this tells you**: [one sentence stating the actual business
conclusion from the real verdict above — e.g. whether gate_40 should ship].

## Simulated teaching examples

5 scenarios (clearly labeled **simulated data** in the app) built on
seeded-RNG synthetic counts, each illustrating a distinct real-world
verdict: borderline significance, a clear win, an inconclusive result, a
harmful variant, and a multi-variant test with multiple-comparison
correction.

## Method

- Two-proportion z-test and Welch's t-test implemented from first
  principles (see `stats/frequentist.py`), cross-validated against
  `statsmodels`/`scipy` reference implementations in `tests/`.
- Power analysis (`stats/power.py`) and O'Brien-Fleming sequential
  monitoring (`stats/sequential.py`) for practical experiment design.
- Bayesian Beta-Binomial analysis (`stats/bayesian.py`) as an alternative
  to the frequentist framework.
- Ship/hold/no-ship recommendation logic (`analysis/verdict.py`) that
  accounts for both significance and achieved power — a non-significant
  underpowered test is flagged "hold", not treated the same as a
  well-powered null result.

## Run it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m data.load_cookie_cats        # fetch real dataset (needs Kaggle API creds)
python -m analysis.cookie_cats_analysis  # print the real analysis
streamlit run app.py                    # interactive app
```

## Live demo

[Streamlit Community Cloud link — added after deployment approval]

## Notes

- Requires Kaggle API credentials (`~/.kaggle/kaggle.json`) to fetch the
  real dataset; the CSV itself is not committed to this repo.
- Everything in this README is generated by the scripts in this
  repository — see `analysis/cookie_cats_analysis.py` for the exact
  computation behind the headline table above.
```

- [ ] **Step 2: Update the root README table row for project 1**

In `~/data-science-portfolio/README.md`, replace the row:
```
| 1 | [A/B Testing & Experimentation](01-ab-testing-experimentation/) | Hypothesis testing, power analysis, ship/no-ship decisions | _pending_ |
```
with:
```
| 1 | [A/B Testing & Experimentation](01-ab-testing-experimentation/) | Hypothesis testing, power analysis, ship/no-ship decisions | [Live demo](README.md) _(update once Streamlit Cloud URL exists)_ |
```

- [ ] **Step 3: Remove the placeholder and commit**

```bash
cd ~/data-science-portfolio
rm 01-ab-testing-experimentation/.gitkeep
git add 01-ab-testing-experimentation/README.md README.md
git rm 01-ab-testing-experimentation/.gitkeep
git commit -m "Project 1: README with real Cookie Cats results"
```

- [ ] **Step 4: Final verification**

Run: `cd ~/data-science-portfolio/01-ab-testing-experimentation && source venv/bin/activate && python -m pytest -v`
Expected: all tests across the whole project directory PASS. This is the completion gate for project 1 — do not mark it done otherwise.
