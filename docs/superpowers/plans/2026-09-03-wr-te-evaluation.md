# WR/TE Weekly Walk-Forward Evaluation MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, reproducible, offline evaluator that answers whether the frozen WR/TE RandomForest probabilities beat overall/position base rates and fixed L2 logistic regression on weekly out-of-sample log loss and Brier score.

**Architecture:** `evaluation.py` owns pure DataFrame logic for eligibility, whole-week folds, fixed models, temporal Platt calibration, pooled metrics, and simple odds/ROI diagnostics. `evaluate_wr.py` owns all local file I/O and atomically writes five deterministic artifacts. Existing production feature engineering, odds matching, and payout conventions are reused without changing `train_wr.py`, `predict_wr.py`, `odds_match.py`, or `ledger.py`.

**Tech Stack:** Python 3.13, pandas, NumPy, scikit-learn, pytest, and standard-library `argparse`, `hashlib`, `json`, and `pathlib`.

**Spec:** `docs/superpowers/specs/2026-09-02-wr-te-evaluation-design.md`

## Global Constraints

- This design does not change production training or prediction behavior.
- The default source is `config.DATA_DIR / "raw_nfl_data.csv"`; evaluation is cached/offline and makes no nflreadpy or HTTP calls.
- Eligibility is computed once, before fitting any model.
- For each outer test week, every model variant receives exactly the same ordered row keys.
- A missing probability is a fold failure, never permission to drop a row for one model.
- The outer test week is never used for hyperparameter selection, calibration, thresholds, feature selection, or eligibility rules.
- A new forest is fitted per fold using the exact parameters in `wr_te_rf_best_params.json`, `WR_TE_FEATURES`, and the run seed. No search.
- ROI is a downstream diagnostic, never a model-selection objective.
- Root-level `wr_te/vegas/week_*` files are excluded because their season is not explicit.
- Legacy ROI is labeled **research-only, timestamp unsafe**. It must not be described as Tuesday-open performance.
- Every historical artifact labels the football evaluation `retrospective_finalish_game_context`.
- Missing odds never remove a football row and never fail football evaluation.
- Optional `--output-dir` resolves beneath `config.EVALUATION_DIR`.
- Rerunning replaces only the five known files in the selected run directory and preserves unrelated files.
- Output files are serialized completely to temporary siblings before any destination is replaced.
- A `(season, week)` group is indivisible. The local cache is treated as frozen only after all games in an in-progress week are final; the README makes this operator requirement explicit.
- Do not add a dashboard, database, notebook, experiment server, network-refresh flag, tuning flag, wager placement, or RB support.

All commands run from `wr_te/`. Do not read or modify `wr_te/claude_transcript.md`. The only production file changed is `config.py`, where the sole addition is `EVALUATION_DIR`.

## File map and dependencies

| File | Change | Responsibility |
|---|---|---|
| `wr_te/config.py` | modify | Add `EVALUATION_DIR = BASE_DIR / "evaluation"`. |
| `wr_te/evaluation.py` | create | Pure eligibility, folds, fixed model predictions, calibration, metrics, odds matching, and ROI. |
| `wr_te/evaluate_wr.py` | create | Offline CLI, local input discovery/loading, manifest creation, deterministic serialization, atomic publication. |
| `wr_te/tests/test_evaluation.py` | create | One synthetic/offline module covering the representative safety and correctness cases. |
| `wr_te/README.md` | modify | Short run command and provenance/cache caveats. |

Task 1 produces canonical predictions. Task 2 consumes those predictions and produces artifact-ready metrics/betting records. Task 3 composes both into files and documentation.

The MVP writes exactly five artifacts:

1. `manifest.json` — arguments, definitions, seed, hashes, features/RF parameters, filter/fold counts, provenance, coverage, and deferred diagnostics.
2. `folds.csv` — one row per test week with fit/calibration/test bounds, counts, event rates, and status.
3. `predictions.csv` — long-form shared-key model/variant predictions plus open-odds coverage/provenance.
4. `metrics.json` — football-context provenance, pooled primary metrics, a ten-bin calibration table, simple betting results, and metric definitions.
5. `summary.csv` — one flat pooled row per model/variant with primary metrics/deltas and betting fields where applicable.

Season/position slice files, PR/ROC AUC, calibration slope/intercept, top-k accuracy, week-cluster bootstrap intervals, closing-price CLV, and maximum drawdown are explicitly deferred from this MVP. Record these names under `manifest.json["deferred"]`; do not emit misleading zeroes for them.

---

### Task 1: Eligible rows, whole-week folds, fixed models, and temporal calibration

**Files:**
- Modify: `wr_te/config.py:10-15`
- Create: `wr_te/evaluation.py`
- Create: `wr_te/tests/test_evaluation.py`

**Interfaces:**

```python
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

GroupKey = tuple[int, int]
ROW_KEY_COLUMNS = ["season", "week", "game_id", "player_id"]
PREGAME_FEATURES = {"implied_total", "spread_line", "depth_chart_rank"}
MODEL_VARIANTS = (
    ("base_rate_overall", "raw"),
    ("base_rate_position", "raw"),
    ("logistic_l2", "raw"),
    ("logistic_l2", "platt"),
    ("random_forest_current", "raw"),
    ("random_forest_current", "platt"),
)

@dataclass(frozen=True)
class FoldSpec:
    test_group: GroupKey
    calibration_groups: tuple[GroupKey, ...]
    fit_groups: tuple[GroupKey, ...]

def prepare_evaluation_rows(
    raw: pd.DataFrame,
    feature_engineer: Callable[[pd.DataFrame], pd.DataFrame],
    feature_columns: Sequence[str] = WR_TE_FEATURES,
) -> tuple[pd.DataFrame, dict[str, int]]: ...

def build_walk_forward_folds(
    rows: pd.DataFrame,
    start_season: int,
    end_season: int,
    calibration_weeks: int,
) -> list[FoldSpec]: ...

def evaluate_folds(
    rows: pd.DataFrame,
    folds: Sequence[FoldSpec],
    rf_params: Mapping[str, Any],
    seed: int,
    estimator_factories: Mapping[str, Callable[[int, Mapping[str, Any]], Any]] | None = None,
    calibrator_factory: Callable[[], Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...
```

`prepare_evaluation_rows` requires the row key, player/team/position/outcome metadata, every raw column read by `train_wr.feature_engineering`, and every resulting `WR_TE_FEATURES` column. It filters once to WR/TE, weeks 1–18, non-null keys, binary `scored_touchdown`, and finite engineered features; rejects duplicate eligible row keys; sets `evaluation_stream` to `retrospective` through 2025 and `prospective_2026` for 2026; and returns rows stably sorted by the row key plus exact filter counts. The feature contract permits the three named same-week pregame fields; every other model feature must be one of the existing shifted player/opponent outputs.

`build_walk_forward_folds` sorts distinct group keys lexicographically. For each requested test group it assigns exactly the immediately preceding `calibration_weeks` whole groups to calibration and every still-earlier group to fit. Before any estimator fits, it rejects the earliest requested group unless it has at least one fit group plus the full calibration window; rejects a `game_id` mapped to multiple groups; and verifies fit, calibration, and test groups/row keys are pairwise disjoint.

`evaluate_folds` uses only fit rows for both base rates and learned-model fitting. The overall baseline is the fit outcome mean; the position baseline requires both WR and TE fit rows. Logistic regression is `StandardScaler()` plus `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)`. Each fold RF is `RandomForestClassifier(**rf_params, random_state=seed, n_jobs=1)` with exactly these saved parameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`. Each learned model predicts the calibration window, fits a separate repository-convention one-feature `LogisticRegression()` Platt model on clipped calibration probabilities, and then emits raw and Platt test probabilities. Calibration must contain both classes. Every probability must be finite and within `[0,1]` before concatenation.

- [ ] **Step 1: Write the representative eligibility/fold test**

In `tests/test_evaluation.py`, create `_raw_rows(groups)` that emits two WR and two TE rows for each group, two distinct game IDs, binary outcomes, every `PLAYER_EWM_STATS` input, opponent-defense inputs, and pregame features. Create `_identity_features` for tests that already supply finite `WR_TE_FEATURES`.

```python
def test_whole_week_folds_cross_season_boundary_and_reject_short_history():
    rows, counts = evaluation.prepare_evaluation_rows(
        _raw_rows([(2021, 16), (2021, 17), (2021, 18), (2022, 1), (2022, 2)]),
        _identity_features,
    )
    folds = evaluation.build_walk_forward_folds(rows, 2022, 2022, 2)
    assert folds[0] == evaluation.FoldSpec(
        test_group=(2022, 1),
        calibration_groups=((2021, 17), (2021, 18)),
        fit_groups=((2021, 16),),
    )
    assert [fold.test_group for fold in folds] == [(2022, 1), (2022, 2)]
    assert counts["eligible_rows"] == 20

    short, _ = evaluation.prepare_evaluation_rows(
        _raw_rows([(2021, 18), (2022, 1)]), _identity_features
    )
    with pytest.raises(ValueError, match=r"earliest requested group.*available history=1.*required=3"):
        evaluation.build_walk_forward_folds(short, 2022, 2022, 2)
```

In the same test, assert every test row from `(2022, 1)` is present, no fit/calibration/test row key overlaps, and no game appears in two partitions. Add one concise duplicate-key assertion. Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -k "whole_week or duplicate" -v
```

Expected: FAIL because `evaluation.py` and `config.EVALUATION_DIR` do not exist.

- [ ] **Step 2: Implement config, eligibility, and fold construction**

Add exactly `EVALUATION_DIR = BASE_DIR / "evaluation"` to `config.py`. In `evaluation.py`, implement the constants, `FoldSpec`, `prepare_evaluation_rows`, and `build_walk_forward_folds` as specified above. Copy the raw frame before calling the mutating feature engineer. Apply feature engineering once for the entire canonical population, never separately by model or fold. Use stable sorts throughout.

Run the Step 1 command. Expected: PASS.

- [ ] **Step 3: Write one shared-keys/no-leakage test**

Use one recording logistic estimator and one recording RF estimator through `estimator_factories`, plus a recording calibrator. Give each group a distinct first-feature marker. Assert learned estimators fit only the earliest fit group, calibrators fit only probabilities from the two calibration groups, and all six model/variant slices have the exact ordered test keys. Perturb only the outer-test outcomes, rerun from raw rows through feature engineering/evaluation, and assert every test probability remains unchanged.

Also assert a one-class calibration partition fails before estimator fitting, and an estimator returning a NaN probability fails instead of dropping its row.

```python
expected_keys = list(map(tuple, test_rows[evaluation.ROW_KEY_COLUMNS].to_numpy()))
for model, variant in evaluation.MODEL_VARIANTS:
    got = predictions.loc[
        (predictions["model"] == model) & (predictions["variant"] == variant),
        evaluation.ROW_KEY_COLUMNS,
    ]
    assert list(map(tuple, got.to_numpy())) == expected_keys
```

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -k "shared_keys or no_leakage or invalid_probability" -v
```

Expected: FAIL because `evaluate_folds` is absent.

- [ ] **Step 4: Implement fixed prediction and temporal calibration**

Before the fold loop, validate every fold's partition invariants, both fit outcome classes, both fit positions, and both calibration outcome classes. Then fit/predict in fixed model order. Build every variant from the same copied test metadata frame and compare its ordered row-key list with the first variant before concatenating. Return predictions sorted by `season,week,game_id,player_id,model,variant` and folds sorted by test group. Include `football_context_provenance="retrospective_finalish_game_context"` in both outputs.

Do not import or call `RandomizedSearchCV`, `train_wr.train_rf_model`, OOB calibration, threshold selection, or outer-test metrics inside model fitting.

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -v
```

Expected: all Task 1 tests PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add config.py evaluation.py tests/test_evaluation.py
git commit -m "feat(wr_te): add weekly evaluation folds and predictions"
```

---

### Task 2: Pooled primary metrics and simple provenance-aware ROI

**Files:**
- Modify: `wr_te/evaluation.py`
- Modify: `wr_te/tests/test_evaluation.py`

**Interfaces:**

```python
def select_open_odds(
    tagged_open: pd.DataFrame | None,
    legacy: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str, list[str]]: ...

def attach_open_odds_and_score_bets(
    predictions: pd.DataFrame,
    odds_by_group: Mapping[GroupKey, tuple[pd.DataFrame, str]],
    team_map: Mapping[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]: ...

def build_metric_records(
    predictions: pd.DataFrame,
    betting_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]: ...
```

`select_open_odds` normalizes the two supported local schemas. A tagged snapshot is `timestamp_safe_open` only when every row has `tag=open`, `in_play=False`, parseable non-empty `last_update`, `fetched_at`, and `commence_time`, and both timestamps precede kickoff. Otherwise the entire tagged file is rejected and a year-scoped legacy file becomes `timestamp_unsafe_legacy`; with neither source it returns `uncovered`. Legacy columns map as `Player→description`, `HomeTeam→home_team`, `AwayTeam→away_team`, `Odds→price`, `Bookmaker→bookmaker`, `Season→season`, `Week→week`. It never reads a file itself.

`attach_open_odds_and_score_bets` calls `odds_match.match_odds_to_players` for actual name/team-in-game matching and best-price selection. It left-joins matches to the unique football row keys and then copies odds fields to every variant, so uncovered rows remain in `predictions.csv`. Coverage is `covered unique eligible row keys / all unique eligible row keys`, overall and by week. For logistic/RF raw and Platt variants only, rank covered rows within each week by probability descending and `player_id` ascending, stake one unit on up to five, and reuse `ledger.decimal_odds`: a hit earns `decimal_odds(price)-1`, a miss loses `1`. Report bets, settled bets, stake, PnL, ROI, hits, hit rate, coverage, and provenance per learned model/variant. Base-rate betting fields stay null.

`build_metric_records` computes pooled out-of-sample log loss, Brier score, and Brier skill `1 - Brier(model) / Brier(base_rate_overall)` on exact matching row keys. It also emits model-minus-reference log-loss/Brier deltas and a simple weighted ten-bin ECE table including zero-count bins. Metrics are separated by `evaluation_stream`; no retrospective/prospective pooling and no refitting occur.

- [ ] **Step 1: Write the hand-computable primary-metric test**

```python
def test_primary_metrics_use_exact_reference_keys_and_weighted_ece():
    model = _prediction_frame([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], "logistic_l2", "raw")
    reference = _prediction_frame([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], "base_rate_overall", "raw")
    metrics, summary = evaluation.build_metric_records(
        pd.concat([model, reference], ignore_index=True), []
    )
    row = summary.query("model == 'logistic_l2' and variant == 'raw'").iloc[0]
    assert row["log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.8, 0.9])))
    assert row["brier"] == pytest.approx(0.025)
    assert row["brier_skill"] == pytest.approx(0.9)
    assert row["brier_reference"] == "base_rate_overall/raw"
    assert row["ece"] == pytest.approx(0.15)
    assert len(metrics["overall"][0]["calibration_bins"]) == 10
```

Remove one reference key and assert `ValueError("reference key mismatch")`. Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -k "primary_metrics or reference_key" -v
```

Expected: FAIL because metric construction is absent.

- [ ] **Step 2: Implement pooled metrics and flat summary records**

Clip probabilities only for numerical log-loss evaluation. Bin `[0.0,0.1), …, [0.9,1.0]`, placing exactly `1.0` in the last bin; empty bins have count zero and JSON-null means/rates/gaps. Add betting fields by `(evaluation_stream, model, variant)`. Sort metric records and summary rows by stream/model/variant. ROI never participates in the definitions or primary deltas.

Run the Step 1 command. Expected: PASS.

- [ ] **Step 3: Write one odds precedence/matching/payout test**

Create a safe tagged open row and a legacy row for the same player; assert tagged wins. Then blank `fetched_at`; assert legacy wins and provenance is `timestamp_unsafe_legacy`. Include two same-named players on different teams, one uncovered player, and two bookmaker prices for one valid match. Assert team-aware matching selects the correct players and higher American price, while all prediction rows remain.

Use three covered learned-model rows with outcomes `[1,0,0]` and prices `[+200,-150,+300]`:

```python
enriched, betting = evaluation.attach_open_odds_and_score_bets(
    predictions, odds_by_group, TEAM_MAP
)
report = next(row for row in betting if row["model"] == "logistic_l2" and row["variant"] == "raw")
assert report["bets"] == 3
assert report["stake"] == 3.0
assert report["pnl"] == pytest.approx(0.0)  # +2, -1, -1
assert report["roi"] == pytest.approx(0.0)
assert report["hits"] == 1
assert report["hit_rate"] == pytest.approx(1 / 3)
assert report["betting_label"] == "research-only, timestamp unsafe"
assert len(enriched) == len(predictions)
```

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -k "odds_and_payout" -v
```

Expected: FAIL because odds/betting logic is absent.

- [ ] **Step 4: Implement simple odds/ROI diagnostics**

Normalize tagged/legacy column names in private helpers used only by `select_open_odds`. Pass each week's unique player metadata to `match_odds_to_players` with a synthetic `evaluation_row_id` rather than the NFL `game_id`; join the selected price/bookmaker back through that ID. Populate `odds_covered`, `price_open`, `bookmaker_open`, and `open_provenance` on every variant. Preserve the exact legacy display label and record rejected-unsafe-tagged warnings. Never inspect root-level odds paths here; Task 3 supplies only year-scoped frames.

Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -v
```

Expected: all Task 1–2 tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add evaluation.py tests/test_evaluation.py
git commit -m "feat(wr_te): add core metrics and ROI diagnostics"
```

---

### Task 3: Offline CLI, five deterministic artifacts, and README

**Files:**
- Create: `wr_te/evaluate_wr.py`
- Modify: `wr_te/tests/test_evaluation.py`
- Modify: `wr_te/README.md:8-16,80-91`

**Interfaces:**

```python
ARTIFACT_NAMES = (
    "manifest.json", "folds.csv", "predictions.csv", "metrics.json", "summary.csv"
)

def resolve_output_dir(
    requested: Path | None,
    start_season: int,
    end_season: int,
    calibration_weeks: int,
    seed: int,
    evaluation_dir: Path = config.EVALUATION_DIR,
) -> Path: ...

def run_evaluation(
    *,
    start_season: int,
    end_season: int,
    calibration_weeks: int,
    seed: int,
    input_path: Path,
    rf_params_path: Path,
    vegas_dir: Path,
    team_path: Path,
    output_dir: Path,
    feature_engineer: Callable[[pd.DataFrame], pd.DataFrame] = feature_engineering,
    estimator_factories: Mapping[str, Callable] | None = None,
) -> dict[str, Path]: ...

def write_artifacts_atomic(
    output_dir: Path,
    payloads: Mapping[str, bytes],
) -> dict[str, Path]: ...

def main(argv: Sequence[str] | None = None) -> int: ...
```

CLI defaults are `--start-season 2022 --end-season 2025 --calibration-weeks 8 --seed 42`. Optional `--output-dir` is relative to or an absolute child of `config.EVALUATION_DIR`; reject escapes. Reject reversed ranges, calibration windows below one, and end seasons after the currently predeclared 2026 stream. There is no input override on the public CLI and no network/tune/refresh flag. The default run directory is `evaluation/walk_forward_{start}_{end}_cal{C}_seed{seed}`.

`run_evaluation` loads only `input_path`, `rf_params_path`, `team_path`, and these exact year-scoped candidates for requested test groups: `{vegas_dir}/{season}/week_{week}_td_odds_open.csv` and `{vegas_dir}/{season}/week_{week}_td_odds.csv`. It never considers `vegas/week_*`. It validates the raw cache, all folds, the RF JSON's exact five-key schema, and all six shared prediction key sets before creating output files.

Artifact columns are fixed:

- `folds.csv`: `football_context_provenance,evaluation_stream,test_season,test_week,fit_start,fit_end,calibration_start,calibration_end,fit_group_count,calibration_group_count,fit_rows,calibration_rows,test_rows,fit_event_rate,calibration_event_rate,test_event_rate,status`.
- `predictions.csv`: `football_context_provenance,evaluation_stream,season,week,game_id,player_id,player_display_name,team,opponent_team,position,scored_touchdown,model,variant,probability,fold,odds_covered,price_open,bookmaker_open,open_provenance`.
- `summary.csv`: `football_context_provenance,evaluation_stream,model,variant,rows,weeks,log_loss,log_loss_delta_vs_reference,brier,brier_delta_vs_reference,brier_skill,brier_reference,ece,bets,settled_bets,stake,pnl,roi,hits,hit_rate,open_coverage_count,open_coverage_total,open_coverage_rate,betting_provenance,betting_label`.

- [ ] **Step 1: Write the focused deterministic CLI test**

Build local temporary raw/RF/team/odds inputs spanning three history groups and two requested groups. Use a five-tree RF parameter file and the identity feature engineer. Monkeypatch `requests.get` and nflreadpy loader functions to raise if invoked. Run `run_evaluation` twice with the same inputs/seed into two output directories and assert all five files are byte-identical.

```python
assert set(first) == set(evaluate_wr.ARTIFACT_NAMES)
for name in evaluate_wr.ARTIFACT_NAMES:
    assert first[name].read_bytes() == second[name].read_bytes()
manifest = json.loads(first["manifest.json"].read_text())
assert manifest["football_context_provenance"] == "retrospective_finalish_game_context"
assert manifest["definitions"]["roi_used_for_selection"] is False
assert manifest["deferred"] == [
    "calibration_intercept_slope", "closing_price_clv", "maximum_drawdown",
    "pr_auc", "roc_auc", "season_position_slices", "top_k_accuracy",
    "week_cluster_bootstrap",
]
```

Pre-create `unrelated.txt`, rerun, and prove it is unchanged. Inject a failure while preparing `metrics.json`; assert no destination artifact is replaced and no temp sibling remains. Add one missing-input test that asserts no artifact is written. Run:

```bash
.venv/bin/pytest tests/test_evaluation.py -k "deterministic_cli or atomic_outputs or missing_input" -v
```

Expected: FAIL because `evaluate_wr.py` does not exist.

- [ ] **Step 2: Implement local orchestration and deterministic serialization**

Load and hash every file actually used with SHA-256. Store hashes under stable logical names such as `raw_nfl_data.csv`, `wr_te_rf_best_params.json`, `nfl_teams.csv`, and `vegas/2025/week_1_td_odds.csv`, never temporary absolute paths. The manifest records statistical arguments (not output path), seed, ordered feature list, parsed RF parameter contents/hash, input hashes, package versions, filter/fold counts, football/odds provenance, coverage, warnings, metric definitions, and the exact deferred list above; omit timestamps, hostnames, and absolute paths. `metrics.json` repeats `football_context_provenance="retrospective_finalish_game_context"` at top level so all five artifacts carry the limitation.

Build all DataFrames/dictionaries in memory and reject non-finite probabilities or JSON numeric values. Sort folds by test season/week, predictions by stream/row key/model/variant, and summary by stream/model/variant. Serialize JSON with `sort_keys=True, indent=2, allow_nan=False` and CSV with explicit columns, `index=False`, `lineterminator="\n"`, and `float_format="%.17g"`.

`write_artifacts_atomic` requires exactly the five known payload keys, writes each to a unique temporary sibling, closes every temp file, and only then replaces the five destinations in fixed `ARTIFACT_NAMES` order. On pre-replacement failure it removes only its temporary siblings. It never deletes the run directory or an unknown file.

Run the Step 1 command. Expected: PASS.

- [ ] **Step 3: Add the short README section**

Replace the stale “No walk-forward backtest yet” limitation with:

```markdown
## Walk-forward evaluation

From `wr_te/`, run the frozen-model comparison against the local cache:

```bash
python evaluate_wr.py \
  --start-season 2022 \
  --end-season 2025 \
  --calibration-weeks 8 \
  --seed 42
```

The evaluator makes no network calls and writes five deterministic artifacts
beneath `evaluation/`. Refresh `data/raw_nfl_data.csv` separately, and only
freeze an in-progress week after all games are final. Historical football
results are labeled `retrospective_finalish_game_context`; legacy odds ROI is
research-only and timestamp unsafe, not Tuesday-open performance. Odds never
change the football evaluation row set or model selection.
```

Also state in Setup that `ODDS_API_KEY` is not required for evaluation.

- [ ] **Step 4: Run the complete gate**

```bash
.venv/bin/pytest tests/test_evaluation.py -v
.venv/bin/pytest tests/ -v
.venv/bin/python evaluate_wr.py --help
git diff --check
git status --short
```

Expected: targeted and full suites PASS; help lists only the four statistical flags and optional output directory; diff check is clean; status contains only `evaluate_wr.py`, `README.md`, and Task 3 changes to `tests/test_evaluation.py` before commit. No generated evaluation directory, cache, odds file, model file, notebook, transcript, or production train/predict/ledger source is changed.

- [ ] **Step 5: Commit Task 3**

```bash
git add evaluate_wr.py tests/test_evaluation.py README.md
git commit -m "feat(wr_te): add offline evaluation artifacts and docs"
```

## MVP acceptance checks

- Every requested test week is one whole test group with exactly `C` immediately prior whole calibration groups and at least one earlier fit group.
- All six variants share the same ordered outer row keys; changing outer outcomes cannot change their probabilities.
- Model selection fields contain only pooled log loss, Brier, Brier skill, and paired deltas; ROI is downstream context only.
- Odds absence or mismatch cannot remove a football row. Safe tagged opens outrank legacy; unsafe tagged snapshots fall back to visibly timestamp-unsafe legacy data.
- The one-unit top-five payout test matches ledger American-odds conventions.
- The CLI performs no network calls, reads no root-level odds file, writes exactly five deterministic artifacts, preserves unrelated files, and publishes nothing before the run validates.
- Production training, prediction, odds matching, and ledger behavior remain unchanged.
