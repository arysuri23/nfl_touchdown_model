# WR/TE Weekly Walk-Forward Evaluation Design

**Status:** Proposed for user review  
**Date:** 2026-09-02  
**Scope:** Retrospective and prospective evaluation of the existing WR/TE anytime-touchdown probability model. This design does not change production training or prediction behavior.

## Objective

Add a lean, reproducible weekly walk-forward evaluator that answers one primary question: do the current RandomForest probabilities beat simple base rates and a fixed L2 logistic regression on out-of-sample log loss and Brier score?

ROI is a downstream diagnostic, never a model-selection objective. Historical market results must expose odds coverage and timestamp limitations. Timestamped 2026 weekly snapshots remain the genuinely untouched evaluation stream.

## Existing Interfaces and Data Inventory

The evaluator reuses the repository's current interfaces instead of creating a parallel feature pipeline.

| Existing asset | What it supports | Limitation relevant to evaluation |
|---|---|---|
| `wr_te/data_collection.py:get_all_historic_data` | Builds one regular-season player-week row for WR/TE players who have weekly stats and a matched snap-count row; includes `player_id`, `season`, `week`, `game_id`, `position`, outcome, raw player/team/defense fields, depth rank, and game context. | The build performs live/cached nflverse reads. Evaluation must not call it by default. Inactive/non-participating candidates are absent, so the retrospective universe is “players who produced a weekly row,” not a true pregame active-roster universe. |
| `wr_te/data/raw_nfl_data.csv` | Default local evaluation input produced by `train_wr.py`. The inspected cache has 21,618 unique player-week rows for 2020-2025, all with `game_id`; 2020 has weeks 1-17 and 2021-2025 have weeks 1-18. | Gitignored/generated rather than a versioned source artifact. The manifest must hash it. It contains raw fields; lagged `avg_*` features are created later. |
| `wr_te/features.py` | Canonical 23-column `WR_TE_FEATURES` and raw `PLAYER_EWM_STATS`. | Must remain the single feature-list source. |
| `wr_te/train_wr.py:feature_engineering` | Sorts by player/week and applies `shift(1)` before player and opponent EWM features. | It fills missing values with zero. Same-week `spread_line`, `implied_total`, and `depth_chart_rank` are pregame-context fields rather than lagged outcome features. |
| `wr_te/models/wr_te_rf_best_params.json` | Frozen definition of the current RF: 300 trees, depth 5, min split 10, min leaf 1, `max_features=log2`. | The file may reflect prior researcher choices. Retrospective results are an audit of this now-frozen configuration, not a pristine historical model-development simulation. |
| nflverse schedule fields used by `schedule_to_team_lines` | `spread_line`, `total_line`, and derived `implied_total` for current RF parity. | These fields are closing/final-ish and lack the Tuesday decision-time provenance required to claim an opening-line backtest. |
| `wr_te/vegas/{season}/week_{week}_td_odds.csv` | Legacy DraftKings ATD prices: all weeks in 2023-2024 and weeks 1-12 in 2025; columns include player, matchup, game date, price, season, week, and book. | No reliable quote retrieval timestamp and no distinct opening/closing observation. Research-only ROI; no CLV. |
| `wr_te/vegas/{season}/week_{week}_td_odds_{open|close}.csv` | New snapshot schema with `game_id`, quote/book timestamps, matchup, price, season/week, tag, and `fetched_at`. One tracked 2025 week-15 open-shaped file currently exists. | That tracked file has an empty `fetched_at`, and no paired close file exists. It is not fully timestamp-safe. |
| `wr_te/odds_match.py` | Team-aware normalized-name match and best available American price selection. | Legacy records still depend on names because they have no stable player ID. |
| `wr_te/ledger.py` | Existing American-odds probability/payout and CLV conventions; funded strategy is `top5_prob`. | The ledger is prospective operational state, not a retrospective experiment tracker. |
| `wr_te/evaluate.ipynb` | Prior exploratory ROI work. | Non-deterministic exploratory artifact; not an input or implementation target. |

### Additional data worth obtaining now

The first two items materially improve the credibility of the primary football evaluation; the third improves only downstream betting evaluation.

1. **Decision-time candidate snapshots:** `season`, `week`, `game_id`, stable `player_id`, team, position, roster status, injury/game-status designation, depth rank, `snapshot_at`, source, and source version. This closes the current played-row survivorship gap.
2. **Decision-time game lines:** book/source, event ID, spread, total, quote timestamp, source update timestamp, kickoff timestamp, and an explicit snapshot label such as Tuesday open or Sunday pre-kick. This replaces final-ish nflverse context for deploy-time fidelity.
3. **Timestamped ATD open and close quotes:** stable event and player identifiers when available, player/team, bookmaker, market/outcome, American price, quote/update/fetch timestamps, kickoff, in-play flag, and source file/version. Paired open/close records enable defensible ROI and CLV.

Multi-book consensus, de-vigging, full line-movement histories, paid providers, weather enrichments, and advanced injury/news features can wait. They are not needed to decide whether the football model beats simple baselines.

## Evaluation Population

The row key is `(season, week, game_id, player_id)`. The default source is `config.DATA_DIR / "raw_nfl_data.csv"`; evaluation is cached/offline and makes no nflreadpy or HTTP calls.

An eligible retrospective row must:

- be a regular-season WR or TE row with `week <= 18`;
- have non-null `season`, `week`, `game_id`, `player_id`, `position`, and binary `scored_touchdown`;
- be unique on the row key; and
- yield finite values for every `WR_TE_FEATURES` column after the existing `feature_engineering` transform.

Eligibility is computed once, before fitting any model. For each outer test week, every model variant receives exactly the same ordered row keys. A missing probability is a fold failure, never permission to drop a row for one model.

Here, a complete `(season, week)` group means the complete set of eligible rows present in the frozen input cache for that group. The entire group is assigned to one partition. For an in-progress season, a week cannot enter evaluation until all games are settled and its candidate/input snapshot is frozen; partial prospective weeks remain pending.

## Walk-Forward Protocol

Order distinct group keys lexicographically by `(season, week)`. For every eligible outer test group `T` within the requested season range:

1. The **test partition** is exactly group `T`.
2. The **calibration partition** is the immediately preceding `C` complete groups, where `C = --calibration-weeks`.
3. The **base-fit partition** is every complete group before that calibration window.

The first requested test group must have at least one earlier base-fit group and exactly `C` earlier calibration groups. Otherwise the CLI fails with the earliest impossible group and the available history; it does not silently shorten the calibration window or skip requested weeks.

For every fold, assert:

- `max(fit_group) < min(calibration_group) <= max(calibration_group) < test_group`;
- fit, calibration, and test row keys and group keys are pairwise disjoint;
- the calibration partition contains exactly `C` whole groups and test exactly one whole group;
- a `game_id` maps to one `(season, week)` and occurs in only one partition;
- every outcome-derived player, team, and opponent feature is strictly lagged through `shift(1)`; and
- the prediction key set is identical for all compared variants.

The outer test week is never used for hyperparameter selection, calibration, thresholds, feature selection, or eligibility rules. All choices below are fixed before the loop.

## Compared Models

| Model ID | Base fit | Calibration fit | Output variants |
|---|---|---|---|
| `base_rate_overall` | Mean touchdown rate over all base-fit rows. | None. | `raw` only. |
| `base_rate_position` | Separate WR and TE touchdown rates over base-fit rows. Both positions must exist; otherwise the fold fails. | None. | `raw` only. |
| `logistic_l2` | `StandardScaler` plus `LogisticRegression` with fixed L2 settings (`C=1.0`, `solver=lbfgs`, `max_iter=2000`) on `WR_TE_FEATURES`. | A separate Platt logistic regression fitted only to the calibration-window raw probabilities and labels. | `raw`, `platt`. |
| `random_forest_current` | A new forest per fold using the exact parameters in `wr_te_rf_best_params.json`, `WR_TE_FEATURES`, and the run seed. No search. | The same temporal Platt procedure on calibration-window raw probabilities and labels. OOB predictions are not used. | `raw`, `platt`. |

Platt fitting follows the repository's current convention: one-feature logistic regression on clipped raw probabilities. The calibration window must contain both outcome classes. Calibration is applied only after the base model is frozen. Base rates are not recalibrated because calibration would add a second intercept-only estimate without a meaningful new ranking.

All stochastic components use `--seed` (default 42). Rows, groups, model IDs, and output columns are sorted before writing. The manifest records the seed, RF parameter file contents/hash, feature list, input hashes, and relevant package versions.

## Metrics and Selection Rule

### Primary

Model selection uses pooled out-of-sample rows across all outer test weeks:

- **Log loss**, with probabilities clipped only for numerical evaluation.
- **Brier score**.
- **Brier skill score:** `1 - Brier(model) / Brier(base_rate_overall)` on the exact same slice and row keys. The reference is named in every result.

Lower log loss and Brier are better; higher Brier skill is better. ROI cannot break a tie or select a model. Summary tables also show paired model-minus-reference differences.

### Secondary

- PR-AUC and ROC-AUC;
- weighted fixed-bin ECE over `[0.0, 0.1), ..., [0.9, 1.0]`, calculated as `sum(bin_count / N * abs(bin_mean_probability - bin_event_rate))`;
- all ten calibration-bin counts, means, event rates, and gaps, including zero-count bins;
- calibration intercept and slope from an unpenalized logistic regression of outcome on clipped probability log-odds (ideal `0` and `1`); and
- weekly top-k accuracy for fixed `k = 3, 5, 10`, defined as hits divided by `min(k, eligible_rows_in_week)`, then averaged equally across outer weeks. Ties break by `player_id`.

Undefined AUC or calibration-regression values are written as JSON `null` with a reason, never coerced to zero.

### Slices and uncertainty

Report pooled overall metrics plus season, position, and season-by-position slices. Slices inherit the same model key set and never refit models.

A deterministic 1,000-replicate paired cluster bootstrap samples outer `(season, week)` groups with replacement, retaining all rows from each sampled week. It produces 95% percentile intervals for log loss, Brier, paired primary-metric differences, and betting ROI. If fewer than eight evaluated weeks are available, intervals are `null` with reason `insufficient_week_clusters`.

## Betting Evaluation

Betting is computed only after football-model predictions are complete and only on outer-test rows with matched player ATD odds. It does not alter model eligibility or primary metrics.

Historical odds precedence is:

1. a year-scoped tagged `open` snapshot when its non-empty timestamps demonstrate it was captured pre-kickoff;
2. otherwise the year-scoped legacy file, labeled `timestamp_unsafe_legacy`.

Root-level `wr_te/vegas/week_*` files are excluded because their season is not explicit. Missing odds are retained in `predictions.csv` as uncovered rows. Matching reuses the team-aware logic in `odds_match.py`; provenance and duplicate-resolution counts are recorded.

The sole initial simulated strategy is the existing `top5_prob`: for each non-constant logistic/RF variant and week, rank only odds-covered rows by probability, break ties by `player_id`, and stake one unit on up to five players at the selected opening price. Baseline models are omitted from betting because their broad ties make top-five selection arbitrary. Raw and calibrated variants are evaluated separately; the evaluator does not assume the fitted calibration slope preserves their ranking.

For each model variant report:

- bets, settled bets, stake, PnL, ROI, hits, and hit rate;
- chronological maximum drawdown from the cumulative one-unit PnL path, including starting equity zero;
- open-price coverage: covered eligible rows / all eligible rows, with counts overall and by week;
- close-price coverage and mean CLV when paired closing quotes exist, using the ledger convention `implied_close - implied_open`; and
- the week-clustered 95% ROI interval described above.

Legacy ROI is labeled **research-only, timestamp unsafe**. It must not be described as Tuesday-open performance. If no safe paired closing data exists, CLV remains `null` with coverage zero rather than being inferred from the legacy price.

Current nflverse game-line fields may remain as inputs solely to reproduce the current RF/logistic feature contract. Every historical artifact must label the resulting football evaluation `retrospective_finalish_game_context`; it is not a deploy-time opening-line fidelity claim. Once timestamped decision-time game lines exist, the same pipeline can produce a separate `decision_time_safe` run without changing model selection rules.

## Prospective 2026 Stream

The retrospective run freezes model definitions and acceptance rules before 2026 outcomes are incorporated. Weekly 2026 predictions, candidate snapshots, open/close odds, and later outcomes are appended only after each week is finalized. They are never used to revise thresholds, RF parameters, calibration-window length, features, or row eligibility during this evaluation phase.

Prospective results are reported separately from retrospective results until a predeclared aggregation date. This is the only stream that can support a genuinely untouched claim, provided snapshots have non-empty fetch/update timestamps and precede kickoff.

## CLI and Artifacts

Add `config.EVALUATION_DIR = config.BASE_DIR / "evaluation"`. The evaluator runs from `wr_te/`, matching existing scripts:

```bash
python evaluate_wr.py \
  --start-season 2022 \
  --end-season 2025 \
  --calibration-weeks 8 \
  --seed 42
```

Optional `--output-dir` must resolve beneath `config.EVALUATION_DIR`. Its deterministic default is `evaluation/walk_forward_{start}_{end}_cal{C}_seed{seed}`. Rerunning replaces only the six known files in that exact run directory; it never deletes unrelated files. There is no network-refresh flag. Users refresh `raw_nfl_data.csv` through the existing training/data workflow as a separate explicit action.

The output stays file-based:

| Artifact | Content |
|---|---|
| `manifest.json` | Arguments, definitions, seeds, versions, input/parameter hashes, feature list, filter counts, fold count, provenance classifications, odds coverage, and warnings. |
| `folds.csv` | One row per test week: fit/calibration/test boundaries, group and row counts, event rates, and status. |
| `predictions.csv` | Long form: row key, player/team/position/outcome, model, variant, probability, fold, odds-covered flag, selected prices/book, and provenance status. |
| `metrics.json` | Full overall/fold/slice metrics, ten-bin calibration tables, uncertainty intervals, metric definitions, and betting metrics. |
| `summary.csv` | Flat overall row per model/variant with primary/secondary metrics, primary deltas, interval bounds, and betting columns where meaningful. |
| `slices.csv` | Flat season, position, and season-position model metrics plus row/week counts. |

No dashboard, database, experiment server, workflow engine, or notebook is added.

## Minimal File Boundaries

| File | Change | Responsibility |
|---|---|---|
| `wr_te/config.py` | Modify | Add only `EVALUATION_DIR`. |
| `wr_te/evaluation.py` | Create | Pure fold construction/assertions, fixed model factories, temporal calibration, metrics, local odds normalization, betting calculations, and artifact-ready records. No I/O except functions explicitly passed paths/dataframes. |
| `wr_te/evaluate_wr.py` | Create | Argument parsing, local input loading, orchestration, provenance classification, and six artifact writes. |
| `wr_te/tests/test_evaluation.py` | Create | Targeted protocol, metric, odds, payout, and determinism tests. |
| `wr_te/README.md` | Modify | Add one short evaluation command and explain research-only historical odds labels. |

Existing `features.py`, `train_wr.feature_engineering`, `odds_match.py`, and `ledger` odds conversion conventions are reused. Production `train_wr.py`, `predict_wr.py`, and ledger behavior do not change.

## Failure Behavior

Fail before fitting and write no partial “successful” run when the input file is missing, required columns/features are absent, keys are duplicated, fold history is insufficient, a partition assertion fails, calibration lacks both classes, a compared model returns a different key set, or any probability is non-finite/outside `[0,1]`.

Missing odds, unmatched odds, or absent close prices do not fail football evaluation. They reduce explicitly reported coverage and produce null betting/CLV values as appropriate. Output files are written to temporary siblings and renamed only after the entire run validates.

## Testing Strategy

Keep tests synthetic and offline in one new test module:

1. **Fold integrity:** shuffled rows spanning a season boundary prove whole-week assignment, immediately preceding contiguous calibration groups, one-week tests, strict ordering, insufficient-history failure, and game isolation.
2. **Same universe and no test tuning:** spy estimators prove every variant predicts identical test keys, calibrators see only calibration keys, and factories/search are never invoked with outer-test rows.
3. **Metrics:** hand-computable probabilities verify log loss, Brier/Brier skill, weighted ECE with empty-bin counts, top-k tie handling, and null behavior for single-class AUC/calibration fits.
4. **Betting:** mixed covered/uncovered rows and open/close fixtures verify team-aware matching, source precedence, coverage, American-odds PnL, ROI, maximum drawdown, CLV, and timestamp-unsafe labels.
5. **Determinism:** two tiny end-to-end runs with the same seed and inputs produce identical manifest, fold, prediction, summary, slice, metric, and bootstrap values. The manifest therefore contains source hashes rather than a wall-clock creation timestamp.

The existing full `wr_te/tests/` suite remains the regression check. No network or full historical retraining is required in unit tests.

## Acceptance Criteria

1. The documented CLI completes using only local cached inputs and writes the six specified artifacts under `config.EVALUATION_DIR`.
2. Every requested test week has one recorded fold with all eligible week rows in test, exactly the configured contiguous calibration groups, strictly earlier expanding fit data, and passing game/key isolation assertions.
3. `base_rate_overall`, `base_rate_position`, `logistic_l2` raw/Platt, and current RF raw/Platt predictions share exactly the same outer row keys.
4. Summary model selection is based on pooled out-of-sample log loss and Brier/Brier skill; ROI is displayed only as downstream context.
5. Secondary metrics include PR-AUC, ROC-AUC, weighted ten-bin ECE with counts, calibration slope/intercept, and top-3/5/10 accuracy with documented null handling.
6. Betting output includes bets, stake, PnL, ROI, hit rate, max drawdown, odds coverage, provenance, CLV/close coverage when available, and week-clustered ROI uncertainty.
7. Historical legacy odds and final-ish nflverse context are visibly labeled timestamp unsafe/research-only in the manifest and flat outputs; no artifact claims Tuesday-open fidelity.
8. No outer-test tuning, model-dependent row dropping, partial-week split, network call, RB row, dashboard, database, paid-provider integration, or automatic wager occurs.
9. Fixed seeds and sorted output make repeated runs numerically deterministic, and all targeted plus existing tests pass.
10. Prospective 2026 observations remain separate and untouched until their pre-recorded week is finalized.

## Decisions and Alternatives

| Decision | Chosen approach | Alternative considered | Reason |
|---|---|---|---|
| Season holdouts vs weekly origins | Strict weekly rolling origins with expanding fit, contiguous temporal calibration, and one outer test week. | One full-season holdout. | Weekly origins match production cadence, expose drift, and yield more independent time clusters. A season holdout is cheaper but too coarse for weekly calibration and operational analysis. |
| OOB vs temporal calibration | Disjoint immediately preceding temporal window. | Current RF OOB calibration. | OOB rows are not a decision-time holdout and mix eras used to fit the forest. A temporal window measures the deployable calibration step and keeps test data untouched. |
| CSV/JSON vs heavier tracking | Six deterministic CSV/JSON artifacts in a config-owned directory. | MLflow, database, dashboard, or orchestration framework. | Current scale needs inspectable, diffable outputs, not infrastructure. |
| Market prices as baseline vs input | Player ATD prices stay outside football-model fitting/selection and are used only for coverage-limited downstream betting. Existing game spread/total remain inputs for current-model parity with a final-ish provenance label. | Treat player implied probability as a primary baseline or model feature. | Player prices change the row universe and lack reliable historical decision-time timestamps. Making them primary would confound football signal with coverage and falsely imply opening-line fidelity. |

## Non-Goals

- RB expansion.
- Paid data-provider integration.
- Dashboards, databases, or orchestration frameworks.
- Automatic wager placement or bankroll management.
- Feature experimentation, feature removal, hyperparameter searches, threshold tuning, or strategy optimization in this first task.
- Changes to the production weekly train/predict/ledger workflow.

## Open Data Requests

No open design choice blocks implementation. The highest-value follow-up is access to historical decision-time game lines and candidate/roster snapshots for the intended weekly decision hour. Timestamped paired ATD open/close quotes are the next priority if ROI and CLV are expected to support more than a research-only claim. Until those fields exist, the evaluator will run but will preserve the explicit retrospective and timestamp-unsafe labels above.
