# NFL Touchdown Model — Session Handoff

Date: 2026-09-02 onward
Active worktree: `.claude/worktrees/2026-season-wr-te`  
Branch/current commit: `worktree-2026-season-wr-te` / `f320303`

## Session outcome

The lean evaluation plan was committed as `ac2500c`. Tasks 1–3 were implemented and reviewed in the feature worktree:

- Task 1: `9f55f67`, `66aeb01`, `8714475` — grouped walk-forward folds, fixed six-variant predictions, temporal calibration, shared-key and leakage validation. Review findings were closed.
- Task 2: `cc7a83d`, `f8404a5` — pooled log loss/Brier metrics, provenance-aware odds matching, and research-only ROI. Review findings closed unsafe tagged-odds and variant-scope gaps.
- Task 3: `310ab55`, `1bba56c`, `40e9779` — offline CLI, deterministic five-artifact serialization, transactional publication rollback, real-cache feature-fill fix, and explicit mixed-odds provenance. Review findings were closed.

The latest evaluator suite had 127 passing tests before the roster alias fix and 131 afterward. The real-cache command for 2022–2025 completed successfully with 72 folds, 87,186 prediction rows, all six model variants, and exactly five artifacts (`manifest.json`, `folds.csv`, `predictions.csv`, `metrics.json`, `summary.csv`) under `wr_te/evaluation/final-verification`. That generated directory remains untracked and is not part of the commits.

Raw-RF production and calibration review landed in `7c29482`. Direct calibration evidence is diagnostic only: raw deployed-forest OOB Brier `0.1277852259`, log loss `0.4103006952`, ECE `0.0201373439`; Platt OOB Brier `0.1283551115`, log loss `0.4128795751`, ECE `0.0236240502`. Empty calibration reports match evaluator undefined values (`NaN`, `NaN`, `None`) and fixed ten-bin weighted ECE semantics (`211dd60`, `6fca23f`).

Cache-only retraining landed in `5cf2a34`, with transactional/validation/provenance fixes in `6912e8b` and stable logical paths/role in `5b1a6a2`. Operational artifacts were committed in `799de6e`: 21,618 rows through 2025, 3,753 latest-season calibration rows, production `random_forest_current/raw`, and calibrator role `diagnostic_only`. Cache SHA-256: `31edb3a2a5fc9d2c5af2056f6f8ca9d085bdb6432129f39ffabb84afbb2743cc`; raw forest: `5a32c06f035f0898ea5600e0fe38953123d3f50670e8ba1651c5addffbcacef6`; calibrator: `d837a2d2592af4c2554bcd112bdc8f43e61da3ac0a58c5811d6a5178eb34022b`; feature importance: `0bd632d735b5128da2695cab18a70f83a93791a5d1aabb9fbfd6acdacf9496e0`. Old model recovery path: `/private/tmp/wrte-model-backup.9tSDK0`.

Phase A collector audit and publication hardening were approved by Sol. `5286454` adds numeric-only final-frame filling in `data_collection.get_all_historic_data` (numeric gaps become zero while string missingness remains missing) and defers network cache writes until validation/training complete. `533adf6` refreshes cache source provenance. `f320303` adds main-orchestration failure coverage and all-position five-target transaction fault coverage. The network path now retains the complete collected source, including active 2026 rows, while the validated prior-season view alone reaches model fitting; cache mode remains read-only. The focused/full suite is 148 passing tests. The current branch is `worktree-2026-season-wr-te` at `f320303`.

## Current evaluator behavior

The evaluator reads only local raw/RF/team caches and requested year-scoped odds files. It makes no network calls and never reads root-level `vegas/week_*` files. Timestamp-safe tagged opens outrank legacy odds; unsafe or incomplete tagged snapshots fall back to legacy odds labeled `timestamp_unsafe_legacy`, with betting labeled research-only/timestamp-unsafe. Mixed aggregates retain the unsafe label, and the manifest records provenance per season/week.

The real-cache smoke exposed 585 nullable/string `surface` values. `train_wr.feature_engineering` now fills only numeric columns, preserving nonnumeric missing values while still filling numeric engineered gaps with zero.

The MVP’s five artifacts defer heavy diagnostics: `calibration_intercept_slope`, `closing_price_clv`, `maximum_drawdown`, `pr_auc`, `roc_auc`, `season_position_slices`, `top_k_accuracy`, and `week_cluster_bootstrap`. They must remain explicitly deferred rather than emitted as misleading zeroes. Model selection remains pooled out-of-sample log loss/Brier/Brier skill and paired deltas; ROI is downstream context only.

## Data availability and xTD-share spike

A successful approved-network preflight against real `nflreadpy` 0.1.5 shows that the Week 1 structural inputs are available: `load_schedules([2026])` returned 272 rows, including 16 regular-season Week 1 rows with complete spread/total fields; `load_rosters([2026])` returned 3,118 rows, and the actual `load_week_roster` fallback produced 302 WR/TE candidates with no null player key, team, or position; the depth-chart helper produced 358 Week 1 rows with no nulls; and the week-lines helper produced 32 team rows with no nulls. `load_rosters_weekly([2026])` remains unsupported before 2026 data publication. Pre-season `load_ff_opportunity(seasons=[2026], stat_type="weekly")` and player stats remain unavailable for postgame outcomes/features. `ODDS_API_KEY` is unset; the only missing prospective Week 1 input is the ATD open-odds snapshot.

The discarded xTD-share experiment used only cached `rec_touchdown_exp` / `rec_touchdown_exp_team`: validated finite non-negative ratios, derived a lagged cross-season player EWM share (`alpha=.3`), and added it as the 24th WR/TE feature without collector or tuning changes. The exact 2022–2025, calibration-8, seed-42 run matched the control’s 72 folds and ordered outer prediction keys. Raw RF log loss was `0.4160999359` control vs `0.4161845481` challenger (delta `+0.0000846122`); Brier was `0.1293563271` vs `0.1293811338` (delta `+0.0000248068`). Under the frozen rule (log-loss improvement ≥ `0.001` and no Brier worsening), the result is **DISCARD**.

The experiment ran in `/private/tmp/xtd-share-experiment`; its discarded feature/test edits remain uncommitted there and have no branch commit. Sol’s next recommendation is to freeze the current 23-feature raw RF, then start prospective 2026 Week 1 snapshots for candidate roster, depth/status, game spread/total, and ATD open odds; after games finalize, retry 2026 player stats and fantasy opportunity through `nflreadpy`.

At `2026-09-04T03:43:52Z`, live `nflreadpy` Week 1 assembly produced 302 active WR/TE candidates across 32 teams, including 10 ARI candidates after the roster alias fix, 358 depth-chart rows, and 32 team-line rows. The assembled prediction frame had finite values for the exact 23 production features, and raw-RF probabilities were `allclose` to direct model output. This is structural preflight evidence only; no prediction or ledger records are authorized until opening ATD odds exist.

Phase B remains gated on a fresh post-Week 1 real-`nflreadpy` spike before any network refresh is automated. That spike must record loader availability and returned coverage/schema, confirm active candidates are available through the requested week, verify regular-season (`REG`, weeks 1–18) filters and join cardinality, and measure snap-count name-match coverage. Until then, network refresh remains a manual dry-run.

## Operating rules and next steps

Any question about future data availability must first use a real `nflreadpy` spike and record the package version, loader/API call, returned schema, coverage, and missingness before planning integration. Freeze an in-progress week only after all games are final; refresh the raw cache separately. Review the five smoke artifacts and decide whether timestamped decision-time odds are available before making any edge claim.

The main worktree remains unchanged on `main` at `4d0fd5d`. The feature worktree is current through `f320303` plus intentionally untracked smoke/evaluation output. The only remaining pregame blocker is `ODDS_API_KEY`/the 2026 Week 1 ATD open-odds snapshot; once available, run the full prediction and ledger flow. Existing exposed historical credentials remain a separate security concern and should be revoked/rotated before reuse; no history rewrite was performed.
