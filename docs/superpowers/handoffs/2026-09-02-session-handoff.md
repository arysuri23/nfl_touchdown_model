# NFL Touchdown Model — Session Handoff

Date: 2026-09-02 onward
Active worktree: `.claude/worktrees/2026-season-wr-te`  
Branch/current commit: `worktree-2026-season-wr-te` / `40e9779`

## Session outcome

The lean evaluation plan was committed as `ac2500c`. Tasks 1–3 were implemented and reviewed in the feature worktree:

- Task 1: `9f55f67`, `66aeb01`, `8714475` — grouped walk-forward folds, fixed six-variant predictions, temporal calibration, shared-key and leakage validation. Review findings were closed.
- Task 2: `cc7a83d`, `f8404a5` — pooled log loss/Brier metrics, provenance-aware odds matching, and research-only ROI. Review findings closed unsafe tagged-odds and variant-scope gaps.
- Task 3: `310ab55`, `1bba56c`, `40e9779` — offline CLI, deterministic five-artifact serialization, transactional publication rollback, real-cache feature-fill fix, and explicit mixed-odds provenance. Review findings were closed.

The latest full suite has 112 passing tests. The real-cache command for 2022–2025 completed successfully with 72 folds, 87,186 prediction rows, all six model variants, and exactly five artifacts (`manifest.json`, `folds.csv`, `predictions.csv`, `metrics.json`, `summary.csv`) under `wr_te/evaluation/final-verification`. That generated directory remains untracked and is not part of the commits.

## Current evaluator behavior

The evaluator reads only local raw/RF/team caches and requested year-scoped odds files. It makes no network calls and never reads root-level `vegas/week_*` files. Timestamp-safe tagged opens outrank legacy odds; unsafe or incomplete tagged snapshots fall back to legacy odds labeled `timestamp_unsafe_legacy`, with betting labeled research-only/timestamp-unsafe. Mixed aggregates retain the unsafe label, and the manifest records provenance per season/week.

The real-cache smoke exposed 585 nullable/string `surface` values. `train_wr.feature_engineering` now fills only numeric columns, preserving nonnumeric missing values while still filling numeric engineered gaps with zero.

The MVP’s five artifacts defer heavy diagnostics: `calibration_intercept_slope`, `closing_price_clv`, `maximum_drawdown`, `pr_auc`, `roc_auc`, `season_position_slices`, `top_k_accuracy`, and `week_cluster_bootstrap`. They must remain explicitly deferred rather than emitted as misleading zeroes. Model selection remains pooled out-of-sample log loss/Brier/Brier skill and paired deltas; ROI is downstream context only.

## Operating rules and next steps

Any question about future data availability must first use a real `nflreadpy` spike and record the package version, loader/API call, returned schema, coverage, and missingness before planning integration. Freeze an in-progress week only after all games are final; refresh the raw cache separately. Review the five smoke artifacts and decide whether timestamped decision-time odds are available before making any edge claim.

The main worktree remains unchanged on `main` at `4d0fd5d`. The feature worktree contains the committed implementation above plus the intentionally untracked smoke output. Existing exposed historical credentials remain a separate security concern and should be revoked/rotated before reuse; no history rewrite was performed.
