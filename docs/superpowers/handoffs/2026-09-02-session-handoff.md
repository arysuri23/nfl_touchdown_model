# NFL Touchdown Model — Session Handoff

Date: 2026-09-02  
Active worktree: `.claude/worktrees/2026-season-wr-te`  
Branch: `worktree-2026-season-wr-te`  
Current commit: `e24ff11`

## What happened, in order

1. The session began by asking Codex to continue work previously done by Claude/Fable, with Sol responsible for planning/review and Luna responsible for implementation. The Claude transcript was inspected only by a Luna subagent; it was empty (0 bytes), so the reliable handoff came from the repository’s SDD ledger and reports.

2. The existing Phase 0 work was found in the feature worktree. Its goal was to make the WR/TE anytime-touchdown RandomForest pipeline 2026-ready: centralized configuration, leakage-safe features, schedule-derived lines, season-agnostic depth charts, calibration, live odds, roster filtering, ledger settlement/CLV/ROI, and a weekly runbook. The initial branch ended at commit `778166f` with 68 tests passing.

3. Sol’s whole-branch review found the pipeline was not ready for 2026. Main findings were: current-season history was never collected for later 2026 weeks; Monday-night games could fall outside the odds window; partial PBP could settle losses prematurely; calibration reporting used raw instead of calibrated probabilities; missing game lines became zero-valued features; feature-importance artifacts still contained `redzone_td_rate`; docs disagreed about betting strategies; and reports merged equal week numbers across seasons. It also identified an exposed historical Odds API credential.

4. The user approved a surgical fix wave and explicitly said not to use LID or over-engineer TDD. Luna implemented the fixes, adding focused regression tests and reusing existing patterns. Sol reviewed the wave and found two additional issues: in-progress schedules with scores could still settle, and completed schedule games without PBP could settle as losses. Those were fixed, along with bye-week roster handling and genuine artifact regeneration.

5. A second Sol review found an ID-namespace bug: The Odds API event ID differed from the nflverse game ID used by settlement. Luna fixed canonical schedule game-ID propagation and changed future feature-importance generation to use the deployed final forest. Sol approved the fixes.

6. The complete fix wave was committed as `80defd2` (`fix(wr_te): harden 2026 weekly pipeline`). Fresh verification after the commit:

   - `88 passed`
   - imports for `data_collection`, `train_wr`, `predict_wr`, `fetch_live_odds`, and `ledger` passed
   - `git diff --check` passed
   - feature-importance artifact has 23 features matching `WR_TE_FEATURES`
   - canonical provider/nflverse game-ID settlement regression passed

7. Sol then researched broader model improvements and evaluation design. The highest-value recommendation was evaluation infrastructure before adding features. Important limitations found: row-level `TimeSeriesSplit` can split an NFL week; OOB calibration is not temporal; historical nflverse lines are closing/final-ish rather than decision-time open lines; the retrospective betting notebook is not a controlled backtest; and current artifacts do not establish live edge.

8. The user approved the recommended lean evaluation direction: a weekly grouped walk-forward evaluator with log loss/Brier as primary metrics and ROI downstream. Sol wrote and self-reviewed the design spec, committed as `e24ff11`:

   [2026-09-02-wr-te-evaluation-design.md](../specs/2026-09-02-wr-te-evaluation-design.md)

## Current evaluation design

- Keep complete `(season, week)` groups intact.
- Use expanding base-fit history, a disjoint contiguous calibration window, and exactly one untouched test week.
- Compare overall/position base-rate baselines, fixed L2 logistic regression, and the current RandomForest; include raw and temporal-Platt variants where applicable.
- Select on pooled out-of-sample log loss and Brier/Brier skill. Also report PR-AUC, ROC-AUC, weighted ECE with bin counts, calibration slope/intercept, top-k accuracy, coverage, and downstream betting metrics.
- Keep artifacts file-based and deterministic: manifest, folds, predictions, metrics, summary, and slices CSV/JSON.
- Historical odds without trustworthy timestamps are research-only and must be labeled; prospective 2026 snapshots are the genuinely untouched stream.
- No RB expansion, paid-provider integration, dashboard, database, automatic wagering, or feature experimentation in the first evaluator task.

## Important security note

The current tree/task report was sanitized, but the exposed credential remains in Git history unless history is rewritten. Rotate/revoke the key before using it again. No history rewrite or credential rotation was performed in this session.

## Resume from here

1. Review/approve the committed evaluation spec above.
2. After approval, have Sol produce the implementation plan.
3. Have Luna implement the evaluator in the feature worktree, with focused offline tests and no broad framework.
4. Have Sol review the implementation and run fresh verification.
5. Before claiming an edge, obtain decision-time candidate/roster snapshots and timestamped game lines; timestamped ATD open/close quotes are the next priority for defensible ROI/CLV.

## Worktree notes

- The main worktree remains on `main` at `4d0fd5d`; it was not modified by the implementation.
- The feature worktree is clean at `e24ff11`.
- Root-only untracked items (`.claude/`, `out/`, and the empty `wr_te/claude_transcript.md`) were not included in feature commits.
- Existing deferred scope remains deferred: inactive-player label rows, de-vigging, multi-book aggregation, walk-forward backtesting as a production feature, and RB support. The new evaluator is the next planned addition, not yet implemented.
