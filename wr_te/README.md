# WR/TE Anytime-Touchdown Model

Predicts anytime-touchdown probability for active WR/TE players for a given
NFL season/week, joins it against sportsbook odds, and tracks bets in a
ledger with closing-line value (CLV).

## Setup

```bash
cd wr_te
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Required only for fetch_live_odds.py (The Odds API). Not needed to train,
# predict, or run the ledger against an odds snapshot that already exists.
export ODDS_API_KEY=...
```

Run tests with `.venv/bin/pytest tests/ -v`.

## Weekly runbook

All commands below assume `WRTE_SEASON`/`WRTE_WEEK` are set (default to the
current `config.SEASON`/`config.WEEK`; see *Overriding season/week*) and are
run from `wr_te/`.

1. **Tue/Wed: fetch opening odds** (≈17 credits)
   ```bash
   python fetch_live_odds.py --tag open
   ```
2. **Train** (first week of the season, and every 4 weeks thereafter; add
   `--tune` only when changing the feature list -- it re-runs
   `RandomizedSearchCV` and overwrites the saved hyperparameters)
   ```bash
   python train_wr.py
   ```
3. **Predict**
   ```bash
   python predict_wr.py
   ```
4. **Record picks to the ledger** -- `top5_prob` is the primary funded
   strategy. `top5_edge_le400` is an optional zero-stake benchmark (tracked
   for its own record, not actually bet). The ledger does not apply a separate
   edge-threshold rule.
   ```bash
   python ledger.py record --strategy top5_prob
   python ledger.py record --strategy top5_edge_le400 --stake 0
   ```
5. **Sun, ~1h before the early kickoff: closing odds**
   ```bash
   python fetch_live_odds.py --tag close
   python ledger.py close
   ```
6. **Tue: settle and report**
   ```bash
   python ledger.py settle
   python ledger.py report
   ```

### Periodic training refresh

Each scheduled training refresh collects all available completed rows from
`config.DATA_SEASONS` (the prior training seasons plus the active prediction
season), writes the feature cache, and fits the deployed forest only on rows
from the configured prior-season `TRAIN_SEASONS` cut. It then recalibrates that
deployed forest on its OOB predictions and refreshes the model, calibrator, and
feature-importance artifacts. The refresh cadence does not change the ledger
strategies or introduce an implicit edge threshold.

### Overriding season/week

Every script reads `config.SEASON`/`config.WEEK`, which default to 2026/1 and
are overridden via environment variables:

```bash
WRTE_SEASON=2025 WRTE_WEEK=15 python predict_wr.py
```

## What changed from 2025

- **Schedule-derived lines**: `implied_total`/`spread_line` now come from
  nflverse schedules (`data_collection.get_odds_data` /
  `schedule_to_team_lines`), not a manually-maintained odds file.
- **Depth charts**: `data_collection.get_depth_chart_data` /
  `get_depth_chart_for_week` are season-agnostic and handle both the old
  (season/week already present) and new (`dt`-snapshot based) nflreadpy
  depth-chart schemas.
- **`redzone_td_rate` removed** from `WR_TE_FEATURES`: it is computed from
  the same-game redzone trips/TDs and leaks information not available before
  the game is played.
- **OOB calibration**: the deployed model is calibrated with Platt scaling
  fit on its own out-of-bag predictions (`train_wr.fit_calibrator_oob`),
  rather than a held-out split.
- **Temporal CV**: hyperparameter tuning (`--tune`) uses `TimeSeriesSplit`
  instead of standard K-fold, and all frames are sorted chronologically
  before any split/EWM computation to avoid leakage.
- **Team-aware odds join**: `odds_match.match_odds_to_players` /
  `predict_wr.join_odds` disambiguate same-named players by requiring the
  player's team to be one of the two teams in the odds row's game.
- **Ledger**: `ledger.py` records picks at open odds, attaches the closing
  line, settles against play-by-play, and reports hit rate / ROI / CLV per
  strategy (`ledger.py record|close|settle|report`).

## Known limitations

- Inactive players are dropped from training data entirely (there is no
  explicit label-0 "inactive" row), so the model only ever sees weeks a
  player actually played.
- No de-vig: `market_implied_prob`/`model_edge` use the raw one-sided
  American-odds implied probability, not a de-vigged (overround-adjusted)
  probability.
- Single book: odds snapshots are pulled from one bookmaker
  (`fetch_live_odds.py --bookmakers draftkings` by default).
- No walk-forward backtest yet -- that is Phase 1 follow-up work; today's
  "informational" validation split in `train_wr.py` (train seasons <
  `config.VALIDATION_SEASON`, validate on `config.VALIDATION_SEASON`) is
  reported for visibility only and is not the deployed model.
