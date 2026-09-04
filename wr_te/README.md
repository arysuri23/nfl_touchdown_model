# WR/TE Anytime-Touchdown Model

Predicts anytime-touchdown probability for active WR/TE players for a given
NFL season/week, joins it against sportsbook odds, and tracks bets in a
ledger with closing-line value (CLV).

## Setup

```bash
cd wr_te
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Required only when a requested weekly snapshot does not exist or you pass
# --refresh to fetch_live_odds.py. Copy .env.example to .env and put the key
# there, or export ODDS_API_KEY in the shell. Shell values take precedence.
cp .env.example .env
# edit .env, then set: ODDS_API_KEY=...
```

Run tests with `.venv/bin/pytest tests/ -v`.

## Weekly runbook

All commands below assume `WRTE_SEASON`/`WRTE_WEEK` are set (default to the
current `config.SEASON`/`config.WEEK`; see *Overriding season/week*) and are
run from `wr_te/`.

1. **Tue/Wed: fetch opening odds** (≈17 credits on a cache miss)
   ```bash
   python fetch_live_odds.py --tag open
   ```
   A valid `vegas/<season>/week_<week>_td_odds_open.csv` is reused without
   spending credits. To intentionally replace it, pass `--refresh`.
   The collector preflights only the exact requested regular-season week,
   maps every scheduled nflverse game to one provider event, and reuses a
   cache only when its game, bookmaker, metadata, and pregame-quote
   attestations match the request. To validate one deterministic event
   without changing the canonical CSV, run with `--canary-one-event` (it
   cannot be combined with `--refresh`).
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

For an offline retrain from an existing cache, use `--from-cache`. This mode
reads `data/raw_nfl_data.csv`, applies the same season/week and feature cuts,
never calls the collector, and never rewrites the cache; hyperparameter tuning
is intentionally disabled. Successful model, calibrator, feature-importance,
and manifest artifacts are published together.

```bash
python train_wr.py --from-cache
```
4. **Record picks to the ledger** -- `top5_prob` is the primary funded
   strategy. `top5_edge_le400` is an optional zero-stake benchmark (tracked
   for its own record, not actually bet). The ledger does not apply a separate
   edge-threshold rule.
   ```bash
   python ledger.py record --strategy top5_prob
   python ledger.py record --strategy top5_edge_le400 --stake 0
   ```
5. **Sun, ~1h before the early kickoff: closing odds** (refresh explicitly if
   the closing snapshot already exists)
   ```bash
   python fetch_live_odds.py --tag close
   python ledger.py close
   ```
   The prediction command loads the opening snapshot through the same cache
   logic; it creates a missing snapshot once, but never refreshes an existing
   one.
6. **Tue: settle and report**
   ```bash
   python ledger.py settle
   python ledger.py report
   ```

### Periodic training refresh

Each scheduled training refresh collects all available completed rows from
`config.DATA_SEASONS` (the prior training seasons plus the active prediction
season), writes the feature cache, and fits the deployed raw RandomForest only
on rows from the configured prior-season `TRAIN_SEASONS` cut. Weekly production
loads `wr_te_rf_final.pkl` and emits the `random_forest_current/raw` probability
variant; the saved calibrator is not part of the production path. OOB Platt
metrics from training are a calibrator-fit-set diagnostic, not held-out
validation. The walk-forward evaluator's temporal Platt metrics are likewise
diagnostic. The refresh cadence does not change the ledger strategies or
introduce an implicit edge threshold.

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
- **Raw RF deployment and calibration diagnostics**: weekly production uses
  the raw deployed forest (`random_forest_current/raw`). Training retains an
  OOB Platt calibrator for diagnostics (`train_wr.fit_calibrator_oob`), and
  walk-forward evaluation reports temporal Platt diagnostics; neither is a
  held-out production calibration claim.
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
