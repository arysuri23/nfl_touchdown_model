# WR/TE Model 2026 Readiness (Phase 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 2025 `wr_te/` RandomForest pipeline run correctly for the 2026 season and log every pick with open/close prices and settled outcomes, so the model's edge becomes measurable from Week 1.

**Architecture:** Keep the three-script pipeline (`data_collection.py` → `train_wr.py` → `predict_wr.py`) and its feature semantics. Extract shared constants into `config.py` and `features.py`, replace hand-maintained inputs (`historic_lines.csv`, `week_N_lines.csv`, hardcoded 2025 depth-chart dates) with nflverse-derived equivalents, fix the confirmed correctness bugs, add a live odds fetcher and a bet ledger. Every nflverse/network call goes through an injectable loader argument so tests run offline.

**Tech Stack:** Python 3.13, pandas, polars, nflreadpy 0.1.5, scikit-learn, joblib, requests, pytest.

## Global Constraints

- All code lives under `wr_te/`. Every command in this plan runs with cwd = `wr_te/` (the scripts use relative paths).
- Interpreter: `/opt/homebrew/bin/python3.13`. Venv at `wr_te/.venv` (already gitignored by the root `.venv/` rule). Invoke as `.venv/bin/python` / `.venv/bin/pytest`.
- Tests live in `wr_te/tests/` and MUST NOT touch the network. Any function that calls `nflreadpy` or `requests` takes the loader as a keyword argument with the real function as default (e.g. `load_schedules=nfl.load_schedules`) so tests pass a fake.
- Never write an API key into code or a committed file. The key is read from the `ODDS_API_KEY` environment variable via `config.odds_api_key()`.
- Do not modify anything outside `wr_te/` except the root `.gitignore` (Task 8) and `docs/`. Do not touch `src/`, `docs/intent/`, `out/`, or branch `2026/v1`.
- Preserve existing feature semantics exactly: `spread_line` is from the team's perspective, **negative = that team is favored**; `implied_total = total_line / 2 - spread_line / 2`. nflverse schedules use the opposite convention (`spread_line` > 0 = home favored), so home team `spread_line = -schedule.spread_line`, away team `spread_line = +schedule.spread_line`.
- Season/week constants exist in exactly one place: `wr_te/config.py`. No other file may contain a literal `2025`/`2026` season constant after this plan (fixture data in tests excepted).
- Feature lists exist in exactly one place: `wr_te/features.py`.
- `redzone_td_rate` is removed from the model (confirmed forward leak + train/serve mismatch). Do not re-add it.
- Commit after every task with a conventional prefix (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`). Do not commit `.venv`, `*.pkl`, `raw_nfl_data.csv`.
- Existing tracked scripts must still import cleanly after every task: `.venv/bin/python -c "import data_collection, train_wr, predict_wr"`.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `wr_te/requirements.txt` | create | pinned dependency floors |
| `wr_te/config.py` | create | season/week/paths/env-key; env overrides `WRTE_SEASON`, `WRTE_WEEK` |
| `wr_te/features.py` | create | `WR_TE_FEATURES`, `PLAYER_EWM_STATS` |
| `wr_te/fetch_live_odds.py` | create | The Odds API live ATD + snapshot writer |
| `wr_te/ledger.py` | create | bet ledger: record / close / settle / report |
| `wr_te/README.md` | create | weekly runbook |
| `wr_te/tests/conftest.py` | create | `sys.path` + shared fixtures |
| `wr_te/tests/test_*.py` | create | one per task |
| `wr_te/data_collection.py` | modify | lines source, depth charts, dead code |
| `wr_te/train_wr.py` | modify | config, features import, calibrator, CV ordering |
| `wr_te/predict_wr.py` | modify | config, features import, roster, lines, depth chart, odds join, output path |
| `wr_te/fetch_historical_odds.py`, `wr_te/fetch_receptions_odds.py` | modify | key from env |
| `wr_te/data/historic_lines.csv` | delete | replaced by nflverse schedule lines |
| `.gitignore` (root) | modify | stop ignoring `**/predictions/` |

---

### Task 1: Environment, config, test scaffold

**Files:**
- Create: `wr_te/requirements.txt`, `wr_te/config.py`, `wr_te/tests/conftest.py`, `wr_te/tests/test_config.py`

**Interfaces:**
- Produces: `config.SEASON: int`, `config.WEEK: int`, `config.TRAIN_SEASONS: list[int]`, `config.MODELS_DIR`, `config.DATA_DIR`, `config.VEGAS_DIR`, `config.PREDICTIONS_DIR`, `config.LEDGER_DIR` (all `pathlib.Path`, relative to `wr_te/`), `config.odds_api_key() -> str`, `config.odds_snapshot_path(season, week, tag) -> Path`, `config.predictions_path(season, week) -> Path`.

- [ ] **Step 1: Create the venv and requirements**

`wr_te/requirements.txt` — one package per line, minimum floors, no upper pins:
`nflreadpy>=0.1.5`, `polars>=1.0`, `pyarrow>=15`, `pandas>=2.2`, `numpy>=1.26`, `scikit-learn>=1.5`, `joblib>=1.3`, `requests>=2.31`, `pytest>=8`.

Run: `/opt/homebrew/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`
Then: `.venv/bin/python -c "import data_collection, train_wr, predict_wr; print('ok')"` — Expected: `ok` (no ImportError; pyarrow is required by polars `.to_pandas()`).

- [ ] **Step 2: Write the failing config tests**

`wr_te/tests/conftest.py`: insert the `wr_te/` directory (parent of `tests/`) at the front of `sys.path` so `import config` works from any cwd.

`wr_te/tests/test_config.py` asserts:
- `config.SEASON == 2026` and `config.WEEK == 1` when env vars `WRTE_SEASON`/`WRTE_WEEK` are unset (use `monkeypatch.delenv(..., raising=False)` then `importlib.reload(config)`).
- With `WRTE_SEASON=2025 WRTE_WEEK=15` set and reloaded, `config.SEASON == 2025`, `config.WEEK == 15`.
- `config.TRAIN_SEASONS == [2020, 2021, 2022, 2023, 2024, 2025]`.
- `config.odds_api_key()` raises `RuntimeError` whose message contains `ODDS_API_KEY` when the env var is unset; returns the value when set.
- `config.odds_snapshot_path(2026, 1, "open") == config.VEGAS_DIR / "2026" / "week_1_td_odds_open.csv"`.
- `config.predictions_path(2026, 1) == config.PREDICTIONS_DIR / "2026" / "week_1.csv"`.

Run: `.venv/bin/pytest tests/test_config.py -v` — Expected: FAIL (`ModuleNotFoundError: config`).

- [ ] **Step 3: Implement `config.py`**

All paths are `Path(__file__).parent / <name>` with names `models`, `data`, `vegas`, `predictions`, `ledger`. `SEASON`/`WEEK` read env with `int(os.environ.get("WRTE_SEASON", 2026))` etc. `odds_api_key()` raises `RuntimeError("Set the ODDS_API_KEY environment variable")` when missing/empty.

- [ ] **Step 4: Run tests** — `.venv/bin/pytest tests/ -v` — Expected: all PASS.

- [ ] **Step 5: Commit** — `git add wr_te/requirements.txt wr_te/config.py wr_te/tests/ && git commit -m "feat(wr_te): add requirements, config module, test scaffold"`

---

### Task 2: Single feature list; remove `redzone_td_rate`

**Files:**
- Create: `wr_te/features.py`, `wr_te/tests/test_features.py`
- Modify: `wr_te/train_wr.py:26-81` (feature lists), `wr_te/train_wr.py:100-104` (player stats list), `wr_te/train_wr.py:129-142` (team-level EWM + `pass_matchup_value`), `wr_te/predict_wr.py:15-70` (feature lists), `wr_te/predict_wr.py:88-92` (player stats list), `wr_te/predict_wr.py:155-189` (non-player feature routing incl. `redzone_td_rate`)

**Interfaces:**
- Produces: `features.WR_TE_FEATURES: list[str]` (23 names — the current active list in `train_wr.py:26-53` minus `'redzone_td_rate'`), `features.PLAYER_EWM_STATS: list[str]` (the 25-name list currently duplicated at `train_wr.py:100-104` and `predict_wr.py:88-92`, verbatim).

- [ ] **Step 1: Write the failing tests** (`tests/test_features.py`)
- `len(features.WR_TE_FEATURES) == 23` and `len(set(...)) == 23`.
- `'redzone_td_rate' not in features.WR_TE_FEATURES`.
- `train_wr.WR_TE_FEATURES is features.WR_TE_FEATURES` and `predict_wr.WR_TE_FEATURES is features.WR_TE_FEATURES` (identity, not equality).
- `train_wr.feature_engineering` on a small synthetic frame (two players, one team, 4 weeks; include every column the function reads — `PLAYER_EWM_STATS`, `opponent_team`, `passing_tds_allowed_to_WR/TE`, `receiving_yards_allowed`, `receiving_epa_allowed`, `receiving_air_yards_allowed`, `explosive_receiving_plays_allowed`, `position`) returns a frame with no `redzone_td_rate`, `pass_rate`, or `pass_matchup_value` column and with every `WR_TE_FEATURES` column except `implied_total`, `spread_line`, `depth_chart_rank` present (those three come from data_collection, not feature_engineering).

Run: `.venv/bin/pytest tests/test_features.py -v` — Expected: FAIL.

- [ ] **Step 2: Implement**
- Create `features.py` with the two lists.
- `train_wr.py`: delete lines 26-81, add `from features import WR_TE_FEATURES, PLAYER_EWM_STATS`; replace the inline list at 100-104 with `PLAYER_EWM_STATS`; delete lines 129-132 (`redzone_td_rate` ×2 and `pass_rate` EWM), delete the `pass_matchup_value` block (139-142). Keep the `is_home` fallback.
- `predict_wr.py`: delete lines 15-70, add the same import; replace 88-92 with `PLAYER_EWM_STATS`; in `predict_touchdown_scorers` remove `'redzone_td_rate'` from the non-player-feature list and delete the `team_history_features` block (lines 165, 183-189).

- [ ] **Step 3: Run** `.venv/bin/pytest tests/ -v` and `.venv/bin/python -c "import data_collection, train_wr, predict_wr"` — Expected: PASS / no error.

- [ ] **Step 4: Commit** — `git commit -am "refactor(wr_te): single feature list; drop leaky redzone_td_rate"` (add `features.py`, tests).

---

### Task 3: Game lines from nflverse schedules

**Files:**
- Modify: `wr_te/data_collection.py:54-83` (`get_odds_data`), `wr_te/data_collection.py:452-508` (`transform_future_odds` — delete), `wr_te/data_collection.py:700-701` (call site)
- Delete: `wr_te/data/historic_lines.csv`
- Create: `wr_te/tests/test_lines.py`

**Interfaces:**
- Produces: `data_collection.schedule_to_team_lines(schedule: pd.DataFrame) -> pd.DataFrame` (pure; columns out: `season, week, team, opponent, spread_line, total_line, implied_total`), `data_collection.get_odds_data(years, team_map=None, load_schedules=nfl.load_schedules) -> pd.DataFrame` (same columns, REG weeks only, `week <= 18`), `data_collection.get_week_lines(season, week, load_schedules=nfl.load_schedules) -> pd.DataFrame` (same columns, one week).
- Input schedule columns used: `season, week, game_type, home_team, away_team, spread_line, total_line`.

- [ ] **Step 1: Write the failing tests** (`tests/test_lines.py`)
Synthetic schedule (pandas): row A `season=2024, week=1, game_type='REG', home_team='KC', away_team='BAL', spread_line=3.0, total_line=46.0`; row B same week `home_team='LA', away_team='SF', spread_line=-2.5, total_line=48.5`; row C `week=19, game_type='WC'`; row D `week=2` with `spread_line=NaN`.
Assert on `schedule_to_team_lines(sched)`:
- KC row: `spread_line == -3.0`, `implied_total == 24.5`, `opponent == 'BAL'`; BAL row: `spread_line == 3.0`, `implied_total == 21.5`.
- LA row: `spread_line == 2.5`, `implied_total == 23.0`; SF: `spread_line == -2.5`, `implied_total == 25.5`.
- For every game, the two `implied_total`s sum to `total_line`.
- Row C (playoff) and row D (null line) produce no output rows.
- `get_odds_data([2024], load_schedules=fake)` where `fake` returns a polars frame from the synthetic schedule → identical to the pure function's output.
- `get_week_lines(2024, 1, load_schedules=fake)` returns only week-1 rows (4 rows).
- Team codes `LAR`/`LVR`, if present in input, are emitted as `LA`/`LV`.

Run — Expected: FAIL (`schedule_to_team_lines` missing).

- [ ] **Step 2: Implement.** Replace `get_odds_data` to load schedules via the injected loader (`.to_pandas()`), filter `game_type == 'REG'`, call `schedule_to_team_lines`, filter `week <= 18`. Delete `transform_future_odds`. Update the call at line 700 (`get_odds_data(years, team_map)` — signature still accepts `team_map` for compatibility; it is unused). Delete `data/historic_lines.csv` with `git rm`.

- [ ] **Step 3: Run** `.venv/bin/pytest tests/ -v`; import check. Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -m "feat(wr_te): derive game lines from nflverse schedules; drop historic_lines.csv"`

---

### Task 4: Depth charts for any season

**Files:**
- Modify: `wr_te/data_collection.py:435-450` (`get_depth_chart_data`), `wr_te/data_collection.py:580-628` (`get_2025_depth_chart_data` — delete), `wr_te/data_collection.py:728-731` (call site)
- Create: `wr_te/tests/test_depth_charts.py`

**Interfaces:**
- Produces: `data_collection.select_week_snapshots(snapshot_dates: list[datetime.date], week_first_gameday: dict[int, datetime.date]) -> dict[int, datetime.date | None]` (pure: for each week, the latest snapshot date `<=` that week's first gameday, else `None`), `data_collection.get_depth_chart_data(seasons: list[int], load_depth_charts=nfl.load_depth_charts, load_schedules=nfl.load_schedules) -> pd.DataFrame` (columns `player_id, season, week, depth_chart_rank`; one row per `(player_id, season, week)`, keep the minimum rank; `depth_chart_rank` int; WR/TE only), `data_collection.get_depth_chart_for_week(season, week, load_depth_charts=..., load_schedules=...) -> pd.DataFrame` (same columns, that week only).
- Schema detection: if the loaded frame has a `dt` column → new schema (`gsis_id`, `pos_abb`, `pos_rank`, `dt` string `YYYY-MM-DD...`); else old schema (`gsis_id`, `depth_position`, `depth_team`, `week`, `season`).
- Week gamedays come from `load_schedules([season])` REG rows: `week_first_gameday[w] = min(gameday)` per week.

- [ ] **Step 1: Write the failing tests**
- `select_week_snapshots([d(2026,9,1), d(2026,9,2), d(2026,9,8), d(2026,9,15)], {1: d(2026,9,9), 2: d(2026,9,17), 3: d(2026,9,24)})` → `{1: d(2026,9,8), 2: d(2026,9,15), 3: d(2026,9,15)}`.
- With no snapshot before week 1's gameday → `{1: None, ...}`.
- New-schema path: fake `load_depth_charts` returns a polars frame with rows for `gsis_id='00-1'` `pos_abb='WR'` `pos_rank=1` at `dt='2026-09-08T00:00:00'`, same player `pos_rank=2` at `dt='2026-09-15...'`, a `QB` row (must be dropped), and a second WR/TE row for the same player+date with `pos_rank=3` (dedupe keeps 1). Fake `load_schedules` gives week gamedays as above. Assert `get_depth_chart_data([2026], ...)` yields `(00-1, 2026, 1, 1)` and `(00-1, 2026, 2, 2)` and nothing for week 3 beyond `(00-1, 2026, 3, 2)`; dtype int.
- Old-schema path: fake frame with `depth_position='WR'`, `depth_team='2'` (string), `week=5`, `season=2024` → row `(gsis, 2024, 5, 2)`.
- `get_depth_chart_for_week(2026, 2, ...)` returns only week 2.

Run — Expected: FAIL.

- [ ] **Step 2: Implement.** One `get_depth_chart_data` handling both schemas; remove the `week_map` dict and `get_2025_depth_chart_data`; at the call site replace lines 728-731 with `depth_chart_df = get_depth_chart_data(years)`.

- [ ] **Step 3: Run tests + import check** — Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -m "feat(wr_te): season-agnostic depth charts mapped to weeks via schedule gamedays"`

---

### Task 5: Training protocol — chronological CV, calibrator on the deployed model, config-driven cut

**Files:**
- Modify: `wr_te/train_wr.py:380-560` (`main`), `wr_te/train_wr.py:150-200` (`train_rf_model`)
- Create: `wr_te/tests/test_train.py`

**Interfaces:**
- Produces: `train_wr.chronological(df: pd.DataFrame) -> pd.DataFrame` (sorted by `['season','week','player_id']`, index reset), `train_wr.fit_calibrator_oob(model, y: pd.Series, mask: pd.Series | None = None, min_rows: int = 500) -> LogisticRegression` (fits Platt on `model.oob_decision_function_[:, 1]` restricted to `mask` rows, dropping NaN OOB entries; if fewer than `min_rows` remain, uses all non-NaN rows), `train_wr.calibration_report(y_true, p) -> dict` with keys `brier`, `log_loss`, `ece` (10 equal-width bins, mean |avg_pred − emp_rate| over populated bins).
- CLI: `python train_wr.py [--tune]`. Default uses `models/wr_te_rf_best_params.json`; `--tune` re-runs `RandomizedSearchCV` and overwrites it.

- [ ] **Step 1: Write the failing tests**
- `chronological` orders a shuffled 3-row frame by season, then week, then player_id, index `0..n-1`.
- `fit_calibrator_oob`: synthetic 2,000-row binary dataset (`numpy` RNG seed 0, 5 features, label = sigmoid of a linear combo > uniform), `RandomForestClassifier(n_estimators=60, max_depth=4, oob_score=True, random_state=0)`; call with `mask` selecting the last 800 rows → returns a fitted `LogisticRegression` (has `coef_`); `predict_proba([[0.1]])[:,1] < predict_proba([[0.9]])[:,1]`; a mask selecting 100 rows with `min_rows=500` falls back to all rows and still returns a fitted model (assert via `n_features_in_ == 1` and no exception).
- `calibration_report(y, p)` on perfectly calibrated synthetic input returns `ece < 0.05` and all three keys.
- `train_rf_model` with `use_saved_params=True` and a params file in a `tmp_path` models dir (monkeypatch cwd) trains without invoking `RandomizedSearchCV` (monkeypatch `train_wr.RandomizedSearchCV` to raise if called).

Run — Expected: FAIL.

- [ ] **Step 2: Implement in `main`:**
- Delete `CURRENT_SEASON`, `CURRENT_WEEK`, `USE_SAVED_PARAMS`; read `config.TRAIN_SEASONS` and `config.SEASON`. Training rows = `season in TRAIN_SEASONS and season < config.SEASON and week <= 18` (i.e. all of 2025 REG for a 2026 run).
- After `feature_engineering`, `df = chronological(df)` **before** any split, so `TimeSeriesSplit` folds inside `train_rf_model` are temporal.
- Keep the informational 2020–2022 → 2023 validation and `evaluate_rf_model` as-is, but label its printout "informational: not the deployed model".
- Final model: unchanged fit on all training rows with `oob_score=True`.
- Replace the entire calibrator block (lines 468-534: `rf_for_calib`, `calib_train_df`, `calib_test_df`, fallback) with `mask = all_data_df['season'] == all_data_df['season'].max()`; `wr_te_calibrator = fit_calibrator_oob(wr_te_final, y_all_wr_te, mask)`; print `calibration_report` on the same OOB rows with the header "Platt on OOB predictions of the deployed forest (metrics in-sample for the 2-parameter calibrator)".
- `argparse` `--tune` flag → `use_saved_params = not args.tune`.
- Replace every stale print string that names seasons/weeks with values derived from config (`f"Training on {min}-{max}"`).

- [ ] **Step 3: Run tests + import check** — Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -m "fix(wr_te): temporal CV ordering, calibrate deployed forest on OOB, config-driven training cut"`

---

### Task 6: Live odds fetcher; keys from env

**Files:**
- Create: `wr_te/fetch_live_odds.py`, `wr_te/tests/test_fetch_live_odds.py`, `wr_te/tests/fixtures/event_odds_sample.json`, `wr_te/tests/fixtures/events_sample.json`
- Modify: `wr_te/fetch_historical_odds.py:18,102`, `wr_te/fetch_receptions_odds.py:18,119`

**Interfaces:**
- Produces: `fetch_live_odds.select_week_events(events: list[dict], window_start: datetime, window_end: datetime) -> list[dict]` (events whose parsed `commence_time` is in `[start, end)`), `fetch_live_odds.parse_event_odds(event: dict, season: int, week: int, tag: str, fetched_at: str) -> list[dict]`, `fetch_live_odds.week_window(season, week, load_schedules=nfl.load_schedules) -> tuple[datetime, datetime]` (`min gameday 00:00 UTC` to `max gameday + 1 day`), `fetch_live_odds.fetch(season, week, tag, bookmakers: list[str], get=requests.get, load_schedules=...) -> pd.DataFrame`, CLI `python fetch_live_odds.py --season S --week W --tag open|close [--bookmakers draftkings,fanduel]` (defaults from config; bookmakers default `draftkings`).
- Output CSV columns, in this order (legacy-compatible): `game_id, commence_time, in_play, bookmaker, last_update, home_team, away_team, market, label, description, price, point, season, week, tag, fetched_at`. One row per (event, bookmaker, player). Written to `config.odds_snapshot_path(season, week, tag)`; parent dir created.
- Endpoints: `GET https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey=K` and `GET https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{id}/odds?apiKey=K&regions=us&markets=player_anytime_td&oddsFormat=american&bookmakers=B`. After the run print `x-requests-used` / `x-requests-remaining` from the last response headers.
- Fixture shape (Odds API v4): event `{id, sport_key, commence_time, home_team, away_team, bookmakers: [{key, title, last_update, markets: [{key: 'player_anytime_td', last_update, outcomes: [{name: 'Yes', description: '<player>', price: <int>}]}]}]}`. Build fixtures with 1 event, 2 bookmakers, 3 players each, one outcome with `name: 'No'` that must be dropped.

- [ ] **Step 1: Write the failing tests**
- `parse_event_odds(fixture, 2026, 1, 'open', '2026-09-08T15:00:00Z')` → 6 rows; `label == 'Yes'` on all; `market == 'player_anytime_td'`; `price` int; `description` player names; `bookmaker` equals bookmaker `title`; `home_team`/`away_team` from the event.
- `select_week_events` keeps an event at `2026-09-13T17:00:00Z` for window `[2026-09-09, 2026-09-15)` and drops one at `2026-09-20T17:00:00Z`.
- `fetch(...)` with a fake `get` (returns objects with `.json()`, `.headers`, `.raise_for_status()`) that serves `events_sample.json` then `event_odds_sample.json` writes the CSV under a `tmp_path` (monkeypatch `config.VEGAS_DIR`) with the exact column order above and 6 rows; assert the fake was called with `markets=player_anytime_td` and the bookmakers string.
- `fetch` raises `RuntimeError` mentioning `ODDS_API_KEY` when the env var is unset (monkeypatch).
- `grep -n "api_key = '" fetch_historical_odds.py fetch_receptions_odds.py` returns nothing (assert via `pathlib.read_text()` that no line matches the regex `api_key\s*=\s*['\"]` or `API_KEY\s*=\s*['\"]`).

Run — Expected: FAIL.

- [ ] **Step 2: Implement** `fetch_live_odds.py`. In the two legacy fetchers replace the literal key assignments with `config.odds_api_key()` (keep the rest untouched).

- [ ] **Step 3: Run tests** — Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -m "feat(wr_te): live Odds API ATD snapshot fetcher; API keys from env"`

---

### Task 7: `predict_wr.py` — roster, lines, depth chart, odds join, output

**Files:**
- Modify: `wr_te/predict_wr.py:122-283`
- Create: `wr_te/tests/test_predict.py`

**Interfaces:**
- Consumes: `data_collection.get_week_lines`, `data_collection.get_depth_chart_for_week` (Tasks 3–4), `config.*`, odds CSV from Task 6.
- Produces: `predict_wr.load_week_roster(season, week, load_rosters_weekly=nfl.load_rosters_weekly, load_rosters=nfl.load_rosters) -> pd.DataFrame` (columns `player_id, player_display_name, position, team`; WR/TE; `status == 'ACT'`; from weekly rosters filtered to `week == week` when available, else — on `ValueError` or empty result — from season rosters; dedupe on `player_id` keep last), `predict_wr.join_odds(predictions: pd.DataFrame, odds: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame` (adds `price, market_implied_prob, model_edge, bookmaker`; join key = `merge_name` AND the prediction `team` must be one of `{team_map[home_team], team_map[away_team]}` for that odds row; if several bookmakers, keep the highest `price`; rows without a price are dropped), `predict_wr.merge_name(s: pd.Series) -> pd.Series` (the existing normalisation at lines 221-222, extracted), `predict_wr.predict_touchdown_scorers(feature_df, model, calibrator, season, week, lines_df, depth_df, roster_df) -> pd.DataFrame`.
- Output file: `config.predictions_path(season, week)`, columns include `season, week, player_id, player_display_name, team, opponent_team, position, predicted_touchdown_probability, price, market_implied_prob, model_edge, bookmaker, odds_snapshot` (`odds_snapshot` = the odds file name).
- Console: top 20 by probability, then top 10 by `model_edge` among `price <= 400` labelled `"informational — edge ranking, not yet validated"`.

- [ ] **Step 1: Write the failing tests**
- `load_week_roster(2026, 1, load_rosters_weekly=raises_valueerror, load_rosters=fake_season)` returns the season-roster WR/TEs with `status == 'ACT'` only.
- With a working weekly fake containing weeks 1 and 3, `load_week_roster(2026, 3, ...)` returns only week-3 rows, and a player listed in week 1 on team A and week 3 on team B comes back with team B.
- `join_odds`: predictions have two "Mike Williams" rows (teams `NYJ`, `LAC`); odds have `description='Mike Williams'` in game `home_team='New York Jets'`, `away_team='Buffalo Bills'`, price `+350` and another in `Los Angeles Chargers` vs `Denver Broncos`, price `+280`; `team_map` maps full names → `NYJ`, `BUF`, `LAC`, `DEN`. Assert NYJ row gets 350, LAC row gets 280, 2 rows out. Add a second bookmaker row for the NYJ game with price `+375` → NYJ keeps 375. A prediction with no odds row is dropped. `market_implied_prob` for `+350` is `100/450`; for `-150` is `150/250`.
- `predict_touchdown_scorers` smoke: build a tiny `feature_df` (two players, 3 prior weeks, all `PLAYER_EWM_STATS` + defense columns + `season/week/team/opponent_team/player_id`), a fake `model` with `predict_proba` returning `[[0.7, 0.3], ...]`, `calibrator=None`, `lines_df` from Task 3's pure function, `depth_df` with one player rank 1, `roster_df` with both players → returns a frame with every `WR_TE_FEATURES` column present and `depth_chart_rank == 4` for the player missing from `depth_df`.

Run — Expected: FAIL.

- [ ] **Step 2: Implement.** In `__main__`: `season, week = config.SEASON, config.WEEK`; models from `config.MODELS_DIR`; odds from `config.odds_snapshot_path(season, week, 'open')` (error message names the path and `fetch_live_odds.py` if missing); `lines_df = data.get_week_lines(season, week)`; `depth_df = data.get_depth_chart_for_week(season, week)`; `roster_df = load_week_roster(season, week)`; write to `config.predictions_path(season, week)` after `mkdir(parents=True, exist_ok=True)`. Delete the `week_{N}_lines.csv` read and `transform_future_odds` call. Fix the final print to name the real output path.

- [ ] **Step 3: Run tests + import check** — Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -m "fix(wr_te): week-scoped roster, schedule lines, season-agnostic depth chart, team-aware odds join"`

---

### Task 8: Bet ledger — record, close, settle, report

**Files:**
- Create: `wr_te/ledger.py`, `wr_te/tests/test_ledger.py`
- Modify: root `.gitignore` (remove the `**/predictions/` line — predictions and the ledger are the record and are committed)

**Interfaces:**
- Ledger file: `config.LEDGER_DIR / "bets.csv"`. Columns, in order: `bet_id, season, week, placed_at, strategy, player_id, player_display_name, team, opponent_team, position, model_prob, bookmaker, price_open, implied_open, price_close, implied_close, stake, outcome, pnl, clv`. `bet_id = f"{season}-{week}-{strategy}-{player_id}"`.
- `ledger.implied_prob(price: int | float) -> float` (American → break-even probability: `100/(p+100)` if `p > 0` else `|p|/(|p|+100)`), `ledger.decimal_odds(price) -> float` (`1 + p/100` if `p > 0` else `1 + 100/|p|`).
- `ledger.record_picks(predictions: pd.DataFrame, season, week, strategy: str, stake: float, ledger_path: Path, now: str) -> int` (appends rows; **idempotent** — existing `bet_id`s are skipped; returns number added). Strategy selectors are in `ledger.STRATEGIES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]]` with two entries: `'top5_prob'` (top 5 by `predicted_touchdown_probability`) and `'top5_edge_le400'` (rows with `price <= 400`, top 5 by `model_edge`).
- `ledger.attach_closing(ledger_path, close_odds: pd.DataFrame, team_map) -> int` (fills `price_close`, `implied_close`, `clv = implied_close - implied_open` for unsettled rows of that season/week, matching by `merge_name` + team-in-game exactly as `predict_wr.join_odds`; returns rows updated).
- `ledger.settle(ledger_path, season, week, load_pbp=nfl.load_pbp) -> int` (scorers = pbp rows of that `season`/`week` with `touchdown == 1` and non-null `td_player_id` — any TD type, matching how the ATD market pays; for each ledger row of that week whose team appears in that week's pbp `posteam`/`defteam`: `outcome = 1 if player_id in scorers else 0`, `pnl = stake * (decimal_odds(price_open) - 1)` on a win else `-stake`; rows whose team has no pbp yet stay unsettled; returns rows settled).
- `ledger.report(ledger_path) -> pd.DataFrame` (per `(strategy, week)` and per `strategy` cumulative: `n_bets, n_settled, hits, hit_rate, staked, pnl, roi, n_with_close, mean_clv`); the CLI prints it.
- CLI: `python ledger.py record --strategy top5_prob [--stake 1.0] [--predictions PATH]`, `python ledger.py close [--odds PATH]`, `python ledger.py settle`, `python ledger.py report` — `--season/--week` default from config; `record` defaults `--predictions` to `config.predictions_path(season, week)`; `close` defaults `--odds` to `config.odds_snapshot_path(season, week, 'close')`.

- [ ] **Step 1: Write the failing tests**
- `implied_prob(350) == pytest.approx(100/450)`, `implied_prob(-150) == pytest.approx(0.6)`, `decimal_odds(350) == 4.5`, `decimal_odds(-150) == pytest.approx(5/3)`.
- `record_picks` on a 7-row predictions frame with `strategy='top5_prob'` writes 5 rows; calling again returns 0 and the file still has 5 rows; the 5 rows are the 5 highest probabilities; `price_open`/`implied_open` populated; `outcome`, `pnl`, `price_close`, `clv` empty.
- `'top5_edge_le400'` on a frame where the highest-edge row has `price = 600` excludes that row.
- `attach_closing` with a close-odds frame moves one player from `+300` to `+250` → `clv == implied_prob(250) - implied_prob(300) > 0`; a player absent from the close file keeps empty close columns; returns 1.
- `settle` with a fake `load_pbp` (polars or pandas frame with `season, week, posteam, defteam, touchdown, td_player_id`) where player A scored (`+300`, stake 1) → `outcome 1, pnl 3.0`; player B didn't → `outcome 0, pnl -1.0`; player C's team has no pbp rows → still unsettled; returns 2. A defensive/return TD (`td_player_id` of a player on `defteam`) counts as a score.
- `report` on that ledger: `top5_prob` cumulative `n_settled == 2, hits == 1, pnl == 2.0, roi == 1.0`.

Run — Expected: FAIL.

- [ ] **Step 2: Implement `ledger.py`**; edit root `.gitignore`.

- [ ] **Step 3: Run tests** — Expected: PASS.

- [ ] **Step 4: Commit** — `git commit -m "feat(wr_te): bet ledger with open/close prices, settlement from pbp, CLV and ROI report"`

---

### Task 9: Runbook and end-to-end verification on 2025 Week 15

**Files:**
- Create: `wr_te/README.md`
- Modify: none (this task runs the pipeline; it may fix small integration breakages it finds, each as its own commit)

**Interfaces:** consumes everything above.

- [ ] **Step 1: Write `README.md`** — sections: *Setup* (venv, `requirements.txt`, `export ODDS_API_KEY=...`), *Weekly runbook* as a numbered list with the exact commands:
  1. Tue/Wed: `python fetch_live_odds.py --tag open` (≈17 credits)
  2. `python train_wr.py` (first week of season and every 4 weeks; `--tune` only when changing features)
  3. `python predict_wr.py`
  4. `python ledger.py record --strategy top5_prob` and `python ledger.py record --strategy top5_edge_le400 --stake 0` (the second is logged at zero stake — tracked, not bet)
  5. Sun ~1h before the early kickoff: `python fetch_live_odds.py --tag close` then `python ledger.py close`
  6. Tue: `python ledger.py settle` then `python ledger.py report`
  Plus *Overriding season/week* (`WRTE_SEASON=2025 WRTE_WEEK=15 python predict_wr.py`), *What changed from 2025* (bullet list: schedule lines, depth charts, `redzone_td_rate` removed, OOB calibration, temporal CV, team-aware odds join, ledger), and *Known limitations* (inactives dropped from training; no de-vig; single-book; no walk-forward backtest yet — Phase 1).

- [ ] **Step 2: End-to-end run against 2025 Week 15 (network required, no API key needed).**
  Prepare the odds snapshot from the legacy file: copy `vegas/week_15_td_odds.csv` to `vegas/2025/week_15_td_odds_open.csv` and append the columns `season=2025, week=15, tag=open, fetched_at=''` (a 5-line pandas one-off; do not commit the script). Then, from `wr_te/`:
  - `WRTE_SEASON=2025 WRTE_WEEK=15 .venv/bin/python train_wr.py` — must complete using saved params. Record in the report: total training rows, seasons covered, the calibration report dict, wall time. (Training cut for `SEASON=2025` is seasons `< 2025`, i.e. 2020–2024 — that is expected for this override.)
  - `WRTE_SEASON=2025 WRTE_WEEK=15 .venv/bin/python predict_wr.py` — must write `predictions/2025/week_15.csv`. Record: row count, the top-5 by probability (name, team, prob, price).
  - `WRTE_SEASON=2025 WRTE_WEEK=15 .venv/bin/python ledger.py record --strategy top5_prob` → 5 rows; `... ledger.py settle` → 5 settled (2025 week 15 pbp exists); `... ledger.py report` → prints; record the hits/pnl.
  Commit the generated `predictions/2025/week_15.csv`, `vegas/2025/week_15_td_odds_open.csv`, and `ledger/bets.csv` (they are the record). Do NOT commit `models/*.pkl` or `data/raw_nfl_data.csv`.

- [ ] **Step 3: Full test suite** — `.venv/bin/pytest tests/ -v` — Expected: all PASS.

- [ ] **Step 4: Commit** — `git commit -m "docs(wr_te): weekly runbook; e2e verification artifacts for 2025 wk15"`

---

## Self-review notes

- Spec coverage: env/manifest (T1), constants (T1), feature list + leak removal (T2), lines (T3), depth charts (T4), CV order + calibrator + `--tune` (T5), key from env + live fetcher (T6), roster/odds-join/output/2026 gate (T7), ledger + CLV (T8), runbook + e2e (T9). Out of scope by decision: label-0 inactives, de-vig, walk-forward backtest, RBs — Phase 1.
- Type consistency: `get_week_lines` / `get_depth_chart_for_week` names match between T3/T4 and T7; `join_odds` matching rule is reused verbatim by `attach_closing` in T8; `config.odds_snapshot_path` / `predictions_path` used by T6, T7, T8, T9.
