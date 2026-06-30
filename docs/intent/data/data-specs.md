# Data Ingestion — EARS Specs

Specs for the `data` leaf (prefix `DATA`). Design: `data-design.md`. Parent HLD:
`../../high-level-design.md`. All markers `[ ]` (greenfield, not yet implemented).

## DATA-ODDS — Historical odds ingestion & schema normalization

- [ ] **DATA-ODDS-001**: The system shall ingest historical Anytime Touchdown odds from `historical_data/td_odds/<season>/week_<week>_td_odds.csv` for seasons 2023–2025, regular-season weeks 1–18.
- [ ] **DATA-ODDS-002**: When ingesting a 2023 or 2024 odds file (Schema A: `Player, Team, HomeTeam, AwayTeam, Season, Week, GameDate, Odds, Bookmaker`), the system shall map its columns into the canonical odds table.
- [ ] **DATA-ODDS-003**: When ingesting a 2025 odds file (Schema B: `game_id, commence_time, in_play, bookmaker, last_update, home_team, away_team, market, label, description, price, point`), the system shall map its columns into the canonical odds table.
- [ ] **DATA-ODDS-004**: When ingesting a Schema B (2025) odds file, the system shall retain only rows whose `market` is `player_anytime_td` and whose `label` is `Yes`.
- [ ] **DATA-ODDS-005**: The system shall record exactly one canonical odds row per `(game_id, player_id)`; where multiple admissible snapshots exist for a player in a game, the system shall keep the latest one (admissibility per DATA-PIT-005).
- [ ] **DATA-ODDS-006**: For each canonical odds row, the system shall store the raw American odds and a derived raw implied probability, and shall not de-vig (de-vigging is a `betting`/`backtest` concern).
- [ ] **DATA-ODDS-007**: If an odds row has a missing or non-numeric price, then the system shall exclude it from the canonical odds table and flag it.

## DATA-NFL — nflverse ingestion

- [ ] **DATA-NFL-001**: The system shall ingest nflverse play-by-play data for the configured seasons via `nfl_data_py`.
- [ ] **DATA-NFL-002**: The system shall ingest nflverse weekly rosters (with `gsis_id`, team, position) for the configured seasons.
- [ ] **DATA-NFL-003**: The system shall ingest nflverse schedules (game metadata, kickoff time, home/away teams) for the configured seasons.
- [ ] **DATA-NFL-004**: The system shall ingest nflverse snap counts for the configured seasons.
- [ ] **DATA-NFL-005**: The system shall derive, per player-game, a binary TD-scoring outcome (1 if the player scored at least one rushing or receiving touchdown, else 0; passing touchdowns excluded) from play-by-play.

## DATA-MATCH — Player identity resolution

- [ ] **DATA-MATCH-001**: When resolving an odds row to a player, the system shall restrict candidate players to the weekly rosters of the row's home and away teams for that season and week.
- [ ] **DATA-MATCH-002**: Before matching, the system shall normalize player names on both the odds and roster sides by stripping generational suffixes (Jr., Sr., II, III, IV, V), removing punctuation from initials (e.g., `C.J.` → `CJ`), and folding case and whitespace.
- [ ] **DATA-MATCH-003**: When a normalized odds player name matches exactly one candidate in the bounded pool, the system shall assign that candidate's `gsis_id` and team to the odds row.
- [ ] **DATA-MATCH-004**: If a normalized odds player name matches no candidate exactly, then the system shall attempt a fuzzy match within the bounded candidate pool and accept the best candidate only if its similarity exceeds the configured threshold.
- [ ] **DATA-MATCH-005**: If an odds player name resolves to no candidate (e.g., a practice-squad call-up or just-signed player absent from both weekly rosters), then the system shall retain the row with null `gsis_id` and team and flag it as unmatched.
- [ ] **DATA-MATCH-006**: If a normalized odds player name matches more than one candidate in the bounded pool, then the system shall flag the row as ambiguous and leave `gsis_id` and team null.
- [ ] **DATA-MATCH-007**: The system shall report the count and rate of unmatched and ambiguous odds rows per season and week.

## DATA-NORM — Team-name normalization, provenance & canonical keys

- [ ] **DATA-NORM-001**: The system shall map sportsbook full team names (e.g., "Atlanta Falcons") to nflverse team abbreviations (e.g., "ATL") for `home_team` and `away_team`.
- [ ] **DATA-NORM-002**: The system shall take `(season, week)` from the odds file path as the authoritative value for every odds row.
- [ ] **DATA-NORM-003**: If a Schema A (2023/2024) odds file's in-file `Season` or `Week` column disagrees with the file path, then the system shall surface a data error rather than silently selecting either value.
- [ ] **DATA-NORM-004**: For Schema A (2023/2024) odds rows, the system shall synthesize a deterministic `game_id` from `(season, week, away_team, home_team)`.
- [ ] **DATA-NORM-005**: The system shall normalize `commence_time` to a UTC timestamp — from `GameDate` for Schema A, and from the date-only `commence_time` (at midnight UTC) for Schema B.
- [ ] **DATA-NORM-006**: The system shall set `snapshot_time` from Schema B's `last_update` and shall leave `snapshot_time` null for Schema A (no snapshot provenance exists).

## DATA-PIT — Point-in-time / leakage guard (Temporal Integrity facet)

- [ ] **DATA-PIT-001**: The system shall tag every emitted play-by-play, roster, snap-count, and odds row with its source `(season, week)`.
- [ ] **DATA-PIT-002**: The system shall emit the per-player-game TD label only for games whose schedule status is final/completed.
- [ ] **DATA-PIT-003**: The system shall store the per-player-game TD label in a field or table separate from feature-input data, so it cannot be consumed as a feature.
- [ ] **DATA-PIT-004**: The system shall restrict all ingestion (nflverse and odds) to regular-season weeks 1–18 and exclude playoff games.
- [ ] **DATA-PIT-005**: Where an odds row's `snapshot_time` is known, the system shall exclude from the canonical odds table any row whose `snapshot_time` is not strictly before the game's `commence_time`, and flag it.
