---
parent: high-level-design
prefix: DATA
---

# Data Ingestion

## Context and Design Philosophy

`data` is the foundation segment: every downstream segment (`features`, `modeling`,
`betting`, `backtest`, `pipeline`) consumes its tables and nothing else reaches around
it to a raw source. Its job is to turn two messy, heterogeneous worlds — open nflverse
football data and sportsbook Anytime Touchdown (ATD) odds — into clean, point-in-time
tables keyed consistently by `(season, week, player_id)` and `(season, week, game_id)`.

Two principles govern the segment, both inherited from the HLD:

- **The backtest is sacred: no leakage.** Every row this segment emits carries enough
  provenance to answer "was this knowable before kickoff?" Where provenance is missing
  (see the historical-odds asymmetry below), the gap is recorded explicitly rather than
  papered over, so `backtest` can decide how to treat it.
- **One canonical shape.** The raw odds files drift in schema across seasons; this
  segment absorbs that drift so no downstream segment ever branches on file format.

## Source Inventory

| Source | Access | Coverage | Role |
|--------|--------|----------|------|
| nflverse (via `nfl_data_py`) | library, free | play-by-play, weekly rosters, schedules, snap counts | feature substrate + the TD-scoring ground truth label |
| Historical ATD odds | on-hand CSVs under `historical_data/td_odds/<year>/week_<n>_td_odds.csv` | 2023–2025, DraftKings only | backtest odds |
| The Odds API | HTTP, paid | current/upcoming slate | live/forward odds (v1: adapter + interface; concrete wiring may follow) |

## Historical Odds: Observed Schemas

The on-hand files carry **two different schemas** across seasons. Both are DraftKings-only
and contain only the "Yes" (anytime) side of the market.

**Schema A — 2023, 2024:**
`Player, Team, HomeTeam, AwayTeam, Season, Week, GameDate, Odds, Bookmaker`
- `Team` is always the literal `N/A` — the player's team is **not** provided.
- `GameDate` is ISO-8601 UTC kickoff (e.g. `2024-10-04T00:15:00Z`).
- `Odds` is American format (e.g. `-125`, `145`).
- No snapshot timestamp: there is no record of *when* during the week these odds were captured.

**Schema B — 2025:**
`game_id, commence_time, in_play, bookmaker, last_update, home_team, away_team, market, label, description, price, point`
- Player name is in `description`; `label` is always `Yes`; `market` is always `player_anytime_td`.
- `commence_time` is a date only (`9/5/2025`, M/D/YYYY) — no kickoff time.
- `last_update` is the odds **snapshot** date (e.g. `9/3/2025`) and `in_play` is `FALSE` — this pair gives real pre-game provenance that Schema A lacks. Observed: a single snapshot per week, ~2 days before kickoff (a mid-week line), **not** a closing line.
- `game_id` is a native stable key; `point` is empty (irrelevant for ATD).
- No `Season`/`Week` columns — those live only in the file path.

### Canonical Odds Table

Both schemas normalize into one table; downstream never sees the difference.

| Field | Type | Source A | Source B | Notes |
|-------|------|----------|----------|-------|
| `season` | int | file path (authoritative) | file path (authoritative) | path wins; validated against `Season` column when present |
| `week` | int | file path (authoritative) | file path | |
| `game_id` | str | synthesized (see below) | native `game_id` | stable per game |
| `commence_time` | UTC datetime | `GameDate` | `commence_time` (date only → midnight UTC) | B has no time-of-day |
| `snapshot_time` | UTC datetime \| null | **null** (unknown) | `last_update` | leakage-provenance marker |
| `bookmaker` | str | `Bookmaker` | `bookmaker` | DraftKings only, all years |
| `player_name_raw` | str | `Player` | `description` | pre-normalization |
| `player_id` | str \| null | resolved (see Identity) | resolved | nflverse `gsis_id`; null if unmatched |
| `team` | str \| null | resolved via roster join | resolved via roster join | nflverse abbrev |
| `home_team` / `away_team` | str | `HomeTeam`/`AwayTeam` | `home_team`/`away_team` | normalized full-name → nflverse abbrev |
| `american_odds` | int | `Odds` | `price` | |
| `implied_prob_raw` | float | derived | derived | from American odds; **not** de-vigged here |
| `market` | const | `anytime_td` | `anytime_td` | |

`game_id` for Schema A is synthesized deterministically from `(season, week, away_team, home_team)`
so it is stable and joinable to schedules; Schema B's native id is kept as-is.

The canonical odds table holds **one row per `(game_id, player_id)`** — a single price per
player per game. All on-hand historical files are single-snapshot, so this is also their
natural shape; if a future source carries multiple intra-week snapshots, the latest
pre-kickoff snapshot wins. Line-history is out of scope for v1.

## Season/Week Provenance

The file path is the **authoritative** source of `(season, week)` for all years, because
Schema B has no such columns. Where Schema A's `Season`/`Week` columns exist, they are
validated against the path and a mismatch is surfaced as a data error rather than
silently trusting either side.

## Player Identity Resolution

The hardest join in the segment. Odds carry a display name (`Bijan Robinson`,
`Brian Robinson Jr.`, `C.J. Ham`) and **no reliable team** (Schema A's `Team` is `N/A`;
Schema B gives only home/away teams, not the player's). Downstream modeling needs the
nflverse `gsis_id`.

Resolution strategy:

1. **Bound the candidate pool by game context.** For a given odds row we know the season,
   week, and the two teams (home/away). The player is on one of those two rosters that
   week. Restricting candidates to those two weekly rosters turns a league-wide name
   match into a ~100-player disambiguation.
2. **Normalize names** on both sides (strip suffixes `Jr./Sr./II/III`, collapse initial
   punctuation `C.J.`→`CJ`, case/whitespace fold) before comparison.
3. **Match** exact-normalized first, then a bounded fuzzy fallback within the candidate
   pool.
4. **Resolve team** as a free byproduct: the matched roster row yields the player's team.
5. **Unmatched rows are retained** with null `player_id`/`team` and flagged, never dropped
   silently — the count of unmatched odds rows per week is a data-quality signal.

## nflverse Ingestion

Standard `nfl_data_py` pulls, normalized and cached locally:

- **Play-by-play** (`import_pbp_data`) — source of the **TD-scoring label** (did a player
  score a rushing or receiving TD in a game) and of usage events (carries, targets,
  red-zone and goal-line touches) consumed by `features`.
- **Weekly rosters** (`import_weekly_rosters`) — the candidate pool for identity
  resolution and the player→team→position map. Weekly (not seasonal) rosters are used so
  mid-season team changes resolve to the correct team for the week.
- **Schedules** (`import_schedules`) — game metadata, kickoff times, home/away; the join
  target for synthesized `game_id`.
- **Snap counts** (`import_snap_counts`) — snap-share inputs for `features`.

All pulls are normalized to nflverse team abbreviations and keyed by `(season, week, ...)`.

## Point-in-Time Guarantees

This segment is the first line of defense for the HLD's **Temporal Integrity** invariant.

- Every emitted row is attributable to a `(season, week)` and, where the source allows,
  a `snapshot_time`, so downstream segments can enforce the as-of boundary.
- The **TD-scoring label is emitted only for completed games** and is kept in a distinct
  field/table from features — it is a target, never a feature input.
- Play-by-play, snaps, and rosters are tagged with their source `(season, week)` so that
  `features` can restrict to weeks _< W_ without re-deriving provenance.
- This segment performs **no de-vigging and no edge computation** — it preserves raw
  American odds and a raw implied probability. De-vig method and bet-time selection are
  `betting`/`backtest` concerns, deliberately kept out so the canonical table stays a
  faithful record of what the book posted.
- **Scope:** regular season only (weeks 1–18); playoffs are out of scope for v1.

## Output Tables

The segment publishes, per season/week where applicable:

- `odds` — the canonical odds table above.
- `pbp` — normalized play-by-play.
- `rosters` — weekly rosters with `gsis_id`, team, position.
- `schedules` — game metadata.
- `snaps` — snap counts.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Schema unification | Normalize A and B into one canonical odds table at ingest | Branch on schema downstream; keep two tables | Confines schema drift to one place; downstream never forks on file format (HLD *one canonical shape*). |
| Season/week source | File path authoritative, validate columns when present | Trust in-file `Season`/`Week` | Schema B has no such columns; path is the only universally available source. |
| Identity resolution | Game-bounded roster join + name normalization + fuzzy fallback | League-wide fuzzy match; manual crosswalk | Home/away teams bound the candidate pool to ~2 rosters, making matching tractable and high-precision; team is recovered as a byproduct. |
| Unmatched odds rows | Retain with null ids, flag, count | Drop unmatched | Dropping hides coverage gaps and biases the backtest; the unmatched count is a quality signal. |
| `game_id` for Schema A | Synthesize deterministically from season/week/teams | Leave null; row-index id | Stable, joinable to schedules, reproducible across runs. |
| De-vig placement | Not here — raw odds preserved | De-vig at ingest | Keeps the canonical table a faithful record; de-vig method is a `betting` decision and varies (single-sided market). |
| Roster granularity | Weekly rosters | Seasonal rosters | Mid-season trades/signings resolve to the correct team for the week. |
| Odds granularity | One price per `(game_id, player_id)` | Full intra-week line history | All historical files are single-snapshot; line history is unneeded for v1 and complicates the canonical key. Latest pre-kickoff snapshot wins if a source ever has many. |
| Off-roster scorers | Unmatched (null id), retained + flagged, not bet | Crosswalk/fallback to resolve call-ups & just-signed players | Practice-squad elevations and just-signed players absent from the weekly roster are a small, accepted coverage gap for v1; the flag surfaces the rate. |
| Season scope | Regular season only (wk 1–18) | Include playoffs | Matches the on-hand odds coverage; playoff usage/role dynamics differ and add little v1 value. |

## Open Questions & Future Decisions

### Resolved
1. ✅ Historical odds are DraftKings-only, "Yes"-side only, 2023–2025 — backtest is
   single-book and single-sided; de-vig must use a field-normalization method rather than
   two-way (Yes/No) de-vig. **Cascade note → `betting`/`backtest`.**
2. ✅ File path is the authoritative season/week source.
3. ✅ One price per `(game_id, player_id)` — single snapshot, no line history in v1.

### Deferred
1. **No closing line in any historical season → no true CLV from on-hand data.**
   2023/2024 prices are a single *undated* number (can't anchor bet-time or close).
   2025 prices are a single *dated* mid-week snapshot (~2 days pre-kickoff) — clean and
   leak-free for ROI/calibration, but still not the closing line. True CLV requires
   capturing a near-kickoff closing snapshot going **forward** (live Odds API pulled
   close to kickoff). The historical backtest **fully supports value/edge testing** —
   model P(TD) vs. the book's de-vigged implied probability → +EV flagging → settled ROI
   and calibration on all three seasons. What it cannot produce is **CLV** (beating the
   *closing* line), a distinct, narrower metric needing a closing snapshot. CLV is not a
   prerequisite for value testing; the HLD's *CLV-is-primary* tenet governs live betting
   from this season on, while the historical backtest leans on value-ROI + calibration.
   `backtest` must also decide how to treat 2023/24's missing bet-time provenance
   (`snapshot_time IS NULL`). **Cascade note → `backtest`.**
2. **Concrete The Odds API wiring** (endpoint, plan tier, book selection) vs. interface-only
   in v1.
3. **Acceptable unmatched-rate threshold** for identity resolution before a week is
   considered low-quality.
4. **How many prior nflverse seasons to ingest for training** (HLD open question; training
   history may exceed the 2023–2025 odds window).

## EARS Facets (to be specified in Phase 3)

This leaf will own facets under the `DATA-` prefix:
- `DATA-ODDS-*` — historical odds ingestion + schema normalization.
- `DATA-NFL-*` — nflverse pulls (pbp, rosters, schedules, snaps).
- `DATA-MATCH-*` — player identity resolution.
- `DATA-NORM-*` — team-name normalization, season/week provenance, canonical keys.
- `DATA-PIT-*` — point-in-time facet of the *Temporal Integrity* invariant: provenance
  tagging, label-only-for-completed-games, regular-season scoping (the segment's leakage
  guard).

## References

- HLD: `docs/high-level-design.md`
- Historical odds: `historical_data/td_odds/<year>/week_<n>_td_odds.csv`
- `nfl_data_py` (nflverse) library
