"""Live Odds API anytime-TD snapshot fetcher.

Fetches player anytime-touchdown odds for the current NFL week from
The Odds API (https://the-odds-api.com/) and writes a flat CSV snapshot
to `config.odds_snapshot_path(season, week, tag)`.

Two-step API flow:
  1. GET /v4/sports/americanfootball_nfl/events
     -> list of this week's (and other weeks') games.
  2. GET /v4/sports/americanfootball_nfl/events/{id}/odds
     -> player_anytime_td market odds for one game.

The events list is filtered down to the target week using the
nflverse schedule (`week_window`) before step 2 is run, to avoid
spending API credits on games outside the requested week.
"""
import argparse
from datetime import datetime, timezone, timedelta
import math
import numbers
import os
import tempfile

import nflreadpy as nfl
import pandas as pd
import requests

import config

EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
EVENT_ODDS_URL_TMPL = (
    "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
)
MARKET = "player_anytime_td"

CSV_COLUMNS = [
    "game_id",
    "provider_event_id",
    "commence_time",
    "in_play",
    "bookmaker",
    "bookmaker_key",
    "last_update",
    "home_team",
    "away_team",
    "market",
    "label",
    "description",
    "price",
    "point",
    "season",
    "week",
    "tag",
    "fetched_at",
    "requested_bookmakers",
    "expected_game_count",
    "schema_version",
]

SCHEMA_VERSION = "2"


def _parse_commence_time(value: str) -> datetime:
    """Parse an Odds API ISO8601 UTC timestamp like '2026-09-13T17:00:00Z'."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a nonblank ISO8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _nonblank(value) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def _normalise_bookmakers(bookmakers) -> list[str]:
    if not isinstance(bookmakers, (list, tuple)):
        raise ValueError("bookmakers must be a non-empty list")
    normalized = [str(bookmaker).strip().lower() for bookmaker in bookmakers]
    if not normalized or any(not bookmaker for bookmaker in normalized):
        raise ValueError("requested bookmaker keys must be nonblank")
    if len(set(normalized)) != len(normalized):
        raise ValueError("requested bookmaker keys must be unique")
    return normalized


def _bookmaker_attestation(bookmakers: list[str]) -> str:
    return ",".join(sorted(bookmakers))


def _load_week_schedule(season: int, week: int, load_schedules) -> pd.DataFrame:
    loaded = load_schedules([season])
    schedule = loaded.to_pandas() if hasattr(loaded, "to_pandas") else loaded.copy()
    required = {"game_id", "home_team", "away_team", "gameday", "season", "week", "game_type"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ValueError(f"Schedule is missing required fields: {', '.join(missing)}")
    schedule = schedule[schedule["game_type"] == "REG"]
    week_df = schedule[(schedule["season"] == season) & (schedule["week"] == week)]
    if week_df.empty:
        raise ValueError(f"No scheduled games found for season={season} week={week}")
    for field in ("game_id", "home_team", "away_team", "gameday"):
        if week_df[field].map(_nonblank).eq(False).any():
            raise ValueError(f"Schedule field {field} must be nonblank")
    game_ids = week_df["game_id"].astype(str).str.strip()
    if game_ids.duplicated().any():
        raise ValueError("Schedule contains duplicate canonical game IDs")
    return week_df


def _window_for_schedule(week_df: pd.DataFrame) -> tuple:
    gamedays = [
        datetime.strptime(str(gd), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        for gd in week_df["gameday"]
    ]
    window_start = min(gamedays)
    # A Monday 8:15 p.m. Eastern kickoff is shortly after midnight UTC on
    # Tuesday. Keep a schedule-derived buffer so it is not lost at the UTC
    # date boundary while still excluding the next week's Thursday slate.
    window_end = max(gamedays) + timedelta(days=2)
    return window_start, window_end


def week_window(season: int, week: int, load_schedules=nfl.load_schedules) -> tuple:
    """Return (window_start, window_end) UTC datetimes bounding a season/week.

    window_start = min gameday for that week, at 00:00 UTC.
    window_end   = max gameday for that week, plus two days, at 00:00 UTC
                    (i.e. the window is half-open: [window_start, window_end)).
    """
    return _window_for_schedule(_load_week_schedule(season, week, load_schedules))


def _team_key(team, team_map: dict) -> str:
    mapped = team_map.get(team, team)
    return str(mapped).strip().upper().replace("LAR", "LA").replace("LVR", "LV")


def _matchup_key(home_team, away_team, team_map: dict) -> frozenset:
    return frozenset({
        _team_key(home_team, team_map),
        _team_key(away_team, team_map),
    })


def _scheduled_matchups(schedule: pd.DataFrame, team_map: dict) -> set | None:
    if not {"home_team", "away_team"}.issubset(schedule.columns):
        return None
    matchups = {
        _matchup_key(row["home_team"], row["away_team"], team_map)
        for _, row in schedule.iterrows()
        if pd.notna(row["home_team"]) and pd.notna(row["away_team"])
    }
    return matchups or None


def _scheduled_game_ids(schedule: pd.DataFrame, team_map: dict) -> dict:
    """Map a normalized schedule matchup to its canonical nflverse game ID."""
    game_ids = {}
    for _, row in schedule.iterrows():
        matchup = _matchup_key(row["home_team"], row["away_team"], team_map)
        game_id = str(row["game_id"]).strip()
        if matchup in game_ids and game_ids[matchup] != game_id:
            raise ValueError("Schedule contains duplicate canonical matchups")
        game_ids[matchup] = game_id
    return game_ids


def _load_team_map() -> dict:
    teams_path = config.DATA_DIR / "nfl_teams.csv"
    if not teams_path.exists():
        return {}
    teams = pd.read_csv(teams_path, usecols=["team_name", "team_id"])
    return dict(zip(teams["team_name"], teams["team_id"]))


def _load_cached_snapshot(
    out_path, expected_season: int, expected_week: int, expected_tag: str
) -> pd.DataFrame | None:
    """Read and validate an existing snapshot, or return None when absent."""
    if not out_path.exists():
        return None
    if out_path.stat().st_size == 0:
        raise ValueError(
            f"Odds snapshot is zero bytes: {out_path}. Remove it or request a refresh."
        )

    try:
        cached = pd.read_csv(out_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Could not parse odds snapshot {out_path}: {exc}. "
            "Remove it or request a refresh."
        ) from exc

    missing = [column for column in CSV_COLUMNS if column not in cached.columns]
    if missing:
        raise ValueError(
            f"Odds snapshot {out_path} is missing required columns: {', '.join(missing)}. "
            "Remove it or request a refresh."
        )

    if cached.empty:
        raise ValueError(f"Odds snapshot {out_path} is header-only or empty")
    for column, expected in (("season", expected_season), ("week", expected_week)):
        values = pd.to_numeric(cached[column], errors="coerce")
        if values.isna().any() or not values.eq(expected).all():
            raise ValueError(f"Odds snapshot metadata does not match requested {column}={expected}")
    if cached["tag"].isna().any() or not cached["tag"].eq(expected_tag).all():
        raise ValueError("Odds snapshot tag metadata does not match the request")
    return cached


def _validate_cached_snapshot(
    cached: pd.DataFrame,
    expected_season: int,
    expected_week: int,
    expected_tag: str,
    expected_game_ids: set[str],
    requested_bookmakers: list[str],
    expected_matchups: dict[str, frozenset] | None = None,
    team_map: dict | None = None,
) -> pd.DataFrame:
    """Validate all attestations and quote semantics before cache reuse."""
    expected_attestation = _bookmaker_attestation(requested_bookmakers)
    for column, expected in (("season", expected_season), ("week", expected_week)):
        values = pd.to_numeric(cached[column], errors="coerce")
        if values.isna().any() or not values.eq(expected).all():
            raise ValueError(f"Odds snapshot metadata does not match requested {column}={expected}")
    if cached["tag"].isna().any() or not cached["tag"].eq(expected_tag).all():
        raise ValueError("Odds snapshot tag metadata does not match the request")

    if cached["requested_bookmakers"].isna().any() or not cached["requested_bookmakers"].eq(expected_attestation).all():
        raise ValueError("Odds snapshot requested bookmaker attestation does not match")
    count = pd.to_numeric(cached["expected_game_count"], errors="coerce")
    if count.isna().any() or not count.eq(len(expected_game_ids)).all():
        raise ValueError("Odds snapshot expected game count attestation is invalid")
    if cached["schema_version"].isna().any() or not cached["schema_version"].astype(str).eq(SCHEMA_VERSION).all():
        raise ValueError("Odds snapshot schema version is unsupported")

    actual_ids = set(cached["game_id"].astype(str).str.strip())
    if actual_ids != expected_game_ids:
        raise ValueError("Odds snapshot canonical game coverage is incomplete or unexpected")
    if cached["provider_event_id"].map(_nonblank).eq(False).any():
        raise ValueError("Odds snapshot has blank provider event IDs")
    if cached["bookmaker_key"].map(_nonblank).eq(False).any():
        raise ValueError("Odds snapshot has blank bookmaker keys")
    provider_ids = cached["provider_event_id"].astype(str).str.strip()
    normalized_cache = cached.assign(_normalized_provider_event_id=provider_ids)
    provider_mapping = normalized_cache.groupby("_normalized_provider_event_id")["game_id"].nunique()
    if provider_mapping.gt(1).any():
        raise ValueError("Odds snapshot provider events map to multiple canonical games")
    canonical_provider_counts = normalized_cache.groupby("game_id")["_normalized_provider_event_id"].nunique()
    if canonical_provider_counts.ne(1).any():
        raise ValueError("Odds snapshot canonical games map to multiple provider events")
    if set(cached["bookmaker_key"].astype(str)) != set(requested_bookmakers):
        raise ValueError("Odds snapshot bookmaker coverage does not match the request")

    for index, row in cached.iterrows():
        _validate_normalized_row(row, index=index)
        if expected_matchups is not None and _matchup_key(row["home_team"], row["away_team"], team_map or {}) != expected_matchups[str(row["game_id"]).strip()]:
            raise ValueError("Odds snapshot canonical game matchup identity is invalid")
    for game_id in expected_game_ids:
        for bookmaker_key in requested_bookmakers:
            mask = (cached["game_id"].astype(str) == game_id) & (cached["bookmaker_key"] == bookmaker_key)
            if not mask.any():
                raise ValueError("Odds snapshot is missing a game/bookmaker quote")
    return cached


def select_week_events(
    events: list,
    window_start: datetime,
    window_end: datetime,
    scheduled_matchups: set | None = None,
    team_map: dict | None = None,
) -> list:
    """Keep in-window events, optionally requiring a scheduled matchup."""
    selected = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if scheduled_matchups:
            if not _nonblank(event.get("home_team")) or not _nonblank(event.get("away_team")):
                continue
            matchup = _matchup_key(
                event.get("home_team"), event.get("away_team"), team_map or {}
            )
            if matchup not in scheduled_matchups:
                continue
        elif not _nonblank(event.get("id")):
            continue
        commence_dt = _parse_commence_time(event.get("commence_time"))
        if not window_start <= commence_dt < window_end:
            continue
        selected.append(event)
    return selected


def _build_match_plan(
    events: list,
    week_schedule: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
    team_map: dict,
    require_full: bool,
) -> list[dict]:
    """Build and validate the provider-event to canonical-game plan."""
    scheduled_ids = _scheduled_game_ids(week_schedule, team_map)
    selected = select_week_events(
        events, window_start, window_end,
        scheduled_matchups=set(scheduled_ids), team_map=team_map,
    )
    by_game = {}
    by_provider = {}
    for event in selected:
        raw_provider_id = event.get("id")
        if not _nonblank(raw_provider_id):
            raise ValueError("provider event ID must be nonblank")
        provider_id = str(raw_provider_id).strip()
        canonical_id = scheduled_ids[_matchup_key(event["home_team"], event["away_team"], team_map)]
        if provider_id in by_provider:
            raise ValueError("multiple provider events or duplicate provider event IDs are not allowed")
        if canonical_id in by_game:
            raise ValueError("Multiple provider events map to one canonical game")
        record = {
            "event": event,
            "provider_event_id": provider_id,
            "canonical_game_id": canonical_id,
        }
        by_provider[provider_id] = record
        by_game[canonical_id] = record
    expected = set(scheduled_ids.values())
    if require_full and set(by_game) != expected:
        missing = sorted(expected - set(by_game))
        raise ValueError(f"Provider event index does not map every scheduled game: {missing}")
    return sorted(by_game.values(), key=lambda record: (record["canonical_game_id"], record["provider_event_id"]))


def _validate_event_identity(selected_event: dict, response_event: dict, record: dict, team_map: dict):
    if not isinstance(response_event, dict):
        raise ValueError("Paid odds response must be one event object")
    if str(response_event.get("id", "")).strip() != record["provider_event_id"]:
        raise ValueError("Paid odds response provider event ID disagrees with preflight")
    for field in ("home_team", "away_team"):
        if not _nonblank(response_event.get(field)):
            raise ValueError(f"Paid odds response has blank {field}")
    if _matchup_key(response_event["home_team"], response_event["away_team"], team_map) != _matchup_key(selected_event["home_team"], selected_event["away_team"], team_map):
        raise ValueError("Paid odds response matchup disagrees with preflight")
    if _parse_commence_time(response_event.get("commence_time")) != _parse_commence_time(selected_event.get("commence_time")):
        raise ValueError("Paid odds response kickoff disagrees with preflight")


def _valid_american_price(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        return False
    if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
        return False
    return float(value).is_integer() and (value <= -100 or value >= 100)


def _validate_normalized_row(row, index=None):
    prefix = f"row {index}: " if index is not None else ""
    if not _nonblank(row.get("game_id")):
        raise ValueError(prefix + "canonical game_id is required")
    if not _nonblank(row.get("provider_event_id")):
        raise ValueError(prefix + "provider event ID is required")
    if str(row.get("game_id")).strip() == str(row.get("provider_event_id")).strip():
        raise ValueError(prefix + "canonical and provider event IDs must remain distinct")
    for field in ("home_team", "away_team", "description", "bookmaker", "bookmaker_key", "market", "label"):
        if not _nonblank(row.get(field)):
            raise ValueError(prefix + f"{field} is required")
    if row.get("market") != MARKET or row.get("label") != "Yes":
        raise ValueError(prefix + "unexpected market outcome")
    if pd.isna(row.get("in_play")) or bool(row.get("in_play")):
        raise ValueError(prefix + "in_play must be explicitly false")
    commence = _parse_commence_time(row.get("commence_time"))
    last_update = _parse_commence_time(row.get("last_update"))
    fetched_at = _parse_commence_time(row.get("fetched_at"))
    if last_update >= commence or fetched_at >= commence:
        raise ValueError(prefix + "quote timestamps must be pregame")
    if not _valid_american_price(row.get("price")):
        raise ValueError(prefix + "American price must be a finite integer at least 100 in magnitude")


def parse_event_odds(
    event: dict,
    season: int,
    week: int,
    tag: str,
    fetched_at: str,
    canonical_game_id: str | None = None,
    requested_bookmakers: set[str] | None = None,
    expected_game_count: int | None = None,
) -> list:
    """Flatten one event-odds API response into player_anytime_td 'Yes' rows."""
    if not _nonblank(canonical_game_id):
        raise ValueError("canonical game_id is required; provider event IDs are not canonical IDs")
    provider_event_id = event.get("id")
    if not _nonblank(provider_event_id):
        raise ValueError("provider event ID is required")
    game_id = str(canonical_game_id).strip()
    provider_event_id = str(provider_event_id).strip()
    commence_time = event.get("commence_time")
    _parse_commence_time(commence_time)
    home_team = event.get("home_team")
    away_team = event.get("away_team")
    if not _nonblank(home_team) or not _nonblank(away_team):
        raise ValueError("event matchup teams are required")
    fetched_dt = _parse_commence_time(fetched_at)
    commence_dt = _parse_commence_time(commence_time)
    if fetched_dt >= commence_dt:
        raise ValueError("fetched_at must be pregame")

    rows = []
    for bookmaker in event.get("bookmakers", []):
        if not isinstance(bookmaker, dict):
            continue
        bookmaker_key = bookmaker.get("key")
        bookmaker_title = bookmaker.get("title")
        for market in bookmaker.get("markets", []):
            if not isinstance(market, dict):
                continue
            if market.get("key") != MARKET:
                continue
            last_update = market.get("last_update")
            for outcome in market.get("outcomes", []):
                if not isinstance(outcome, dict):
                    continue
                if outcome.get("name") != "Yes":
                    continue
                row = {
                    "game_id": game_id,
                    "provider_event_id": provider_event_id,
                    "commence_time": commence_time,
                    "in_play": False,
                    "bookmaker": bookmaker_title,
                    "bookmaker_key": str(bookmaker_key).strip().lower() if _nonblank(bookmaker_key) else bookmaker_key,
                    "last_update": last_update,
                    "home_team": str(home_team).strip(),
                    "away_team": str(away_team).strip(),
                    "market": market.get("key"),
                    "label": outcome.get("name"),
                    "description": outcome.get("description"),
                    "price": outcome.get("price"),
                    "point": outcome.get("point"),
                    "season": season,
                    "week": week,
                    "tag": tag,
                    "fetched_at": fetched_at,
                    "requested_bookmakers": _bookmaker_attestation(sorted(requested_bookmakers)) if requested_bookmakers is not None else "",
                    "expected_game_count": expected_game_count if expected_game_count is not None else "",
                    "schema_version": SCHEMA_VERSION,
                }
                _validate_normalized_row(row)
                if requested_bookmakers is not None and str(bookmaker_key).strip().lower() not in requested_bookmakers:
                    continue
                rows.append(row)
    return rows


def fetch(
    season: int,
    week: int,
    tag: str,
    bookmakers: list,
    get=requests.get,
    load_schedules=nfl.load_schedules,
    refresh: bool = False,
    canary_one_event: bool = False,
) -> pd.DataFrame:
    """Load a cached snapshot or fetch and publish a new one.

    Existing valid snapshots are returned without resolving credentials or
    touching HTTP dependencies. Set ``refresh=True`` to explicitly replace
    one. Publication is atomic, so a failed refresh leaves any prior snapshot
    untouched.
    """
    bookmakers = _normalise_bookmakers(bookmakers)
    if canary_one_event and refresh:
        raise ValueError("--canary-one-event cannot be combined with --refresh")
    out_path = config.odds_snapshot_path(season, week, tag)

    cached = None
    if not refresh and not canary_one_event:
        cached = _load_cached_snapshot(out_path, season, week, tag)
        if cached is not None:
            week_schedule = _load_week_schedule(season, week, load_schedules)
            team_map = _load_team_map()
            return _validate_cached_snapshot(
                cached, season, week, tag,
                expected_game_ids=set(_scheduled_game_ids(week_schedule, team_map).values()),
                requested_bookmakers=bookmakers,
                expected_matchups={
                    str(row["game_id"]).strip(): _matchup_key(row["home_team"], row["away_team"], team_map)
                    for _, row in week_schedule.iterrows()
                },
                team_map=team_map,
            )

    api_key = config.odds_api_key()
    week_schedule = _load_week_schedule(season, week, load_schedules)
    window_start, window_end = _window_for_schedule(week_schedule)
    team_map = _load_team_map()

    events_resp = get(EVENTS_URL, params={"apiKey": api_key})
    events_resp.raise_for_status()
    events = events_resp.json()

    plan = _build_match_plan(
        events, week_schedule, window_start, window_end, team_map,
        require_full=not canary_one_event,
    )
    if not plan:
        raise ValueError("No matched provider event found for requested week")
    if canary_one_event:
        plan = plan[:1]
    scheduled_game_ids = _scheduled_game_ids(week_schedule, team_map)

    bookmakers_str = ",".join(bookmakers)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    last_resp = events_resp
    for record in plan:
        event = record["event"]
        url = EVENT_ODDS_URL_TMPL.format(event_id=record["provider_event_id"])
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": MARKET,
            "oddsFormat": "american",
            "bookmakers": bookmakers_str,
        }
        resp = get(url, params=params)
        resp.raise_for_status()
        event_odds = resp.json()
        _validate_event_identity(event, event_odds, record, team_map)
        rows.extend(
            parse_event_odds(
                event_odds,
                season,
                week,
                tag,
                fetched_at,
                canonical_game_id=record["canonical_game_id"],
                requested_bookmakers=set(bookmakers),
                expected_game_count=len(scheduled_game_ids),
            )
        )
        last_resp = resp

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    if df.empty:
        raise ValueError("Paid odds responses contained no valid Yes quotes")
    for index, row in df.iterrows():
        _validate_normalized_row(row, index=index)
    if not canary_one_event:
        expected_ids = set(scheduled_game_ids.values())
        actual_ids = set(df["game_id"].astype(str))
        if actual_ids != expected_ids:
            raise ValueError("Full weekly snapshot is missing canonical games")
        for game_id in expected_ids:
            for bookmaker_key in bookmakers:
                mask = (df["game_id"] == game_id) & (df["bookmaker_key"] == bookmaker_key)
                if not mask.any():
                    raise ValueError("Full weekly snapshot is missing a requested bookmaker quote")

    if not canary_one_event:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=out_path.parent,
                prefix=f".{out_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
            df.to_csv(temp_path, index=False)
            os.replace(temp_path, out_path)
            temp_path = None
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    used = last_resp.headers.get("x-requests-used")
    remaining = last_resp.headers.get("x-requests-remaining")
    print(f"x-requests-used: {used}  x-requests-remaining: {remaining}")

    return df


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch a live anytime-TD odds snapshot from The Odds API"
    )
    parser.add_argument("--season", type=int, default=config.SEASON)
    parser.add_argument("--week", type=int, default=config.WEEK)
    parser.add_argument("--tag", required=True, choices=["open", "close"])
    parser.add_argument("--bookmakers", default="draftkings")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch and replace the cached snapshot (spends API credits)",
    )
    parser.add_argument(
        "--canary-one-event",
        action="store_true",
        help="Fetch and validate one deterministic event without publishing a snapshot",
    )
    args = parser.parse_args(argv)

    bookmakers = [b.strip() for b in args.bookmakers.split(",")]
    kwargs = {"refresh": args.refresh}
    if args.canary_one_event:
        kwargs["canary_one_event"] = True
    fetch(args.season, args.week, args.tag, bookmakers, **kwargs)


if __name__ == "__main__":
    main()
