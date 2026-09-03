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
    "commence_time",
    "in_play",
    "bookmaker",
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
]


def _parse_commence_time(value: str) -> datetime:
    """Parse an Odds API ISO8601 UTC timestamp like '2026-09-13T17:00:00Z'."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _load_week_schedule(season: int, week: int, load_schedules) -> pd.DataFrame:
    schedule = load_schedules([season]).to_pandas()
    schedule = schedule[schedule["game_type"] == "REG"]
    week_df = schedule[(schedule["season"] == season) & (schedule["week"] == week)]
    if week_df.empty:
        raise ValueError(f"No scheduled games found for season={season} week={week}")
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
    required = {"game_id", "home_team", "away_team"}
    if not required.issubset(schedule.columns):
        return {}

    game_ids = {}
    for _, row in schedule.iterrows():
        if (
            pd.isna(row["game_id"])
            or pd.isna(row["home_team"])
            or pd.isna(row["away_team"])
        ):
            continue
        game_ids[_matchup_key(row["home_team"], row["away_team"], team_map)] = str(
            row["game_id"]
        )
    return game_ids


def _load_team_map() -> dict:
    teams_path = config.DATA_DIR / "nfl_teams.csv"
    if not teams_path.exists():
        return {}
    teams = pd.read_csv(teams_path, usecols=["team_name", "team_id"])
    return dict(zip(teams["team_name"], teams["team_id"]))


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
        commence_dt = _parse_commence_time(event["commence_time"])
        if not window_start <= commence_dt < window_end:
            continue
        if scheduled_matchups:
            matchup = _matchup_key(
                event.get("home_team"), event.get("away_team"), team_map or {}
            )
            if matchup not in scheduled_matchups:
                continue
        selected.append(event)
    return selected


def parse_event_odds(
    event: dict,
    season: int,
    week: int,
    tag: str,
    fetched_at: str,
    canonical_game_id: str | None = None,
) -> list:
    """Flatten one event-odds API response into player_anytime_td 'Yes' rows."""
    # The provider ID is only an API endpoint key.  Persist the schedule's
    # nflverse game ID when the caller matched the event to a scheduled game;
    # falling back to the provider ID preserves legacy direct-parser callers.
    game_id = canonical_game_id if canonical_game_id is not None else event.get("id")
    commence_time = event.get("commence_time")
    home_team = event.get("home_team")
    away_team = event.get("away_team")

    rows = []
    for bookmaker in event.get("bookmakers", []):
        bookmaker_title = bookmaker.get("title")
        for market in bookmaker.get("markets", []):
            if market.get("key") != MARKET:
                continue
            last_update = market.get("last_update")
            for outcome in market.get("outcomes", []):
                if outcome.get("name") != "Yes":
                    continue
                rows.append(
                    {
                        "game_id": game_id,
                        "commence_time": commence_time,
                        "in_play": False,
                        "bookmaker": bookmaker_title,
                        "last_update": last_update,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market": market.get("key"),
                        "label": outcome.get("name"),
                        "description": outcome.get("description"),
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                        "season": season,
                        "week": week,
                        "tag": tag,
                        "fetched_at": fetched_at,
                    }
                )
    return rows


def fetch(
    season: int,
    week: int,
    tag: str,
    bookmakers: list,
    get=requests.get,
    load_schedules=nfl.load_schedules,
) -> pd.DataFrame:
    """Fetch a live ATD odds snapshot and write it to the season/week/tag CSV."""
    api_key = config.odds_api_key()

    week_schedule = _load_week_schedule(season, week, load_schedules)
    window_start, window_end = _window_for_schedule(week_schedule)
    team_map = _load_team_map()
    scheduled_matchups = _scheduled_matchups(week_schedule, team_map)
    scheduled_game_ids = _scheduled_game_ids(week_schedule, team_map)

    events_resp = get(EVENTS_URL, params={"apiKey": api_key})
    events_resp.raise_for_status()
    events = events_resp.json()

    week_events = select_week_events(
        events,
        window_start,
        window_end,
        scheduled_matchups=scheduled_matchups,
        team_map=team_map,
    )

    bookmakers_str = ",".join(bookmakers)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    last_resp = events_resp
    for event in week_events:
        url = EVENT_ODDS_URL_TMPL.format(event_id=event["id"])
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
        canonical_game_id = scheduled_game_ids.get(
            _matchup_key(event.get("home_team"), event.get("away_team"), team_map)
        )
        rows.extend(
            parse_event_odds(
                event_odds,
                season,
                week,
                tag,
                fetched_at,
                canonical_game_id=canonical_game_id,
            )
        )
        last_resp = resp

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    out_path = config.odds_snapshot_path(season, week, tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

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
    args = parser.parse_args(argv)

    bookmakers = [b.strip() for b in args.bookmakers.split(",") if b.strip()]
    fetch(args.season, args.week, args.tag, bookmakers)


if __name__ == "__main__":
    main()
