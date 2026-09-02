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


def week_window(season: int, week: int, load_schedules=nfl.load_schedules) -> tuple:
    """Return (window_start, window_end) UTC datetimes bounding a season/week.

    window_start = min gameday for that week, at 00:00 UTC.
    window_end   = max gameday for that week, plus one day, at 00:00 UTC
                    (i.e. the window is half-open: [window_start, window_end)).
    """
    schedule = load_schedules([season]).to_pandas()
    schedule = schedule[schedule["game_type"] == "REG"]
    week_df = schedule[(schedule["season"] == season) & (schedule["week"] == week)]

    if week_df.empty:
        raise ValueError(f"No scheduled games found for season={season} week={week}")

    gamedays = [
        datetime.strptime(str(gd), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        for gd in week_df["gameday"]
    ]
    window_start = min(gamedays)
    window_end = max(gamedays) + timedelta(days=1)
    return window_start, window_end


def select_week_events(events: list, window_start: datetime, window_end: datetime) -> list:
    """Keep events whose commence_time falls in [window_start, window_end)."""
    selected = []
    for event in events:
        commence_dt = _parse_commence_time(event["commence_time"])
        if window_start <= commence_dt < window_end:
            selected.append(event)
    return selected


def parse_event_odds(event: dict, season: int, week: int, tag: str, fetched_at: str) -> list:
    """Flatten one event-odds API response into player_anytime_td 'Yes' rows."""
    game_id = event.get("id")
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

    window_start, window_end = week_window(season, week, load_schedules=load_schedules)

    events_resp = get(EVENTS_URL, params={"apiKey": api_key})
    events_resp.raise_for_status()
    events = events_resp.json()

    week_events = select_week_events(events, window_start, window_end)

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
        rows.extend(parse_event_odds(event_odds, season, week, tag, fetched_at))
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
