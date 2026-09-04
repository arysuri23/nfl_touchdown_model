"""Shared player-name / team-in-game matching between prediction rows and
sportsbook odds rows.

Used by both `predict_wr.join_odds` (predictions vs. open odds) and
`ledger.attach_closing` (unsettled ledger rows vs. close odds) so both call
sites resolve ambiguous names (e.g. two active "Mike Williams") the same
way: normalise the name, then require the player's team to be one of the
two teams playing in that odds row's game.
"""
import re

import pandas as pd

_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]")
_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv)\s*$")


def merge_name(s: pd.Series) -> pd.Series:
    """Normalise player names for joining: lowercase, strip punctuation,
    drop a trailing generational suffix (jr/sr/ii/iii/iv), strip whitespace.
    """
    out = s.astype(str).str.lower()
    out = out.str.replace(_NON_ALNUM_SPACE_RE, "", regex=True)
    out = out.str.replace(_SUFFIX_RE, "", regex=True)
    out = out.str.strip()
    return out


def match_odds_to_players(players: pd.DataFrame, odds: pd.DataFrame, team_map: dict) -> pd.DataFrame:
    """Join `players` to `odds` on normalised name, keeping only odds rows
    whose game (home_team/away_team, mapped through `team_map` to abbrs)
    includes the player's `team`. When several bookmakers match the same
    player, keeps the highest `price`. Returns `players`' original columns
    plus `price` and `bookmaker`; players with no match are dropped.
    """
    player_cols = list(players.columns)

    left = players.reset_index(drop=True).copy()
    left["_pidx"] = left.index
    left["_merge_name"] = merge_name(left["player_display_name"])

    # Keep only the odds columns this function actually needs. Real odds
    # snapshots (from fetch_live_odds.py, or the legacy-CSV conversion) carry
    # their own `season`/`week` columns that collide with `players`' -- an
    # unrestricted merge would suffix both sides (season_x/season_y) and
    # break the `player_cols` lookup below.
    odds_columns = ["description", "home_team", "away_team", "price", "bookmaker"]
    if "game_id" in odds.columns:
        odds_columns.append("game_id")
    right = odds[odds_columns].copy()
    right["_merge_name"] = merge_name(right["description"])
    # team_map (built from data/nfl_teams.csv) maps "Los Angeles Rams" / "Las
    # Vegas Raiders" to the legacy "LAR"/"LVR" codes, but rosters and
    # schedule-derived lines (data_collection.schedule_to_team_lines) both
    # use the modern "LA"/"LV" codes -- normalise here the same way, or
    # every Rams/Raiders player fails the in-game check below and is
    # silently dropped from the matched output.
    right["_home_abbr"] = right["home_team"].map(team_map).replace({"LAR": "LA", "LVR": "LV"})
    right["_away_abbr"] = right["away_team"].map(team_map).replace({"LAR": "LA", "LVR": "LV"})

    merged = left.merge(right, on="_merge_name", how="inner")
    in_game = (merged["team"] == merged["_home_abbr"]) | (merged["team"] == merged["_away_abbr"])
    merged = merged[in_game]

    merged = merged.sort_values("price", ascending=False)
    merged = merged.drop_duplicates(subset="_pidx", keep="first")
    merged = merged.sort_values("_pidx")

    result_columns = player_cols + ["price", "bookmaker"]
    if "game_id" in right.columns:
        result_columns.append("game_id")
    result = merged[result_columns].reset_index(drop=True)
    return result
