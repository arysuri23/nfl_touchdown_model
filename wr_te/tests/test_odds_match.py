import pandas as pd
import pytest

import odds_match


TEAM_MAP = {
    "New York Jets": "NYJ",
    "Buffalo Bills": "BUF",
    "Los Angeles Chargers": "LAC",
    "Denver Broncos": "DEN",
}


def test_merge_name_normalises_punctuation_case_and_suffix():
    s = pd.Series(["Mike Williams", "A.J. Brown", "Odell Beckham Jr.", "Michael Pittman II"])
    out = odds_match.merge_name(s)
    assert list(out) == ["mike williams", "aj brown", "odell beckham", "michael pittman"]


def test_merge_name_strips_leading_trailing_whitespace():
    out = odds_match.merge_name(pd.Series([" Mike Williams  Sr. "]))
    assert list(out) == ["mike williams"]


def _two_mike_williams_odds():
    # Two different real players share the name "Mike Williams": one plays
    # for NYJ, one for LAC, in two separate games this week.
    return pd.DataFrame([
        {
            "description": "Mike Williams",
            "home_team": "New York Jets",
            "away_team": "Buffalo Bills",
            "price": 350,
            "bookmaker": "book1",
        },
        {
            "description": "Mike Williams",
            "home_team": "Los Angeles Chargers",
            "away_team": "Denver Broncos",
            "price": 280,
            "bookmaker": "book1",
        },
    ])


def _players():
    return pd.DataFrame([
        {"player_id": "P1", "player_display_name": "Mike Williams", "team": "NYJ"},
        {"player_id": "P2", "player_display_name": "Mike Williams", "team": "LAC"},
    ])


def test_match_odds_to_players_disambiguates_by_team_in_game():
    matched = odds_match.match_odds_to_players(_players(), _two_mike_williams_odds(), TEAM_MAP)

    assert len(matched) == 2
    nyj = matched[matched["team"] == "NYJ"].iloc[0]
    lac = matched[matched["team"] == "LAC"].iloc[0]
    assert nyj["price"] == 350
    assert lac["price"] == 280


def test_match_odds_to_players_keeps_highest_price_across_bookmakers():
    odds = _two_mike_williams_odds()
    # Second bookmaker offers a better (higher) price for the NYJ game only.
    odds = pd.concat([odds, pd.DataFrame([{
        "description": "Mike Williams",
        "home_team": "New York Jets",
        "away_team": "Buffalo Bills",
        "price": 375,
        "bookmaker": "book2",
    }])], ignore_index=True)

    matched = odds_match.match_odds_to_players(_players(), odds, TEAM_MAP)

    assert len(matched) == 2
    nyj = matched[matched["team"] == "NYJ"].iloc[0]
    lac = matched[matched["team"] == "LAC"].iloc[0]
    assert nyj["price"] == 375
    assert nyj["bookmaker"] == "book2"
    assert lac["price"] == 280


def test_match_odds_to_players_drops_players_without_a_match():
    players = pd.concat([_players(), pd.DataFrame([
        {"player_id": "P3", "player_display_name": "Nobody Here", "team": "SF"},
    ])], ignore_index=True)

    matched = odds_match.match_odds_to_players(players, _two_mike_williams_odds(), TEAM_MAP)

    assert len(matched) == 2
    assert "P3" not in matched["player_id"].values


def test_match_odds_to_players_preserves_extra_player_columns():
    players = _players()
    players["position"] = ["WR", "WR"]

    matched = odds_match.match_odds_to_players(players, _two_mike_williams_odds(), TEAM_MAP)

    assert "position" in matched.columns
    assert set(matched["position"]) == {"WR"}
