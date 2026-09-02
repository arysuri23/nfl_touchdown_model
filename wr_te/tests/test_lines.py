import pandas as pd
import polars as pl
import pytest

import data_collection


def _synthetic_schedule_pd():
    return pd.DataFrame([
        # Row A: KC (home) favored by 3, total 46
        {
            "season": 2024, "week": 1, "game_type": "REG",
            "home_team": "KC", "away_team": "BAL",
            "spread_line": 3.0, "total_line": 46.0,
        },
        # Row B: LA (home) is the underdog (spread_line negative => away favored)
        {
            "season": 2024, "week": 1, "game_type": "REG",
            "home_team": "LA", "away_team": "SF",
            "spread_line": -2.5, "total_line": 48.5,
        },
        # Row C: playoff game, should be excluded
        {
            "season": 2024, "week": 19, "game_type": "WC",
            "home_team": "KC", "away_team": "BUF",
            "spread_line": 1.5, "total_line": 44.0,
        },
        # Row D: null spread_line, should be excluded
        {
            "season": 2024, "week": 2, "game_type": "REG",
            "home_team": "DAL", "away_team": "NYG",
            "spread_line": float("nan"), "total_line": 41.0,
        },
    ])


def test_schedule_to_team_lines_kc_bal():
    sched = _synthetic_schedule_pd()
    out = data_collection.schedule_to_team_lines(sched)

    kc = out[(out["team"] == "KC") & (out["season"] == 2024) & (out["week"] == 1)].iloc[0]
    assert kc["spread_line"] == -3.0
    assert kc["implied_total"] == 24.5
    assert kc["opponent"] == "BAL"

    bal = out[(out["team"] == "BAL") & (out["season"] == 2024) & (out["week"] == 1)].iloc[0]
    assert bal["spread_line"] == 3.0
    assert bal["implied_total"] == 21.5
    assert bal["opponent"] == "KC"


def test_schedule_to_team_lines_la_sf():
    sched = _synthetic_schedule_pd()
    out = data_collection.schedule_to_team_lines(sched)

    la = out[(out["team"] == "LA") & (out["season"] == 2024) & (out["week"] == 1)].iloc[0]
    assert la["spread_line"] == 2.5
    assert la["implied_total"] == 23.0

    sf = out[(out["team"] == "SF") & (out["season"] == 2024) & (out["week"] == 1)].iloc[0]
    assert sf["spread_line"] == -2.5
    assert sf["implied_total"] == 25.5


def test_implied_totals_sum_to_total_line():
    sched = _synthetic_schedule_pd()
    out = data_collection.schedule_to_team_lines(sched)

    # Pair up team/opponent rows per game and check implied totals sum to total_line
    for _, row in out.iterrows():
        opp_row = out[
            (out["season"] == row["season"])
            & (out["week"] == row["week"])
            & (out["team"] == row["opponent"])
            & (out["opponent"] == row["team"])
        ].iloc[0]
        assert row["implied_total"] + opp_row["implied_total"] == pytest.approx(row["total_line"])


def test_playoff_and_null_rows_excluded():
    sched = _synthetic_schedule_pd()
    out = data_collection.schedule_to_team_lines(sched)

    assert out[out["week"] == 19].empty
    assert out[out["week"] == 2].empty
    assert len(out) == 4  # 2 games * 2 teams each


def test_get_odds_data_matches_pure_function():
    sched_pd = _synthetic_schedule_pd()

    def fake_load_schedules(years):
        return pl.from_pandas(sched_pd)

    expected = data_collection.schedule_to_team_lines(
        sched_pd[sched_pd["game_type"] == "REG"]
    )
    expected = expected[expected["week"] <= 18].reset_index(drop=True)

    out = data_collection.get_odds_data([2024], load_schedules=fake_load_schedules)
    out = out.reset_index(drop=True)

    pd.testing.assert_frame_equal(
        out.sort_values(["season", "week", "team"]).reset_index(drop=True),
        expected.sort_values(["season", "week", "team"]).reset_index(drop=True),
    )


def test_get_week_lines_filters_to_one_week():
    sched_pd = _synthetic_schedule_pd()

    def fake_load_schedules(years):
        return pl.from_pandas(sched_pd)

    out = data_collection.get_week_lines(2024, 1, load_schedules=fake_load_schedules)
    assert len(out) == 4
    assert (out["week"] == 1).all()


def test_lar_lvr_renamed_to_la_lv():
    sched = pd.DataFrame([
        {
            "season": 2024, "week": 1, "game_type": "REG",
            "home_team": "LAR", "away_team": "LVR",
            "spread_line": 1.0, "total_line": 40.0,
        },
    ])
    out = data_collection.schedule_to_team_lines(sched)
    assert set(out["team"]) == {"LA", "LV"}
    assert "LAR" not in out["team"].values
    assert "LVR" not in out["team"].values
