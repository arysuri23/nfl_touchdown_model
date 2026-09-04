import datetime as dt

import pandas as pd
import polars as pl
import pytest

import data_collection


def d(year, month, day):
    return dt.date(year, month, day)


def test_select_week_snapshots_picks_latest_snapshot_before_gameday():
    snapshot_dates = [d(2026, 9, 1), d(2026, 9, 2), d(2026, 9, 8), d(2026, 9, 15)]
    week_first_gameday = {1: d(2026, 9, 9), 2: d(2026, 9, 17), 3: d(2026, 9, 24)}

    out = data_collection.select_week_snapshots(snapshot_dates, week_first_gameday)

    assert out == {1: d(2026, 9, 8), 2: d(2026, 9, 15), 3: d(2026, 9, 15)}


def test_select_week_snapshots_no_snapshot_before_gameday_is_none():
    snapshot_dates = [d(2026, 9, 10), d(2026, 9, 20)]
    week_first_gameday = {1: d(2026, 9, 9)}

    out = data_collection.select_week_snapshots(snapshot_dates, week_first_gameday)

    assert out == {1: None}


def _fake_new_schema_depth_charts(seasons):
    return pl.DataFrame({
        "gsis_id": ["00-1", "00-1", "00-2", "00-1"],
        "pos_abb": ["WR", "WR", "QB", "WR"],
        "pos_rank": [1, 2, 1, 3],
        "dt": [
            "2026-09-08T00:00:00",
            "2026-09-15T00:00:00",
            "2026-09-08T00:00:00",
            "2026-09-08T00:00:00",  # duplicate player+date, worse rank -> deduped away
        ],
    })


def _fake_schedules_2026(seasons):
    return pl.DataFrame({
        "season": [2026] * 3,
        "week": [1, 2, 3],
        "game_type": ["REG", "REG", "REG"],
        "gameday": ["2026-09-09", "2026-09-17", "2026-09-24"],
    })


def _unused_load_schedules(seasons):
    raise AssertionError("load_schedules should not be called for old-schema depth charts")


def test_get_depth_chart_data_new_schema():
    out = data_collection.get_depth_chart_data(
        [2026],
        load_depth_charts=_fake_new_schema_depth_charts,
        load_schedules=_fake_schedules_2026,
    )

    out = out.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # QB dropped; player 00-1 present for weeks 1-3 only
    assert set(out["player_id"]) == {"00-1"}

    week1 = out[out["week"] == 1].iloc[0]
    assert (week1["player_id"], week1["season"], week1["week"], week1["depth_chart_rank"]) == ("00-1", 2026, 1, 1)

    week2 = out[out["week"] == 2].iloc[0]
    assert (week2["player_id"], week2["season"], week2["week"], week2["depth_chart_rank"]) == ("00-1", 2026, 2, 2)

    week3 = out[out["week"] == 3].iloc[0]
    assert (week3["player_id"], week3["season"], week3["week"], week3["depth_chart_rank"]) == ("00-1", 2026, 3, 2)

    assert out["depth_chart_rank"].dtype.kind in "iu"  # int dtype


def _fake_old_schema_depth_charts(seasons):
    return pl.DataFrame({
        "gsis_id": ["00-9"],
        "depth_position": ["WR"],
        "depth_team": ["2"],
        "week": [5],
        "season": [2024],
    })


def test_get_depth_chart_data_old_schema():
    out = data_collection.get_depth_chart_data(
        [2024],
        load_depth_charts=_fake_old_schema_depth_charts,
        load_schedules=_unused_load_schedules,
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert (row["player_id"], row["season"], row["week"], row["depth_chart_rank"]) == ("00-9", 2024, 5, 2)
    assert out["depth_chart_rank"].dtype.kind in "iu"


def test_get_depth_chart_for_week_filters_to_one_week():
    out = data_collection.get_depth_chart_for_week(
        2026,
        2,
        load_depth_charts=_fake_new_schema_depth_charts,
        load_schedules=_fake_schedules_2026,
    )

    assert len(out) == 1
    assert (out.iloc[0]["week"]) == 2
