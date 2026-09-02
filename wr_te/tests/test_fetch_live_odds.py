import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import polars as pl
import pytest

import config
import fetch_live_odds

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name):
    return json.loads((FIXTURES_DIR / name).read_text())


# ---------------------------------------------------------------------------
# parse_event_odds
# ---------------------------------------------------------------------------


def test_parse_event_odds_returns_six_rows():
    fixture = _load_fixture("event_odds_sample.json")
    rows = fetch_live_odds.parse_event_odds(
        fixture, 2026, 1, "open", "2026-09-08T15:00:00Z"
    )
    assert len(rows) == 6


def test_parse_event_odds_row_fields():
    fixture = _load_fixture("event_odds_sample.json")
    rows = fetch_live_odds.parse_event_odds(
        fixture, 2026, 1, "open", "2026-09-08T15:00:00Z"
    )

    assert all(row["label"] == "Yes" for row in rows)
    assert all(row["market"] == "player_anytime_td" for row in rows)
    assert all(isinstance(row["price"], int) for row in rows)
    assert all(row["home_team"] == "Kansas City Chiefs" for row in rows)
    assert all(row["away_team"] == "Baltimore Ravens" for row in rows)

    descriptions = {row["description"] for row in rows}
    assert descriptions == {"Player One", "Player Two", "Player Three"}

    bookmakers = {row["bookmaker"] for row in rows}
    assert bookmakers == {"DraftKings", "FanDuel"}

    # "No" outcomes must be dropped entirely.
    assert not any(row["description"] == "Player Four" for row in rows)

    assert all(row["season"] == 2026 for row in rows)
    assert all(row["week"] == 1 for row in rows)
    assert all(row["tag"] == "open" for row in rows)
    assert all(row["fetched_at"] == "2026-09-08T15:00:00Z" for row in rows)


# ---------------------------------------------------------------------------
# select_week_events
# ---------------------------------------------------------------------------


def test_select_week_events_keeps_in_window_and_drops_out_of_window():
    events = [
        {"id": "keep", "commence_time": "2026-09-13T17:00:00Z"},
        {"id": "drop", "commence_time": "2026-09-20T17:00:00Z"},
    ]
    window_start = datetime(2026, 9, 9, tzinfo=timezone.utc)
    window_end = datetime(2026, 9, 15, tzinfo=timezone.utc)

    selected = fetch_live_odds.select_week_events(events, window_start, window_end)

    ids = [event["id"] for event in selected]
    assert ids == ["keep"]


def test_select_week_events_window_is_half_open():
    # Event exactly at window_end should be dropped ([start, end)).
    events = [{"id": "edge", "commence_time": "2026-09-15T00:00:00Z"}]
    window_start = datetime(2026, 9, 9, tzinfo=timezone.utc)
    window_end = datetime(2026, 9, 15, tzinfo=timezone.utc)

    selected = fetch_live_odds.select_week_events(events, window_start, window_end)

    assert selected == []


# ---------------------------------------------------------------------------
# week_window
# ---------------------------------------------------------------------------


def _fake_load_schedules_factory(season):
    def fake_load_schedules(seasons):
        return pl.DataFrame(
            {
                "season": [season, season, season],
                "week": [1, 1, 2],
                "game_type": ["REG", "REG", "REG"],
                "gameday": ["2026-09-10", "2026-09-13", "2026-09-20"],
            }
        )

    return fake_load_schedules


def test_week_window_spans_min_to_max_plus_one_day():
    fake_load_schedules = _fake_load_schedules_factory(2026)

    start, end = fetch_live_odds.week_window(
        2026, 1, load_schedules=fake_load_schedules
    )

    assert start == datetime(2026, 9, 10, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 14, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeGet:
    """Serves events_sample.json then event_odds_sample.json, in order."""

    def __init__(self):
        self.calls = []
        events_payload = _load_fixture("events_sample.json")
        odds_payload = _load_fixture("event_odds_sample.json")
        headers = {"x-requests-used": "10", "x-requests-remaining": "490"}
        self._responses = [
            _FakeResponse(events_payload, headers=headers),
            _FakeResponse(odds_payload, headers=headers),
        ]

    def __call__(self, url, params=None):
        self.calls.append((url, params))
        return self._responses[len(self.calls) - 1]


def test_fetch_writes_csv_with_expected_columns_and_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)

    fake_get = _FakeGet()
    fake_load_schedules = _fake_load_schedules_factory(2026)

    df = fetch_live_odds.fetch(
        2026,
        1,
        "open",
        ["draftkings", "fanduel"],
        get=fake_get,
        load_schedules=fake_load_schedules,
    )

    expected_columns = [
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
    assert list(df.columns) == expected_columns
    assert len(df) == 6

    out_path = config.odds_snapshot_path(2026, 1, "open")
    assert out_path.exists()
    on_disk = pd.read_csv(out_path)
    assert list(on_disk.columns) == expected_columns
    assert len(on_disk) == 6

    # Second call: fetching odds for the selected event.
    assert len(fake_get.calls) == 2
    odds_url, odds_params = fake_get.calls[1]
    assert odds_params["markets"] == "player_anytime_td"
    assert odds_params["bookmakers"] == "draftkings,fanduel"


def test_fetch_raises_runtime_error_when_api_key_unset(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        fetch_live_odds.fetch(2026, 1, "open", ["draftkings"])


# ---------------------------------------------------------------------------
# No hardcoded API key literals in the legacy fetchers.
# ---------------------------------------------------------------------------


def test_no_hardcoded_api_key_literals():
    pattern = re.compile(r"""api_key\s*=\s*['"]|API_KEY\s*=\s*['"]""")
    base_dir = Path(__file__).parent.parent
    for filename in ("fetch_historical_odds.py", "fetch_receptions_odds.py"):
        text = (base_dir / filename).read_text()
        assert not pattern.search(text), f"hardcoded key literal found in {filename}"
