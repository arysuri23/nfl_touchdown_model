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


def test_select_week_events_requires_scheduled_matchup_when_provided():
    events = [
        {
            "id": "target",
            "commence_time": "2026-09-13T17:00:00Z",
            "home_team": "Kansas City Chiefs",
            "away_team": "Baltimore Ravens",
        },
        {
            "id": "wrong-game",
            "commence_time": "2026-09-13T17:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "New York Jets",
        },
    ]
    scheduled_matchups = {frozenset({"KC", "BAL"})}
    team_map = {
        "Kansas City Chiefs": "KC",
        "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF",
        "New York Jets": "NYJ",
    }

    selected = fetch_live_odds.select_week_events(
        events,
        datetime(2026, 9, 9, tzinfo=timezone.utc),
        datetime(2026, 9, 15, tzinfo=timezone.utc),
        scheduled_matchups=scheduled_matchups,
        team_map=team_map,
    )

    assert [event["id"] for event in selected] == ["target"]


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


def test_week_window_spans_min_to_max_plus_two_days():
    fake_load_schedules = _fake_load_schedules_factory(2026)

    start, end = fetch_live_odds.week_window(
        2026, 1, load_schedules=fake_load_schedules
    )

    assert start == datetime(2026, 9, 10, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 15, tzinfo=timezone.utc)


def test_week_window_includes_late_monday_kickoff_but_not_adjacent_week():
    def fake_load_schedules(seasons):
        return pl.DataFrame(
            {
                "season": [2026] * 4,
                "week": [1, 1, 1, 2],
                "game_type": ["REG"] * 4,
                "gameday": ["2026-09-10", "2026-09-13", "2026-09-14", "2026-09-17"],
            }
        )

    start, end = fetch_live_odds.week_window(
        2026, 1, load_schedules=fake_load_schedules
    )
    events = [
        {"id": "monday", "commence_time": "2026-09-15T00:15:00Z"},
        {"id": "next-week", "commence_time": "2026-09-18T00:20:00Z"},
    ]

    selected = fetch_live_odds.select_week_events(events, start, end)

    assert [event["id"] for event in selected] == ["monday"]


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


def test_fetch_writes_schedule_game_id_when_provider_event_id_differs(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)

    class _DifferentIdGet(_FakeGet):
        def __init__(self):
            super().__init__()
            self._responses[1]._payload = dict(self._responses[1]._payload)
            self._responses[1]._payload["id"] = "provider-event-abc123"

    def fake_load_schedules(seasons):
        return pl.DataFrame(
            {
                "season": [2026],
                "week": [1],
                "game_type": ["REG"],
                "gameday": ["2026-09-13"],
                "game_id": ["2026_01_BAL_KC"],
                "home_team": ["Kansas City Chiefs"],
                "away_team": ["Baltimore Ravens"],
            }
        )

    out = fetch_live_odds.fetch(
        2026,
        1,
        "open",
        ["draftkings"],
        get=_DifferentIdGet(),
        load_schedules=fake_load_schedules,
    )

    assert set(out["game_id"]) == {"2026_01_BAL_KC"}
    assert "provider-event-abc123" not in set(out["game_id"])


def test_fetch_raises_runtime_error_when_api_key_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path / "vegas")

    def fail(*args, **kwargs):
        raise AssertionError("missing key must fail before schedule/API access")

    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        fetch_live_odds.fetch(
            2026, 1, "open", ["draftkings"], get=fail, load_schedules=fail
        )


def _valid_snapshot_frame(extra=False):
    row = {column: None for column in fetch_live_odds.CSV_COLUMNS}
    row.update({"season": 2026, "week": 1, "tag": "open", "description": "Player One"})
    frame = pd.DataFrame([row], columns=fetch_live_odds.CSV_COLUMNS)
    if extra:
        frame["provider_extra"] = "kept"
    return frame


def test_fetch_cache_hit_returns_csv_without_key_schedule_or_http(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)
    path = config.odds_snapshot_path(2026, 1, "open")
    path.parent.mkdir(parents=True)
    _valid_snapshot_frame(extra=True).to_csv(path, index=False)

    def fail(*args, **kwargs):
        raise AssertionError("cache hit must not resolve dependencies")

    monkeypatch.setattr(config, "odds_api_key", fail)
    out = fetch_live_odds.fetch(
        2026, 1, "open", ["draftkings"], get=fail, load_schedules=fail
    )

    assert len(out) == 1
    assert "provider_extra" in out.columns


@pytest.mark.parametrize(
    "contents, expected",
    [("", "zero bytes"), ("description\nPlayer One\n", "missing required columns")],
)
def test_fetch_invalid_cache_fails_before_network(tmp_path, monkeypatch, contents, expected):
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)
    path = config.odds_snapshot_path(2026, 1, "open")
    path.parent.mkdir(parents=True)
    path.write_text(contents)

    def fail(*args, **kwargs):
        raise AssertionError("invalid cache must not use network")

    with pytest.raises(ValueError, match=expected):
        fetch_live_odds.fetch(
            2026, 1, "open", ["draftkings"], get=fail, load_schedules=fail
        )


def test_fetch_header_only_cache_is_valid_and_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)
    path = config.odds_snapshot_path(2026, 1, "open")
    path.parent.mkdir(parents=True)
    path.write_text(",".join(fetch_live_odds.CSV_COLUMNS) + "\n")

    out = fetch_live_odds.fetch(
        2026, 1, "open", ["draftkings"],
        get=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
        load_schedules=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    assert out.empty
    assert list(out.columns) == fetch_live_odds.CSV_COLUMNS


@pytest.mark.parametrize(
    "metadata",
    [
        {"season": 2025, "week": 1, "tag": "open"},
        {"season": 2026, "week": 2, "tag": "open"},
        {"season": 2026, "week": 1, "tag": "close"},
        {"season": "not-a-season", "week": 1, "tag": "open"},
        {"season": 2026, "week": None, "tag": "open"},
    ],
)
def test_fetch_rejects_nonmatching_or_invalid_cache_metadata_before_dependencies(
    tmp_path, monkeypatch, metadata
):
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)
    path = config.odds_snapshot_path(2026, 1, "open")
    path.parent.mkdir(parents=True)
    _valid_snapshot_frame().assign(**metadata).to_csv(path, index=False)

    def fail(*args, **kwargs):
        raise AssertionError("invalid cache must not resolve dependencies")

    with pytest.raises(ValueError, match="season|week|tag"):
        fetch_live_odds.fetch(
            2026, 1, "open", ["draftkings"], get=fail, load_schedules=fail
        )


def test_fetch_rejects_mixed_cache_metadata_before_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)
    path = config.odds_snapshot_path(2026, 1, "open")
    path.parent.mkdir(parents=True)
    cached = pd.concat(
        [_valid_snapshot_frame(), _valid_snapshot_frame().assign(week=2)],
        ignore_index=True,
    )
    cached.to_csv(path, index=False)

    def fail(*args, **kwargs):
        raise AssertionError("invalid cache must not resolve dependencies")

    with pytest.raises(ValueError, match="week"):
        fetch_live_odds.fetch(
            2026, 1, "open", ["draftkings"], get=fail, load_schedules=fail
        )


def test_fetch_refresh_replaces_existing_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)
    path = config.odds_snapshot_path(2026, 1, "open")
    path.parent.mkdir(parents=True)
    path.write_text("old-bytes")
    fake_get = _FakeGet()

    out = fetch_live_odds.fetch(
        2026, 1, "open", ["draftkings"], get=fake_get,
        load_schedules=_fake_load_schedules_factory(2026), refresh=True,
    )

    assert len(out) == 6
    assert path.read_text().startswith(",".join(fetch_live_odds.CSV_COLUMNS))
    assert not list(path.parent.glob("*.tmp"))


def test_fetch_refresh_failure_preserves_existing_bytes_and_cleans_temp(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)
    path = config.odds_snapshot_path(2026, 1, "open")
    path.parent.mkdir(parents=True)
    original = ",".join(fetch_live_odds.CSV_COLUMNS) + "\nold\n"
    path.write_text(original)

    def failing_get(*args, **kwargs):
        raise RuntimeError("simulated API failure")

    with pytest.raises(RuntimeError, match="simulated API failure"):
        fetch_live_odds.fetch(
            2026, 1, "open", ["draftkings"], get=failing_get,
            load_schedules=_fake_load_schedules_factory(2026), refresh=True,
        )
    assert path.read_text() == original
    assert not list(path.parent.glob("*.tmp"))


def test_fetch_refresh_to_csv_failure_preserves_existing_bytes_and_cleans_temp(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)
    path = config.odds_snapshot_path(2026, 1, "open")
    path.parent.mkdir(parents=True)
    original = ",".join(fetch_live_odds.CSV_COLUMNS) + "\nold\n"
    path.write_text(original)

    def fail_to_csv(*args, **kwargs):
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(RuntimeError, match="serialization failure"):
        fetch_live_odds.fetch(
            2026, 1, "open", ["draftkings"], get=_FakeGet(),
            load_schedules=_fake_load_schedules_factory(2026), refresh=True,
        )
    assert path.read_text() == original
    assert not list(path.parent.glob("*.tmp"))


def test_fetch_refresh_replace_failure_preserves_existing_bytes_and_cleans_temp(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setattr(config, "VEGAS_DIR", tmp_path)
    path = config.odds_snapshot_path(2026, 1, "open")
    path.parent.mkdir(parents=True)
    original = ",".join(fetch_live_odds.CSV_COLUMNS) + "\nold\n"
    path.write_text(original)

    def fail_replace(*args, **kwargs):
        raise RuntimeError("simulated replace failure")

    monkeypatch.setattr(fetch_live_odds.os, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="replace failure"):
        fetch_live_odds.fetch(
            2026, 1, "open", ["draftkings"], get=_FakeGet(),
            load_schedules=_fake_load_schedules_factory(2026), refresh=True,
        )
    assert path.read_text() == original
    assert not list(path.parent.glob("*.tmp"))


def test_fetch_cli_forwards_refresh(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fetch_live_odds, "fetch",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    fetch_live_odds.main(["--season", "2026", "--week", "1", "--tag", "open", "--refresh"])

    assert calls == [((2026, 1, "open", ["draftkings"]), {"refresh": True})]


# ---------------------------------------------------------------------------
# No hardcoded API key literals in the legacy fetchers.
# ---------------------------------------------------------------------------


def test_no_hardcoded_api_key_literals():
    pattern = re.compile(r"""api_key\s*=\s*['"]|API_KEY\s*=\s*['"]""")
    base_dir = Path(__file__).parent.parent
    for filename in ("fetch_historical_odds.py", "fetch_receptions_odds.py"):
        text = (base_dir / filename).read_text()
        assert not pattern.search(text), f"hardcoded key literal found in {filename}"
