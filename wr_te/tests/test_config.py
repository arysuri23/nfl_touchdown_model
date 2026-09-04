import importlib

import pytest

import config


def _reload_with_env(monkeypatch, season=None, week=None):
    if season is None:
        monkeypatch.delenv("WRTE_SEASON", raising=False)
    else:
        monkeypatch.setenv("WRTE_SEASON", str(season))
    if week is None:
        monkeypatch.delenv("WRTE_WEEK", raising=False)
    else:
        monkeypatch.setenv("WRTE_WEEK", str(week))
    importlib.reload(config)


def test_default_season_and_week(monkeypatch):
    _reload_with_env(monkeypatch)
    assert config.SEASON == 2026
    assert config.WEEK == 1


def test_env_override_season_and_week(monkeypatch):
    _reload_with_env(monkeypatch, season=2025, week=15)
    assert config.SEASON == 2025
    assert config.WEEK == 15


def test_train_seasons(monkeypatch):
    _reload_with_env(monkeypatch)
    assert config.TRAIN_SEASONS == [2020, 2021, 2022, 2023, 2024, 2025]


def test_data_seasons_include_current_prediction_season_without_changing_train_cut(monkeypatch):
    _reload_with_env(monkeypatch, season=2026, week=4)
    assert config.DATA_SEASONS == [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    assert config.TRAIN_SEASONS == [2020, 2021, 2022, 2023, 2024, 2025]


def test_validation_season(monkeypatch):
    _reload_with_env(monkeypatch)
    assert config.VALIDATION_SEASON == 2023


def test_odds_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        config.odds_api_key()


def test_odds_api_key_present_returns_value(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "secret-value")
    assert config.odds_api_key() == "secret-value"


def test_odds_api_key_loads_project_env_lazily(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    (tmp_path / ".env").write_text("ODDS_API_KEY=dotenv-value\n")

    assert config.odds_api_key() == "dotenv-value"


def test_odds_api_key_process_environment_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "process-value")
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    (tmp_path / ".env").write_text("ODDS_API_KEY=dotenv-value\n")

    assert config.odds_api_key() == "process-value"


def test_odds_api_key_explicit_empty_does_not_fall_back_to_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "")
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    (tmp_path / ".env").write_text("ODDS_API_KEY=dotenv-value\n")

    with pytest.raises(RuntimeError, match="empty"):
        config.odds_api_key()


def test_odds_snapshot_path():
    assert config.odds_snapshot_path(2026, 1, "open") == (
        config.VEGAS_DIR / "2026" / "week_1_td_odds_open.csv"
    )


def test_predictions_path():
    assert config.predictions_path(2026, 1) == (
        config.PREDICTIONS_DIR / "2026" / "week_1.csv"
    )
