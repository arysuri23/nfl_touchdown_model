"""Central configuration for the wr_te pipeline.

Stdlib-only (no nflreadpy, no pandas) so importing/reloading this module
in tests stays fast and side-effect free.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
VEGAS_DIR = BASE_DIR / "vegas"
PREDICTIONS_DIR = BASE_DIR / "predictions"
LEDGER_DIR = BASE_DIR / "ledger"

SEASON = int(os.environ.get("WRTE_SEASON", 2026))
WEEK = int(os.environ.get("WRTE_WEEK", 1))

TRAIN_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]


def odds_api_key() -> str:
    """Return the ODDS_API_KEY environment variable, or raise if unset/empty."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Set the ODDS_API_KEY environment variable")
    return key


def odds_snapshot_path(season: int, week: int, tag: str) -> Path:
    return VEGAS_DIR / str(season) / f"week_{week}_td_odds_{tag}.csv"


def predictions_path(season: int, week: int) -> Path:
    return PREDICTIONS_DIR / str(season) / f"week_{week}.csv"
