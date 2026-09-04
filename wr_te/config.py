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
EVALUATION_DIR = BASE_DIR / "evaluation"

SEASON = int(os.environ.get("WRTE_SEASON", 2026))
WEEK = int(os.environ.get("WRTE_WEEK", 1))

TRAIN_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]

# Seasons to collect for the feature cache. The active prediction season is
# included so completed current-season rows are available to prediction;
# train_wr.main applies the prior-season fit cut separately.
DATA_SEASONS = sorted(set(TRAIN_SEASONS) | {SEASON})

# Season used for the informational (not deployed) train/validation split in
# train_wr.py. Fixed regardless of config.SEASON so the reported validation
# metrics stay comparable run over run.
VALIDATION_SEASON = 2023


def odds_api_key() -> str:
    """Return the API key from the process environment or this project's .env.

    dotenv loading is deliberately lazy so importing config never reads files or
    changes the process environment.  A key explicitly supplied in the process
    environment always wins; an explicitly empty value is an error rather than
    an invitation to fall back to a stale .env value.
    """
    key = os.environ.get("ODDS_API_KEY")
    if key is not None:
        if not key:
            raise RuntimeError("ODDS_API_KEY is set but empty")
        return key

    # Keep the dependency and its file read out of module import.  Re-read on
    # each lookup so a user may add or change .env during a long-lived process;
    # override=False preserves values already present in os.environ.
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            f"Set ODDS_API_KEY in the environment or {BASE_DIR / '.env'}"
        )
    return key


def odds_snapshot_path(season: int, week: int, tag: str) -> Path:
    return VEGAS_DIR / str(season) / f"week_{week}_td_odds_{tag}.csv"


def predictions_path(season: int, week: int) -> Path:
    return PREDICTIONS_DIR / str(season) / f"week_{week}.csv"
