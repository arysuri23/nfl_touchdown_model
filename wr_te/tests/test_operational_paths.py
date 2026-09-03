import argparse
from datetime import datetime

import pandas as pd

import config
import ledger


def test_cli_record_uses_timezone_aware_timestamp(monkeypatch, tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame([{"player_id": "P1"}]).to_csv(predictions_path, index=False)
    captured = {}

    def fake_record_picks(predictions, season, week, strategy, stake, ledger_path, now):
        captured["now"] = now
        return 0

    monkeypatch.setattr(ledger, "record_picks", fake_record_picks)
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path)

    class DeprecatedTimestamp:
        @classmethod
        def utcnow(cls):
            raise AssertionError("deprecated utcnow called")

    monkeypatch.setattr(ledger.pd, "Timestamp", DeprecatedTimestamp)
    args = argparse.Namespace(
        predictions=predictions_path,
        season=2026,
        week=1,
        strategy="top5_prob",
        stake=1.0,
    )

    ledger._cli_record(args)

    assert datetime.fromisoformat(captured["now"]).tzinfo is not None
