import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import train_wr
import config


# --- chronological ---

def test_chronological_sorts_by_season_week_player_id():
    df = pd.DataFrame(
        {
            "season": [2020, 2020, 2019],
            "week": [1, 1, 5],
            "player_id": ["B", "A", "Z"],
            "value": [1, 2, 3],
        },
        index=[5, 2, 9],
    )

    out = train_wr.chronological(df)

    assert list(out["season"]) == [2019, 2020, 2020]
    assert list(out["week"]) == [5, 1, 1]
    assert list(out["player_id"]) == ["Z", "A", "B"]
    assert list(out.index) == [0, 1, 2]


# --- fit_calibrator_oob ---

def _make_synthetic_oob_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    coef = np.array([1.0, -0.5, 0.3, 0.8, -0.2])
    linear = X @ coef
    prob = 1.0 / (1.0 + np.exp(-linear))
    y = (prob > rng.uniform(size=n)).astype(int)
    return X, y


def test_fit_calibrator_oob_fits_monotonic_logistic_on_masked_rows():
    X, y = _make_synthetic_oob_data()
    model = RandomForestClassifier(
        n_estimators=60, max_depth=4, oob_score=True, random_state=0, n_jobs=-1
    )
    model.fit(X, y)

    mask = np.zeros(len(y), dtype=bool)
    mask[-800:] = True

    calibrator = train_wr.fit_calibrator_oob(model, pd.Series(y), pd.Series(mask))

    assert hasattr(calibrator, "coef_")
    low = calibrator.predict_proba([[0.1]])[:, 1]
    high = calibrator.predict_proba([[0.9]])[:, 1]
    assert low[0] < high[0]


def test_fit_calibrator_oob_falls_back_to_all_rows_when_mask_too_small():
    X, y = _make_synthetic_oob_data()
    model = RandomForestClassifier(
        n_estimators=60, max_depth=4, oob_score=True, random_state=0, n_jobs=-1
    )
    model.fit(X, y)

    mask = np.zeros(len(y), dtype=bool)
    mask[:100] = True

    calibrator = train_wr.fit_calibrator_oob(
        model, pd.Series(y), pd.Series(mask), min_rows=500
    )

    assert calibrator.n_features_in_ == 1


# --- calibration_report ---

def test_calibration_report_on_perfectly_calibrated_input():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)

    report = train_wr.calibration_report(y, p)

    assert set(report.keys()) == {"brier", "log_loss", "ece"}
    assert report["ece"] < 0.05


def test_calibration_report_uses_weighted_fixed_probability_bins():
    # Bin gaps are .45 (2 rows), .15 (2 rows), and .05 (1 row), so the
    # evaluator-compatible weighted ECE is (2*.45 + 2*.15 + 1*.05) / 5.
    y = np.array([0, 1, 0, 0, 1])
    p = np.array([0.05, 0.05, 0.15, 0.15, 0.95])

    report = train_wr.calibration_report(y, p)

    assert report["ece"] == pytest.approx(0.25)


def test_calibration_report_handles_empty_input_before_sklearn_metrics():
    report = train_wr.calibration_report(np.array([]), np.array([]))

    assert report == {"brier": 0.0, "log_loss": 0.0, "ece": 0.0}


def test_calibration_report_assigns_exact_probability_boundaries_to_evaluator_bins():
    # The 0.0 and 0.1 rows must be separate bins; 1.0 belongs to the final bin.
    y = np.array([0, 0, 1])
    p = np.array([0.0, 0.1, 1.0])

    report = train_wr.calibration_report(y, p)

    assert report["ece"] == pytest.approx(0.1 / 3)


# --- train_rf_model ---

def test_train_rf_model_with_saved_params_skips_randomized_search(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(train_wr.config, "MODELS_DIR", models_dir)
    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    (models_dir / "wr_te_rf_best_params.json").write_text(json.dumps(params))

    def _raise(*args, **kwargs):
        raise AssertionError(
            "RandomizedSearchCV should not be called when use_saved_params=True"
        )

    monkeypatch.setattr(train_wr, "RandomizedSearchCV", _raise)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = pd.Series((rng.uniform(size=50) > 0.5).astype(int))

    model, best_params, elapsed = train_wr.train_rf_model(
        X, y, "wr_te", use_saved_params=True
    )

    assert best_params == params
    assert hasattr(model, "predict_proba")


def test_train_rf_model_reads_params_from_configured_model_path(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    (models_dir / "wr_te_rf_best_params.json").write_text(json.dumps(params))

    def _raise(*args, **kwargs):
        raise AssertionError("RandomizedSearchCV should not be called")

    monkeypatch.setattr(train_wr, "RandomizedSearchCV", _raise)
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = pd.Series((rng.uniform(size=50) > 0.5).astype(int))

    _, best_params, _ = train_wr.train_rf_model(
        X, y, "wr_te", use_saved_params=True
    )

    assert best_params == params


def test_feature_importance_table_uses_the_supplied_deployed_model():
    class _Model:
        feature_importances_ = np.array([0.2, 0.8])

    out = train_wr.feature_importance_table(_Model(), ["first", "second"])

    assert list(out["feature"]) == ["second", "first"]
    np.testing.assert_allclose(out["importance"], [0.8, 0.2])
    np.testing.assert_allclose(out["importance_pct"], [80.0, 20.0])


class _FakeCalibrator:
    def __init__(self):
        self.seen = None

    def predict_proba(self, X):
        self.seen = np.asarray(X).copy()
        calibrated = np.asarray(X).ravel() * 0.5 + 0.25
        return np.column_stack([1.0 - calibrated, calibrated])


def test_deployed_calibration_report_uses_calibrated_oob_array(monkeypatch):
    calls = []

    def capture_report(y_true, probabilities):
        calls.append(np.asarray(probabilities).copy())
        return {"brier": 0.0, "log_loss": 0.0, "ece": 0.0}

    monkeypatch.setattr(train_wr, "calibration_report", capture_report)
    calibrator = _FakeCalibrator()
    raw_oob = np.array([0.1, 0.8])

    reports = train_wr.deployed_calibration_reports(
        np.array([0, 1]), raw_oob, calibrator
    )

    assert len(calls) == 2
    np.testing.assert_allclose(calls[0], [0.1, 0.8])
    np.testing.assert_allclose(calls[1], [0.30, 0.65])
    assert set(reports) == {"raw", "calibrated"}
