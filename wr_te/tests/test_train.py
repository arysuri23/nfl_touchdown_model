import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import train_wr
import config
import evaluation


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
    empty = pd.DataFrame({
        "probability": [],
        "scored_touchdown": [],
        "season": [],
        "week": [],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = evaluation._metric_values(empty)
    report = train_wr.calibration_report(np.array([]), np.array([]))

    assert np.isnan(report["brier"]) == np.isnan(expected["brier"])
    assert np.isnan(report["log_loss"]) == np.isnan(expected["log_loss"])
    assert report["ece"] is expected["ece"]


def test_calibration_report_assigns_exact_probability_boundaries_to_evaluator_bins():
    # The 0.0 and 0.1 rows must be separate bins; 1.0 belongs to the final bin.
    y = np.array([0, 0, 1])
    p = np.array([0.0, 0.1, 1.0])

    report = train_wr.calibration_report(y, p)

    assert report["ece"] == pytest.approx(0.1 / 3)


def test_from_cache_requires_existing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(train_wr.config, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="training cache not found"):
        train_wr._load_training_data(True, "missing.csv", {})


def test_from_cache_does_not_collect_or_rewrite_cache(tmp_path, monkeypatch):
    cache = tmp_path / "raw_nfl_data.csv"
    cache.write_bytes(b"season,week,player_id\n2020,1,P1\n")
    original = cache.read_bytes()

    monkeypatch.setattr(train_wr.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(train_wr.data, "get_all_historic_data", lambda *_: pytest.fail("collector called"))
    train_wr._load_training_data(True, "raw_nfl_data.csv", {})

    assert cache.read_bytes() == original


def test_cache_path_must_be_relative_and_beneath_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(train_wr.config, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="beneath config.DATA_DIR"):
        train_wr._resolve_cache_path(tmp_path / "raw.csv")
    with pytest.raises(ValueError, match="beneath config.DATA_DIR"):
        train_wr._resolve_cache_path("../raw.csv")
    outside = tmp_path.parent / "outside-cache.csv"
    outside.write_bytes(b"x")
    (tmp_path / "link.csv").symlink_to(outside)
    with pytest.raises(ValueError, match="beneath config.DATA_DIR"):
        train_wr._resolve_cache_path("link.csv")


def test_cache_validation_requires_exact_seasons_both_positions_and_non_null_keys():
    row = {column: 0 for column in train_wr.RAW_REQUIRED_COLUMNS}
    row.update({"season": 2020, "week": 1, "player_id": "P1", "position": "WR", "team": "A", "opponent_team": "B"})
    frame = pd.DataFrame([row])
    with pytest.raises(ValueError, match="exactly match"):
        train_wr._validate_training_cache(frame, [2020, 2021])
    row["position"] = "TE"
    frame = pd.DataFrame([row])
    with pytest.raises(ValueError, match="both WR and TE"):
        train_wr._validate_training_cache(frame, [2020])


def test_network_data_loader_remains_callable(tmp_path, monkeypatch):
    source = pd.DataFrame({"season": [2020], "week": [1], "player_id": ["P1"]})
    calls = []
    monkeypatch.setattr(train_wr.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(train_wr.data, "get_all_historic_data", lambda years, team_map: calls.append((years, team_map)) or source)

    out, _, path = train_wr._load_training_data(False, None, {"A": "A"})

    assert calls == [(train_wr.config.DATA_SEASONS, {"A": "A"})]
    assert out.equals(source)
    assert path == str(tmp_path / "raw_nfl_data.csv")


def test_manifest_artifact_hashes_match_published_files(tmp_path):
    artifacts = {
        "wr_te_rf_final.pkl": {"model": 1},
        "wr_te_rf_calibrator.pkl": {"calibrator": 1},
        "wr_te_rf_feature_importance.csv": pd.DataFrame({"feature": ["x"]}),
    }
    manifest = {"production_variant": "random_forest_current/raw"}

    train_wr._publish_training_artifacts(tmp_path, artifacts, manifest)

    published = json.loads((tmp_path / "wr_te_rf_manifest.json").read_text())
    for name, digest in published["artifact_hashes"].items():
        assert digest == train_wr._sha256_file(tmp_path / name)


def test_atomic_publication_rolls_back_all_known_outputs_on_replacement_failure(tmp_path, monkeypatch):
    names = ["wr_te_rf_final.pkl", "wr_te_rf_calibrator.pkl", "wr_te_rf_feature_importance.csv", "wr_te_rf_manifest.json"]
    for name in names:
        (tmp_path / name).write_bytes(b"old-" + name.encode())
    artifacts = {
        "wr_te_rf_final.pkl": {"model": 2},
        "wr_te_rf_calibrator.pkl": {"calibrator": 2},
        "wr_te_rf_feature_importance.csv": pd.DataFrame({"feature": ["new"]}),
    }
    real_replace = train_wr.os.replace
    replacements = {"count": 0}

    def fail_on_third(source, target):
        replacements["count"] += 1
        if replacements["count"] == 3:
            raise OSError("injected replacement failure")
        return real_replace(source, target)

    monkeypatch.setattr(train_wr.os, "replace", fail_on_third)
    with pytest.raises(OSError, match="injected replacement failure"):
        train_wr._publish_training_artifacts(tmp_path, artifacts, {"version": 2})

    for name in names:
        assert (tmp_path / name).read_bytes() == b"old-" + name.encode()


def test_strict_oob_calibration_mask_never_falls_back_to_other_seasons():
    model = type("Model", (), {"oob_decision_function_": np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])})()
    mask = np.array([False, True, False])

    _, candidate = train_wr._oob_calibration_rows(model, mask, min_rows=500, fallback=False)

    assert candidate.tolist() == [False, True, False]


def test_rf_params_hash_matches_evaluator_canonical_json():
    import evaluate_wr
    params = {"max_depth": 3, "n_estimators": 10}

    assert train_wr._canonical_hash(params) == train_wr._sha256_bytes(evaluate_wr._json_bytes(params))


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
