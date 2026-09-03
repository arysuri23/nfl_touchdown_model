import numpy as np
import pandas as pd
import pytest

import evaluation
from features import PLAYER_EWM_STATS, WR_TE_FEATURES


def _raw_rows(groups):
    rows = []
    defense = [
        "passing_tds_allowed_to_WR",
        "passing_tds_allowed_to_TE",
        "receiving_yards_allowed",
        "receiving_epa_allowed",
        "receiving_air_yards_allowed",
        "explosive_receiving_plays_allowed",
    ]
    for season, week in groups:
        for idx, position in enumerate(("WR", "TE", "WR", "TE")):
            row = {
                "season": season,
                "week": week,
                "game_id": f"{season}_{week}_{idx // 2}",
                "player_id": f"p{idx}",
                "player_display_name": f"Player {idx}",
                "team": "AAA",
                "opponent_team": "BBB",
                "position": position,
                "scored_touchdown": idx % 2,
                "implied_total": 21.0 + idx,
                "spread_line": float(idx),
                "depth_chart_rank": idx + 1,
            }
            row.update({stat: float(idx + 1) for stat in PLAYER_EWM_STATS})
            row["scored_touchdown"] = idx % 2
            row.update({stat: 1.0 for stat in defense})
            row.update({feature: float(idx + 1) for feature in WR_TE_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def _identity_features(frame):
    return frame.copy()


def _prediction_frame(outcomes, probabilities, model, variant, stream="retrospective"):
    """Small shared-key prediction fixture for pooled metric tests."""
    rows = []
    for index, (outcome, probability) in enumerate(zip(outcomes, probabilities)):
        rows.append({
            "season": 2025,
            "week": index // 2 + 1,
            "game_id": f"g{index}",
            "player_id": f"p{index}",
            "player_display_name": f"Player {index}",
            "team": "AAA",
            "opponent_team": "BBB",
            "position": "WR",
            "scored_touchdown": outcome,
            "evaluation_stream": stream,
            "model": model,
            "variant": variant,
            "probability": probability,
            "fold": 0,
        })
    return pd.DataFrame(rows)


def test_primary_metrics_use_exact_reference_keys_and_weighted_ece():
    model = _prediction_frame([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], "logistic_l2", "raw")
    reference = _prediction_frame(
        [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], "base_rate_overall", "raw"
    )
    metrics, summary = evaluation.build_metric_records(
        pd.concat([model, reference], ignore_index=True), []
    )
    row = summary.query("model == 'logistic_l2' and variant == 'raw'").iloc[0]
    assert row["log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.8, 0.9])))
    assert row["brier"] == pytest.approx(0.025)
    assert row["brier_skill"] == pytest.approx(0.9)
    assert row["brier_reference"] == "base_rate_overall/raw"
    assert row["ece"] == pytest.approx(0.15)
    assert len(metrics["overall"][0]["calibration_bins"]) == 10


def test_primary_metrics_reject_reference_key_mismatch():
    model = _prediction_frame([0], [0.2], "logistic_l2", "raw")
    reference = _prediction_frame([0, 1], [0.5, 0.5], "base_rate_overall", "raw")
    with pytest.raises(ValueError, match="reference key mismatch"):
        evaluation.build_metric_records(pd.concat([model, reference], ignore_index=True), [])


def test_odds_and_payout_use_precedence_team_matching_and_preserve_rows():
    tagged = pd.DataFrame([{
        "game_id": "tagged-game",
        "commence_time": "2025-09-07T13:00:00Z",
        "in_play": False,
        "bookmaker": "safe-book",
        "last_update": "2025-09-07T10:00:00Z",
        "home_team": "New York Jets",
        "away_team": "Buffalo Bills",
        "description": "Mike Williams",
        "price": 200,
        "season": 2025,
        "week": 1,
        "tag": "open",
        "fetched_at": "2025-09-07T10:05:00Z",
    }])
    legacy = pd.DataFrame([{
        "Player": "Mike Williams", "HomeTeam": "New York Jets", "AwayTeam": "Buffalo Bills",
        "Odds": 150, "Bookmaker": "legacy-book", "Season": 2025, "Week": 1,
    }])
    selected, provenance, warnings = evaluation.select_open_odds(tagged, legacy)
    assert provenance == "timestamp_safe_open"
    assert selected.iloc[0]["price"] == 200
    assert warnings == []

    unsafe = tagged.copy()
    unsafe["fetched_at"] = ""
    selected, provenance, warnings = evaluation.select_open_odds(unsafe, legacy)
    assert provenance == "timestamp_unsafe_legacy"
    assert selected.iloc[0]["price"] == 150
    assert warnings

    predictions = pd.DataFrame([
        {"season": 2025, "week": 1, "game_id": "g1", "player_id": "p1",
         "player_display_name": "Mike Williams", "team": "NYJ", "opponent_team": "BUF",
         "position": "WR", "scored_touchdown": 1, "evaluation_stream": "retrospective",
         "model": "logistic_l2", "variant": "raw", "probability": 0.9},
        {"season": 2025, "week": 1, "game_id": "g2", "player_id": "p2",
         "player_display_name": "Mike Williams", "team": "LAC", "opponent_team": "DEN",
         "position": "WR", "scored_touchdown": 0, "evaluation_stream": "retrospective",
         "model": "logistic_l2", "variant": "raw", "probability": 0.8},
        {"season": 2025, "week": 1, "game_id": "g3", "player_id": "p3",
         "player_display_name": "Nobody Here", "team": "SF", "opponent_team": "SEA",
         "position": "WR", "scored_touchdown": 0, "evaluation_stream": "retrospective",
         "model": "logistic_l2", "variant": "raw", "probability": 0.7},
        {"season": 2025, "week": 1, "game_id": "g4", "player_id": "p4",
         "player_display_name": "Other Player", "team": "NYJ", "opponent_team": "BUF",
         "position": "WR", "scored_touchdown": 0, "evaluation_stream": "retrospective",
         "model": "logistic_l2", "variant": "raw", "probability": 0.6},
    ])
    odds = pd.DataFrame([
            {"description": "Mike Williams", "home_team": "New York Jets", "away_team": "Buffalo Bills",
             "price": 150, "bookmaker": "book1"},
            {"description": "Mike Williams", "home_team": "New York Jets", "away_team": "Buffalo Bills",
             "price": 200, "bookmaker": "book2"},
            {"description": "Mike Williams", "home_team": "Los Angeles Chargers", "away_team": "Denver Broncos",
             "price": 300, "bookmaker": "book1"},
            {"description": "Other Player", "home_team": "New York Jets", "away_team": "Buffalo Bills",
             "price": 300, "bookmaker": "book1"},
    ])
    enriched, betting = evaluation.attach_open_odds_and_score_bets(
        predictions, {(2025, 1): (odds, "timestamp_unsafe_legacy")},
        {"New York Jets": "NYJ", "Buffalo Bills": "BUF", "Los Angeles Chargers": "LAC", "Denver Broncos": "DEN"},
    )
    report = next(row for row in betting if row["model"] == "logistic_l2" and row["variant"] == "raw")
    assert report["bets"] == 3
    assert report["stake"] == 3.0
    assert report["pnl"] == pytest.approx(0.0)
    assert report["roi"] == pytest.approx(0.0)
    assert report["hits"] == 1
    assert report["hit_rate"] == pytest.approx(1 / 3)
    assert report["betting_label"] == "research-only, timestamp unsafe"
    assert len(enriched) == len(predictions)
    assert enriched.loc[enriched["player_id"] == "p1", "price_open"].iloc[0] == 200
    assert enriched.loc[enriched["player_id"] == "p2", "price_open"].iloc[0] == 300
    assert pd.isna(enriched.loc[enriched["player_id"] == "p3", "price_open"].iloc[0])
    assert enriched.loc[enriched["player_id"] == "p3", "open_provenance"].iloc[0] == "timestamp_unsafe_legacy"


def test_whole_week_folds_cross_season_boundary_and_reject_short_history():
    rows, counts = evaluation.prepare_evaluation_rows(
        _raw_rows([(2021, 16), (2021, 17), (2021, 18), (2022, 1), (2022, 2)]),
        _identity_features,
    )
    folds = evaluation.build_walk_forward_folds(rows, 2022, 2022, 2)
    assert folds[0] == evaluation.FoldSpec(
        test_group=(2022, 1),
        calibration_groups=((2021, 17), (2021, 18)),
        fit_groups=((2021, 16),),
    )
    assert [fold.test_group for fold in folds] == [(2022, 1), (2022, 2)]
    assert counts["eligible_rows"] == 20
    fit_rows, calibration_rows, test_rows = evaluation.split_and_assert_fold(rows, folds[0])
    test_keys = set(map(tuple, test_rows[evaluation.ROW_KEY_COLUMNS].to_numpy()))
    expected_test_keys = set(
        map(
            tuple,
            rows.loc[(rows.season == 2022) & (rows.week == 1), evaluation.ROW_KEY_COLUMNS].to_numpy(),
        )
    )
    assert test_keys == expected_test_keys
    assert len(test_rows) == 4
    key_sets = [set(map(tuple, part[evaluation.ROW_KEY_COLUMNS].to_numpy())) for part in (
        fit_rows, calibration_rows, test_rows
    )]
    game_sets = [set(part["game_id"]) for part in (fit_rows, calibration_rows, test_rows)]
    assert not (key_sets[0] & key_sets[1] or key_sets[0] & key_sets[2] or key_sets[1] & key_sets[2])
    assert not (game_sets[0] & game_sets[1] or game_sets[0] & game_sets[2] or game_sets[1] & game_sets[2])

    short, _ = evaluation.prepare_evaluation_rows(
        _raw_rows([(2021, 18), (2022, 1)]), _identity_features
    )
    with pytest.raises(ValueError, match=r"earliest requested group.*available history=1.*required=3"):
        evaluation.build_walk_forward_folds(short, 2022, 2022, 2)


def test_duplicate_eligible_row_key_is_rejected():
    raw = pd.concat([_raw_rows([(2022, 1)]), _raw_rows([(2022, 1)]).iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate eligible row key"):
        evaluation.prepare_evaluation_rows(raw, _identity_features)


def test_player_display_name_is_required():
    raw = _raw_rows([(2022, 1)]).drop(columns=["player_display_name"])
    with pytest.raises(ValueError, match="player_display_name"):
        evaluation.prepare_evaluation_rows(raw, _identity_features)


class _RecordingEstimator:
    fits = []

    def __init__(self, kind):
        self.kind = kind

    def fit(self, features, outcomes):
        self.__class__.fits.append(
            (self.kind, features.iloc[:, 0].to_numpy().copy(), np.asarray(outcomes).copy())
        )
        return self

    def predict_proba(self, features):
        # Group markers make calibration inputs auditable per fold.
        p = (features.iloc[:, 0].to_numpy() % 100) / 100
        return np.column_stack([1 - p, p])


class _RecordingCalibrator:
    fits = []

    def fit(self, features, outcomes):
        self.__class__.fits.append((features[:, 0].copy(), np.asarray(outcomes).copy()))
        return self

    def predict_proba(self, features):
        p = np.full(len(features), 0.5)
        return np.column_stack([1 - p, p])


def test_shared_keys_and_temporal_calibration_do_not_use_test_rows():
    raw = _raw_rows([(2021, 16), (2021, 17), (2021, 18), (2022, 1), (2022, 2)])
    rows, _ = evaluation.prepare_evaluation_rows(raw, _identity_features)
    rows[WR_TE_FEATURES[0]] = rows["season"] * 100 + rows["week"]
    folds = evaluation.build_walk_forward_folds(rows, 2022, 2022, 2)
    _RecordingEstimator.fits = []
    _RecordingCalibrator.fits = []
    params = {
        "n_estimators": 5,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    predictions, fold_rows = evaluation.evaluate_folds(
        rows,
        folds,
        params,
        seed=42,
        estimator_factories={
            "logistic_l2": lambda seed, params: _RecordingEstimator("logistic_l2"),
            "random_forest_current": lambda seed, params: _RecordingEstimator("random_forest_current"),
        },
        calibrator_factory=_RecordingCalibrator,
    )
    test_rows = rows[rows.season == 2022]
    expected_keys = list(map(tuple, test_rows[evaluation.ROW_KEY_COLUMNS].to_numpy()))
    for model, variant in evaluation.MODEL_VARIANTS:
        got = predictions.loc[
            (predictions["model"] == model) & (predictions["variant"] == variant),
            evaluation.ROW_KEY_COLUMNS,
        ]
        assert list(map(tuple, got.to_numpy())) == expected_keys
    assert set(_RecordingEstimator.fits[0][1]) == {202116}
    assert set(_RecordingEstimator.fits[0][2]) == {0, 1}
    assert len(_RecordingEstimator.fits) == 4
    assert len(_RecordingCalibrator.fits) == 4
    assert [set(record[1]) for record in _RecordingEstimator.fits] == [
        {202116},
        {202116},
        {202116, 202117},
        {202116, 202117},
    ]
    expected_calibration_outcomes = []
    for fold in folds:
        _, calibration, _ = evaluation.split_and_assert_fold(rows, fold)
        expected_calibration_outcomes.extend([calibration["scored_touchdown"].to_numpy()] * 2)
    assert all(
        np.array_equal(record[1], expected_calibration_outcomes[index // 2])
        for index, record in enumerate(_RecordingCalibrator.fits)
    )
    expected_calibration_probabilities = []
    for fold in folds:
        _, calibration, _ = evaluation.split_and_assert_fold(rows, fold)
        expected = (calibration[WR_TE_FEATURES[0]].to_numpy() % 100) / 100
        expected_calibration_probabilities.extend([expected] * 2)
    assert all(
        np.array_equal(record[0], expected_calibration_probabilities[index])
        for index, record in enumerate(_RecordingCalibrator.fits)
    )
    assert all(set(outcomes) == {0, 1} for _, outcomes in _RecordingCalibrator.fits)
    assert set(fold_rows["football_context_provenance"]) == {
        "retrospective_finalish_game_context"
    }

    perturbed = raw.copy()
    outer_test = (perturbed["season"] == 2022) & (perturbed["week"] == 1)
    perturbed.loc[outer_test, "scored_touchdown"] = 1 - perturbed.loc[
        outer_test, "scored_touchdown"
    ]
    rows_perturbed, _ = evaluation.prepare_evaluation_rows(perturbed, _identity_features)
    rows_perturbed[WR_TE_FEATURES[0]] = rows_perturbed["season"] * 100 + rows_perturbed["week"]
    rerun, _ = evaluation.evaluate_folds(
        rows_perturbed,
        folds,
        params,
        seed=42,
        estimator_factories={
            "logistic_l2": lambda seed, params: _RecordingEstimator("logistic_l2"),
            "random_forest_current": lambda seed, params: _RecordingEstimator("random_forest_current"),
        },
        calibrator_factory=_RecordingCalibrator,
    )
    pd.testing.assert_series_equal(
        predictions.loc[predictions["fold"] == 0, "probability"].reset_index(drop=True),
        rerun.loc[rerun["fold"] == 0, "probability"].reset_index(drop=True),
    )


def test_calibration_class_and_probability_failures_happen_before_concat():
    raw = _raw_rows([(2021, 16), (2021, 17), (2021, 18), (2022, 1)])
    raw.loc[(raw["season"] == 2021) & (raw["week"] == 17), "scored_touchdown"] = 0
    raw.loc[(raw["season"] == 2021) & (raw["week"] == 18), "scored_touchdown"] = 0
    rows, _ = evaluation.prepare_evaluation_rows(raw, _identity_features)
    folds = evaluation.build_walk_forward_folds(rows, 2022, 2022, 2)
    params = {
        "n_estimators": 5,
        "max_depth": 2,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    _RecordingEstimator.fits = []
    with pytest.raises(ValueError, match="calibration partition.*both outcome classes"):
        evaluation.evaluate_folds(
            rows,
            folds,
            params,
            seed=42,
            estimator_factories={
                "logistic_l2": lambda seed, params: _RecordingEstimator("logistic_l2"),
                "random_forest_current": lambda seed, params: _RecordingEstimator("random_forest_current"),
            },
            calibrator_factory=_RecordingCalibrator,
        )
    assert _RecordingEstimator.fits == []

    class _NaNEstimator(_RecordingEstimator):
        def predict_proba(self, features):
            return np.column_stack([np.full(len(features), np.nan), np.full(len(features), np.nan)])

    rows, _ = evaluation.prepare_evaluation_rows(_raw_rows([(2021, 16), (2021, 17), (2021, 18), (2022, 1)]), _identity_features)
    folds = evaluation.build_walk_forward_folds(rows, 2022, 2022, 2)
    with pytest.raises(ValueError, match="non-finite"):
        evaluation.evaluate_folds(
            rows,
            folds,
            params,
            seed=42,
            estimator_factories={
                "logistic_l2": lambda seed, params: _NaNEstimator("logistic_l2"),
                "random_forest_current": lambda seed, params: _RecordingEstimator("random_forest_current"),
            },
            calibrator_factory=_RecordingCalibrator,
        )
