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
        p = np.full(len(features), 0.25 if self.kind == "logistic_l2" else 0.35)
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
