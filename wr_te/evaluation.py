"""Pure, offline helpers for the weekly WR/TE model evaluation.

This module deliberately has no data-loading or training-pipeline side effects.
Feature engineering is supplied by the caller so the evaluation population is
created once and is shared by every fold and model variant.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import PLAYER_EWM_STATS, WR_TE_FEATURES


GroupKey = tuple[int, int]
ROW_KEY_COLUMNS = ["season", "week", "game_id", "player_id"]
PREGAME_FEATURES = {"implied_total", "spread_line", "depth_chart_rank"}
MODEL_VARIANTS = (
    ("base_rate_overall", "raw"),
    ("base_rate_position", "raw"),
    ("logistic_l2", "raw"),
    ("logistic_l2", "platt"),
    ("random_forest_current", "raw"),
    ("random_forest_current", "platt"),
)

_DEFENSE_INPUTS = (
    "passing_tds_allowed_to_WR",
    "passing_tds_allowed_to_TE",
    "receiving_yards_allowed",
    "receiving_epa_allowed",
    "receiving_air_yards_allowed",
    "explosive_receiving_plays_allowed",
)
_RF_PARAMETER_NAMES = (
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
)
_MODEL_METADATA = (
    "player_display_name",
    "team",
    "opponent_team",
    "position",
    "scored_touchdown",
)
_PROVENANCE = "retrospective_finalish_game_context"


@dataclass(frozen=True)
class FoldSpec:
    test_group: GroupKey
    calibration_groups: tuple[GroupKey, ...]
    fit_groups: tuple[GroupKey, ...]


def _required_raw_columns() -> set[str]:
    return {
        *ROW_KEY_COLUMNS,
        "team",
        "player_display_name",
        "position",
        "scored_touchdown",
        "opponent_team",
        *PLAYER_EWM_STATS,
        *_DEFENSE_INPUTS,
    }


def _finite_feature_mask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return a row mask for numeric finite feature values."""
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    return pd.Series(np.isfinite(values).all(axis=1), index=frame.index)


def prepare_evaluation_rows(
    raw: pd.DataFrame,
    feature_engineer: Callable[[pd.DataFrame], pd.DataFrame],
    feature_columns: Sequence[str] = WR_TE_FEATURES,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build one canonical, finite WR/TE evaluation population.

    The supplied feature engineer is called exactly once on a deep copy of
    ``raw``.  Eligibility is then applied to the engineered result and the
    surviving rows are stably sorted by their immutable row key.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw must be a pandas DataFrame")
    feature_columns = tuple(feature_columns)
    unknown_features = set(feature_columns) - set(WR_TE_FEATURES)
    if unknown_features:
        raise ValueError(
            "feature contract permits only WR_TE_FEATURES; "
            f"unknown features={sorted(unknown_features)}"
        )

    missing = sorted(_required_raw_columns() - set(raw.columns))
    if missing:
        raise ValueError(f"raw frame is missing required columns: {missing}")

    engineered = feature_engineer(raw.copy(deep=True))
    if not isinstance(engineered, pd.DataFrame):
        raise TypeError("feature_engineer must return a pandas DataFrame")
    missing_features = sorted(set(WR_TE_FEATURES) - set(engineered.columns))
    if missing_features:
        raise ValueError(f"engineered frame is missing WR_TE_FEATURES: {missing_features}")

    counts: dict[str, int] = {"input_rows": len(raw)}
    position_mask = engineered["position"].isin(["WR", "TE"])
    counts["wr_te_rows"] = int(position_mask.sum())
    frame = engineered.loc[position_mask].copy()

    week_numeric = pd.to_numeric(frame["week"], errors="coerce")
    week_mask = week_numeric.between(1, 18, inclusive="both")
    counts["regular_season_rows"] = int(week_mask.sum())
    frame = frame.loc[week_mask].copy()

    key_mask = frame[ROW_KEY_COLUMNS].notna().all(axis=1)
    counts["non_null_key_rows"] = int(key_mask.sum())
    frame = frame.loc[key_mask].copy()

    outcome = pd.to_numeric(frame["scored_touchdown"], errors="coerce")
    outcome_mask = outcome.isin([0, 1])
    counts["binary_outcome_rows"] = int(outcome_mask.sum())
    frame = frame.loc[outcome_mask].copy()
    frame["scored_touchdown"] = pd.to_numeric(frame["scored_touchdown"], errors="raise").astype(int)

    finite_mask = _finite_feature_mask(frame, WR_TE_FEATURES)
    counts["finite_feature_rows"] = int(finite_mask.sum())
    frame = frame.loc[finite_mask].copy()
    counts["eligible_rows"] = len(frame)

    duplicate_mask = frame.duplicated(ROW_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_keys = frame.loc[duplicate_mask, ROW_KEY_COLUMNS].drop_duplicates().head(3)
        raise ValueError(
            "duplicate eligible row key(s): "
            f"{[tuple(x) for x in duplicate_keys.to_numpy()]}"
        )

    # The model only consumes numeric features. Converting numeric strings here
    # also makes the finite-value contract explicit without changing raw input.
    for column in WR_TE_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["evaluation_stream"] = np.where(
        pd.to_numeric(frame["season"], errors="coerce") == 2026,
        "prospective_2026",
        "retrospective",
    )
    frame = frame.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    return frame, counts


def _group_keys(rows: pd.DataFrame) -> list[GroupKey]:
    return sorted(
        {
            (int(season), int(week))
            for season, week in rows[["season", "week"]].itertuples(index=False, name=None)
        }
    )


def _group_mask(rows: pd.DataFrame, group: GroupKey) -> pd.Series:
    season, week = group
    return (rows["season"] == season) & (rows["week"] == week)


def _validate_game_group_mapping(rows: pd.DataFrame) -> None:
    game_groups = rows.groupby("game_id", sort=False)[["season", "week"]].nunique(dropna=False)
    bad = game_groups[(game_groups["season"] > 1) | (game_groups["week"] > 1)]
    if not bad.empty:
        raise ValueError("game_id maps to multiple season/week groups")


def _row_keys(rows: pd.DataFrame) -> set[tuple[Any, ...]]:
    return {tuple(value) for value in rows[ROW_KEY_COLUMNS].to_numpy()}


def build_walk_forward_folds(
    rows: pd.DataFrame,
    start_season: int,
    end_season: int,
    calibration_weeks: int,
) -> list[FoldSpec]:
    """Construct contiguous whole-week expanding walk-forward folds."""
    if calibration_weeks < 1:
        raise ValueError("calibration_weeks must be at least one")
    if start_season > end_season:
        raise ValueError("start_season must not exceed end_season")
    missing = sorted(set(ROW_KEY_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"rows is missing required columns: {missing}")
    if rows.duplicated(ROW_KEY_COLUMNS).any():
        raise ValueError("duplicate row key(s) in rows")
    _validate_game_group_mapping(rows)
    groups = _group_keys(rows)
    requested = [group for group in groups if start_season <= group[0] <= end_season]
    if not requested:
        return []

    earliest = requested[0]
    earliest_index = groups.index(earliest)
    available_history = earliest_index
    required_history = calibration_weeks + 1
    if available_history < required_history:
        raise ValueError(
            "earliest requested group "
            f"{earliest} has available history={available_history}, "
            f"required={required_history}"
        )

    folds: list[FoldSpec] = []
    for test_group in requested:
        index = groups.index(test_group)
        if index < required_history:
            raise ValueError(
                "earliest requested group "
                f"{test_group} has available history={index}, required={required_history}"
            )
        calibration_groups = tuple(groups[index - calibration_weeks : index])
        fit_groups = tuple(groups[: index - calibration_weeks])
        fold = FoldSpec(test_group, calibration_groups, fit_groups)
        _validate_fold_partitions(rows, fold)
        folds.append(fold)
    return folds


def _partition_rows(rows: pd.DataFrame, groups: Sequence[GroupKey]) -> pd.DataFrame:
    if not groups:
        return rows.iloc[0:0].copy()
    mask = pd.Series(False, index=rows.index)
    for group in groups:
        mask |= _group_mask(rows, group)
    return rows.loc[mask].copy()


def _validate_fold_partitions(rows: pd.DataFrame, fold: FoldSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_groups = tuple(fold.fit_groups)
    cal_groups = tuple(fold.calibration_groups)
    test_groups = (tuple(fold.test_group),)
    all_groups = [*fit_groups, *cal_groups, *test_groups]
    if len(set(all_groups)) != len(all_groups):
        raise ValueError(f"fold {fold.test_group} has overlapping partition groups")
    if len(cal_groups) == 0 or len(fit_groups) == 0:
        raise ValueError(f"fold {fold.test_group} must have fit and calibration groups")
    if tuple(sorted(fit_groups)) != fit_groups or tuple(sorted(cal_groups)) != cal_groups:
        raise ValueError(f"fold {fold.test_group} groups must be lexicographically ordered")
    if not (fit_groups[-1] < cal_groups[0] <= cal_groups[-1] < fold.test_group):
        raise ValueError(f"fold {fold.test_group} partitions are not strictly chronological")

    fit = _partition_rows(rows, fit_groups)
    calibration = _partition_rows(rows, cal_groups)
    test = _partition_rows(rows, test_groups)
    if fit.empty or calibration.empty or test.empty:
        raise ValueError(f"fold {fold.test_group} contains an empty partition")
    partitions = (fit, calibration, test)
    key_sets = [_row_keys(part) for part in partitions]
    if key_sets[0] & key_sets[1] or key_sets[0] & key_sets[2] or key_sets[1] & key_sets[2]:
        raise ValueError(f"fold {fold.test_group} row keys overlap across partitions")
    game_sets = [set(part["game_id"]) for part in partitions]
    if game_sets[0] & game_sets[1] or game_sets[0] & game_sets[2] or game_sets[1] & game_sets[2]:
        raise ValueError(f"fold {fold.test_group} game IDs overlap across partitions")
    return fit, calibration, test


def split_and_assert_fold(
    rows: pd.DataFrame, fold: FoldSpec
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return validated fit, calibration, and test frames for one fold."""
    return _validate_fold_partitions(rows, fold)


def _positive_probabilities(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features))
    if probabilities.ndim == 1:
        result = probabilities
    elif probabilities.ndim == 2 and probabilities.shape[1] == 2:
        classes = getattr(estimator, "classes_", None)
        positive_column = 1
        if classes is not None and 1 in classes:
            positive_column = int(np.flatnonzero(np.asarray(classes) == 1)[0])
        result = probabilities[:, positive_column]
    else:
        raise ValueError("estimator predict_proba must return one or two probability columns")
    return np.asarray(result, dtype=float)


def _validate_probabilities(probabilities: Any, expected_rows: int, label: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(values) != expected_rows:
        raise ValueError(f"{label} returned {len(values)} probabilities; expected {expected_rows}")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{label} returned a non-finite or out-of-range probability")
    return values


def _default_estimator_factory(model: str, seed: int, params: Mapping[str, Any]) -> Any:
    if model == "logistic_l2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
        )
    if model == "random_forest_current":
        return RandomForestClassifier(**dict(params), random_state=seed, n_jobs=1)
    raise ValueError(f"unknown learned model: {model}")


def evaluate_folds(
    rows: pd.DataFrame,
    folds: Sequence[FoldSpec],
    rf_params: Mapping[str, Any],
    seed: int,
    estimator_factories: Mapping[str, Callable[[int, Mapping[str, Any]], Any]] | None = None,
    calibrator_factory: Callable[[], Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fixed models on each expanding fit partition and predict its test week."""
    missing = sorted(set(ROW_KEY_COLUMNS + ["position", "scored_touchdown"] + list(WR_TE_FEATURES)) - set(rows.columns))
    if missing:
        raise ValueError(f"rows is missing required evaluation columns: {missing}")
    if not isinstance(rf_params, Mapping):
        raise TypeError("rf_params must be a mapping")
    missing_rf = [name for name in _RF_PARAMETER_NAMES if name not in rf_params]
    if missing_rf:
        raise ValueError(f"rf_params is missing required parameters: {missing_rf}")
    effective_rf_params = {name: rf_params[name] for name in _RF_PARAMETER_NAMES}
    if rows.duplicated(ROW_KEY_COLUMNS).any():
        raise ValueError("duplicate row key(s) in rows")
    _validate_game_group_mapping(rows)

    # Validate every fold completely before fitting any estimator.
    ordered_folds = sorted(folds, key=lambda fold: tuple(fold.test_group))
    if len({tuple(fold.test_group) for fold in ordered_folds}) != len(ordered_folds):
        raise ValueError("duplicate test group in folds")
    validated: list[tuple[FoldSpec, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for fold in ordered_folds:
        fit, calibration, test = _validate_fold_partitions(rows, fold)
        fit_outcomes = set(fit["scored_touchdown"])
        if fit_outcomes != {0, 1}:
            raise ValueError(f"fit partition for {fold.test_group} must contain both outcome classes")
        if not {"WR", "TE"}.issubset(set(fit["position"])):
            raise ValueError(f"fit partition for {fold.test_group} must contain both WR and TE rows")
        if set(calibration["scored_touchdown"]) != {0, 1}:
            raise ValueError(
                f"calibration partition for {fold.test_group} must contain both outcome classes"
            )
        validated.append((fold, fit, calibration, test))

    factories = estimator_factories or {}
    calibrator_ctor = calibrator_factory or LogisticRegression
    prediction_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    feature_columns = list(WR_TE_FEATURES)
    metadata_columns = [column for column in ROW_KEY_COLUMNS + ["evaluation_stream"] + list(_MODEL_METADATA) if column in rows.columns]

    for fold_number, (fold, fit, calibration, test) in enumerate(validated):
        test = test.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
        fit = fit.sort_values(ROW_KEY_COLUMNS, kind="mergesort")
        calibration = calibration.sort_values(ROW_KEY_COLUMNS, kind="mergesort")
        x_fit, y_fit = fit[feature_columns], fit["scored_touchdown"].to_numpy()
        x_cal, y_cal = calibration[feature_columns], calibration["scored_touchdown"].to_numpy()
        x_test = test[feature_columns]
        test_keys = list(map(tuple, test[ROW_KEY_COLUMNS].to_numpy()))
        fold_stream = str(test["evaluation_stream"].iloc[0]) if "evaluation_stream" in test else "retrospective"
        variant_parts: list[pd.DataFrame] = []
        first_variant_keys: list[tuple[Any, ...]] | None = None

        def add_variant(model: str, variant: str, probabilities: Any) -> None:
            nonlocal first_variant_keys
            checked = _validate_probabilities(probabilities, len(test), f"{model}/{variant}")
            part = test.loc[:, metadata_columns].copy()
            part["model"] = model
            part["variant"] = variant
            part["probability"] = checked
            part["fold"] = fold_number
            part["football_context_provenance"] = _PROVENANCE
            variant_keys = list(map(tuple, part[ROW_KEY_COLUMNS].to_numpy()))
            if first_variant_keys is None:
                first_variant_keys = variant_keys
            elif variant_keys != first_variant_keys:
                raise ValueError(f"{model}/{variant} returned a different test row-key set")
            variant_parts.append(part)

        overall_rate = float(y_fit.mean())
        position_rates = fit.groupby("position")["scored_touchdown"].mean().to_dict()
        add_variant("base_rate_overall", "raw", np.full(len(test), overall_rate))
        add_variant(
            "base_rate_position",
            "raw",
            test["position"].map(position_rates).to_numpy(dtype=float),
        )

        for model, params in (("logistic_l2", {}), ("random_forest_current", effective_rf_params)):
            factory = factories.get(model)
            estimator = factory(seed, params) if factory is not None else _default_estimator_factory(model, seed, params)
            estimator.fit(x_fit, y_fit)
            calibration_raw = _validate_probabilities(
                _positive_probabilities(estimator, x_cal), len(calibration), f"{model}/calibration"
            )
            test_raw = _validate_probabilities(
                _positive_probabilities(estimator, x_test), len(test), f"{model}/test"
            )
            calibrator = calibrator_ctor()
            calibrator.fit(np.clip(calibration_raw, 1e-15, 1 - 1e-15).reshape(-1, 1), y_cal)
            test_platt = _validate_probabilities(
                _positive_probabilities(calibrator, test_raw.reshape(-1, 1)),
                len(test),
                f"{model}/platt",
            )
            add_variant(model, "raw", test_raw)
            add_variant(model, "platt", test_platt)
        prediction_parts.extend(variant_parts)

        fold_records.append(
            {
                "football_context_provenance": _PROVENANCE,
                "evaluation_stream": fold_stream,
                "fold": fold_number,
                "test_group": fold.test_group,
                "test_season": fold.test_group[0],
                "test_week": fold.test_group[1],
                "fit_groups": fold.fit_groups,
                "calibration_groups": fold.calibration_groups,
                "fit_group_count": len(fold.fit_groups),
                "calibration_group_count": len(fold.calibration_groups),
                "fit_rows": len(fit),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "fit_event_rate": float(y_fit.mean()),
                "calibration_event_rate": float(y_cal.mean()),
                "test_event_rate": float(test["scored_touchdown"].mean()),
                "status": "ok",
            }
        )

    if prediction_parts:
        predictions = pd.concat(prediction_parts, ignore_index=True)
        predictions = predictions.sort_values(
            ROW_KEY_COLUMNS + ["model", "variant"], kind="mergesort"
        ).reset_index(drop=True)
    else:
        predictions = pd.DataFrame(
            columns=metadata_columns + ["model", "variant", "probability", "fold", "football_context_provenance"]
        )
    fold_frame = pd.DataFrame(fold_records)
    if not fold_frame.empty:
        fold_frame = fold_frame.sort_values("test_group", kind="mergesort").reset_index(drop=True)
    return predictions, fold_frame
