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
import ledger
import odds_match


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


# The evaluation odds helpers intentionally accept frames rather than paths.
# This keeps source selection and provenance decisions in the offline runner,
# while leaving the production prediction and ledger flows unchanged.
_ODDS_COLUMNS = ("description", "home_team", "away_team", "price", "bookmaker")
_LEGACY_ODDS_COLUMNS = {
    "Player": "description",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "Odds": "price",
    "Bookmaker": "bookmaker",
    "Season": "season",
    "Week": "week",
}


def _empty_odds_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_ODDS_COLUMNS, "season", "week"])


def _normalize_odds_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return the common odds_match schema without mutating ``frame``."""
    if frame is None:
        return _empty_odds_frame()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("odds source must be a pandas DataFrame")
    result = frame.rename(columns=_LEGACY_ODDS_COLUMNS).copy(deep=True)
    missing = [column for column in _ODDS_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"odds source is missing required columns: {missing}")
    result["price"] = pd.to_numeric(result["price"], errors="coerce")
    for column in ("season", "week"):
        if column not in result.columns:
            result[column] = pd.NA
    return result


def _timestamp_safe_open(frame: pd.DataFrame) -> bool:
    required = {"tag", "in_play", "last_update", "fetched_at", "commence_time"}
    if frame.empty or not required.issubset(frame.columns):
        return False

    def nonempty(series: pd.Series) -> pd.Series:
        return series.notna() & series.astype(str).str.strip().ne("")

    if not (nonempty(frame["tag"]).all() and frame["tag"].astype(str).str.lower().eq("open").all()):
        return False
    def explicit_false(value: Any) -> bool:
        # CSV round-trips of the fetcher's boolean column are represented as
        # numpy.bool_, while a string "False" is the only accepted textual
        # representation.  Numeric and unrecognised values are unsafe.
        if isinstance(value, (bool, np.bool_)):
            return not bool(value)
        return isinstance(value, str) and value == "False"

    if not frame["in_play"].map(explicit_false).all():
        return False
    if not all(nonempty(frame[column]).all() for column in ("last_update", "fetched_at", "commence_time")):
        return False
    updates = pd.to_datetime(frame["last_update"], errors="coerce", utc=True)
    fetched = pd.to_datetime(frame["fetched_at"], errors="coerce", utc=True)
    commence = pd.to_datetime(frame["commence_time"], errors="coerce", utc=True)
    if updates.isna().any() or fetched.isna().any() or commence.isna().any():
        return False
    return bool((updates < commence).all() and (fetched < commence).all())


def select_open_odds(
    tagged_open: pd.DataFrame | None,
    legacy: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Select and normalize a timestamp-safe open or legacy odds snapshot.

    A malformed tagged snapshot is rejected as a whole.  The warning is kept
    as a record rather than logging so callers can publish deterministic
    manifests without making this pure helper perform I/O.
    """
    warnings: list[str] = []
    if tagged_open is not None:
        if _timestamp_safe_open(tagged_open):
            return _normalize_odds_frame(tagged_open), "timestamp_safe_open", warnings
        if not tagged_open.empty:
            warnings.append("rejected tagged open snapshot: timestamp/open fields were unsafe")

    if legacy is not None and not legacy.empty:
        return _normalize_odds_frame(legacy), "timestamp_unsafe_legacy", warnings
    return _empty_odds_frame(), "uncovered", warnings


def _provenance_label(provenance: str) -> str:
    if provenance == "timestamp_unsafe_legacy":
        return "research-only, timestamp unsafe"
    if provenance == "timestamp_safe_open":
        return "timestamp-safe open"
    if provenance == "uncovered":
        return "uncovered"
    return "mixed provenance"


def _key_tuple(row: pd.Series | Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in ROW_KEY_COLUMNS)


def _coverage_record(
    covered_keys: set[tuple[Any, ...]],
    eligible: pd.DataFrame,
) -> tuple[int, int, float | None, list[dict[str, Any]]]:
    total = len(eligible)
    covered = len(covered_keys)
    overall = covered / total if total else None
    by_week: list[dict[str, Any]] = []
    for (season, week), group in eligible.groupby(["season", "week"], sort=True):
        keys = {_key_tuple(row) for _, row in group.iterrows()}
        n_covered = len(keys & covered_keys)
        n_total = len(keys)
        by_week.append({
            "season": int(season),
            "week": int(week),
            "open_coverage_count": n_covered,
            "open_coverage_total": n_total,
            "open_coverage_rate": n_covered / n_total if n_total else None,
        })
    return covered, total, overall, by_week


def attach_open_odds_and_score_bets(
    predictions: pd.DataFrame,
    odds_by_group: Mapping[GroupKey, tuple[pd.DataFrame, str]],
    team_map: Mapping[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Attach matched open odds while preserving every prediction row.

    Matching is done once per unique football row key and then broadcast to
    each model variant.  Bets are only scored for learned variants and are
    selected independently within each test week.
    """
    required = set(ROW_KEY_COLUMNS + [
        "player_display_name", "team", "probability", "scored_touchdown", "model", "variant"
    ])
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions is missing required odds columns: {missing}")
    result = predictions.copy(deep=True)
    result["odds_covered"] = False
    result["price_open"] = np.nan
    result["bookmaker_open"] = pd.NA
    result["open_provenance"] = pd.NA

    unique_columns = list(dict.fromkeys(ROW_KEY_COLUMNS + [
        "player_display_name", "team", "opponent_team", "position", "evaluation_stream"
    ]))
    unique_columns = [column for column in unique_columns if column in result.columns]
    eligible = result[unique_columns].drop_duplicates(ROW_KEY_COLUMNS, keep="first").copy()
    eligible = eligible.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    eligible["evaluation_row_id"] = np.arange(len(eligible), dtype=int)
    odds_for_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    provenance_by_group = {
        (int(group[0]), int(group[1])): str(source[1])
        for group, source in odds_by_group.items()
    }

    for group, source in sorted(odds_by_group.items(), key=lambda item: tuple(item[0])):
        odds, provenance = source
        group_rows = eligible[
            (pd.to_numeric(eligible["season"], errors="coerce") == int(group[0]))
            & (pd.to_numeric(eligible["week"], errors="coerce") == int(group[1]))
        ].copy()
        if group_rows.empty:
            continue
        normalized = _normalize_odds_frame(odds)
        normalized = normalized[np.isfinite(normalized["price"].to_numpy(dtype=float))]
        if normalized.empty:
            continue
        players = group_rows[[
            "evaluation_row_id", "player_display_name", "team"
        ]].copy()
        matched = odds_match.match_odds_to_players(players, normalized, dict(team_map))
        for _, match in matched.iterrows():
            player_row = group_rows[group_rows["evaluation_row_id"] == match["evaluation_row_id"]].iloc[0]
            odds_for_key[_key_tuple(player_row)] = {
                "price": match["price"],
                "bookmaker": match["bookmaker"],
                "provenance": str(provenance),
            }

    for index, row in result.iterrows():
        key = _key_tuple(row)
        quote = odds_for_key.get(key)
        if quote is None:
            group = (int(row["season"]), int(row["week"]))
            result.at[index, "open_provenance"] = provenance_by_group.get(group, "uncovered")
            continue
        result.at[index, "odds_covered"] = True
        result.at[index, "price_open"] = quote["price"]
        result.at[index, "bookmaker_open"] = quote["bookmaker"]
        result.at[index, "open_provenance"] = quote["provenance"]

    football = eligible
    all_covered = set(odds_for_key)
    result_records: list[dict[str, Any]] = []
    for stream, stream_rows in result.groupby(
        "evaluation_stream" if "evaluation_stream" in result else pd.Series("retrospective", index=result.index),
        sort=True,
    ):
        stream = str(stream)
        stream_eligible = football[
            football.get("evaluation_stream", pd.Series("retrospective", index=football.index)).astype(str) == stream
        ]
        stream_keys = {_key_tuple(row) for _, row in stream_eligible.iterrows()}
        stream_covered = stream_keys & all_covered
        coverage_count, coverage_total, coverage_rate, by_week = _coverage_record(stream_covered, stream_eligible)
        stream_groups = {
            (int(row["season"]), int(row["week"])) for _, row in stream_eligible.iterrows()
        }
        source_values = {provenance_by_group[group] for group in stream_groups if group in provenance_by_group}
        if not source_values:
            provenance = "uncovered"
        elif len(source_values) == 1:
            provenance = next(iter(source_values))
        else:
            provenance = "mixed"

        stream_prediction = stream_rows.copy()
        for model, variant in sorted(
            {(str(model), str(variant)) for model, variant in stream_prediction[["model", "variant"]].itertuples(index=False, name=None)}
        ):
            if model in {"logistic_l2", "random_forest_current"} and variant not in {"raw", "platt"}:
                continue
            model_rows = stream_prediction[
                (stream_prediction["model"] == model) & (stream_prediction["variant"] == variant)
            ].copy()
            common = {
                "evaluation_stream": stream,
                "model": model,
                "variant": variant,
                "bets": None,
                "settled_bets": None,
                "stake": None,
                "pnl": None,
                "roi": None,
                "hits": None,
                "hit_rate": None,
                "open_coverage_count": None,
                "open_coverage_total": None,
                "open_coverage_rate": None,
                "coverage_by_week": None,
                "betting_provenance": None,
                "betting_label": None,
            }
            if model in {"logistic_l2", "random_forest_current"} and variant in {"raw", "platt"}:
                common.update({
                    "open_coverage_count": coverage_count,
                    "open_coverage_total": coverage_total,
                    "open_coverage_rate": coverage_rate,
                    "coverage_by_week": by_week,
                    "betting_provenance": provenance,
                    "betting_label": _provenance_label(provenance),
                })
                selected_parts: list[pd.DataFrame] = []
                model_rows["_covered"] = model_rows["odds_covered"].fillna(False).astype(bool)
                for _, week_rows in model_rows[model_rows["_covered"]].groupby(["season", "week"], sort=True):
                    selected_parts.append(
                        week_rows.sort_values(
                            ["probability", "player_id"], ascending=[False, True], kind="mergesort"
                        ).head(5)
                    )
                selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else model_rows.iloc[0:0]
                bets = len(selected)
                settled_mask = pd.to_numeric(selected["scored_touchdown"], errors="coerce").isin([0, 1])
                settled = selected.loc[settled_mask]
                pnl = 0.0
                hits = 0
                for _, bet in settled.iterrows():
                    hit = int(bet["scored_touchdown"]) == 1
                    hits += int(hit)
                    pnl += ledger.decimal_odds(bet["price_open"]) - 1 if hit else -1.0
                stake = float(bets)
                common.update({
                    "bets": bets,
                    "settled_bets": len(settled),
                    "stake": stake,
                    "pnl": pnl,
                    "roi": pnl / stake if stake else None,
                    "hits": hits,
                    "hit_rate": hits / len(settled) if len(settled) else None,
                })
            result_records.append(common)
    result_records.sort(key=lambda row: (row["evaluation_stream"], row["model"], row["variant"]))
    return result, result_records


def _calibration_bins(probabilities: np.ndarray, outcomes: np.ndarray) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    bin_ids = np.minimum((probabilities * 10).astype(int), 9)
    for index in range(10):
        selected = bin_ids == index
        count = int(selected.sum())
        if count:
            mean_probability = float(probabilities[selected].mean())
            event_rate = float(outcomes[selected].mean())
            gap = mean_probability - event_rate
        else:
            mean_probability = None
            event_rate = None
            gap = None
        bins.append({
            "bin": index,
            "lower": index / 10,
            "upper": 1.0 if index == 9 else (index + 1) / 10,
            "count": count,
            "mean_probability": mean_probability,
            "event_rate": event_rate,
            "gap": gap,
        })
    return bins


def _metric_values(frame: pd.DataFrame) -> dict[str, Any]:
    probabilities = pd.to_numeric(frame["probability"], errors="coerce").to_numpy(dtype=float)
    outcomes = pd.to_numeric(frame["scored_touchdown"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.isfinite(outcomes).all():
        raise ValueError("metrics require finite probabilities and outcomes")
    if ((outcomes < 0) | (outcomes > 1) | (outcomes != outcomes.astype(int))).any():
        raise ValueError("metrics require binary outcomes")
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    log_loss = float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log1p(-clipped)))
    brier = float(np.mean((probabilities - outcomes) ** 2))
    calibration_bins = _calibration_bins(probabilities, outcomes)
    ece = float(sum(
        bin_record["count"] / len(frame) * abs(bin_record["gap"])
        for bin_record in calibration_bins
        if bin_record["count"]
    )) if len(frame) else None
    return {
        "rows": len(frame),
        "weeks": int(frame[["season", "week"]].drop_duplicates().shape[0]),
        "log_loss": log_loss,
        "brier": brier,
        "ece": ece,
        "calibration_bins": calibration_bins,
    }


def build_metric_records(
    predictions: pd.DataFrame,
    betting_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build pooled, stream-separated primary metrics and flat summaries."""
    required = set(ROW_KEY_COLUMNS + ["model", "variant", "probability", "scored_touchdown", "season", "week"])
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions is missing required metric columns: {missing}")
    frame = predictions.copy()
    if "evaluation_stream" not in frame.columns:
        frame["evaluation_stream"] = "retrospective"
    duplicate = frame.duplicated(ROW_KEY_COLUMNS + ["evaluation_stream", "model", "variant"])
    if duplicate.any():
        raise ValueError("duplicate prediction metric key")

    betting_by_key = {
        (str(record.get("evaluation_stream", "retrospective")), str(record["model"]), str(record["variant"])): dict(record)
        for record in betting_records
    }
    metric_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for stream, stream_frame in frame.groupby("evaluation_stream", sort=True):
        stream = str(stream)
        reference = stream_frame[
            (stream_frame["model"] == "base_rate_overall") & (stream_frame["variant"] == "raw")
        ]
        if reference.empty:
            raise ValueError("reference key mismatch")
        reference = reference.sort_values(ROW_KEY_COLUMNS, kind="mergesort")
        reference_keys = set(map(tuple, reference[ROW_KEY_COLUMNS].to_numpy()))
        ref_values = _metric_values(reference)
        for (model, variant), model_frame in sorted(
            stream_frame.groupby(["model", "variant"], sort=True), key=lambda item: (str(item[0][0]), str(item[0][1]))
        ):
            model_frame = model_frame.sort_values(ROW_KEY_COLUMNS, kind="mergesort")
            model_keys = set(map(tuple, model_frame[ROW_KEY_COLUMNS].to_numpy()))
            if model_keys != reference_keys or len(model_frame) != len(reference):
                raise ValueError("reference key mismatch")
            values = _metric_values(model_frame)
            reference_name = "base_rate_overall/raw"
            denominator = ref_values["brier"]
            record = {
                "evaluation_stream": stream,
                "model": str(model),
                "variant": str(variant),
                **values,
                "log_loss_delta_vs_reference": values["log_loss"] - ref_values["log_loss"],
                "brier_delta_vs_reference": values["brier"] - ref_values["brier"],
                "brier_skill": 1 - values["brier"] / denominator if denominator else None,
                "brier_reference": reference_name,
            }
            metric_records.append(record)
            flat = {key: value for key, value in record.items() if key != "calibration_bins"}
            flat.update({
                "football_context_provenance": str(stream_frame["football_context_provenance"].iloc[0])
                if "football_context_provenance" in stream_frame.columns else _PROVENANCE,
            })
            betting_columns = (
                "bets", "settled_bets", "stake", "pnl", "roi", "hits", "hit_rate",
                "open_coverage_count", "open_coverage_total", "open_coverage_rate",
                "betting_provenance", "betting_label",
            )
            betting_record = betting_by_key.get((stream, str(model), str(variant)), {})
            flat.update({key: betting_record.get(key) for key in betting_columns})
            summary_records.append(flat)

    metric_records.sort(key=lambda row: (row["evaluation_stream"], row["model"], row["variant"]))
    summary_records.sort(key=lambda row: (row["evaluation_stream"], row["model"], row["variant"]))
    return {"overall": metric_records}, pd.DataFrame(summary_records)
