"""Offline, deterministic walk-forward evaluation runner.

The pure evaluation and model logic lives in :mod:`evaluation`.  This module
only reads the explicitly supplied local cache, assembles artifact payloads in
memory, and publishes the five known files as one atomic batch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
import evaluation
from features import WR_TE_FEATURES
from train_wr import feature_engineering


ARTIFACT_NAMES = (
    "manifest.json",
    "folds.csv",
    "predictions.csv",
    "metrics.json",
    "summary.csv",
)
_MAX_PREDECLARED_SEASON = 2026
_DEFERRED = [
    "calibration_intercept_slope",
    "closing_price_clv",
    "maximum_drawdown",
    "pr_auc",
    "roc_auc",
    "season_position_slices",
    "top_k_accuracy",
    "week_cluster_bootstrap",
]
_FOLD_COLUMNS = [
    "football_context_provenance", "evaluation_stream", "test_season", "test_week",
    "fit_start", "fit_end", "calibration_start", "calibration_end",
    "fit_group_count", "calibration_group_count", "fit_rows", "calibration_rows",
    "test_rows", "fit_event_rate", "calibration_event_rate", "test_event_rate", "status",
]
_PREDICTION_COLUMNS = [
    "football_context_provenance", "evaluation_stream", "season", "week", "game_id",
    "player_id", "player_display_name", "team", "opponent_team", "position",
    "scored_touchdown", "model", "variant", "probability", "fold", "odds_covered",
    "price_open", "bookmaker_open", "open_provenance",
]
_SUMMARY_COLUMNS = [
    "football_context_provenance", "evaluation_stream", "model", "variant", "rows", "weeks",
    "log_loss", "log_loss_delta_vs_reference", "brier", "brier_delta_vs_reference",
    "brier_skill", "brier_reference", "ece", "bets", "settled_bets", "stake", "pnl",
    "roi", "hits", "hit_rate", "open_coverage_count", "open_coverage_total",
    "open_coverage_rate", "betting_provenance", "betting_label",
]


def _validate_range(start_season: int, end_season: int, calibration_weeks: int) -> None:
    if start_season > end_season:
        raise ValueError("start_season must not exceed end_season")
    if calibration_weeks < 1:
        raise ValueError("calibration_weeks must be at least one")
    if end_season > _MAX_PREDECLARED_SEASON:
        raise ValueError(
            f"end_season cannot exceed the predeclared {_MAX_PREDECLARED_SEASON} stream"
        )


def resolve_output_dir(
    requested: Path | None,
    start_season: int,
    end_season: int,
    calibration_weeks: int,
    seed: int,
    evaluation_dir: Path = config.EVALUATION_DIR,
) -> Path:
    """Resolve a requested run directory, rejecting paths outside evaluation_dir."""
    _validate_range(start_season, end_season, calibration_weeks)
    root = Path(evaluation_dir).expanduser().resolve()
    if requested is None:
        candidate = root / f"walk_forward_{start_season}_{end_season}_cal{calibration_weeks}_seed{seed}"
    else:
        requested_path = Path(requested).expanduser()
        candidate = requested_path if requested_path.is_absolute() else root / requested_path
        candidate = candidate.resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("output directory must be beneath evaluation directory")
    return candidate


def _read_bytes(path: Path, logical_name: str, hashes: dict[str, str]) -> bytes:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing required input: {path}")
    payload = path.read_bytes()
    hashes[logical_name] = hashlib.sha256(payload).hexdigest()
    return payload


def _read_csv(path: Path, logical_name: str, hashes: dict[str, str]) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_read_bytes(path, logical_name, hashes)))


def _plain(value: Any) -> Any:
    """Convert numpy/pandas scalars while rejecting non-finite JSON numbers."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("JSON payload contains a non-finite number")
        return number
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _plain(payload), sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"artifact frame is missing columns: {missing}")
    return frame.loc[:, list(columns)].to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")


def write_artifacts_atomic(
    output_dir: Path,
    payloads: Mapping[str, bytes],
) -> dict[str, Path]:
    """Publish exactly the known artifacts after all temporary writes succeed."""
    if set(payloads) != set(ARTIFACT_NAMES):
        raise ValueError("payloads must contain exactly the known artifacts")
    if any(not isinstance(payloads[name], (bytes, bytearray)) for name in ARTIFACT_NAMES):
        raise TypeError("artifact payloads must be bytes")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    temporary: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    backup_temporary: set[Path] = set()
    published: set[str] = set()
    try:
        for name in ARTIFACT_NAMES:
            fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=destination)
            temp_path = Path(temp_name)
            temporary[name] = temp_path
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(bytes(payloads[name]))
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

        # Keep a private copy of every predecessor before replacing anything.
        # Copies avoid disturbing the visible old artifact set until the first
        # new destination is atomically installed.
        for name in ARTIFACT_NAMES:
            target = destination / name
            if target.exists():
                fd, backup_temp_name = tempfile.mkstemp(
                    prefix=f".{name}.", suffix=".backup-tmp", dir=destination
                )
                os.close(fd)
                backup_temp = Path(backup_temp_name)
                backup_temporary.add(backup_temp)
                shutil.copyfile(target, backup_temp)
                backup_path = backup_temp.with_name(
                    backup_temp.name.removesuffix(".backup-tmp") + ".backup"
                )
                os.replace(backup_temp, backup_path)
                backup_temporary.remove(backup_temp)
                backups[name] = backup_path

        for name in ARTIFACT_NAMES:
            os.replace(temporary[name], destination / name)
            temporary.pop(name, None)
            published.add(name)
    except BaseException:
        # Restore every predecessor, including destinations that were not yet
        # reached by the replacement loop.  A destination without a backup did
        # not exist before this transaction and must not survive a rollback.
        for name in reversed(ARTIFACT_NAMES):
            target = destination / name
            backup_path = backups.get(name)
            if backup_path is not None and backup_path.exists():
                try:
                    os.replace(backup_path, target)
                except OSError:
                    # If the injected/system failure also affects restore,
                    # copy the predecessor back before final cleanup.
                    shutil.copyfile(backup_path, target)
            elif name in published and target.exists():
                target.unlink()
        raise
    finally:
        for temp_path in temporary.values():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        for backup_path in backups.values():
            try:
                backup_path.unlink()
            except FileNotFoundError:
                pass
        for backup_path in backup_temporary:
            try:
                backup_path.unlink()
            except FileNotFoundError:
                pass
    return {name: destination / name for name in ARTIFACT_NAMES}


def _format_group(group: Sequence[int]) -> str:
    return f"{int(group[0])}-W{int(group[1]):02d}"


def _fold_artifact_frame(folds: Sequence[evaluation.FoldSpec], fold_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    rows_by_group = {
        (int(row.test_season), int(row.test_week)): row
        for row in fold_rows.itertuples(index=False)
    }
    for fold in sorted(folds, key=lambda item: tuple(item.test_group)):
        source = rows_by_group[tuple(fold.test_group)]
        records.append({
            "football_context_provenance": "retrospective_finalish_game_context",
            "evaluation_stream": source.evaluation_stream,
            "test_season": int(source.test_season),
            "test_week": int(source.test_week),
            "fit_start": _format_group(fold.fit_groups[0]),
            "fit_end": _format_group(fold.fit_groups[-1]),
            "calibration_start": _format_group(fold.calibration_groups[0]),
            "calibration_end": _format_group(fold.calibration_groups[-1]),
            "fit_group_count": int(source.fit_group_count),
            "calibration_group_count": int(source.calibration_group_count),
            "fit_rows": int(source.fit_rows),
            "calibration_rows": int(source.calibration_rows),
            "test_rows": int(source.test_rows),
            "fit_event_rate": float(source.fit_event_rate),
            "calibration_event_rate": float(source.calibration_event_rate),
            "test_event_rate": float(source.test_event_rate),
            "status": str(source.status),
        })
    return pd.DataFrame(records, columns=_FOLD_COLUMNS)


def _validate_shared_prediction_keys(predictions: pd.DataFrame) -> None:
    required = set(evaluation.ROW_KEY_COLUMNS + ["model", "variant", "probability"])
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions are missing required columns: {missing}")
    probabilities = pd.to_numeric(predictions["probability"], errors="coerce").to_numpy(float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("predictions contain a non-finite or out-of-range probability")
    expected = set(evaluation.MODEL_VARIANTS)
    actual = set(zip(predictions["model"], predictions["variant"]))
    if actual != expected:
        raise ValueError(f"predictions must contain exactly the six model variants: {sorted(actual)}")
    reference: list[tuple[Any, ...]] | None = None
    for model, variant in evaluation.MODEL_VARIANTS:
        part = predictions[(predictions["model"] == model) & (predictions["variant"] == variant)]
        keys = list(map(tuple, part.sort_values(evaluation.ROW_KEY_COLUMNS, kind="mergesort")[evaluation.ROW_KEY_COLUMNS].to_numpy()))
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ValueError(f"{model}/{variant} returned a different shared row-key set")


def _load_team_map(team: pd.DataFrame) -> dict[str, str]:
    required = {"team_name", "team_id"}
    missing = sorted(required - set(team.columns))
    if missing:
        raise ValueError(f"team cache is missing required columns: {missing}")
    return dict(zip(team["team_name"].astype(str), team["team_id"].astype(str)))


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scikit-learn")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unknown"
    return result


def run_evaluation(
    *,
    start_season: int,
    end_season: int,
    calibration_weeks: int,
    seed: int,
    input_path: Path,
    rf_params_path: Path,
    vegas_dir: Path,
    team_path: Path,
    output_dir: Path,
    feature_engineer: Callable[[pd.DataFrame], pd.DataFrame] = feature_engineering,
    estimator_factories: Mapping[str, Callable] | None = None,
) -> dict[str, Path]:
    """Run the complete local evaluation and atomically publish its artifacts."""
    _validate_range(start_season, end_season, calibration_weeks)
    hashes: dict[str, str] = {}
    raw = _read_csv(Path(input_path), "raw_nfl_data.csv", hashes)
    rf_bytes = _read_bytes(Path(rf_params_path), "wr_te_rf_best_params.json", hashes)
    team = _read_csv(Path(team_path), "nfl_teams.csv", hashes)
    try:
        rf_params = json.loads(rf_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("RF parameter file must contain valid JSON") from exc
    if not isinstance(rf_params, dict) or set(rf_params) != {
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"
    }:
        raise ValueError("RF parameter file must contain exactly the five saved parameters")
    _plain(rf_params)
    rf_hash = hashlib.sha256(_json_bytes(rf_params)).hexdigest()

    rows, filter_counts = evaluation.prepare_evaluation_rows(raw, feature_engineer)
    folds = evaluation.build_walk_forward_folds(rows, start_season, end_season, calibration_weeks)
    predictions, fold_rows = evaluation.evaluate_folds(
        rows, folds, rf_params, seed, estimator_factories=estimator_factories
    )
    _validate_shared_prediction_keys(predictions)

    odds_by_group: dict[evaluation.GroupKey, tuple[pd.DataFrame, str]] = {}
    warnings: list[str] = []
    requested_groups = [tuple(fold.test_group) for fold in sorted(folds, key=lambda item: tuple(item.test_group))]
    for season, week in requested_groups:
        group_dir = Path(vegas_dir) / str(season)
        open_path = group_dir / f"week_{week}_td_odds_open.csv"
        legacy_path = group_dir / f"week_{week}_td_odds.csv"
        tagged = _read_csv(open_path, f"vegas/{season}/week_{week}_td_odds_open.csv", hashes) if open_path.is_file() else None
        legacy = _read_csv(legacy_path, f"vegas/{season}/week_{week}_td_odds.csv", hashes) if legacy_path.is_file() else None
        selected, provenance, odds_warnings = evaluation.select_open_odds(tagged, legacy)
        odds_by_group[(int(season), int(week))] = (selected, provenance)
        warnings.extend(f"{season}/{week}: {warning}" for warning in odds_warnings)
    predictions, betting_records = evaluation.attach_open_odds_and_score_bets(
        predictions, odds_by_group, _load_team_map(team)
    )
    _validate_shared_prediction_keys(predictions)
    metrics, summary = evaluation.build_metric_records(predictions, betting_records)

    predictions = predictions.sort_values(
        ["evaluation_stream", *evaluation.ROW_KEY_COLUMNS, "model", "variant"], kind="mergesort"
    ).reset_index(drop=True)
    fold_frame = _fold_artifact_frame(folds, fold_rows)
    summary = summary.copy()
    summary = summary.sort_values(["evaluation_stream", "model", "variant"], kind="mergesort").reset_index(drop=True)
    summary = summary.reindex(columns=_SUMMARY_COLUMNS)
    for column in _SUMMARY_COLUMNS:
        if column not in summary:
            summary[column] = None
    summary = summary.loc[:, _SUMMARY_COLUMNS]

    metric_definitions = {
        "primary_selection_metrics": ["log_loss", "brier", "brier_skill", "paired_deltas"],
        "roi_used_for_selection": False,
        "log_loss": "pooled out-of-sample binary log loss",
        "brier": "pooled out-of-sample mean squared probability error",
        "ece": "weighted ten-bin absolute calibration gap",
    }
    metrics_payload = {
        "football_context_provenance": "retrospective_finalish_game_context",
        "metrics": metrics,
        "overall": metrics.get("overall", []),
        "definitions": metric_definitions,
        "betting": betting_records,
    }
    odds_provenance_by_group = {
        f"{season}/{week}": str(provenance)
        for (season, week), (_, provenance) in sorted(odds_by_group.items())
    }
    source_provenance = sorted(set(odds_provenance_by_group.values()))
    manifest = {
        "start_season": int(start_season),
        "end_season": int(end_season),
        "calibration_weeks": int(calibration_weeks),
        "seed": int(seed),
        "arguments": {
            "start_season": int(start_season), "end_season": int(end_season),
            "calibration_weeks": int(calibration_weeks), "seed": int(seed),
        },
        "football_context_provenance": "retrospective_finalish_game_context",
        "evaluation_streams": sorted(map(str, predictions["evaluation_stream"].dropna().unique())),
        "features": list(WR_TE_FEATURES),
        "rf_params": rf_params,
        "rf_params_hash": rf_hash,
        "input_hashes": dict(sorted(hashes.items())),
        "package_versions": _package_versions(),
        "filter_counts": filter_counts,
        "fold_counts": {"folds": len(folds), "test_groups": len(requested_groups)},
        "odds_provenance": source_provenance,
        "odds_provenance_by_group": odds_provenance_by_group,
        "coverage": [
            {key: value for key, value in record.items() if key in {
                "evaluation_stream", "open_coverage_count", "open_coverage_total", "open_coverage_rate",
            }}
            for record in betting_records if record.get("open_coverage_total") is not None
        ],
        "warnings": sorted(set(warnings)),
        "definitions": metric_definitions,
        "deferred": list(_DEFERRED),
    }

    payloads = {
        "manifest.json": _json_bytes(manifest),
        "folds.csv": _csv_bytes(fold_frame, _FOLD_COLUMNS),
        "predictions.csv": _csv_bytes(predictions, _PREDICTION_COLUMNS),
        "metrics.json": _json_bytes(metrics_payload),
        "summary.csv": _csv_bytes(summary, _SUMMARY_COLUMNS),
    }
    return write_artifacts_atomic(Path(output_dir), payloads)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline WR/TE walk-forward evaluation.")
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--calibration-weeks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        output_dir = resolve_output_dir(
            args.output_dir, args.start_season, args.end_season, args.calibration_weeks, args.seed
        )
        run_evaluation(
            start_season=args.start_season,
            end_season=args.end_season,
            calibration_weeks=args.calibration_weeks,
            seed=args.seed,
            input_path=config.DATA_DIR / "raw_nfl_data.csv",
            rf_params_path=config.MODELS_DIR / "wr_te_rf_best_params.json",
            vegas_dir=config.VEGAS_DIR,
            team_path=config.DATA_DIR / "nfl_teams.csv",
            output_dir=output_dir,
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
