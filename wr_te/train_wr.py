# NFL Touchdown Scorer Prediction Model - WR/TE RandomForest Only
# Simplified version using only RandomForest (no stacking)

# --- 1. Importing Libraries ---
import argparse
import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import data_collection as data
import joblib
import os
import sys
import json
import time
from datetime import datetime

import config
from features import WR_TE_FEATURES, PLAYER_EWM_STATS


### CONSTANTS ###

# -- Hyperparameter Distributions --
RF_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}


# --- 2. Feature Engineering ---
def feature_engineering(df):
    """Engineers features from the raw data to improve model performance."""

    # Ensure strict chronological ordering per player before lag/EWM to avoid leakage
    df.sort_values(by=['player_id', 'season', 'week'], inplace=True, ignore_index=True)

    player_stats = PLAYER_EWM_STATS

    for stat in player_stats:
        df[f'avg_{stat}'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())

    pos_defense_cols = [col for col in df.columns if 'tds_allowed_to' in col] + ['receiving_yards_allowed', 'receiving_epa_allowed', 'receiving_air_yards_allowed', 'explosive_receiving_plays_allowed']

    opponent_stats_df = (
        df[['season', 'week', 'opponent_team'] + pos_defense_cols]
          .groupby(['season', 'week', 'opponent_team'], as_index=False)[pos_defense_cols]
          .mean()
    )
    # Ensure chronological order within each opponent for lag/EWM
    opponent_stats_df.sort_values(by=['opponent_team', 'season', 'week'], inplace=True)

    for col in pos_defense_cols:
         opponent_stats_df[col] = opponent_stats_df.groupby('opponent_team')[col].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())

    # Rename columns to add 'avg_' prefix for defensive stats (excluding tds_allowed_to columns)
    rename_dict = {col: f'avg_{col}' for col in pos_defense_cols if 'tds_allowed_to' not in col}
    opponent_stats_df.rename(columns=rename_dict, inplace=True)

    df.drop(columns=pos_defense_cols, inplace=True)
    df = pd.merge(df, opponent_stats_df, on=['season', 'week', 'opponent_team'], how='left')

    # is_home is already a binary/static feature per game, no need to lag/smooth it for the player
    # But we need to ensure it exists. It comes from get_game_data -> merged in data_collection.
    if 'is_home' not in df.columns:
        df['is_home'] = 0 # Fallback

    df.fillna(0, inplace=True)

    return df


# --- 2.5 Training Protocol Utilities ---
def chronological(df):
    """Return `df` sorted by (season, week, player_id) with the index reset.

    Ensures TimeSeriesSplit folds computed downstream (e.g. inside
    train_rf_model) respect temporal order.
    """
    return df.sort_values(by=['season', 'week', 'player_id'], ignore_index=True)


def _oob_calibration_rows(model, mask=None, min_rows=500):
    """Return (oob_proba, candidate_mask) for calibrating on OOB predictions.

    `candidate_mask` selects rows with a non-NaN OOB prediction, further
    restricted to `mask` when given. If fewer than `min_rows` rows survive
    that restriction, falls back to all non-NaN OOB rows.
    """
    oob_proba = model.oob_decision_function_[:, 1]
    valid = ~np.isnan(oob_proba)
    if mask is not None:
        candidate = valid & np.asarray(mask)
    else:
        candidate = valid
    if candidate.sum() < min_rows:
        candidate = valid
    return oob_proba, candidate


def fit_calibrator_oob(model, y, mask=None, min_rows=500):
    """Fit a Platt-scaling LogisticRegression on a model's OOB predictions.

    `model` must be a fitted RandomForestClassifier with oob_score=True.
    Restricts to `mask` rows (if given) with valid (non-NaN) OOB predictions;
    if fewer than `min_rows` remain, falls back to using all non-NaN rows.
    """
    oob_proba, candidate = _oob_calibration_rows(model, mask, min_rows)
    X = oob_proba[candidate].reshape(-1, 1)
    y_arr = np.asarray(y)[candidate]
    calibrator = LogisticRegression()
    calibrator.fit(X, y_arr)
    return calibrator


def calibration_report(y_true, p):
    """Return {'brier', 'log_loss', 'ece'} for predictions `p` against `y_true`.

    ECE uses 10 equal-width probability bins: mean of |avg_pred - emp_rate|
    over populated bins.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)

    brier = brier_score_loss(y_true, p)
    logloss = log_loss(y_true, p)

    bins = pd.cut(p, bins=10, labels=False)
    gaps = []
    for bin_idx in range(10):
        bin_mask = bins == bin_idx
        if bin_mask.sum() > 0:
            gaps.append(abs(p[bin_mask].mean() - y_true[bin_mask].mean()))
    ece = float(np.mean(gaps)) if gaps else 0.0

    return {'brier': brier, 'log_loss': logloss, 'ece': ece}


# --- 3. RandomForest Training Function ---
def train_rf_model(X_train, y_train, position_name, use_saved_params=False):
    """Trains a single RandomForest model with optional hyperparameter tuning."""

    print(f"\n{'='*60}\nTRAINING RANDOMFOREST MODEL: {position_name}\n{'='*60}")

    start_time = time.time()
    param_file = f'models/{position_name}_rf_best_params.json'

    if use_saved_params and os.path.exists(param_file):
        # Load saved params
        with open(param_file, 'r') as f:
            best_params = json.load(f)
        print(f"✓ Using saved hyperparameters from {param_file}")
        print(f"  Parameters: {best_params}")
    else:
        # Perform hyperparameter tuning with TimeSeriesSplit
        print("Performing hyperparameter tuning with TimeSeriesSplit CV...")
        print("  (Ensures no temporal leakage during cross-validation)")

        rf = RandomForestClassifier(random_state=42, class_weight=None, n_jobs=-1)

        # Use TimeSeriesSplit instead of standard K-Fold to respect temporal order
        tscv = TimeSeriesSplit(n_splits=5)

        random_search = RandomizedSearchCV(
            rf, RF_PARAM_DIST, n_iter=20, cv=tscv, scoring='average_precision',
            random_state=42, n_jobs=-1, verbose=1
        )
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_

        print(f"\n✓ Best hyperparameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"  Best CV score: {random_search.best_score_:.4f}")

        # Save params
        os.makedirs('models', exist_ok=True)
        with open(param_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"✓ Saved parameters to {param_file}")

    # Train final model with best params on all training data
    print("\nTraining final RandomForest model on all training data...")
    final_rf = RandomForestClassifier(**best_params, random_state=42, class_weight=None, n_jobs=-1)
    final_rf.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"✓ Training complete in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")

    return final_rf, best_params, training_time


# --- 4. Evaluation Function ---
def evaluate_rf_model(model, model_name, validation_df, features, k_values=[3, 5, 10, 15, 20]):
    """Evaluates a single RandomForest model."""
    print(f"\n{'='*60}\nEVALUATION FOR: {model_name}\n{'='*60}")

    if validation_df.empty:
        print("Validation data is empty. Skipping evaluation.")
        return None

    # Prepare X_val from the validation dataframe
    X_val = validation_df[features].copy()

    # Get predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    results_df = validation_df[['player_display_name', 'week', 'scored_touchdown']].copy()
    results_df['predicted_prob'] = y_pred_proba

    # Evaluate at multiple K values
    print(f"\n--- Performance at Multiple K Values ---")
    k_summary = []

    for k in k_values:
        weekly_metrics = []
        for week_num in sorted(results_df['week'].unique()):
            week_df = results_df[results_df['week'] == week_num].copy()
            week_df.sort_values('predicted_prob', ascending=False, inplace=True)

            top_k = week_df.head(k)
            true_positives = top_k['scored_touchdown'].sum()
            precision = true_positives / k if k > 0 else 0
            total_scorers = week_df['scored_touchdown'].sum()
            recall = true_positives / total_scorers if total_scorers > 0 else 0

            weekly_metrics.append({
                'week': week_num,
                'precision_at_k': precision,
                'recall_at_k': recall,
                'successful_picks': int(true_positives)
            })

        weekly_df = pd.DataFrame(weekly_metrics)
        avg_precision = weekly_df['precision_at_k'].mean()
        avg_recall = weekly_df['recall_at_k'].mean()
        avg_picks = weekly_df['successful_picks'].mean()

        k_summary.append({
            'K': k,
            'Avg_Precision': avg_precision,
            'Avg_Recall': avg_recall,
            'Avg_Hits_Per_Week': avg_picks
        })

    k_summary_df = pd.DataFrame(k_summary)
    print(k_summary_df.to_string(index=False))

    # Detailed weekly breakdown for K=5 (default)
    print(f"\n--- Weekly Performance Breakdown @ K=5 ---")
    weekly_metrics_k5 = []
    for week_num in sorted(results_df['week'].unique()):
        week_df = results_df[results_df['week'] == week_num].copy()
        week_df.sort_values('predicted_prob', ascending=False, inplace=True)

        top_k = week_df.head(5)
        true_positives = top_k['scored_touchdown'].sum()
        precision = true_positives / 5
        total_scorers = week_df['scored_touchdown'].sum()
        recall = true_positives / total_scorers if total_scorers > 0 else 0

        weekly_metrics_k5.append({
            'week': week_num,
            'precision_at_k': precision,
            'recall_at_k': recall,
            'successful_picks': int(true_positives)
        })

    weekly_df_k5 = pd.DataFrame(weekly_metrics_k5)
    print(weekly_df_k5.to_string(index=True))

    # Store precision at k=10 for final summary
    avg_precision_k5 = k_summary_df[k_summary_df['K'] == 5]['Avg_Precision'].values[0]

    # Threshold-based evaluation
    print(f"\n--- Threshold-Based Performance (Flexible Picks) ---")
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    threshold_results = []

    total_scorers = validation_df['scored_touchdown'].sum()

    for thresh in thresholds:
        picks = results_df[results_df['predicted_prob'] > thresh]
        if len(picks) > 0:
            thresh_precision = picks['scored_touchdown'].mean()
            thresh_recall = picks['scored_touchdown'].sum() / total_scorers if total_scorers > 0 else 0
            thresh_picks = len(picks)
            thresh_hits = picks['scored_touchdown'].sum()
            threshold_results.append({
                'threshold': f'>{thresh:.0%}',
                'precision': thresh_precision,
                'recall': thresh_recall,
                'picks': thresh_picks,
                'hits': int(thresh_hits)
            })

    if threshold_results:
        thresh_df = pd.DataFrame(threshold_results)
        print(thresh_df.to_string(index=False))

        # Find optimal threshold (highest precision with reasonable picks)
        optimal = max(threshold_results, key=lambda x: x['precision'])
        print(f"\n  Best threshold: {optimal['threshold']} → {optimal['precision']:.3f} precision ({optimal['hits']}/{optimal['picks']} picks)")

    # Probability quality metrics
    y_true = validation_df['scored_touchdown'].values
    brier = brier_score_loss(y_true, y_pred_proba)
    logloss = log_loss(y_true, y_pred_proba)

    prob_bins = pd.cut(y_pred_proba, bins=10, labels=False)
    calibration_data = []
    for bin_idx in range(10):
        mask = prob_bins == bin_idx
        if mask.sum() > 0:
            avg_pred = y_pred_proba[mask].mean()
            emp_rate = y_true[mask].mean()
            calibration_data.append({
                'bin': bin_idx,
                'count': mask.sum(),
                'avg_pred': avg_pred,
                'emp_rate': emp_rate,
                'abs_gap': abs(avg_pred - emp_rate)
            })

    calib_df = pd.DataFrame(calibration_data)
    ece = calib_df['abs_gap'].mean() if len(calib_df) > 0 else 0

    print(f"\n--- Probability Quality ({model_name}) ---")
    print(f"{{'brier': {brier:.4f}, 'log_loss': {logloss:.4f}, 'ece': {ece:.4f}}}")
    print("Calibration table (first 10 bins):")
    print(calib_df.to_string(index=False))

    # Feature importance analysis
    print(f"\n{'='*60}\nFEATURE IMPORTANCE ANALYSIS: {model_name}\n{'='*60}")

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Normalize
    feature_importance['importance_pct'] = (feature_importance['importance'] / feature_importance['importance'].sum()) * 100

    print("\n--- Top 20 Most Important Features ---")
    print(feature_importance.head(20).to_string(index=False))

    print("\n--- Bottom 10 Least Important Features ---")
    print(feature_importance.tail(10).to_string(index=False))

    top_5_pct = feature_importance.head(5)['importance_pct'].sum()
    top_10_pct = feature_importance.head(10)['importance_pct'].sum()
    bottom_10_pct = feature_importance.tail(10)['importance_pct'].sum()

    print(f"\n--- Feature Importance Summary ---")
    print(f"Total features: {len(features)}")
    print(f"Top 5 features account for: {top_5_pct:.1f}% of importance")
    print(f"Top 10 features account for: {top_10_pct:.1f}% of importance")
    print(f"Bottom 10 features account for: {bottom_10_pct:.1f}% of importance")

    # Flag low-importance features
    low_importance = feature_importance[feature_importance['importance_pct'] < 0.5]
    if len(low_importance) > 0:
        print(f"\n⚠️  Features with <0.5% importance (consider removing): {len(low_importance)}")
        print(low_importance[['feature', 'importance_pct']].to_string(index=False))

    return feature_importance, avg_precision_k5


# --- 5. Main Training Function ---
def main():
    parser = argparse.ArgumentParser(description="Train the WR/TE touchdown RandomForest model.")
    parser.add_argument(
        '--tune', action='store_true',
        help="Re-run RandomizedSearchCV hyperparameter tuning and overwrite the saved params "
             "file (default: reuse models/wr_te_rf_best_params.json)."
    )
    args = parser.parse_args()
    use_saved_params = not args.tune

    train_years = sorted(s for s in config.TRAIN_SEASONS if s < config.SEASON)

    print("="*60)
    print("WR/TE RANDOMFOREST TD SCORER PREDICTION - RETRAINED")
    print("="*60)
    print(f"🔄 Training on {train_years[0]}-{train_years[-1]} (config.TRAIN_SEASONS < config.SEASON={config.SEASON})")
    print(f"📊 Informational validation on {config.VALIDATION_SEASON}")
    print("="*60)

    nfl_teams = pd.read_csv('data/nfl_teams.csv')
    team_map = dict(zip(nfl_teams['team_name'], nfl_teams['team_id']))

    # Load data
    print("\nLoading and preparing data...")
    df = data.get_all_historic_data(config.TRAIN_SEASONS, team_map)
    df.to_csv('data/raw_nfl_data.csv', index=False)
    df = df[df['week'] <= 18]
    df = df[df['season'].isin(config.TRAIN_SEASONS) & (df['season'] < config.SEASON)]

    print(df.tail())

    print(f"✓ Loaded {len(df)} rows of historical data")

    # Feature engineering
    print("\nApplying feature engineering...")
    df = feature_engineering(df)
    print("✓ Feature engineering complete")

    # Chronological ordering before any split, so TimeSeriesSplit folds inside
    # train_rf_model respect temporal order.
    df = chronological(df)

    # Data quality checks
    print("\nRunning data quality checks...")
    assert not df[WR_TE_FEATURES].isnull().any().any(), \
        "❌ NaN values detected in features!"
    assert df['scored_touchdown'].isin([0, 1]).all(), \
        "❌ Target variable contains non-binary values!"
    assert len(df) > 10000, \
        f"❌ Suspiciously small dataset: {len(df)} rows"
    print(f"✓ Data quality verified: {len(df)} rows, no NaNs, binary target")

    # Informational train/validation split (2020-2022 -> 2023). This is NOT
    # the deployed model — the deployed model is retrained on all rows below
    # and calibrated on its own OOB predictions.
    train_df = df[df['season'] < config.VALIDATION_SEASON].copy()
    val_df = df[df['season'] == config.VALIDATION_SEASON].copy()

    print(f"\n[informational] Train set: {len(train_df)} rows ({train_df['season'].min()}-{train_df['season'].max()})")
    print(f"[informational] Validation set: {len(val_df)} rows ({config.VALIDATION_SEASON})")

    # Filter for WR/TE only
    train_df_wr_te = train_df[train_df['position'].isin(['WR','TE'])].copy()
    val_df_wr_te = val_df[val_df['position'].isin(['WR','TE'])].copy()

    print(f"\nPosition splits:")
    print(f"  WR/TE: {len(train_df_wr_te)} train, {len(val_df_wr_te)} val")

    # Class imbalance reporting
    print(f"\nClass imbalance (TD rate):")
    print(f"  WR/TE: {train_df_wr_te['scored_touchdown'].mean():.1%} train, {val_df_wr_te['scored_touchdown'].mean():.1%} val")

    # --- Train WR/TE Model (informational only) ---
    X_train_wr_te = train_df_wr_te[WR_TE_FEATURES]
    y_train_wr_te = train_df_wr_te['scored_touchdown']

    wr_te_model, wr_te_params, wr_te_training_time = train_rf_model(X_train_wr_te, y_train_wr_te, 'wr_te', use_saved_params=use_saved_params)
    wr_te_importance, wr_te_precision = evaluate_rf_model(
        wr_te_model,
        f"informational: not the deployed model (WR/TE, validated on {config.VALIDATION_SEASON})",
        val_df_wr_te,
        WR_TE_FEATURES,
    )

    # --- Retrain on ALL Data for Production ---
    print(f"\n{'='*60}\nRETRAINING FINAL MODEL ON ALL HISTORICAL DATA FOR PRODUCTION\n{'='*60}")

    print(f"\nRetraining final WR/TE model on all data ({train_years[0]}-{train_years[-1]})...")
    all_data_df = df[df['position'].isin(['WR','TE'])].copy()
    # Reset index to ensure alignment with OOB predictions later
    all_data_df.reset_index(drop=True, inplace=True)

    X_all_wr_te = all_data_df[WR_TE_FEATURES]
    y_all_wr_te = all_data_df['scored_touchdown']

    latest_season = all_data_df['season'].max()
    print(f"  Total training rows: {len(X_all_wr_te)}")
    print(f"  {latest_season} rows: {len(all_data_df[all_data_df['season'] == latest_season])}")

    # Enable OOB score to get out-of-sample estimates for calibration
    wr_te_final = RandomForestClassifier(**wr_te_params, oob_score=True, random_state=42, class_weight=None, n_jobs=-1)
    wr_te_final.fit(X_all_wr_te, y_all_wr_te)
    print("✓ WR/TE retraining complete")

    # --- Calibrate the deployed model on its own OOB predictions ---
    print(f"\n{'='*60}\nCALIBRATING DEPLOYED MODEL (PLATT ON OOB PREDICTIONS)\n{'='*60}")

    mask = all_data_df['season'] == all_data_df['season'].max()
    wr_te_calibrator = fit_calibrator_oob(wr_te_final, y_all_wr_te, mask)

    oob_proba, candidate = _oob_calibration_rows(wr_te_final, mask)
    report = calibration_report(np.asarray(y_all_wr_te)[candidate], oob_proba[candidate])
    print("Platt on OOB predictions of the deployed forest (metrics in-sample for the 2-parameter calibrator)")
    print(report)

    # --- Save Models ---
    print(f"\n{'='*60}\nSAVING MODEL ARTIFACTS LOCALLY\n{'='*60}")

    os.makedirs('models', exist_ok=True)

    def save_model(model, filename):
        print(f"Saving model artifact to '{filename}'...")
        joblib.dump(model, filename)
        print("Save successful.")

    save_model(wr_te_final, 'models/wr_te_rf_final.pkl')
    save_model(wr_te_calibrator, 'models/wr_te_rf_calibrator.pkl')

    # Save feature importance
    wr_te_importance.to_csv('models/wr_te_rf_feature_importance.csv', index=False)
    print("Feature importance saved to models/ directory")

    print(f"\n{'='*60}\nWR/TE TRAINING COMPLETE - NEW MODEL READY\n{'='*60}")
    print("\n✅ Models saved:")
    print("  - models/wr_te_rf_final.pkl")
    print("  - models/wr_te_rf_calibrator.pkl")
    print("  - models/wr_te_rf_best_params.json")
    print("  - models/wr_te_rf_feature_importance.csv")

    print(f"\n{'='*60}\nPERFORMANCE SUMMARY\n{'='*60}")
    print(f"Position: WR/TE")
    print(f"[informational] Validation Set ({config.VALIDATION_SEASON}) Precision@5: {wr_te_precision:.3f}")
    print(f"Training Data: {train_years[0]}-{train_years[-1]}")
    print(f"Calibration: Platt scaling on OOB predictions ({latest_season} rows)")

    print(f"\n{'='*60}\nTRAINING TIME SUMMARY\n{'='*60}")
    print(f"WR/TE model:  {wr_te_training_time/60:.1f} minutes ({wr_te_training_time:.0f} seconds)")

    print("\n" + "="*60)
    print(f"🎯 NEXT STEPS FOR {config.SEASON} WEEK {config.WEEK}")
    print("="*60)
    print(f"  1. ✅ Model retrained on {train_years[0]}-{train_years[-1]}")
    print("  2. ✅ Calibrator fitted on OOB predictions of the deployed forest")
    print("  3. 📊 Run predict_wr.py for weekly predictions")
    print("  4. 🎲 ONLY BET when model_edge > 0.07 (7%)")
    print("  5. 📈 Track weekly: hit rate, edge, calibration")
    print("\n⚠️  BETTING STRATEGY CHANGE:")
    print("     OLD: Bet top 5 WRs regardless of edge")
    print("     NEW: Only bet picks with >7% model edge")
    print("     Expected: Higher ROI, fewer but better bets")


if __name__ == '__main__':
    main()
