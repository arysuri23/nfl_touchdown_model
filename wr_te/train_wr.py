# NFL Touchdown Scorer Prediction Model - WR/TE RandomForest Only
# Simplified version using only RandomForest (no stacking)

# --- 1. Importing Libraries ---
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


### CONSTANTS ###

# -- Position-Specific Feature Lists --

WR_TE_FEATURES = [
    'avg_offense_snap_share',
    'avg_wopr',
    'avg_target_share',
    'avg_receiving_epa',
    'avg_racr',
    'avg_endzone_targets', 
    'avg_endzone_target_share',  
    #'avg_redzone_target_share',
    'redzone_td_rate',  
    'passing_tds_allowed_to_WR', 
    'passing_tds_allowed_to_TE',
    'implied_total',
    'spread_line',
    'depth_chart_rank',
    #'avg_total_tds',
    'avg_scored_touchdown',
    'avg_receptions',
    'avg_receiving_yards',
    'avg_receiving_air_yards',
    'avg_receiving_yards_allowed',  
    'avg_receiving_epa_allowed',  
    'avg_receiving_air_yards_allowed',  
    'avg_explosive_receiving_plays',
    'avg_explosive_receiving_plays_allowed',
    'avg_rec_touchdown_exp', 
    'avg_rec_touchdown_exp_team'
]


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
    
    player_stats = ['receptions', 'receiving_yards', 'wopr','receiving_epa', 'target_share',
                      'receiving_air_yards', 'racr', 'scored_touchdown', 'redzone_target_share', 'total_tds',
                      'endzone_targets', 'endzone_target_share', 'inside_5_target_share', 'inside_10_targets', 'offense_snap_share',
                      'avg_cushion', 'avg_separation', 'avg_intended_air_yards', 'percent_share_of_intended_air_yards',
                    'catch_percentage', 'avg_expected_yac', 'avg_yac_above_expectation', 'explosive_receiving_plays', 'rec_touchdown_exp', 'rec_touchdown_exp_team']
    
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

    # Team-level stats
    df['redzone_td_rate'] = df.groupby('team')['redzone_td_rate'].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())
    df['redzone_td_rate'] = df.groupby('team')['redzone_td_rate'].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())
    df['pass_rate'] = df.groupby('team')['pass_rate'].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())
    
    # is_home is already a binary/static feature per game, no need to lag/smooth it for the player
    # But we need to ensure it exists. It comes from get_game_data -> merged in data_collection.
    if 'is_home' not in df.columns:
        df['is_home'] = 0 # Fallback
   
    df['pass_matchup_value'] = np.select(
        [df['position'] == 'WR', df['position'] == 'TE'],
        [df['avg_redzone_target_share'] * df['passing_tds_allowed_to_WR'], df['avg_redzone_target_share'] * df['passing_tds_allowed_to_TE']],
        default=0)
    
    df.fillna(0, inplace=True)

    return df


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
    print("="*60)
    print("WR/TE RANDOMFOREST TD SCORER PREDICTION - RETRAINED")
    print("="*60)
    print("🔄 Retraining on 2020-2024 + 2025 weeks 1-9")
    print("📊 Validation on 2023 (avoiding 2025 contamination)")
    print("="*60)
    
    CURRENT_SEASON = 2025
    CURRENT_WEEK = 12  # Updated: we have data through week 9
    # Configuration
    USE_SAVED_PARAMS = False  # Re-tune hyperparameters with new data

    nfl_teams = pd.read_csv('data/nfl_teams.csv')
    team_map = dict(zip(nfl_teams['team_name'], nfl_teams['team_id']))
    
    # Load data
    print("\nLoading and preparing data...")
    df = data.get_all_historic_data([2020, 2021, 2022, 2023, 2024, 2025], team_map)
    df.to_csv('data/raw_nfl_data.csv', index=False)
    df = df[df['week'] <= 18]
    df = df[(df['season'] < CURRENT_SEASON) | ((df['season'] == CURRENT_SEASON) & (df['week'] < CURRENT_WEEK))]

    print(df.tail())

    print(f"✓ Loaded {len(df)} rows of historical data")
    
    # Feature engineering
    print("\nApplying feature engineering...")
    df = feature_engineering(df)
    print("✓ Feature engineering complete")
    
    # Data quality checks
    print("\nRunning data quality checks...")
    assert not df[WR_TE_FEATURES].isnull().any().any(), \
        "❌ NaN values detected in features!"
    assert df['scored_touchdown'].isin([0, 1]).all(), \
        "❌ Target variable contains non-binary values!"
    assert len(df) > 10000, \
        f"❌ Suspiciously small dataset: {len(df)} rows"
    print(f"✓ Data quality verified: {len(df)} rows, no NaNs, binary target")
    
    # Train/validation split (2023 as validation to avoid 2025 contamination)
    train_df = df[df['season'] < 2023].copy()
    val_df = df[df['season'] == 2023].copy()
    
    print(f"\nTrain set: {len(train_df)} rows ({train_df['season'].min()}-{train_df['season'].max()})")
    print(f"Validation set: {len(val_df)} rows (2023)")
    print(f"⚠️  Note: 2024 and 2025 weeks 1-9 will be used in final production model")
    
    # Filter for WR/TE only
    train_df_wr_te = train_df[train_df['position'].isin(['WR','TE'])].copy()
    val_df_wr_te = val_df[val_df['position'].isin(['WR','TE'])].copy()
    
    print(f"\nPosition splits:")
    print(f"  WR/TE: {len(train_df_wr_te)} train, {len(val_df_wr_te)} val")
    
    # Class imbalance reporting
    print(f"\nClass imbalance (TD rate):")
    print(f"  WR/TE: {train_df_wr_te['scored_touchdown'].mean():.1%} train, {val_df_wr_te['scored_touchdown'].mean():.1%} val")
    
    # --- Train WR/TE Model ---
    X_train_wr_te = train_df_wr_te[WR_TE_FEATURES]
    y_train_wr_te = train_df_wr_te['scored_touchdown']
    
    wr_te_model, wr_te_params, wr_te_training_time = train_rf_model(X_train_wr_te, y_train_wr_te, 'wr_te', use_saved_params=USE_SAVED_PARAMS)
    wr_te_importance, wr_te_precision = evaluate_rf_model(wr_te_model, "WR/TE Model (Validated on 2023)", val_df_wr_te, WR_TE_FEATURES)
    
    # --- Retrain on ALL Data for Production ---
    print(f"\n{'='*60}\nRETRAINING FINAL MODEL ON ALL HISTORICAL DATA FOR PRODUCTION\n{'='*60}")
    
    # WR/TE - Train on 2020-2024 + 2025 weeks 1-9
    print("\nRetraining final WR/TE model on all data (2020-2024 + 2025 weeks 1-9)...")
    all_data_df = df[df['position'].isin(['WR','TE'])].copy()
    # Reset index to ensure alignment with OOB predictions later
    all_data_df.reset_index(drop=True, inplace=True)
    
    X_all_wr_te = all_data_df[WR_TE_FEATURES]
    y_all_wr_te = all_data_df['scored_touchdown']
    
    print(f"  Total training rows: {len(X_all_wr_te)}")
    print(f"  2025 weeks 1-9 rows: {len(all_data_df[all_data_df['season'] == 2025])}")
    
    # Enable OOB score to get out-of-sample estimates for calibration
    wr_te_final = RandomForestClassifier(**wr_te_params, oob_score=True, random_state=42, class_weight=None, n_jobs=-1)
    wr_te_final.fit(X_all_wr_te, y_all_wr_te)
    print("✓ WR/TE retraining complete")
    
    # --- Fit Calibrator on 2025 weeks 1-9 ---
    print(f"\n{'='*60}\nFITTING CALIBRATOR ON 2025 WEEKS 1-9 (USING OOB PREDICTIONS)\n{'='*60}")
    
    # Identify indices for 2025 calibration data
    calib_mask = (all_data_df['season'] == 2025) & (all_data_df['week'] < CURRENT_WEEK)
    calibration_rows = all_data_df[calib_mask]
    print(f"\n{'='*60}\nFITTING CALIBRATOR ON 2025 WEEKS 1-9\n{'='*60}")
    
    # Split data for calibration: Train on < 2025, Calibrate on 2025
    calib_train_df = all_data_df[all_data_df['season'] < 2025].copy()
    calib_test_df = all_data_df[(all_data_df['season'] == 2025) & (all_data_df['week'] < CURRENT_WEEK)].copy()
    
    print(f"  Calibration Train (2020-2024): {len(calib_train_df)} rows")
    print(f"  Calibration Test (2025): {len(calib_test_df)} rows")
    
    if len(calib_test_df) > 0:
        print("  Training temporary model on 2020-2024 to generate unbiased 2025 predictions...")
        rf_for_calib = RandomForestClassifier(**wr_te_params, random_state=42, class_weight=None, n_jobs=-1)
        rf_for_calib.fit(calib_train_df[WR_TE_FEATURES], calib_train_df['scored_touchdown'])
        
        # Get unbiased predictions for 2025
        calib_probs_2025 = rf_for_calib.predict_proba(calib_test_df[WR_TE_FEATURES])[:, 1].reshape(-1, 1)
        
        print("  Fitting LogisticRegression calibrator on 2025 predictions vs actuals...")
        wr_te_calibrator = LogisticRegression()
        wr_te_calibrator.fit(calib_probs_2025, calib_test_df['scored_touchdown'])
        
        print("✓ Calibrator trained on 2025 out-of-sample predictions")
        
        # --- Evaluate Calibration ---
        print(f"\n{'='*60}\nCALIBRATION METRICS (2025 OOS)\n{'='*60}")
        
        # Predict calibrated probabilities for 2025
        calib_probs_final = wr_te_calibrator.predict_proba(calib_probs_2025)[:, 1]
        y_true_calib = calib_test_df['scored_touchdown'].values
        
        brier_calib = brier_score_loss(y_true_calib, calib_probs_final)
        logloss_calib = log_loss(y_true_calib, calib_probs_final)
        
        # ECE Calculation
        prob_bins = pd.cut(calib_probs_final, bins=10, labels=False)
        calibration_data = []
        for bin_idx in range(10):
            mask = prob_bins == bin_idx
            if mask.sum() > 0:
                avg_pred = calib_probs_final[mask].mean()
                emp_rate = y_true_calib[mask].mean()
                calibration_data.append({
                    'bin': bin_idx,
                    'count': mask.sum(),
                    'avg_pred': avg_pred,
                    'emp_rate': emp_rate,
                    'abs_gap': abs(avg_pred - emp_rate)
                })
        
        calib_summary_df = pd.DataFrame(calibration_data)
        ece_calib = calib_summary_df['abs_gap'].mean() if len(calib_summary_df) > 0 else 0
        
        print(f"{{'brier': {brier_calib:.4f}, 'log_loss': {logloss_calib:.4f}, 'ece': {ece_calib:.4f}}}")
        print("Calibration table (first 10 bins):")
        print(calib_summary_df.to_string(index=False))
        
    else:
        print("⚠️  No 2025 data found for calibration, using default")
        wr_te_calibrator = LogisticRegression()
        # Use 2024 as fallback (still better to do OOB but for fallback we just fit on what we have)
        fallback_df = all_data_df[all_data_df['season'] == 2024].copy()
        fallback_probs = wr_te_final.predict_proba(fallback_df[WR_TE_FEATURES])[:, 1].reshape(-1, 1)
        wr_te_calibrator.fit(fallback_probs, fallback_df['scored_touchdown'])
    
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
    print(f"Validation Set (2023) Precision@5: {wr_te_precision:.3f}")
    print(f"Training Data: 2020-2024 + 2025 weeks 1-9")
    print(f"Calibration: 2025 weeks 1-9 actual outcomes")
    
    print(f"\n{'='*60}\nTRAINING TIME SUMMARY\n{'='*60}")
    print(f"WR/TE model:  {wr_te_training_time/60:.1f} minutes ({wr_te_training_time:.0f} seconds)")
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS FOR WEEK 10")
    print("="*60)
    print("  1. ✅ Model retrained with 2025 weeks 1-9 data")
    print("  2. ✅ Calibrator fitted on 2025 actual outcomes")
    print("  3. 📊 Run predict_wr.py for week 10 predictions")
    print("  4. 🎲 ONLY BET when model_edge > 0.07 (7%)")
    print("  5. 📈 Track weekly: hit rate, edge, calibration")
    print("\n⚠️  BETTING STRATEGY CHANGE:")
    print("     OLD: Bet top 5 WRs regardless of edge")
    print("     NEW: Only bet picks with >7% model edge")
    print("     Expected: Higher ROI, fewer but better bets")


if __name__ == '__main__':
    main()

