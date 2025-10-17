# NFL Touchdown Scorer Prediction Model - LightGBM Only
# Simplified version using only LightGBM (no stacking)

# --- 1. Importing Libraries ---
import nflreadpy as nfl
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from lightgbm import LGBMClassifier
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

RB_FEATURES = [
    'avg_offense_snap_share',
    'avg_wopr',
    'avg_rushing_epa',
    'avg_receiving_epa',
    'avg_racr',
    'avg_redzone_carry_share',
    'avg_redzone_target_share',
    'avg_inside_5_carry_share',
    'rush_matchup_value',
    'redzone_td_rate',
    'rushing_tds_allowed_to_RB',
    'passing_tds_allowed_to_RB',
    'implied_total',
    'spread_line',
    'depth_chart_rank',
    'avg_scored_touchdown',
    'avg_rushing_yards_allowed',
    'avg_receiving_yards_allowed',
    'avg_rushing_epa_allowed',
    'avg_receiving_epa_allowed',
    'avg_receiving_air_yards_allowed',
    'avg_explosive_rushing_plays',
    'avg_explosive_receiving_plays',
    'avg_explosive_rushing_plays_allowed',
    'avg_explosive_receiving_plays_allowed',
]

WR_TE_FEATURES = [
    'avg_offense_snap_share',
    'avg_wopr',
    'avg_target_share',
    'avg_receiving_epa',
    'avg_racr',
    'avg_endzone_targets',
    'avg_endzone_target_share',
    'redzone_td_rate',
    'passing_tds_allowed_to_WR',
    'passing_tds_allowed_to_TE',
    'implied_total',
    'spread_line',
    'depth_chart_rank',
    'avg_scored_touchdown',
    'avg_receptions',
    'avg_receiving_yards',
    'avg_receiving_air_yards',
    'avg_receiving_yards_allowed',
    'avg_receiving_epa_allowed',
    'avg_receiving_air_yards_allowed',
    'avg_explosive_receiving_plays',
    'avg_explosive_receiving_plays_allowed',
]

QB_FEATURES = [
    'avg_offense_snap_share',
    'avg_carries', 'avg_rushing_yards', 'avg_rushing_epa',
    'avg_scored_touchdown',
    'avg_redzone_carry_share', 'avg_inside_5_carry_share',
    'rush_matchup_value',
    'rushing_tds_allowed_to_QB', 'implied_total', 'spread_line', 'depth_chart_rank',
    'avg_explosive_rushing_plays',  
]


# -- Hyperparameter Distributions --
LGBM_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, -1],
    'num_leaves': [15, 31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0]
}


# --- 2. Feature Engineering ---
def feature_engineering(df):
    """Engineers features from the raw data to improve model performance."""
    
    # Ensure strict chronological ordering per player before lag/EWM to avoid leakage
    df.sort_values(by=['player_id', 'season', 'week'], inplace=True, ignore_index=True)
    
    player_stats = ['carries', 'rushing_yards', 'receptions', 'receiving_yards', 'wopr', 'rushing_epa', 'receiving_epa', 'target_share',
                      'receiving_air_yards', 'racr', 'scored_touchdown', 'redzone_carry_share', 'redzone_target_share',
                      'endzone_targets', 'endzone_target_share', 'inside_5_carry_share', 'inside_5_target_share', 'offense_snap_share',
                      'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 'avg_time_to_los', 'percent_attempts_gte_eight_defenders',
                      'avg_cushion', 'avg_separation', 'avg_intended_air_yards', 'percent_share_of_intended_air_yards',
                    'catch_percentage', 'avg_expected_yac', 'avg_yac_above_expectation', 'explosive_rushing_plays', 'explosive_receiving_plays']
    
    for stat in player_stats:
        df[f'avg_{stat}'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())
   
    pos_defense_cols = [col for col in df.columns if 'tds_allowed_to' in col] + ['rushing_yards_allowed', 'receiving_yards_allowed', 'rushing_epa_allowed', 'receiving_epa_allowed', 'receiving_air_yards_allowed', 'explosive_rushing_plays_allowed', 'explosive_receiving_plays_allowed']

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
    df['pass_rate'] = df.groupby('team')['pass_rate'].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())
    df['rush_rate'] = df.groupby('team')['rush_rate'].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())
    
    df['rush_matchup_value'] = np.select(
        [df['position'] == 'RB', df['position'] == 'QB'],
        [df['avg_redzone_carry_share'] * df['rushing_tds_allowed_to_RB'], df['avg_redzone_carry_share'] * df['rushing_tds_allowed_to_QB']],
        default=0)
    df['pass_matchup_value'] = np.select(
        [df['position'] == 'RB', df['position'] == 'WR', df['position'] == 'TE'],
        [df['avg_redzone_target_share'] * df['passing_tds_allowed_to_RB'], df['avg_redzone_target_share'] * df['passing_tds_allowed_to_WR'], df['avg_redzone_target_share'] * df['passing_tds_allowed_to_TE']],
        default=0)
    
    df.fillna(0, inplace=True)

    return df


# --- 3. LightGBM Training Function ---
def train_lgbm_model(X_train, y_train, position_name, use_saved_params=False):
    """Trains a single LightGBM model with optional hyperparameter tuning."""
    
    print(f"\n{'='*60}\nTRAINING LIGHTGBM MODEL: {position_name}\n{'='*60}")
    
    start_time = time.time()
    param_file = f'models/{position_name}_lgbm_best_params.json'
    
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
        
        lgbm = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
        
        # Use TimeSeriesSplit instead of standard K-Fold to respect temporal order
        tscv = TimeSeriesSplit(n_splits=5)
        
        random_search = RandomizedSearchCV(
            lgbm, LGBM_PARAM_DIST, n_iter=20, cv=tscv, scoring='average_precision',
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
    print("\nTraining final LightGBM model on all training data...")
    final_lgbm = LGBMClassifier(**best_params, random_state=42, verbose=-1, n_jobs=-1)
    final_lgbm.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"✓ Training complete in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    return final_lgbm, best_params, training_time


# --- 4. Evaluation Function ---
def evaluate_lgbm_model(model, model_name, validation_df, features, k=25, baseline_precision=None):
    """Evaluates a single LightGBM model."""
    print(f"\n{'='*60}\nEVALUATION FOR: {model_name}\n{'='*60}")
    
    if validation_df.empty:
        print("Validation data is empty. Skipping evaluation.")
        return None
    
    # Prepare X_val from the validation dataframe
    X_val = validation_df[features].copy()
    
    # Get predictions
    print(f"\n--- Weekly Performance @ K={k} ---")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    results_df = validation_df[['player_display_name', 'week', 'scored_touchdown']].copy()
    results_df['predicted_prob'] = y_pred_proba
    
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
    print(weekly_df.to_string(index=True))
    
    avg_precision = weekly_df['precision_at_k'].mean()
    avg_recall = weekly_df['recall_at_k'].mean()
    avg_picks = weekly_df['successful_picks'].mean()
    
    print(f"\n--- Average Season Performance ---")
    print(f"Average Precision@{k}: {avg_precision:.3f}")
    print(f"Average Recall@{k}:    {avg_recall:.3f}")
    print(f"Average Successful Picks Per Week: {avg_picks:.1f}")
    
    if baseline_precision is not None:
        diff = avg_precision - baseline_precision
        status = "✅ IMPROVEMENT" if diff > 0 else "⚠️ DECLINE"
        print(f"\n--- Comparison to RF Baseline ---")
        print(f"Baseline (RF): {baseline_precision:.3f}")
        print(f"LightGBM:      {avg_precision:.3f}")
        print(f"Difference:    {diff:+.3f} ({status})")
    
    # Threshold-based evaluation
    print(f"\n--- Threshold-Based Performance (Flexible Picks) ---")
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
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
    
    return feature_importance, avg_precision


# --- 5. Main Training Function ---
def main():
    print("="*60)
    print("LIGHTGBM-ONLY TD SCORER PREDICTION")
    print("="*60)
    CURRENT_SEASON = 2025
    CURRENT_WEEK = 7
    # Configuration
    USE_SAVED_PARAMS = True  # Set to False to tune hyperparameters

    nfl_teams = pd.read_csv('nfl_teams.csv')
    team_map = dict(zip(nfl_teams['team_name'], nfl_teams['team_id']))
    
    # Load data
    print("\nLoading and preparing data...")
    #df = data.get_all_historic_data([2020, 2021, 2022, 2023, 2024], team_map)
    df = pd.read_csv('raw_nfl_data.csv')
    df = df[df['week'] <= 18]
    df = df[(df['season'] < CURRENT_SEASON) | ((df['season'] == CURRENT_SEASON) & (df['week'] < CURRENT_WEEK))]


    print(f"✓ Loaded {len(df)} rows of historical data")
    
    # Feature engineering
    print("\nApplying feature engineering...")
    df = feature_engineering(df)
    print("✓ Feature engineering complete")
    
    # Data quality checks
    print("\nRunning data quality checks...")
    all_features = list(set(RB_FEATURES + WR_TE_FEATURES + QB_FEATURES))
    assert not df[all_features].isnull().any().any(), \
        "❌ NaN values detected in features!"
    assert df['scored_touchdown'].isin([0, 1]).all(), \
        "❌ Target variable contains non-binary values!"
    assert len(df) > 20000, \
        f"❌ Suspiciously small dataset: {len(df)} rows"
    print(f"✓ Data quality verified: {len(df)} rows, no NaNs, binary target")
    
    # Train/validation split (2024 as validation)
    train_df = df[df['season'] < 2024].copy()
    val_df = df[df['season'] == 2024].copy()
    
    print(f"\nTrain set: {len(train_df)} rows ({train_df['season'].min()}-{train_df['season'].max()})")
    print(f"Validation set: {len(val_df)} rows (2024)")
    
    # Split by position
    train_df_rb = train_df[train_df['position'] == 'RB'].copy()
    train_df_wr_te = train_df[train_df['position'].isin(['WR', 'TE'])].copy()
    train_df_qb = train_df[train_df['position'] == 'QB'].copy()
    
    val_df_rb = val_df[val_df['position'] == 'RB'].copy()
    val_df_wr_te = val_df[val_df['position'].isin(['WR', 'TE'])].copy()
    val_df_qb = val_df[val_df['position'] == 'QB'].copy()
    
    print(f"\nPosition splits:")
    print(f"  RB: {len(train_df_rb)} train, {len(val_df_rb)} val")
    print(f"  WR/TE: {len(train_df_wr_te)} train, {len(val_df_wr_te)} val")
    print(f"  QB: {len(train_df_qb)} train, {len(val_df_qb)} val")
    
    # Class imbalance reporting
    print(f"\nClass imbalance (TD rate):")
    print(f"  RB:    {train_df_rb['scored_touchdown'].mean():.1%} train, {val_df_rb['scored_touchdown'].mean():.1%} val")
    print(f"  WR/TE: {train_df_wr_te['scored_touchdown'].mean():.1%} train, {val_df_wr_te['scored_touchdown'].mean():.1%} val")
    print(f"  QB:    {train_df_qb['scored_touchdown'].mean():.1%} train, {val_df_qb['scored_touchdown'].mean():.1%} val")
    
    # --- Train RB Model ---
    X_train_rb = train_df_rb[RB_FEATURES]
    y_train_rb = train_df_rb['scored_touchdown']
    
    rb_model, rb_params, rb_training_time = train_lgbm_model(X_train_rb, y_train_rb, 'rb', use_saved_params=USE_SAVED_PARAMS)
    rb_importance, rb_precision = evaluate_lgbm_model(rb_model, "RB Model", val_df_rb, RB_FEATURES, k=10, baseline_precision=0.539)
    
    # --- Train WR/TE Model ---
    X_train_wr_te = train_df_wr_te[WR_TE_FEATURES]
    y_train_wr_te = train_df_wr_te['scored_touchdown']
    
    wr_te_model, wr_te_params, wr_te_training_time = train_lgbm_model(X_train_wr_te, y_train_wr_te, 'wr_te', use_saved_params=USE_SAVED_PARAMS)
    wr_te_importance, wr_te_precision = evaluate_lgbm_model(wr_te_model, "WR/TE Model", val_df_wr_te, WR_TE_FEATURES, k=10, baseline_precision=0.411)
    
    # --- Train QB Model ---
    X_train_qb = train_df_qb[QB_FEATURES]
    y_train_qb = train_df_qb['scored_touchdown']
    
    qb_model, qb_params, qb_training_time = train_lgbm_model(X_train_qb, y_train_qb, 'qb', use_saved_params=USE_SAVED_PARAMS)
    qb_importance, qb_precision = evaluate_lgbm_model(qb_model, "QB Model", val_df_qb, QB_FEATURES, k=5, baseline_precision=0.356)
    
    # --- Unified Evaluation ---
    print(f"\n{'='*60}\nUNIFIED EVALUATION ON 2024 SEASON\n{'='*60}")
    
    val_df['predicted_prob'] = 0.0
    
    val_df.loc[val_df['position'] == 'RB', 'predicted_prob'] = rb_model.predict_proba(val_df[val_df['position'] == 'RB'][RB_FEATURES])[:, 1]
    val_df.loc[val_df['position'].isin(['WR', 'TE']), 'predicted_prob'] = wr_te_model.predict_proba(val_df[val_df['position'].isin(['WR', 'TE'])][WR_TE_FEATURES])[:, 1]
    val_df.loc[val_df['position'] == 'QB', 'predicted_prob'] = qb_model.predict_proba(val_df[val_df['position'] == 'QB'][QB_FEATURES])[:, 1]
    
    # Weekly unified metrics
    k_unified = 16
    unified_metrics = []
    for week_num in sorted(val_df['week'].unique()):
        week_df = val_df[val_df['week'] == week_num].copy()
        week_df.sort_values('predicted_prob', ascending=False, inplace=True)
        
        top_k = week_df.head(k_unified)
        true_positives = top_k['scored_touchdown'].sum()
        precision = true_positives / k_unified
        total_scorers = week_df['scored_touchdown'].sum()
        recall = true_positives / total_scorers if total_scorers > 0 else 0
        
        unified_metrics.append({
            'week': week_num,
            'precision_at_k': precision,
            'recall_at_k': recall,
            'successful_picks': int(true_positives)
        })
    
    unified_df = pd.DataFrame(unified_metrics)
    print(f"\n--- Weekly Performance @ K={k_unified} (All Positions) ---")
    print(unified_df.to_string(index=True))
    
    avg_unified_precision = unified_df['precision_at_k'].mean()
    avg_unified_recall = unified_df['recall_at_k'].mean()
    avg_unified_picks = unified_df['successful_picks'].mean()
    
    print(f"\n--- Average Season Performance (All Positions) ---")
    print(f"Average Precision@{k_unified}: {avg_unified_precision:.3f}")
    print(f"Average Recall@{k_unified}:    {avg_unified_recall:.3f}")
    print(f"Average Successful Picks Per Week: {avg_unified_picks:.1f}")
    
    print(f"\n--- Comparison to RF Baseline ---")
    print(f"Baseline (RF): 0.542")
    print(f"LightGBM:      {avg_unified_precision:.3f}")
    diff = avg_unified_precision - 0.542
    status = "✅ IMPROVEMENT" if diff > 0 else "⚠️ DECLINE"
    print(f"Difference:    {diff:+.3f} ({status})")
    
    # Threshold-based unified evaluation
    print(f"\n--- Threshold-Based Performance (All Positions, Flexible) ---")
    thresholds_unified = [0.35, 0.40, 0.45, 0.50, 0.55]
    threshold_unified_results = []
    
    total_scorers_all = val_df['scored_touchdown'].sum()
    
    for thresh in thresholds_unified:
        picks = val_df[val_df['predicted_prob'] > thresh]
        if len(picks) > 0:
            thresh_precision = picks['scored_touchdown'].mean()
            thresh_recall = picks['scored_touchdown'].sum() / total_scorers_all if total_scorers_all > 0 else 0
            thresh_picks = len(picks)
            thresh_hits = picks['scored_touchdown'].sum()
            
            # Average picks per week
            weeks = picks['week'].nunique()
            avg_picks_per_week = thresh_picks / weeks if weeks > 0 else 0
            
            threshold_unified_results.append({
                'threshold': f'>{thresh:.0%}',
                'precision': thresh_precision,
                'recall': thresh_recall,
                'total_picks': thresh_picks,
                'avg_picks_per_week': avg_picks_per_week,
                'total_hits': int(thresh_hits)
            })
    
    if threshold_unified_results:
        thresh_unified_df = pd.DataFrame(threshold_unified_results)
        print(thresh_unified_df.to_string(index=False))
        
        # Find optimal threshold (balance precision and reasonable pick count)
        # Look for peak precision with at least 10 avg picks per week
        reasonable_picks = [r for r in threshold_unified_results if r['avg_picks_per_week'] >= 10]
        if reasonable_picks:
            optimal_unified = max(reasonable_picks, key=lambda x: x['precision'])
            print(f"\n  Optimal threshold (≥10 picks/week): {optimal_unified['threshold']} → {optimal_unified['precision']:.3f} precision")
            print(f"    ({optimal_unified['avg_picks_per_week']:.1f} picks/week, {optimal_unified['total_hits']} total hits)")
        
        # Also show absolute best precision regardless of volume
        best_precision = max(threshold_unified_results, key=lambda x: x['precision'])
        print(f"  Highest precision overall: {best_precision['threshold']} → {best_precision['precision']:.3f} precision")
        print(f"    ({best_precision['avg_picks_per_week']:.1f} picks/week, {best_precision['total_hits']} total hits)")
    
    # --- Fit Calibrators on 2024 Validation ---
    print(f"\n{'='*60}\nFITTING CALIBRATORS ON 2024 VALIDATION\n{'='*60}")
    
    val_probs_rb = rb_model.predict_proba(val_df_rb[RB_FEATURES])[:, 1].reshape(-1, 1)
    val_probs_wr_te = wr_te_model.predict_proba(val_df_wr_te[WR_TE_FEATURES])[:, 1].reshape(-1, 1)
    val_probs_qb = qb_model.predict_proba(val_df_qb[QB_FEATURES])[:, 1].reshape(-1, 1)
    
    rb_calibrator = LogisticRegression()
    rb_calibrator.fit(val_probs_rb, val_df_rb['scored_touchdown'])
    
    wr_te_calibrator = LogisticRegression()
    wr_te_calibrator.fit(val_probs_wr_te, val_df_wr_te['scored_touchdown'])
    
    qb_calibrator = LogisticRegression()
    qb_calibrator.fit(val_probs_qb, val_df_qb['scored_touchdown'])
    
    print("✓ Calibrators trained")
    
    # --- Retrain on ALL Data for Production ---
    print(f"\n{'='*60}\nRETRAINING FINAL MODELS ON ALL HISTORICAL DATA FOR PRODUCTION\n{'='*60}")
    
    # RB
    print("\nRetraining final RB model on all data (2020-2024)...")
    X_all_rb = df[df['position'] == 'RB'][RB_FEATURES]
    y_all_rb = df[df['position'] == 'RB']['scored_touchdown']
    rb_final = LGBMClassifier(**rb_params, random_state=42, verbose=-1, n_jobs=-1)
    rb_final.fit(X_all_rb, y_all_rb)
    print("✓ RB retraining complete")
    
    # WR/TE
    print("\nRetraining final WR/TE model on all data (2020-2024)...")
    X_all_wr_te = df[df['position'].isin(['WR', 'TE'])][WR_TE_FEATURES]
    y_all_wr_te = df[df['position'].isin(['WR', 'TE'])]['scored_touchdown']
    wr_te_final = LGBMClassifier(**wr_te_params, random_state=42, verbose=-1, n_jobs=-1)
    wr_te_final.fit(X_all_wr_te, y_all_wr_te)
    print("✓ WR/TE retraining complete")
    
    # QB
    print("\nRetraining final QB model on all data (2020-2024)...")
    X_all_qb = df[df['position'] == 'QB'][QB_FEATURES]
    y_all_qb = df[df['position'] == 'QB']['scored_touchdown']
    qb_final = LGBMClassifier(**qb_params, random_state=42, verbose=-1, n_jobs=-1)
    qb_final.fit(X_all_qb, y_all_qb)
    print("✓ QB retraining complete")
    
    # --- Save Models ---
    print(f"\n{'='*60}\nSAVING MODEL ARTIFACTS LOCALLY\n{'='*60}")
    
    os.makedirs('models', exist_ok=True)
    
    def save_model(model, filename):
        print(f"Saving model artifact to '{filename}'...")
        joblib.dump(model, filename)
        print("Save successful.")
    
    save_model(rb_final, 'models/rb_lgbm_final.pkl')
    save_model(wr_te_final, 'models/wr_te_lgbm_final.pkl')
    save_model(qb_final, 'models/qb_lgbm_final.pkl')
    save_model(rb_calibrator, 'models/rb_lgbm_calibrator.pkl')
    save_model(wr_te_calibrator, 'models/wr_te_lgbm_calibrator.pkl')
    save_model(qb_calibrator, 'models/qb_lgbm_calibrator.pkl')
    
    # Save feature importance
    rb_importance.to_csv('models/rb_lgbm_feature_importance.csv', index=False)
    wr_te_importance.to_csv('models/wr_te_lgbm_feature_importance.csv', index=False)
    qb_importance.to_csv('models/qb_lgbm_feature_importance.csv', index=False)
    print("Feature importance saved to models/ directory")
    
    # Calculate total training time
    total_training_time = rb_training_time + wr_te_training_time + qb_training_time
    
    print(f"\n{'='*60}\nLIGHTGBM-ONLY TRAINING COMPLETE\n{'='*60}")
    print("\nModels saved:")
    print("  - models/rb_lgbm_final.pkl")
    print("  - models/wr_te_lgbm_final.pkl")
    print("  - models/qb_lgbm_final.pkl")
    print("  - models/*_lgbm_calibrator.pkl (3 files)")
    print("  - models/*_lgbm_best_params.json (3 files)")
    print("  - models/*_lgbm_feature_importance.csv (3 files)")
    
    print(f"\n{'='*60}\nPERFORMANCE SUMMARY\n{'='*60}")
    print(f"{'Position':<10} {'RF Baseline':<12} {'LightGBM':<12} {'Change':<12} {'Status'}")
    print("-" * 60)
    print(f"{'RB':<10} {0.539:<12.3f} {rb_precision:<12.3f} {rb_precision-0.539:+.3f}")
    print(f"{'WR/TE':<10} {0.411:<12.3f} {wr_te_precision:<12.3f} {wr_te_precision-0.411:+.3f}")
    print(f"{'QB':<10} {0.356:<12.3f} {qb_precision:<12.3f} {qb_precision-0.356:+.3f}")
    print("-" * 60)
    print(f"{'Unified':<10} {0.542:<12.3f} {avg_unified_precision:<12.3f} {diff:+.3f}       {status}")
    
    print(f"\n{'='*60}\nTRAINING TIME SUMMARY\n{'='*60}")
    print(f"RB model:     {rb_training_time/60:.1f} minutes")
    print(f"WR/TE model:  {wr_te_training_time/60:.1f} minutes")
    print(f"QB model:     {qb_training_time/60:.1f} minutes")
    print(f"{'─'*60}")
    print(f"Total time:   {total_training_time/60:.1f} minutes ({total_training_time:.0f} seconds)")
    
    print("\n🎯 Next steps:")
    print("  1. Use predict_lgbm_only.py for predictions")
    print("  2. Compare results with RandomForest models")
    print("  3. Analyze feature importances for insights")


if __name__ == '__main__':
    main()

