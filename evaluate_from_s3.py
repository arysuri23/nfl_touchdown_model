# evaluate_from_s3.py
#
# Description:
# This script pulls pre-trained models and data from an S3 bucket
# to evaluate model performance on a hold-out validation set.
# This avoids the need to retrain models for every performance analysis.

import os
import boto3
import joblib
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError

###
### CONSTANTS - CONFIGURE THESE
###
S3_BUCKET_NAME = "nfl-touchdown-model-data"
LOCAL_ARTIFACT_DIR = "s3_artifacts" # A local folder to store downloaded files
VALIDATION_YEAR = 2024

# --- Feature lists must match the ones used in training ---
CATEGORICAL_FEATURES = ['opponent_encoded']

RB_FEATURES = [
    'avg_offense_snap_share', 'team_continuity', 
    #'avg_carries', 'avg_rushing_yards', 'avg_receptions', 'avg_receiving_yards', 
    # 'avg_wopr', 'avg_rushing_epa', 'avg_receiving_epa', 'target_share', 
    'avg_receiving_air_yards', 'avg_racr', 'avg_scored_touchdown', 'avg_redzone_carry_share',
    'avg_redzone_target_share', 'avg_endzone_targets', 'avg_endzone_target_share', 'avg_inside_5_carry_share', 
    'avg_inside_5_target_share', 'rush_matchup_value', 'pass_matchup_value', 'redzone_td_rate', 
    'rushing_tds_allowed_to_RB', 'passing_tds_allowed_to_RB', 'implied_total','depth_chart_rank',
    'avg_rush_yards_over_expected_per_att', 'avg_rush_pct_over_expected', 'avg_avg_time_to_los', 'avg_percent_attempts_gte_eight_defenders' ]
WR_TE_FEATURES = [
    'avg_offense_snap_share', 'team_continuity', 
    #'avg_receptions', 'avg_receiving_yards', 
    'avg_wopr', 
    'avg_receiving_epa', 
    #'target_share', 'avg_receiving_air_yards', 
    'avg_racr', 
    'avg_scored_touchdown', 'avg_redzone_target_share', 'avg_endzone_targets', 'avg_endzone_target_share',
    #'avg_inside_5_target_share', 
    'pass_matchup_value', 'redzone_td_rate', 'passing_tds_allowed_to_WR',
    'passing_tds_allowed_to_TE', 'implied_total','depth_chart_rank',
    'avg_avg_cushion', 'avg_avg_separation', 'avg_avg_intended_air_yards', 'avg_percent_share_of_intended_air_yards', 
    'avg_catch_percentage', 'avg_avg_expected_yac', 'avg_avg_yac_above_expectation']
QB_FEATURES = [
    'avg_offense_snap_share', 'team_continuity', 'avg_carries', 'avg_rushing_yards', 'avg_rushing_epa', 
    'avg_scored_touchdown', 'avg_redzone_carry_share', 'avg_inside_5_carry_share',
    'rush_matchup_value', 'redzone_td_rate', 'rushing_tds_allowed_to_QB', 'implied_total', 'depth_chart_rank',
    'avg_rush_yards_over_expected_per_att', 'avg_rush_pct_over_expected', 'avg_avg_time_to_los',]

# --- S3 Artifact Paths ---
ARTIFACT_PATHS = {
    # Data
    "feature_df": "data/feature_df.csv",
    "opponent_encoder": "models/opponent_encoder.pkl",
    # RB Model
    "rb_base": "models/rb_base_final.pkl",
    "rb_meta": "models/rb_meta_final.pkl",
    "rb_scaler": "models/rb_scaler.pkl",
    # WR/TE Model
    "wr_te_base": "models/wr_te_base_final.pkl",
    "wr_te_meta": "models/wr_te_meta_final.pkl",
    "wr_te_scaler": "models/wr_te_scaler.pkl",
    # QB Model
    "qb_base": "models/qb_base_final.pkl",
    "qb_meta": "models/qb_meta_final.pkl",
    "qb_scaler": "models/qb_scaler.pkl",
}


###
### HELPER AND EVALUATION FUNCTIONS (COPIED FROM TRAIN SCRIPT)
###
def download_from_s3(bucket_name, s3_key, local_path):
    """Downloads a file from S3, creating local directory if needed."""
    s3_client = boto3.client('s3')
    local_dir = os.path.dirname(local_path)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    try:
        print(f"Downloading s3://{bucket_name}/{s3_key} to {local_path}...")
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"ERROR: The object s3://{bucket_name}/{s3_key} does not exist.")
        else:
            print(f"An unexpected error occurred: {e}")
        return False

def predict_stacked_proba(X, base_models, meta_model):
    """Generates final probabilities from a manually stacked model."""
    meta_features = np.column_stack([
        model.predict_proba(X)[:, 1] for model in base_models
    ])
    final_predictions = meta_model.predict_proba(meta_features)[:, 1]
    return final_predictions

def evaluate_model_at_k(predictions_df: pd.DataFrame, k: int = 25):
    """Calculates Precision@k and Recall@k for weekly NFL touchdown predictions."""
    print(rb_results_df.groupby('week')['scored_touchdown'].sum())
    weekly_results = []
    for week in sorted(predictions_df['week'].unique()):
        week_df = predictions_df[predictions_df['week'] == week]

        actual_scorers = set(week_df[week_df['scored_touchdown'] == 1]['player_display_name'])
        top_k_predictions = week_df.sort_values(by='predicted_prob', ascending=False).head(k)
        predicted_scorers = set(top_k_predictions['player_display_name'])

        hits = len(predicted_scorers.intersection(actual_scorers))
        precision_at_k = hits / k if k > 0 else 0
        recall_at_k = hits / len(actual_scorers) if len(actual_scorers) > 0 else 0

        weekly_results.append({'week': week, 'precision_at_k': precision_at_k, 'recall_at_k': recall_at_k, 'successful_picks': hits})
        
    return pd.DataFrame(weekly_results)


###
### MAIN EXECUTION
###
if __name__ == "__main__":
    # --- 1. Download all required artifacts from S3 ---
    print("\n" + "="*60 + "\nDownloading Artifacts from S3...\n" + "="*60)
    local_paths = {}
    for name, s3_key in ARTIFACT_PATHS.items():
        local_path = os.path.join(LOCAL_ARTIFACT_DIR, s3_key)
        if download_from_s3(S3_BUCKET_NAME, s3_key, local_path):
            local_paths[name] = local_path
        else:
            # Exit if a critical file is missing
            exit()
            
    # --- 2. Load artifacts into memory ---
    print("\n" + "="*60 + "\nLoading Models and Data...\n" + "="*60)
    # Load Models
    rb_base_models = joblib.load(local_paths["rb_base"])
    rb_meta_model = joblib.load(local_paths["rb_meta"])
    wr_te_base_models = joblib.load(local_paths["wr_te_base"])
    wr_te_meta_model = joblib.load(local_paths["wr_te_meta"])
    qb_base_models = joblib.load(local_paths["qb_base"])
    qb_meta_model = joblib.load(local_paths["qb_meta"])
    # Load Scalers
    rb_scaler = joblib.load(local_paths["rb_scaler"])
    wr_te_scaler = joblib.load(local_paths["wr_te_scaler"])
    qb_scaler = joblib.load(local_paths["qb_scaler"])
    # Load Data
    feature_df = pd.read_csv(local_paths["feature_df"])



    # --- 3. Prepare Validation Data ---
    print("\n" + "="*60 + f"\nPreparing Validation Data for Season {VALIDATION_YEAR}...\n" + "="*60)
    validation_df = feature_df[feature_df['season'] == VALIDATION_YEAR].copy()
    val_df_rb = validation_df[validation_df['position'] == 'RB'].copy()
    val_df_wr_te = validation_df[validation_df['position'].isin(['WR', 'TE'])].copy()
    val_df_qb = validation_df[validation_df['position'] == 'QB'].copy()


    # --- 4. Evaluate Each Model ---
    all_results = []
    
    # RB Evaluation
    print("\n" + "="*60 + "\nEVALUATING: RB Model\n" + "="*60)
    X_val_rb = val_df_rb[RB_FEATURES].copy()
    numerical_features_rb = [f for f in RB_FEATURES if f not in CATEGORICAL_FEATURES]
    X_val_rb.loc[:, numerical_features_rb] = rb_scaler.transform(X_val_rb[numerical_features_rb])
    rb_preds = predict_stacked_proba(X_val_rb, rb_base_models, rb_meta_model)
    rb_results_df = val_df_rb[['player_display_name', 'week', 'scored_touchdown']].copy()
    rb_results_df['predicted_prob'] = rb_preds
    rb_results_df.to_csv('rb.csv', index=False)

    rb_perf = evaluate_model_at_k(rb_results_df)
    print(rb_perf)
    print("\n--- Average Season Performance ---")
    print(rb_perf.mean())
    all_results.append(rb_results_df)
    

    # WR/TE Evaluation
    print("\n" + "="*60 + "\nEVALUATING: WR/TE Model\n" + "="*60)
    X_val_wr_te = val_df_wr_te[WR_TE_FEATURES].copy()
    numerical_features_wr_te = [f for f in WR_TE_FEATURES if f not in CATEGORICAL_FEATURES]
    X_val_wr_te.loc[:, numerical_features_wr_te] = wr_te_scaler.transform(X_val_wr_te[numerical_features_wr_te])
    wr_te_preds = predict_stacked_proba(X_val_wr_te, wr_te_base_models, wr_te_meta_model)
    wr_te_results_df = val_df_wr_te[['player_display_name', 'week', 'scored_touchdown']].copy()
    wr_te_results_df['predicted_prob'] = wr_te_preds
    wr_te_perf = evaluate_model_at_k(wr_te_results_df)
    print(wr_te_perf)
    print("\n--- Average Season Performance ---")
    print(wr_te_perf.mean())
    all_results.append(wr_te_results_df)

    # QB Evaluation
    print("\n" + "="*60 + "\nEVALUATING: QB Model\n" + "="*60)
    X_val_qb = val_df_qb[QB_FEATURES].copy()
    numerical_features_qb = [f for f in QB_FEATURES if f not in CATEGORICAL_FEATURES]
    X_val_qb.loc[:, numerical_features_qb] = qb_scaler.transform(X_val_qb[numerical_features_qb])
    qb_preds = predict_stacked_proba(X_val_qb, qb_base_models, qb_meta_model)
    qb_results_df = val_df_qb[['player_display_name', 'week', 'scored_touchdown']].copy()
    qb_results_df['predicted_prob'] = qb_preds
    qb_perf = evaluate_model_at_k(qb_results_df)
    print(qb_perf)
    print("\n--- Average Season Performance ---")
    print(qb_perf.mean())
    all_results.append(qb_results_df)
    
    # --- 5. Unified Evaluation ---
    print("\n" + "="*60 + "\nUNIFIED EVALUATION ON 2024 SEASON\n" + "="*60)
    unified_df = pd.concat(all_results)
    unified_perf = evaluate_model_at_k(unified_df)
    print(unified_perf)
    print("\n--- Average Season Performance (All Positions) ---")
    print(unified_perf.mean())

    print("\n\nEvaluation complete.")