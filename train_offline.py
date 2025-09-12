# NFL Touchdown Scorer Prediction Model
# Final Version with Position-Specific Models & Manual Time-Series Stacking


# --- 1. Importing Libraries ---
import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from sklearn.base import clone
import nfl_td_lambda.data_collection as data
import joblib
import boto3
from io import BytesIO
import os
import sys



### CONSTANTS ###
S3_BUCKET_NAME = "nfl-touchdown-model-data"

# -- Position-Specific Feature Lists --

CATEGORICAL_FEATURES = ['opponent_encoded']

RB_FEATURES = [
    'avg_offense_snap_share',
    #'team_continuity', # Creative feature, monitor its importance as it could be noisy.

    # --- Usage & Opportunity Metrics ---
    # avg_wopr is a powerful composite metric. It's calculated from target share and air yards share.
    # It might be correlated with target_share. Test model performance with one or the other commented out.
    'avg_wopr',
    #'target_share', # Potentially correlated with avg_wopr.

    # --- Efficiency Metrics ---
    'avg_rushing_epa', # Advanced metric for rushing efficiency.
    'avg_receiving_epa', # Advanced metric for receiving efficiency.
    'avg_racr', # Measures efficiency in converting air yards to receiving yards.
    
    # These Next Gen Stats measure rushing efficiency and style. They might be correlated with avg_rushing_epa.
    # Consider testing the model with avg_rushing_epa vs. this block of features.
    'avg_rush_yards_over_expected_per_att', # Correlated with avg_rushing_epa.
    'avg_rush_pct_over_expected',
    'avg_avg_time_to_los',
    'avg_percent_attempts_gte_eight_defenders', # Measures ability against stacked boxes.

    # --- High-Value Touches ---
    # These features are crucial but can be correlated. For example, a high redzone carry share
    # often leads to a high inside_5 carry share. The model should handle this, but it's worth noting.
    'avg_redzone_carry_share',
    'avg_redzone_target_share',
    'avg_endzone_targets',
    'avg_endzone_target_share',
    'avg_inside_5_carry_share',
    'avg_inside_5_target_share',

    # --- Contextual & Matchup Features ---
    'rush_matchup_value',
    'pass_matchup_value',
    #'redzone_td_rate', # Team-level efficiency in the red zone.
    'rushing_tds_allowed_to_RB', # Opponent tendency.
    'passing_tds_allowed_to_RB', # Opponent tendency.
    'implied_total', # Game script proxy.
    'depth_chart_rank',

    # --- Target Variable Lag ---
    # This is a lagged version of the target. Highly predictive, but ensure no data leakage.
    'avg_scored_touchdown',

    # --- Commented Out: Base Volume Stats ---
    # These are likely redundant given the advanced metrics above (e.g., avg_wopr, avg_rushing_epa).
    #'avg_carries',
    #'avg_rushing_yards',
    #'avg_receptions',
   # 'avg_receiving_yards',
    #'avg_receiving_air_yards',
]
WR_TE_FEATURES = [
    'avg_offense_snap_share',
    #'team_continuity', # Creative feature, monitor its importance.

    # --- Opportunity & Usage Metrics ---
    # avg_wopr is a composite of target share and air yards share. It's highly correlated with
    # avg_percent_share_of_intended_air_yards. Test them against each other.
    'avg_wopr',
    'target_share', # Consider testing this uncommented. It's a fundamental metric and might offer value alongside wopr.

    # --- Efficiency & Route-Running Metrics ---
    'avg_receiving_epa', # Advanced efficiency metric.
    'avg_racr', # Efficiency in converting air yards to real yards.

    # avg_avg_cushion and avg_avg_separation both measure a receiver's ability to get open.
    # They are likely correlated. Test with one or both.
    #'avg_avg_cushion',
    'avg_avg_separation',

    # These metrics describe how a player is used and how well they perform on their targets.
    # They are generally complementary.
    'avg_percent_share_of_intended_air_yards', # Correlated with avg_wopr.
    'avg_catch_percentage',
    'avg_avg_expected_yac',
    'avg_avg_yac_above_expectation',

    # --- High-Value Touches ---
    # Crucial for touchdown prediction.
    'avg_redzone_target_share',
    'avg_endzone_targets',
    'avg_endzone_target_share',
    'avg_inside_5_target_share',

    # --- Contextual & Matchup Features ---
    'pass_matchup_value',
    #'redzone_td_rate', # Team-level efficiency.
    'passing_tds_allowed_to_WR', # Opponent tendency.
    'passing_tds_allowed_to_TE', # Opponent tendency.
    'implied_total', # Game script proxy.
    'depth_chart_rank',

    # --- Target Variable Lag ---
    # Highly predictive, but ensure no data leakage.
    'avg_scored_touchdown',

    # --- Commented Out: Base Volume & Air Yard Stats ---
    # Mostly captured by more advanced metrics.
    'avg_receptions',
    'avg_receiving_yards',
    'avg_receiving_air_yards', # Raw air yards; avg_percent_share_of_intended_air_yards is often more predictive.
    'avg_avg_intended_air_yards', # Player's aDOT; consider testing this uncommented. It shows role (deep threat vs. possession).
]
QB_FEATURES = [
    'avg_offense_snap_share',
    #'team_continuity',
    'avg_carries', 'avg_rushing_yards', 'avg_rushing_epa', 
    'avg_scored_touchdown', 'avg_redzone_carry_share', 'avg_inside_5_carry_share',
    'rush_matchup_value', 
    #'redzone_td_rate', 
    'rushing_tds_allowed_to_QB', 'implied_total', 'depth_chart_rank',
    'avg_rush_yards_over_expected_per_att', 'avg_rush_pct_over_expected', 'avg_avg_time_to_los',]



# -- Hyperparameter Distributions --
RF_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}



LGBM_PARAM_DIST = {
    'n_estimators': [100, 200, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 40, 50],
    'max_depth': [-1, 10, 20],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# --- 3. Feature Engineering ---
# [Feature engineering function from the original script is included here]
#def feature_engineering(df, redzone_df, redzone_td_rate, ez_target_df, odds_df, goal_line_df, positional_defense_df, depth_chart_df, snap_counts_df, ngs_rushing_df, ngs_receiving_df):
def feature_engineering(df): 
    """Engineers features from the raw data to improve model performance."""
   
    # df = pd.merge(df, redzone_df, on=['player_id', 'week', 'season'], how='left')
    # #df = pd.merge(df, redzone_td_rate, on=['recent_team', 'season', 'week'], how='left')
    # df = pd.merge(df, ez_target_df, on=['player_id', 'season', 'week'], how='left')
    # df = pd.merge(df, odds_df, left_on=['recent_team', 'season', 'week'], right_on=['team', 'season', 'week'], how='left')
    # df = pd.merge(df, goal_line_df, on=['player_id', 'week', 'season'], how='left')
    # df = pd.merge(df, positional_defense_df, on=['opponent_team', 'season', 'week'], how='left')
    # df = pd.merge(df, depth_chart_df, on=['player_id', 'season', 'week'], how='left')
    # df['depth_chart_rank'] = pd.to_numeric(df['depth_chart_rank'], errors='coerce').fillna(4).astype(int)
    
    

    # df = pd.merge(df, snap_counts_df, on=['merge_name', 'season', 'week'])
    # df['offense_snap_share'].fillna(0, inplace=True)


    # df = pd.merge(df, ngs_rushing_df, on=['player_id', 'season', 'week'], how='left')
    # df = pd.merge(df, ngs_receiving_df, on=['player_id', 'season', 'week'], how='left')

    # df.fillna(0, inplace=True)

    # df.sort_values(by=['season', 'week', 'player_id'], inplace=True, ignore_index=True)
    
    
 

    player_stats = ['carries', 'rushing_yards', 'receptions', 'receiving_yards', 'wopr', 'rushing_epa', 'receiving_epa', 'target_share',
                      'receiving_air_yards', 'racr', 'scored_touchdown', 'redzone_carry_share', 'redzone_target_share',
                      'endzone_targets', 'endzone_target_share', 'inside_5_carry_share', 'inside_5_target_share', 'offense_snap_share', 
                      'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 'avg_time_to_los', 'percent_attempts_gte_eight_defenders',
                      'avg_cushion', 'avg_separation', 'avg_intended_air_yards', 'percent_share_of_intended_air_yards', 
                    'catch_percentage', 'avg_expected_yac', 'avg_yac_above_expectation']
    
    for stat in player_stats:
        df[f'avg_{stat}'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
   
    pos_defense_cols = [col for col in df.columns if 'tds_allowed_to' in col]

    opponent_stats_df = df[['season', 'week', 'opponent_team'] + pos_defense_cols].drop_duplicates()
    opponent_stats_df.sort_values(by=['season', 'week'], inplace=True)
    
    for col in pos_defense_cols:
         opponent_stats_df[col] = opponent_stats_df.groupby('opponent_team')[col].transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())


    df.drop(columns=pos_defense_cols, inplace=True)
    df = pd.merge(df, opponent_stats_df, on=['season', 'week', 'opponent_team'], how='left')


    
   # df['redzone_td_rate'] = df.groupby('recent_team')['redzone_td_rate'].transform(lambda x: x.shift(1).ewm(span=4, min_periods=1).mean())
    df['rush_matchup_value'] = np.select(
        [df['position'] == 'RB', df['position'] == 'QB'],
        [df['avg_redzone_carry_share'] * df['rushing_tds_allowed_to_RB'], df['avg_redzone_carry_share'] * df['rushing_tds_allowed_to_QB']],
        default=0)
    df['pass_matchup_value'] = np.select(
        [df['position'] == 'RB', df['position'] == 'WR', df['position'] == 'TE'],
        [df['avg_redzone_target_share'] * df['passing_tds_allowed_to_RB'], df['avg_redzone_target_share'] * df['passing_tds_allowed_to_WR'], df['avg_redzone_target_share'] * df['passing_tds_allowed_to_TE']],
        default=0)
    df.fillna(0, inplace=True)
    df['position_encoded'] = LabelEncoder().fit_transform(df['position'])
    df['opponent_encoded'] = LabelEncoder().fit_transform(df['opponent_team'])

    return df
    



# --- 4. Position-Specific Model Training ---


###
### NEW: MANUAL TIME-SERIES STACKING IMPLEMENTATION
###
def train_stacked_model_timeseries(X, y, base_estimators, meta_estimator, n_splits=5):
    """
    Trains a stacked model using time-series cross-validation to generate meta-features.


    Returns:
        - A list of base estimators trained on the full dataset.
        - The meta-estimator trained on the out-of-fold predictions.
    """
    print("Generating out-of-fold predictions for meta-model training...")
    # Initialize an array for meta-features, with one column per base estimator
    meta_features = np.full((len(X), len(base_estimators)), np.nan)
    
    # Use TimeSeriesSplit to respect chronological order
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    first_test_fold_start = 0 # To find where our predictions start
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        if i == 0:
            first_test_fold_start = test_index[0]
            
        # For each base model, fit on past data and predict on future data
        for j, estimator in enumerate(base_estimators):
            # Clone the estimator to ensure it's fresh for each fold
            model = clone(estimator)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            predictions = model.predict_proba(X.iloc[test_index])[:, 1]
            meta_features[test_index, j] = predictions


    # Trim the data to only include rows for which we have out-of-fold predictions
    valid_indices = np.arange(first_test_fold_start, len(X))
    meta_features_for_training = meta_features[valid_indices]
    y_for_training = y.iloc[valid_indices]


    print("Training meta-model on out-of-fold predictions...")
    trained_meta_estimator = clone(meta_estimator)
    trained_meta_estimator.fit(meta_features_for_training, y_for_training)


    print("Training final base models on all available data...")
    trained_base_estimators = []
    for estimator in base_estimators:
        final_base_model = clone(estimator)
        final_base_model.fit(X, y)
        trained_base_estimators.append(final_base_model)
        
    return trained_base_estimators, trained_meta_estimator



def tune_and_train_specialist_model(df_position, features, rf_param_dist, lgbm_param_dist, validation_year=2024):
    """
    Tunes hyperparameters and trains a stacked model using manual time-series logic.
    """
    model_type = df_position['position'].unique()[0]
    if len(df_position['position'].unique()) > 1: model_type = "WR/TE"
            
    print("\n" + "="*60 + f"\nTUNING AND TRAINING FOR: {model_type} Model\n" + "="*60)


    # --- 1. Split data for hyperparameter tuning ---
    train_df = df_position[df_position['season'] < validation_year]
    X_train = train_df[features]
    y_train = train_df['scored_touchdown']

    # --- NEW: Feature Scaling ---
    numerical_features = [f for f in features if f not in CATEGORICAL_FEATURES]
    scaler = StandardScaler()
    
    # Fit the scaler ONLY on the training data
    X_train.loc[:, numerical_features] = scaler.fit_transform(X_train[numerical_features])
    print(f"Scaler fitted for {model_type} model.")

    tscv = TimeSeriesSplit(n_splits=5)


    # --- 2. Tune RandomForest ---
    print(f"\nTuning RandomForest for {model_type}s...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_dist, n_iter=25, cv=tscv, scoring='average_precision', n_jobs=-1, random_state=42)
    rf_search.fit(X_train, y_train)
    best_rf_params = rf_search.best_params_
    print(f"Best RF Params: {best_rf_params}")


    # --- 3. Tune LightGBM ---
    print(f"\nTuning LightGBM for {model_type}s...")
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1)
    lgbm_search = RandomizedSearchCV(estimator=lgbm, param_distributions=lgbm_param_dist, n_iter=25, cv=tscv, scoring='average_precision', n_jobs=-1, random_state=42)
    lgbm_search.fit(X_train, y_train)
    best_lgbm_params = lgbm_search.best_params_
    print(f"Best LGBM Params: {best_lgbm_params}")


    # --- 4. Train Final Stacked Model using Manual Time-Series Logic ---
    print(f"\nTraining {model_type} Stacked Model for validation...")
    base_estimators = [
        RandomForestClassifier(random_state=42, class_weight='balanced', **best_rf_params),
        lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1, **best_lgbm_params)
    ]
    meta_estimator = LogisticRegression(class_weight='balanced')
    
    # This trains the models needed for evaluation on the validation set
    base_models, meta_model = train_stacked_model_timeseries(X_train, y_train, base_estimators, meta_estimator)
        
    print("Final model training complete.")
    return base_models, meta_model, best_rf_params, best_lgbm_params, scaler



# --- 5. Model Evaluation ---
###
### UPDATED: EVALUATION FOR MANUAL STACKING
###
def predict_stacked_proba(X, base_models, meta_model):
    """Generates final probabilities from a manually stacked model."""
    # Generate predictions from each base model
    meta_features = np.column_stack([
        model.predict_proba(X)[:, 1] for model in base_models
    ])
    # Use the meta-model to make the final prediction
    final_predictions = meta_model.predict_proba(meta_features)[:, 1]
    return final_predictions


def evaluate_model_at_k(predictions_df: pd.DataFrame, k: int = 25):
    """Calculates Precision@k and Recall@k for weekly NFL touchdown predictions."""
    weekly_results = []
    for week in sorted(predictions_df['week'].unique()):
        week_df = predictions_df[predictions_df['week'] == week]

        actual_scorers = set(week_df[week_df['scored_touchdown'] == 1]['player_display_name'])
        top_k_predictions = week_df.sort_values(by='predicted_prob', ascending=False).head(k)
        predicted_scorers = set(top_k_predictions['player_display_name'])
        # print(f"Week: {week}")
        # print(predicted_scorers)
        # print(actual_scorers)

        hits = len(predicted_scorers.intersection(actual_scorers))
        precision_at_k = hits / k if k > 0 else 0
        recall_at_k = hits / len(actual_scorers) if len(actual_scorers) > 0 else 0

        weekly_results.append({'week': week, 'precision_at_k': precision_at_k, 'recall_at_k': recall_at_k, 'successful_picks': hits})
        
    return pd.DataFrame(weekly_results)


def evaluate_specialist_model(base_models, meta_model,scaler, model_name, validation_df, features, k=25):
    """Calculates performance metrics for a manually stacked model."""
    print("\n" + "="*60 + f"\nEVALUATION FOR: {model_name}\n" + "="*60)
    if validation_df.empty:
        print("Validation data is empty. Skipping evaluation.")
        return

    # Prepare X_val from the validation dataframe
    X_val = validation_df[features].copy() # Use .copy() to avoid SettingWithCopyWarning
    y_val = validation_df['scored_touchdown']

    # --- THIS IS THE FIX ---
    # Scale X_val using the scaler that was FIT ON THE TRAINING DATA
    numerical_features = [f for f in features if f not in CATEGORICAL_FEATURES]
    X_val.loc[:, numerical_features] = scaler.transform(X_val[numerical_features])
    # -----------------------

    # --- 1. Performance Metrics (Precision@k) ---
    print(f"\n--- Weekly Performance @ K={k} ---")
    y_pred_proba = predict_stacked_proba(X_val, base_models, meta_model)
    
    results_df = validation_df[['player_display_name', 'week', 'scored_touchdown']].copy()
    results_df['predicted_prob'] = y_pred_proba
    
    weekly_performance = evaluate_model_at_k(results_df, k=k)
    print(weekly_performance)
    
    average_performance = weekly_performance.mean()
    print("\n--- Average Season Performance ---")
    print(f"Average Precision@{k}: {average_performance['precision_at_k']:.3f}")
    print(f"Average Recall@{k}:    {average_performance['recall_at_k']:.3f}")
    print(f"Average Successful Picks Per Week: {average_performance['successful_picks']:.1f}")


    # --- 2. Base Model Importance (Meta-Model Coefficients) ---
    print("\n--- Base Model Importance (Final Estimator Weights) ---")
    final_estimator_coefs = meta_model.coef_[0]
    # The order of base models is preserved from training
    base_model_names = ['RandomForest', 'LightGBM']
    model_importance_df = pd.DataFrame({
        'Base Model': base_model_names,
        'Coefficient (Weight)': final_estimator_coefs
    }).sort_values(by='Coefficient (Weight)', ascending=False)
    print(model_importance_df)


###
### UPDATED: RETRAINING FUNCTION FOR MANUAL STACKING
###
def train_model_on_all_data(df_position, features, best_rf_params, best_lgbm_params):
    """Retrains a final stacked model on all data using the best hyperparameters."""
    model_type = df_position['position'].unique()[0]
    if len(df_position['position'].unique()) > 1: model_type = "WR/TE"
    print(f"\nRetraining final {model_type} model on all data (2020-2024)...")


    X_full = df_position[features]
    y_full = df_position['scored_touchdown']

    numerical_features = [f for f in features if f not in CATEGORICAL_FEATURES]
    scaler = StandardScaler()
    X_full.loc[:, numerical_features] = scaler.fit_transform(X_full[numerical_features])
    


    base_estimators = [
        RandomForestClassifier(random_state=42, class_weight='balanced', **best_rf_params),
        lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1, **best_lgbm_params)
    ]
    meta_estimator = LogisticRegression()


    # Use the same robust training logic on the full dataset
    final_base_models, final_meta_model = train_stacked_model_timeseries(X_full, y_full, base_estimators, meta_estimator)


    print(f"{model_type} retraining complete.")
    return final_base_models, final_meta_model, scaler



# --- S3 Upload Helper Function ---
def write_joblib_to_s3(python_object, bucket_name, s3_key):
    """
    Serializes a Python object with joblib and uploads it to an S3 bucket.
    """
    s3_client = boto3.client('s3')
    print(f"Uploading model artifact to 's3://{bucket_name}/{s3_key}'...")
    try:
        with BytesIO() as buffer:
            joblib.dump(python_object, buffer)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, bucket_name, s3_key)
        print("Upload successful.")
        return True
    except Exception as e:
        print(f"An error occurred during S3 upload: {e}")
        return False
    
def upload_csv_to_s3(local_file_path, bucket_name, s3_key):
    """
    Uploads a local CSV file to an S3 bucket.
    """
    s3_client = boto3.client('s3')
    print(f"Uploading data file to 's3://{bucket_name}/{s3_key}'...")
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        print("Upload successful.")
        return True
    except FileNotFoundError:
        print(f"Error: The file '{local_file_path}' was not found in the current directory.")
        return False
    except Exception as e:
        print(f"An error occurred during S3 data upload: {e}")
        return False


if __name__ == '__main__':
    # -- Configuration --
    all_years_to_load = range(2020, 2026) ### TODO: change to 2026

    



    # -- Data Loading --
    print("Loading data...")
    #pbp = nfl.import_pbp_data(all_years_to_load, downcast=True)
    
    #rosters = nfl.import_seasonal_rosters(all_years_to_load)
    nfl_teams = pd.read_csv('nfl_teams.csv')
    team_map = dict(zip(nfl_teams['team_name'], nfl_teams['team_id']))
    
    # Ensure we only load data for weeks 1-18
    # pbp = pbp[pbp['week'] <= 18]

    # nfl_df = data.get_nfl_data([2020,2021,2022,2023,2024])
    # nfl_2025_df = data.get_nfl_2025_weekly_data()

    # nfl_df = pd.concat([nfl_df, nfl_2025_df], ignore_index=True)

    # nfl_df = nfl_df[nfl_df['week'] <= 18]

    # nfl_df['merge_name'] = nfl_df['player_display_name'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()


    
    # redzone_df = data.get_redzone_data(pbp)
    # redzone_df = redzone_df[redzone_df['week'] <= 18]
    # redzone_td_df = data.get_redzone_td_rate(pbp)
    # redzone_td_df = redzone_td_df[redzone_td_df['week'] <= 18]
    # ez_target_df = data.get_endzone_target_data(pbp)
    # ez_target_df = ez_target_df[ez_target_df['week'] <= 18]
    # odds_df = data.get_odds_data(all_years_to_load, team_map)
    # odds_df = odds_df[odds_df['week'] <= 18]
    # goal_line_df = data.get_goal_line_data(pbp)
    # goal_line_df = goal_line_df[goal_line_df['week'] <= 18]
    # positional_defense_df = data.get_opponent_positional_data(pbp, rosters)
    # positional_defense_df = positional_defense_df[positional_defense_df['week'] <= 18]
    # depth_chart_df = data.get_depth_chart_data([2020, 2021, 2022, 2023, 2024])
    # depth_chart_df_2025 = data.get_2025_depth_chart_data()

    # depth_chart_df = pd.concat([depth_chart_df, depth_chart_df_2025], ignore_index=True)
    # depth_chart_df = depth_chart_df[depth_chart_df['week'] <= 18]

    # snap_counts_df = data.get_snap_counts(all_years_to_load)
    # snap_counts_df = snap_counts_df[snap_counts_df['week']<=18]
    # ngs_rushing_df = data.get_ngs_data_rushing(all_years_to_load)
    # ngs_receiving_df = data.get_ngs_data_receiving(all_years_to_load)
    # ngs_receiving_df = ngs_receiving_df[ngs_receiving_df['week'] <= 18]

    # -- Feature Engineering --
    print("Engineering features...")
    nfl_df = data.get_all_historic_data(all_years_to_load, team_map)
    nfl_df.to_csv("raw_nfl_data.csv", index=False)
    feature_df = feature_engineering(nfl_df)
    #feature_df = feature_engineering(nfl_df, redzone_df, redzone_td_df, ez_target_df, odds_df, goal_line_df, positional_defense_df, depth_chart_df, snap_counts_df, ngs_rushing_df, ngs_receiving_df)
    feature_df.to_csv("feature_df.csv", index=False)

    

    


    df_rb = feature_df[feature_df['position'] == 'RB'].copy()
    df_wr_te = feature_df[feature_df['position'].isin(['WR', 'TE'])].copy()
    df_qb = feature_df[feature_df['position'] == 'QB'].copy()


    # --- Phase 1: Tune, Train, and Evaluate on 2024 Season ---
    # The function now returns the trained base/meta models and the best params
    rb_models, rb_meta_model, rb_rf_params, rb_lgbm_params, rb_scaler = tune_and_train_specialist_model(df_rb, RB_FEATURES, RF_PARAM_DIST, LGBM_PARAM_DIST)
    wr_te_models, wr_te_meta_model, wr_te_rf_params, wr_te_lgbm_params, wr_te_scaler = tune_and_train_specialist_model(df_wr_te, WR_TE_FEATURES, RF_PARAM_DIST, LGBM_PARAM_DIST)
    qb_models, qb_meta_model, qb_rf_params, qb_lgbm_params, qb_scaler = tune_and_train_specialist_model(df_qb, QB_FEATURES, RF_PARAM_DIST, LGBM_PARAM_DIST)


    # --- MODEL EVALUATION ON 2024 SEASON ---
    validation_year = 2024
    validation_df = feature_df[feature_df['season'] == validation_year]
    val_df_rb = validation_df[validation_df['position'] == 'RB'].copy()
    val_df_wr_te = validation_df[validation_df['position'].isin(['WR', 'TE'])].copy()
    val_df_qb = validation_df[validation_df['position'] == 'QB'].copy()

    # Pass the scaler object during the evaluation call
    evaluate_specialist_model(rb_models, rb_meta_model, rb_scaler, "RB Model", val_df_rb, RB_FEATURES, k=15)
    evaluate_specialist_model(wr_te_models, wr_te_meta_model, wr_te_scaler, "WR/TE Model", val_df_wr_te, WR_TE_FEATURES)
    evaluate_specialist_model(qb_models, qb_meta_model, qb_scaler, "QB Model", val_df_qb, QB_FEATURES,k=5)

    



     # --- UNIFIED MODEL EVALUATION ON 2024 SEASON ---
    print("\n" + "="*60 + "\nUNIFIED EVALUATION ON 2024 SEASON\n" + "="*60)
    validation_year = 2024
    validation_df = feature_df[feature_df['season'] == validation_year].copy()

    # Get predictions for each position group
    val_df_rb = validation_df[validation_df['position'] == 'RB'].copy()
    val_df_rb['predicted_prob'] = predict_stacked_proba(val_df_rb[RB_FEATURES], rb_models, rb_meta_model)

    val_df_wr_te = validation_df[validation_df['position'].isin(['WR', 'TE'])].copy()
    val_df_wr_te['predicted_prob'] = predict_stacked_proba(val_df_wr_te[WR_TE_FEATURES], wr_te_models, wr_te_meta_model)

    val_df_qb = validation_df[validation_df['position'] == 'QB'].copy()
    val_df_qb['predicted_prob'] = predict_stacked_proba(val_df_qb[QB_FEATURES], qb_models, qb_meta_model)

    # Combine all predictions into a single DataFrame
    combined_results_df = pd.concat([val_df_rb, val_df_wr_te, val_df_qb])

    # Now, evaluate the combined results
    # This will give a true measure of performance across all positions
    unified_weekly_performance = evaluate_model_at_k(combined_results_df, k=25)
    print("\n--- Weekly Performance @ K=25 (All Positions) ---")
    print(unified_weekly_performance)

    average_performance = unified_weekly_performance.mean()
    print("\n--- Average Season Performance (All Positions) ---")
    print(f"Average Precision@25: {average_performance['precision_at_k']:.3f}")
    print(f"Average Recall@25:    {average_performance['recall_at_k']:.3f}")
    print(f"Average Successful Picks Per Week: {average_performance['successful_picks']:.1f}")


    
  
   
    # --- Phase 2: Retrain Final Models on All Data (2020-2024) ---
    print("\n" + "="*60 + "\nRETRAINING FINAL MODELS ON ALL HISTORICAL DATA FOR PREDICTION\n" + "="*60)
    rb_base_final, rb_meta_final, rb_scaler_final = train_model_on_all_data(df_rb, RB_FEATURES, rb_rf_params, rb_lgbm_params)
    wr_te_base_final, wr_te_meta_final, wr_te_scaler_final = train_model_on_all_data(df_wr_te, WR_TE_FEATURES, wr_te_rf_params, wr_te_lgbm_params)
    qb_base_final, qb_meta_final, qb_scaler_final = train_model_on_all_data(df_qb, QB_FEATURES, qb_rf_params, qb_lgbm_params)

    print("\n" + "="*60 + "\nUPLOADING MODEL ARTIFACTS TO S3\n" + "="*60)
    # Save RB models
    write_joblib_to_s3(rb_base_final, S3_BUCKET_NAME, 'models/rb_base_final.pkl')
    write_joblib_to_s3(rb_meta_final, S3_BUCKET_NAME, 'models/rb_meta_final.pkl')
    # Save WR/TE models
    write_joblib_to_s3(wr_te_base_final, S3_BUCKET_NAME, 'models/wr_te_base_final.pkl')
    write_joblib_to_s3(wr_te_meta_final, S3_BUCKET_NAME, 'models/wr_te_meta_final.pkl')
    # Save QB models
    write_joblib_to_s3(qb_base_final, S3_BUCKET_NAME, 'models/qb_base_final.pkl')
    write_joblib_to_s3(qb_meta_final, S3_BUCKET_NAME, 'models/qb_meta_final.pkl')
    # Save the crucial opponent label encoder
    opponent_le = LabelEncoder().fit(feature_df['opponent_team'].unique())
    write_joblib_to_s3(opponent_le, S3_BUCKET_NAME, 'models/opponent_encoder.pkl')

    write_joblib_to_s3(rb_scaler_final, S3_BUCKET_NAME, 'models/rb_scaler.pkl')
    write_joblib_to_s3(wr_te_scaler_final, S3_BUCKET_NAME, 'models/wr_te_scaler.pkl')
    write_joblib_to_s3(qb_scaler_final, S3_BUCKET_NAME, 'models/qb_scaler.pkl')
    
    print("\n" + "="*60 + "\nUPLOADING DATA FILES TO S3\n" + "="*60)
    # List of data files required by the prediction app
    required_data_files = ['nfl_teams.csv', 'data/week_2_lines.csv', 'data/week_2_td_odds.csv', 'feature_df.csv', 'raw_nfl_data.csv']
    for file_name in required_data_files:
        s3_key = f"data/{file_name}"
        upload_csv_to_s3(file_name, S3_BUCKET_NAME, s3_key)

    print("\n" + "="*60 + "\nOFFLINE TRAINING AND DEPLOYMENT COMPLETE.\n" + "="*60)
    

