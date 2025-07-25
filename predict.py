# NFL Touchdown Scorer Prediction Model
# Final Version with Position-Specific Models & Manual Time-Series Stacking


# --- 1. Importing Libraries ---
import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from sklearn.base import clone
import data_collection as data



### CONSTANTS ###

# -- Position-Specific Feature Lists --
RB_FEATURES = [
    'avg_carries', 'avg_rushing_yards', 'avg_receptions', 'avg_receiving_yards', 'avg_wopr', 'avg_rushing_epa', 'avg_receiving_epa',
    'target_share', 'avg_receiving_air_yards', 'avg_air_yards_share', 'avg_racr', 'avg_scored_touchdown', 'avg_redzone_carry_share',
    'avg_redzone_target_share', 'avg_endzone_targets', 'avg_endzone_target_share', 'avg_inside_5_carry_share', 'avg_inside_5_target_share',
    'rush_matchup_value', 'pass_matchup_value', 'redzone_td_rate', 'rushing_tds_allowed_to_RB', 'passing_tds_allowed_to_RB', 'implied_total', 'opponent_encoded']
WR_TE_FEATURES = [
    'avg_receptions', 'avg_receiving_yards', 'avg_wopr', 'avg_receiving_epa', 'target_share', 'avg_receiving_air_yards',
    'avg_air_yards_share', 'avg_racr', 'avg_scored_touchdown', 'avg_redzone_target_share', 'avg_endzone_targets', 'avg_endzone_target_share',
    'avg_inside_5_target_share', 'pass_matchup_value', 'redzone_td_rate', 'passing_tds_allowed_to_WR', 'passing_tds_allowed_to_TE', 'implied_total', 'opponent_encoded']
QB_FEATURES = [
    'avg_carries', 'avg_rushing_yards', 'avg_rushing_epa', 'avg_scored_touchdown', 'avg_redzone_carry_share', 'avg_inside_5_carry_share',
    'rush_matchup_value', 'redzone_td_rate', 'rushing_tds_allowed_to_QB', 'implied_total', 'opponent_encoded']



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
def feature_engineering(df, redzone_df, redzone_td_rate, ez_target_df, odds_df, goal_line_df, positional_defense_df):
    """Engineers features from the raw data to improve model performance."""
    df = pd.merge(df, redzone_df, on=['player_id', 'week', 'season'], how='left')
    df = pd.merge(df, redzone_td_rate, on=['recent_team', 'season', 'week'], how='left')
    df = pd.merge(df, ez_target_df, on=['player_id', 'season', 'week'], how='left')
    df = pd.merge(df, odds_df, left_on=['recent_team', 'season', 'week'], right_on=['team', 'season', 'week'], how='left')
    df = pd.merge(df, goal_line_df, on=['player_id', 'week', 'season'], how='left')
    df = pd.merge(df, positional_defense_df, on=['opponent_team', 'season', 'week'], how='left')
    player_stats = ['carries', 'rushing_yards', 'receptions', 'receiving_yards', 'wopr', 'rushing_epa', 'receiving_epa', 'target_share',
                      'receiving_air_yards', 'air_yards_share', 'racr', 'scored_touchdown', 'redzone_carry_share', 'redzone_target_share',
                      'endzone_targets', 'endzone_target_share', 'inside_5_carry_share', 'inside_5_target_share']
    for stat in player_stats:
        df[f'avg_{stat}'] = df.groupby('player_display_name')[stat].transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    pos_defense_cols = [col for col in df.columns if 'tds_allowed_to' in col]
    for col in pos_defense_cols:
        df[col] = df.groupby('opponent_team')[col].transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    df['redzone_td_rate'] = df.groupby('recent_team')['redzone_td_rate'].transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
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
    return base_models, meta_model, best_rf_params, best_lgbm_params



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
        top_k_predictions = week_df.sort_values(by='predicted_prob', ascending=False).head(k)
        actual_scorers = set(week_df[week_df['scored_touchdown'] == 1]['player_display_name'])
        predicted_scorers = set(top_k_predictions['player_display_name'])
        hits = len(predicted_scorers.intersection(actual_scorers))
        precision_at_k = hits / k if k > 0 else 0
        recall_at_k = hits / len(actual_scorers) if actual_scorers else 0
        weekly_results.append({'week': week, 'precision_at_k': precision_at_k, 'recall_at_k': recall_at_k, 'successful_picks': hits})
    return pd.DataFrame(weekly_results)


def evaluate_specialist_model(base_models, meta_model, model_name, validation_df, features, k=25):
    """Calculates performance metrics for a manually stacked model."""
    print("\n" + "="*60 + f"\nEVALUATION FOR: {model_name}\n" + "="*60)
    if validation_df.empty:
        print("Validation data is empty. Skipping evaluation.")
        return


    X_val = validation_df[features]
    y_val = validation_df['scored_touchdown']


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


    base_estimators = [
        RandomForestClassifier(random_state=42, class_weight='balanced', **best_rf_params),
        lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1, **best_lgbm_params)
    ]
    meta_estimator = LogisticRegression()


    # Use the same robust training logic on the full dataset
    final_base_models, final_meta_model = train_stacked_model_timeseries(X_full, y_full, base_estimators, meta_estimator)


    print(f"{model_type} retraining complete.")
    return final_base_models, final_meta_model


# --- 6. Prediction Pipeline ---

def predict_touchdown_scorers(feature_df, rb_models, wr_te_models, qb_models, rb_features, wr_te_features, qb_features, opponent_le, year, week, future_odds_df):
    """Predicts touchdown scorers using manually stacked, position-specific models."""
    print("\nAssembling features for prediction with corrected logic...")


    # [Steps 1-3 from the original script for building prediction_df are included here]
    history_df = feature_df[(feature_df['season'] < year) | ((feature_df['season'] == year) & (feature_df['week'] < week))].copy()
    schedule = nfl.import_schedules([year])
    week_schedule = schedule[schedule['week'] == week]
    rosters = nfl.import_seasonal_rosters([year])
    opponent_map = {row['home_team']: row['away_team'] for _, row in week_schedule.iterrows()}
    opponent_map.update({row['away_team']: row['home_team'] for _, row in week_schedule.iterrows()})
    teams_playing = list(opponent_map.keys())
    week_rosters = rosters[rosters['team'].isin(teams_playing) & rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()
    prediction_df = week_rosters[['player_id', 'player_name', 'position', 'team']].copy()
    prediction_df.rename(columns={'player_name': 'player_display_name'}, inplace=True)
    prediction_df['opponent_team'] = prediction_df['team'].map(opponent_map)


    # [Step 4 from the original script for assembling features is included here]
    player_history_features = [f for f in rb_features + wr_te_features + qb_features if f.startswith('avg_') or f in ['target_share', 'avg_racr']]
    team_history_features = ['redzone_td_rate']
    opponent_history_features = [f for f in rb_features + wr_te_features + qb_features if 'allowed_to' in f]
    player_history_features = sorted(list(set(player_history_features)))
    team_history_features = sorted(list(set(team_history_features)))
    opponent_history_features = sorted(list(set(opponent_history_features)))
    features_from_player_history = player_history_features + team_history_features
    latest_player_data = history_df.groupby('player_id')[features_from_player_history].last().reset_index()
    prediction_df = pd.merge(prediction_df, latest_player_data, on='player_id', how='left')
    latest_opponent_data = history_df.groupby('recent_team')[opponent_history_features].last().reset_index()
    prediction_df = pd.merge(prediction_df, latest_opponent_data, left_on='opponent_team', right_on='recent_team', how='left')
    prediction_df = pd.merge(prediction_df, future_odds_df[['team', 'implied_total']], on='team', how='left')
    prediction_df['opponent_encoded'] = opponent_le.transform(prediction_df['opponent_team'])
    prediction_df['rush_matchup_value'] = np.select(
        [prediction_df['position'] == 'RB', prediction_df['position'] == 'QB'],
        [prediction_df['avg_redzone_carry_share'] * prediction_df['rushing_tds_allowed_to_RB'], prediction_df['avg_redzone_carry_share'] * prediction_df['rushing_tds_allowed_to_QB']],
        default=0)
    prediction_df['pass_matchup_value'] = np.select(
        [prediction_df['position'] == 'RB', prediction_df['position'] == 'WR', prediction_df['position'] == 'TE'],
        [prediction_df['avg_redzone_target_share'] * prediction_df['passing_tds_allowed_to_RB'], prediction_df['avg_redzone_target_share'] * prediction_df['passing_tds_allowed_to_WR'], prediction_df['avg_redzone_target_share'] * prediction_df['passing_tds_allowed_to_TE']],
        default=0)
    prediction_df.fillna(0, inplace=True)
    print("Feature assembly complete.")


    # --- Step 5: Split Prediction Data and Apply Correct Model ---
    pred_df_rb = prediction_df[prediction_df['position'] == 'RB'].copy()
    pred_df_wr_te = prediction_df[prediction_df['position'].isin(['WR', 'TE'])].copy()
    pred_df_qb = prediction_df[prediction_df['position'] == 'QB'].copy()


    # The models are now tuples: (list_of_base_models, meta_model)
    if not pred_df_rb.empty:
        pred_df_rb['predicted_touchdown_probability'] = predict_stacked_proba(pred_df_rb[rb_features], rb_models[0], rb_models[1])
    if not pred_df_wr_te.empty:
        pred_df_wr_te['predicted_touchdown_probability'] = predict_stacked_proba(pred_df_wr_te[wr_te_features], wr_te_models[0], wr_te_models[1])
    if not pred_df_qb.empty:
        pred_df_qb['predicted_touchdown_probability'] = predict_stacked_proba(pred_df_qb[qb_features], qb_models[0], qb_models[1])


    # --- Step 6: Combine results and find market edge ---
    # [Step 6 from original script for combining results is included here]
    final_predictions = pd.concat([pred_df_rb, pred_df_wr_te, pred_df_qb])
    td_odds = pd.read_csv('week_1_td_odds.csv')
    td_odds['merge_name'] = td_odds['description'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()
    final_predictions['merge_name'] = final_predictions['player_display_name'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()
    prob_if_pos = 100 / (td_odds['price'] + 100)
    prob_if_neg = abs(td_odds['price']) / (abs(td_odds['price']) + 100)
    td_odds['market_implied_prob'] = np.where(td_odds['price'] > 0, prob_if_pos, prob_if_neg)
    final_predictions = pd.merge(final_predictions, td_odds[['merge_name', 'price', 'market_implied_prob']], on='merge_name', how='left')
    final_predictions['model_edge'] = final_predictions['predicted_touchdown_probability'] - final_predictions['market_implied_prob']
    display_cols = ['player_display_name', 'team', 'position', 'predicted_touchdown_probability', 'price', 'market_implied_prob', 'model_edge']
    print('\n--- Top 25 Predicted Touchdown Scorers (Position-Specific Models) ---')
    final_predictions.dropna(subset=['price'], inplace=True)
    print(final_predictions[display_cols].sort_values(by='predicted_touchdown_probability', ascending=False).head(25))
    final_predictions[display_cols].to_csv('final_predictions_week_1.csv', index=False)
    
    return final_predictions


# --- 7. Main Execution Block ---
if __name__ == '__main__':
    # -- Configuration --
    all_years_to_load = range(2020, 2025)
    prediction_year = 2025
    prediction_week = 1


    # -- Data Loading --
    print("Loading data...")
    pbp = nfl.import_pbp_data(all_years_to_load, downcast=True)
    pbp = pbp[pbp['week'] <= 18]
    rosters = nfl.import_seasonal_rosters(all_years_to_load)
    nfl_teams = pd.read_csv('nfl_teams.csv')
    team_map = dict(zip(nfl_teams['team_name'], nfl_teams['team_id']))
    
    nfl_df = data.get_nfl_data(all_years_to_load)
    redzone_df = data.get_redzone_data(pbp)
    redzone_td_df = data.get_redzone_td_rate(pbp)
    ez_target_df = data.get_endzone_target_data(pbp)
    odds_df = data.get_odds_data(all_years_to_load, team_map)
    goal_line_df = data.get_goal_line_data(pbp)
    positional_defense_df = data.get_opponent_positional_data(pbp, rosters)
    
    # -- Feature Engineering --
    print("Engineering features...")
    feature_df = feature_engineering(nfl_df, redzone_df, redzone_td_df, ez_target_df, odds_df, goal_line_df, positional_defense_df)


    df_rb = feature_df[feature_df['position'] == 'RB'].copy()
    df_wr_te = feature_df[feature_df['position'].isin(['WR', 'TE'])].copy()
    df_qb = feature_df[feature_df['position'] == 'QB'].copy()


    # --- Phase 1: Tune, Train, and Evaluate on 2024 Season ---
    # The function now returns the trained base/meta models and the best params
    rb_models, rb_meta_model, rb_rf_params, rb_lgbm_params = tune_and_train_specialist_model(df_rb, RB_FEATURES, RF_PARAM_DIST, LGBM_PARAM_DIST)
    wr_te_models, wr_te_meta_model, wr_te_rf_params, wr_te_lgbm_params = tune_and_train_specialist_model(df_wr_te, WR_TE_FEATURES, RF_PARAM_DIST, LGBM_PARAM_DIST)
    qb_models, qb_meta_model, qb_rf_params, qb_lgbm_params = tune_and_train_specialist_model(df_qb, QB_FEATURES, RF_PARAM_DIST, LGBM_PARAM_DIST)


    # --- MODEL EVALUATION ON 2024 SEASON ---
    validation_year = 2024
    validation_df = feature_df[feature_df['season'] == validation_year]
    val_df_rb = validation_df[validation_df['position'] == 'RB'].copy()
    val_df_wr_te = validation_df[validation_df['position'].isin(['WR', 'TE'])].copy()
    val_df_qb = validation_df[validation_df['position'] == 'QB'].copy()


    evaluate_specialist_model(rb_models, rb_meta_model, "RB Model", val_df_rb, RB_FEATURES)
    evaluate_specialist_model(wr_te_models, wr_te_meta_model, "WR/TE Model", val_df_wr_te, WR_TE_FEATURES)
    evaluate_specialist_model(qb_models, qb_meta_model, "QB Model", val_df_qb, QB_FEATURES)


    # --- Phase 2: Retrain Final Models on All Data (2020-2024) ---
    print("\n" + "="*60 + "\nRETRAINING FINAL MODELS ON ALL HISTORICAL DATA FOR PREDICTION\n" + "="*60)
    rb_base_final, rb_meta_final = train_model_on_all_data(df_rb, RB_FEATURES, rb_rf_params, rb_lgbm_params)
    wr_te_base_final, wr_te_meta_final = train_model_on_all_data(df_wr_te, WR_TE_FEATURES, wr_te_rf_params, wr_te_lgbm_params)
    qb_base_final, qb_meta_final = train_model_on_all_data(df_qb, QB_FEATURES, qb_rf_params, qb_lgbm_params)


    # -- Prediction for Future Week --
    print("\n--- Generating Predictions for Week 1, 2025 ---")
    future_odds_raw = pd.read_csv('week_1_lines.csv')
    future_odds_df = data.transform_future_odds(future_odds_raw, team_map)
    opponent_le = LabelEncoder().fit(feature_df['opponent_team'].unique())


    predict_touchdown_scorers(feature_df,
                              (rb_base_final, rb_meta_final),
                              (wr_te_base_final, wr_te_meta_final),
                              (qb_base_final, qb_meta_final),
                              RB_FEATURES, WR_TE_FEATURES, QB_FEATURES,
                              opponent_le, prediction_year, prediction_week, future_odds_df)