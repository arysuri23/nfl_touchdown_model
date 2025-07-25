# NFL Touchdown Scorer Prediction Model
# Neural Network Version - v5 with Probability Calibration

# --- 1. Importing Libraries ---
import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
# You may need to install scikeras: pip install scikeras
from scikeras.wrappers import KerasClassifier
import kerastuner as kt
import data_collection as data
import os
import joblib
import random

# --- 2. Constants and Configuration ---

# --- For Reproducibility ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
    'avg_carries', 'avg_rushing_yards', 'avg_rushing_epa','avg_scored_touchdown', 'avg_redzone_carry_share', 'avg_inside_5_carry_share',
    'rush_matchup_value', 'redzone_td_rate', 'rushing_tds_allowed_to_QB', 'implied_total', 'opponent_encoded']

# -- Neural Network Hyperparameters --
NN_PARAMS = {
    'epochs': 100,
    'batch_size': 64,
    'patience': 10
}

# --- 3. Feature Engineering ---
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
    df['opponent_encoded'] = LabelEncoder().fit_transform(df['opponent_team'])
    return df


# --- 4. Neural Network Model Training, Tuning, and Calibration ---

def build_hypermodel(hp, input_shape):
    """Builds a Keras model with tunable hyperparameters."""
    model = Sequential()
    model.add(Input(shape=(input_shape,))) # Use input_shape directly
    
    model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    
    model.add(Dense(1, activation='sigmoid'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name="AUC")]) # FIX: Explicitly name the metric
    return model

def tune_and_train_nn_model(df_position, features, nn_params, validation_year=2024):
    """
    Tunes hyperparameters, trains the best model, and then calibrates it.
    """
    model_type = df_position['position'].unique()[0]
    if len(df_position['position'].unique()) > 1: model_type = "WR_TE"
            
    print("\n" + "="*60 + f"\nTUNING, TRAINING, & CALIBRATING: {model_type} Model\n" + "="*60)

    # --- 1. Data Preparation ---
    train_df = df_position[df_position['season'] < validation_year]
    X_train = train_df[features]
    y_train = train_df['scored_touchdown']

    if X_train.empty: return None, None, None, None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    class_weights = dict(zip(np.unique(y_train), compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)))

    tscv = TimeSeriesSplit(n_splits=5)
    train_indices, val_indices = list(tscv.split(X_train_scaled))[-1]
    X_train_fold, X_val_fold = X_train_scaled[train_indices], X_train_scaled[val_indices]
    y_train_fold, y_val_fold = y_train.iloc[train_indices], y_train.iloc[val_indices]

    # --- 2. Hyperparameter Tuning ---
    print(f"Starting hyperparameter tuning for {model_type}...")
    
    def build_model_for_tuner(hp):
        input_shape = X_train_scaled.shape[1]
        return build_hypermodel(hp, input_shape=input_shape)

    tuner = kt.RandomSearch(build_model_for_tuner, 
                            objective='val_AUC', # FIX: Match the metric name (val_ + name)
                            max_trials=10, 
                            executions_per_trial=1, 
                            directory='keras_tuner', 
                            project_name=f'nfl_td_{model_type}', 
                            seed=SEED)
                            
    early_stopping = EarlyStopping(monitor='val_loss', patience=nn_params['patience'], verbose=0)
    tuner.search(X_train_fold, y_train_fold, epochs=nn_params['epochs'], validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], verbose=0)
    best_hps = tuner.get_best_hyperparameters(num_models=1)[0]
    print("Best hyperparameters found.")

    # --- 3. Train the Best Model ---
    print(f"Training final {model_type} model with best hyperparameters...")
    best_model = build_model_for_tuner(best_hps)
    history = best_model.fit(X_train_fold, y_train_fold, epochs=nn_params['epochs'], batch_size=nn_params['batch_size'], validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], class_weight=class_weights, verbose=0)
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"Best model trained for {best_epoch} epochs.")

    # --- 4. Calibrate Probabilities ---
    print(f"Calibrating {model_type} model probabilities...")
    keras_clf = KerasClassifier(model=best_model, verbose=0)
    calibrated_clf = CalibratedClassifierCV(keras_clf, method='isotonic', cv='prefit')
    calibrated_clf.fit(X_val_fold, y_val_fold)
    print("Calibration complete.")

    return best_model, scaler, calibrated_clf, best_hps, best_epoch

def train_nn_on_all_data(df_position, features, best_hps, best_epoch, model_name):
    """Retrains a final model and calibrator on all data."""
    model_type = df_position['position'].unique()[0]
    if len(df_position['position'].unique()) > 1: model_type = "WR_TE"
    print(f"\nRetraining final {model_type} model and calibrator on all data...")

    X_full = df_position[features]
    y_full = df_position['scored_touchdown']

    if X_full.empty: return None, None

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    class_weights = dict(zip(np.unique(y_full), compute_class_weight(class_weight='balanced', classes=np.unique(y_full), y=y_full)))

    input_shape = X_full_scaled.shape[1]
    final_model = build_hypermodel(best_hps, input_shape=input_shape)
    final_model.fit(X_full_scaled, y_full, epochs=best_epoch, batch_size=NN_PARAMS['batch_size'], class_weight=class_weights, verbose=0)

    keras_clf_full = KerasClassifier(model=final_model, verbose=0)
    final_calibrator = CalibratedClassifierCV(keras_clf_full, method='isotonic', cv='prefit')
    final_calibrator.fit(X_full_scaled, y_full)

    final_model.save(f'{model_name}_final_model.keras')
    joblib.dump(scaler, f'{model_name}_final_scaler.gz')
    joblib.dump(final_calibrator, f'{model_name}_final_calibrator.gz')
    print(f"{model_type} retraining complete. Model, scaler, and calibrator saved.")
    
    return final_model, scaler, final_calibrator

# --- 5. Model Evaluation ---
def evaluate_model_at_k(predictions_df: pd.DataFrame, k: int):
    """Calculates Precision@k and Recall@k for weekly NFL touchdown predictions."""
    weekly_results = []
    for week in sorted(predictions_df['week'].unique()):
        week_df = predictions_df[predictions_df['week'] == week]
        top_k_predictions = week_df.sort_values(by='predicted_prob', ascending=False).head(k)
        actual_scorers = set(week_df[week_df['scored_touchdown'] == 1]['player_display_name'])
        predicted_scorers = set(top_k_predictions['player_display_name'])
        hits = len(predicted_scorers.intersection(actual_scorers))
        precision_at_k = hits / k if k > 0 else 0
        recall_at_k = hits / len(actual_scorers) if len(actual_scorers) > 0 else 0
        weekly_results.append({'week': week, 'precision_at_k': precision_at_k, 'recall_at_k': recall_at_k, 'successful_picks': hits})
    return pd.DataFrame(weekly_results)

def evaluate_nn_model(calibrated_clf, scaler, model_name, validation_df, features, k=25):
    """Calculates performance metrics for a trained and calibrated model."""
    print("\n" + "="*60 + f"\nEVALUATION FOR: {model_name} (k={k})\n" + "="*60)
    if validation_df.empty: return

    X_val = validation_df[features]
    X_val_scaled = scaler.transform(X_val)

    print(f"\n--- Weekly Performance @ K={k} ---")
    y_pred_proba = calibrated_clf.predict_proba(X_val_scaled)[:, 1]
    
    results_df = validation_df[['player_display_name', 'week', 'scored_touchdown']].copy()
    results_df['predicted_prob'] = y_pred_proba
    
    weekly_performance = evaluate_model_at_k(results_df, k=k)
    print(weekly_performance)
    
    average_performance = weekly_performance.mean()
    print("\n--- Average Season Performance ---")
    print(f"Average Precision@{k}: {average_performance['precision_at_k']:.3f}")
    print(f"Average Recall@{k}:    {average_performance['recall_at_k']:.3f}")
    print(f"Average Successful Picks Per Week: {average_performance['successful_picks']:.1f}")

# --- 6. Prediction Pipeline ---
def predict_touchdown_scorers(feature_df, model_names, rb_features, wr_te_features, qb_features, opponent_le, year, week, future_odds_df):
    """Predicts touchdown scorers using saved calibrated models."""
    print("\nAssembling features for prediction...")

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
    
    known_opponents = opponent_le.classes_
    prediction_df['opponent_team'] = prediction_df['opponent_team'].apply(lambda x: x if x in known_opponents else 'UNKNOWN')
    if 'UNKNOWN' not in opponent_le.classes_:
        opponent_le.classes_ = np.append(opponent_le.classes_, 'UNKNOWN')
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

    pred_df_rb = prediction_df[prediction_df['position'] == 'RB'].copy()
    pred_df_wr_te = prediction_df[prediction_df['position'].isin(['WR', 'TE'])].copy()
    pred_df_qb = prediction_df[prediction_df['position'] == 'QB'].copy()

    # Load and predict for each position
    for pos, df, features in [('RB', pred_df_rb, rb_features), ('WR_TE', pred_df_wr_te, wr_te_features), ('QB', pred_df_qb, qb_features)]:
        if not df.empty:
            model_name = model_names[pos]
            scaler = joblib.load(f'{model_name}_final_scaler.gz')
            calibrator = joblib.load(f'{model_name}_final_calibrator.gz')
            X_scaled = scaler.transform(df[features])
            df['predicted_touchdown_probability'] = calibrator.predict_proba(X_scaled)[:, 1]

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
    print('\n--- Top 25 Predicted Touchdown Scorers (Calibrated NN Models) ---')
    final_predictions.dropna(subset=['price'], inplace=True)
    print(final_predictions[display_cols].sort_values(by='predicted_touchdown_probability', ascending=False).head(25))
    final_predictions[display_cols].to_csv('final_predictions_week_1_nn_calibrated.csv', index=False)
    
    return final_predictions

# --- 7. Main Execution Block ---
if __name__ == '__main__':
    all_years_to_load = range(2020, 2025)
    validation_year = 2024
    prediction_year = 2025
    prediction_week = 1
    
    MODEL_NAMES = {"RB": "rb_model", "WR_TE": "wr_te_model", "QB": "qb_model"}

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
    
    print("Engineering features...")
    feature_df = feature_engineering(nfl_df, redzone_df, redzone_td_df, ez_target_df, odds_df, goal_line_df, positional_defense_df)

    df_rb = feature_df[feature_df['position'] == 'RB'].copy()
    df_wr_te = feature_df[feature_df['position'].isin(['WR', 'TE'])].copy()
    df_qb = feature_df[feature_df['position'] == 'QB'].copy()

    # --- Phase 1: Tune, Train, Calibrate, and Evaluate ---
    rb_model, rb_scaler, rb_calibrator, rb_hps, rb_epochs = tune_and_train_nn_model(df_rb, RB_FEATURES, NN_PARAMS, validation_year)
    wr_te_model, wr_te_scaler, wr_te_calibrator, wr_te_hps, wr_te_epochs = tune_and_train_nn_model(df_wr_te, WR_TE_FEATURES, NN_PARAMS, validation_year)
    qb_model, qb_scaler, qb_calibrator, qb_hps, qb_epochs = tune_and_train_nn_model(df_qb, QB_FEATURES, NN_PARAMS, validation_year)

    validation_df = feature_df[feature_df['season'] == validation_year]
    val_df_rb = validation_df[validation_df['position'] == 'RB'].copy()
    val_df_wr_te = validation_df[validation_df['position'].isin(['WR', 'TE'])].copy()
    val_df_qb = validation_df[validation_df['position'] == 'QB'].copy()

    if rb_calibrator: evaluate_nn_model(rb_calibrator, rb_scaler, "RB Model", val_df_rb, RB_FEATURES)
    if wr_te_calibrator: evaluate_nn_model(wr_te_calibrator, wr_te_scaler, "WR/TE Model", val_df_wr_te, WR_TE_FEATURES)
    if qb_calibrator: evaluate_nn_model(qb_calibrator, qb_scaler, "QB Model", val_df_qb, QB_FEATURES, k=10)

    # --- Phase 2: Retrain Final Models on All Data ---
    print("\n" + "="*60 + "\nRETRAINING FINAL MODELS ON ALL HISTORICAL DATA\n" + "="*60)
    if rb_hps: train_nn_on_all_data(df_rb, RB_FEATURES, rb_hps, rb_epochs, MODEL_NAMES["RB"])
    if wr_te_hps: train_nn_on_all_data(df_wr_te, WR_TE_FEATURES, wr_te_hps, wr_te_epochs, MODEL_NAMES["WR_TE"])
    if qb_hps: train_nn_on_all_data(df_qb, QB_FEATURES, qb_hps, qb_epochs, MODEL_NAMES["QB"])

    # -- Prediction for Future Week --
    print(f"\n--- Generating Predictions for Week {prediction_year}, Week {prediction_week} ---")
    future_odds_raw = pd.read_csv('week_1_lines.csv')
    future_odds_df = data.transform_future_odds(future_odds_raw, team_map)
    
    opponent_le = LabelEncoder().fit(feature_df['opponent_team'].unique())

    predict_touchdown_scorers(feature_df,
                              MODEL_NAMES,
                              RB_FEATURES, WR_TE_FEATURES, QB_FEATURES,
                              opponent_le, prediction_year, prediction_week, future_odds_df)
