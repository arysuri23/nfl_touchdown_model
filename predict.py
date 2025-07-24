# NFL Touchdown Scorer Prediction Model
# Final Version with Position-Specific Models

# --- 1. Importing Libraries ---
import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import lightgbm as lgb

# --- 2. Data Acquisition and Preprocessing ---

def get_nfl_data(years):
    """Fetches and preprocesses NFL weekly data for a given list of years."""
    df = nfl.import_weekly_data(years, downcast=True)
    df = df[df['week'] <= 18]

    # Select all relevant columns, including advanced receiving metrics
    df = df[['player_id', 'player_display_name', 'position', 'recent_team', 'season', 'week',
             'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
             'receiving_yards', 'receiving_tds', 'opponent_team', 'wopr', 'rushing_epa',
             'receiving_epa', 'target_share', 'receiving_air_yards', 'air_yards_share', 'racr']]
             
    df = df[df['position'].isin(['QB', 'RB', 'TE', 'WR'])]
    df['scored_touchdown'] = ((df['rushing_tds'] > 0) | (df['receiving_tds'] > 0)).astype(int)
    df.fillna(0, inplace=True)
    return df

def get_odds_data(years, team_map):
    """Loads and processes historical betting odds data."""
    df_odds = pd.read_csv('spreadspoke_scores.csv', low_memory=False)
    df_odds = df_odds[['schedule_season', 'schedule_week', 'team_home', 'team_away',
                       'team_favorite_id', 'spread_favorite', 'over_under_line', 'schedule_playoff']]
    df_odds.rename(columns={'schedule_season': 'season', 'schedule_week': 'week', 'over_under_line': 'total_line'}, inplace=True)
    df_odds = df_odds[df_odds['season'].isin(years) & (df_odds['schedule_playoff'] == False)]
    
    for col in ['total_line', 'spread_favorite', 'season', 'week']:
        df_odds[col] = pd.to_numeric(df_odds[col], errors='coerce')
    df_odds.dropna(subset=['week', 'total_line', 'spread_favorite'], inplace=True)
    df_odds['week'] = df_odds['week'].astype(int)

    df_odds['home_team_abbr'] = df_odds['team_home'].map(team_map)
    df_odds['home_spread'] = np.where(df_odds['team_favorite_id'] == df_odds['home_team_abbr'], df_odds['spread_favorite'], -df_odds['spread_favorite'])

    df_home = df_odds[['season', 'week', 'home_team_abbr', 'home_spread', 'total_line']].rename(columns={'home_team_abbr': 'team', 'home_spread': 'spread_line'})
    df_away = df_odds[['season', 'week', 'team_away', 'home_spread', 'total_line']].rename(columns={'team_away': 'team_full_name'})
    df_away['team'] = df_away['team_full_name'].map(team_map)
    df_away['spread_line'] = -df_away['home_spread']
    df_away.drop(columns=['home_spread', 'team_full_name'], inplace=True)

    df_processed_odds = pd.concat([df_home, df_away]).dropna(subset=['team'])
    df_processed_odds['implied_total'] = (df_processed_odds['total_line'] / 2) - (df_processed_odds['spread_line'] / 2)
    return df_processed_odds

def get_redzone_data(pbp):
    """Calculates each player's share of their team's red zone carries and targets."""
    redzone_df = pbp[pbp['yardline_100'] <= 20].copy()
    team_rz_plays = redzone_df.groupby(['posteam', 'season', 'week']).agg(team_rz_rushes=('rush_attempt', 'sum'), team_rz_targets=('pass_attempt', 'sum')).reset_index()
    player_rz_rushes = redzone_df.groupby(['rusher_player_id', 'posteam', 'season', 'week']).agg(player_rz_rushes=('rush_attempt', 'sum')).reset_index().rename(columns={'rusher_player_id': 'player_id'})
    player_rz_targets = redzone_df.groupby(['receiver_player_id', 'posteam', 'season', 'week']).agg(player_rz_targets=('pass_attempt', 'sum')).reset_index().rename(columns={'receiver_player_id': 'player_id'})
    player_usage = pd.merge(player_rz_rushes, player_rz_targets, on=['player_id', 'posteam', 'season', 'week'], how='outer')
    final_rz_df = pd.merge(player_usage, team_rz_plays, on=['posteam', 'season', 'week'], how='left')
    final_rz_df['redzone_carry_share'] = (final_rz_df['player_rz_rushes'] / final_rz_df['team_rz_rushes']).fillna(0)
    final_rz_df['redzone_target_share'] = (final_rz_df['player_rz_targets'] / final_rz_df['team_rz_targets']).fillna(0)
    return final_rz_df[['player_id', 'season', 'week', 'redzone_carry_share', 'redzone_target_share']]

def get_goal_line_data(pbp):
    """Calculates shares of carries and targets inside the 5-yard line."""
    goal_line_df = pbp[pbp['yardline_100'] <= 5].copy()
    team_gl_plays = goal_line_df.groupby(['posteam', 'season', 'week']).agg(team_gl_rushes=('rush_attempt', 'sum'), team_gl_targets=('pass_attempt', 'sum')).reset_index()
    player_gl_rushes = goal_line_df.groupby(['rusher_player_id', 'posteam', 'season', 'week']).agg(player_gl_rushes=('rush_attempt', 'sum')).reset_index().rename(columns={'rusher_player_id': 'player_id'})
    player_gl_targets = goal_line_df.groupby(['receiver_player_id', 'posteam', 'season', 'week']).agg(player_gl_targets=('pass_attempt', 'sum')).reset_index().rename(columns={'receiver_player_id': 'player_id'})
    player_usage = pd.merge(player_gl_rushes, player_gl_targets, on=['player_id', 'posteam', 'season', 'week'], how='outer')
    final_gl_df = pd.merge(player_usage, team_gl_plays, on=['posteam', 'season', 'week'], how='left')
    final_gl_df['inside_5_carry_share'] = (final_gl_df['player_gl_rushes'] / final_gl_df['team_gl_rushes']).fillna(0)
    final_gl_df['inside_5_target_share'] = (final_gl_df['player_gl_targets'] / final_gl_df['team_gl_targets']).fillna(0)
    return final_gl_df[['player_id', 'season', 'week', 'inside_5_carry_share', 'inside_5_target_share']]

def get_endzone_target_data(pbp):
    """Calculates each player's end zone targets and share of team targets."""
    pass_plays = pbp[(pbp['pass_attempt'] == 1) & pbp['air_yards'].notna()].copy()
    pass_plays['is_endzone_target'] = pass_plays['air_yards'] >= pass_plays['yardline_100']
    team_targets = pass_plays.groupby(['posteam', 'season', 'week']).agg(team_total_targets=('pass_attempt', 'sum')).reset_index()
    ez_target_df = pass_plays[pass_plays['is_endzone_target'] == True]
    player_ez_targets = ez_target_df.groupby(['receiver_player_id', 'posteam', 'season', 'week']).agg(endzone_targets=('is_endzone_target', 'sum')).reset_index().rename(columns={'receiver_player_id': 'player_id'})
    final_ez_df = pd.merge(player_ez_targets, team_targets, on=['posteam', 'season', 'week'], how='left')
    final_ez_df['endzone_target_share'] = (final_ez_df['endzone_targets'] / final_ez_df['team_total_targets']).fillna(0)
    return final_ez_df[['player_id', 'season', 'week', 'endzone_targets', 'endzone_target_share']]

def get_redzone_td_rate(pbp):
    """Calculates team-level red zone touchdown conversion rate."""
    redzone = pbp[pbp['yardline_100'] <= 20]
    redzone_trips = redzone.groupby(['posteam', 'season', 'week', 'drive']).size().reset_index(name='plays')
    redzone_tds = redzone[redzone['touchdown'] == 1].groupby(['posteam', 'season', 'week', 'drive']).size().reset_index(name='tds')
    drive_summary = pd.merge(redzone_trips[['posteam', 'season', 'week', 'drive']], redzone_tds, on=['posteam', 'season', 'week', 'drive'], how='left').fillna(0)
    redzone_summary = drive_summary.groupby(['posteam', 'season', 'week']).agg(redzone_trips=('drive', 'nunique'), redzone_tds=('tds', lambda x: (x > 0).sum())).reset_index()
    redzone_summary['redzone_td_rate'] = redzone_summary['redzone_tds'] / redzone_summary['redzone_trips']
    redzone_summary.rename(columns={'posteam': 'recent_team'}, inplace=True)
    return redzone_summary

def get_opponent_positional_data(pbp, rosters):
    """Calculates how many TDs each defense allows to specific offensive positions."""
    roster_positions = rosters[['player_id', 'position']].drop_duplicates()
    rush_tds = pbp[pbp['rush_touchdown'] == 1][['defteam', 'season', 'week', 'rusher_player_id']]
    rush_tds = pd.merge(rush_tds, roster_positions, left_on='rusher_player_id', right_on='player_id', how='left')
    rush_tds_allowed = rush_tds.groupby(['defteam', 'season', 'week', 'position']).size().unstack(fill_value=0).add_prefix('rushing_tds_allowed_to_')
    pass_tds = pbp[pbp['pass_touchdown'] == 1][['defteam', 'season', 'week', 'receiver_player_id']]
    pass_tds = pd.merge(pass_tds, roster_positions, left_on='receiver_player_id', right_on='player_id', how='left')
    pass_tds_allowed = pass_tds.groupby(['defteam', 'season', 'week', 'position']).size().unstack(fill_value=0).add_prefix('passing_tds_allowed_to_')
    positional_defense_df = pd.merge(rush_tds_allowed, pass_tds_allowed, on=['defteam', 'season', 'week'], how='outer').fillna(0).reset_index()
    for pos in ['RB', 'WR', 'TE', 'QB']:
        if f'rushing_tds_allowed_to_{pos}' not in positional_defense_df.columns: positional_defense_df[f'rushing_tds_allowed_to_{pos}'] = 0
        if f'passing_tds_allowed_to_{pos}' not in positional_defense_df.columns: positional_defense_df[f'passing_tds_allowed_to_{pos}'] = 0
    positional_defense_df.rename(columns={'defteam': 'opponent_team'}, inplace=True)
    return positional_defense_df

# --- 3. Feature Engineering ---

def feature_engineering(df, redzone_df, redzone_td_rate, ez_target_df, odds_df, goal_line_df, positional_defense_df):
    """Engineers features from the raw data to improve model performance."""
    # Merge all data sources
    df = pd.merge(df, redzone_df, on=['player_id', 'week', 'season'], how='left')
    df = pd.merge(df, redzone_td_rate, on=['recent_team', 'season', 'week'], how='left')
    df = pd.merge(df, ez_target_df, on=['player_id', 'season', 'week'], how='left')
    df = pd.merge(df, odds_df, left_on=['recent_team', 'season', 'week'], right_on=['team', 'season', 'week'], how='left')
    df = pd.merge(df, goal_line_df, on=['player_id', 'week', 'season'], how='left')
    df = pd.merge(df, positional_defense_df, on=['opponent_team', 'season', 'week'], how='left')

    # Rolling averages for player stats
    player_stats = ['carries', 'rushing_yards', 'receptions', 'receiving_yards', 'wopr', 'rushing_epa', 'receiving_epa', 'target_share',
                    'receiving_air_yards', 'air_yards_share', 'racr', 'scored_touchdown', 'redzone_carry_share', 'redzone_target_share',
                    'endzone_targets', 'endzone_target_share', 'inside_5_carry_share', 'inside_5_target_share']
    for stat in player_stats:
        df[f'avg_{stat}'] = df.groupby('player_display_name')[stat].transform(lambda x: x.shift(1).rolling(5, 1).mean())

    # Rolling averages for opponent defensive stats
    pos_defense_cols = [col for col in df.columns if 'tds_allowed_to' in col]
    for col in pos_defense_cols:
        df[col] = df.groupby('opponent_team')[col].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['redzone_td_rate'] = df.groupby('recent_team')['redzone_td_rate'].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    
    # Decomposed Interaction Features
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

def tune_and_train_specialist_model(df_position, features, rf_param_dist, lgbm_param_dist, validation_year=2024):
    """
    Tunes hyperparameters on a validation set and then trains a final specialist model on all data.
    """
    model_type = df_position['position'].unique()[0]
    if len(df_position['position'].unique()) > 1:
        model_type = "WR/TE"
        
    print("\n" + "="*60)
    print(f"TUNING AND TRAINING FOR: {model_type} Model")
    print("="*60)

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

    # --- 4. Train Final Stacking Model on ALL Data ---
    print(f"\nTraining final {model_type} Stacking Model on all available data...")
    final_estimators = [
        ('rf', RandomForestClassifier(random_state=42, class_weight='balanced', **best_rf_params)),
        ('lgbm', lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1, **best_lgbm_params))
    ]
    
    final_stacking_model = StackingClassifier(estimators=final_estimators, final_estimator=LogisticRegression(), cv=5)
    
    X_full = df_position[features]
    y_full = df_position['scored_touchdown']
    
    final_stacking_model.fit(X_full, y_full)
    print("Final model training complete.")
    
    return final_stacking_model


# --- 5. Model Evaluation ---

def evaluate_model_at_k(predictions_df: pd.DataFrame, k: int = 25):
    """
    Calculates Precision@k and Recall@k for weekly NFL touchdown predictions.
    This is the NEW evaluation function.
    """
    weekly_results = []
    for week in sorted(predictions_df['week'].unique()):
        week_df = predictions_df[predictions_df['week'] == week]
        top_k_predictions = week_df.sort_values(by='predicted_prob', ascending=False).head(k)
        
        actual_scorers = set(week_df[week_df['scored_touchdown'] == 1]['player_display_name'])
        predicted_scorers = set(top_k_predictions['player_display_name'])
        
        hits = len(predicted_scorers.intersection(actual_scorers))
        
        precision_at_k = hits / k if k > 0 else 0
        recall_at_k = hits / len(actual_scorers) if actual_scorers else 0
        
        weekly_results.append({
            'week': week,
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'successful_picks': hits
        })
    return pd.DataFrame(weekly_results)

def evaluate_specialist_model(model, model_name, validation_df, features, k=25):
    """
    Calculates and prints performance metrics and feature importance for a single specialist model.
    """
    print("\n" + "="*60)
    print(f"EVALUATION FOR: {model_name}")
    print("="*60)

    if validation_df.empty:
        print("Validation data is empty. Skipping evaluation.")
        return

    X_val = validation_df[features]
    y_val = validation_df['scored_touchdown']

    # --- 1. Performance Metrics (Precision@k) ---
    print(f"\n--- Weekly Performance @ K={k} ---")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    results_df = validation_df[['player_display_name', 'week', 'scored_touchdown']].copy()
    results_df['predicted_prob'] = y_pred_proba
    
    # We can reuse your existing evaluate_model_at_k function for this part!
    weekly_performance = evaluate_model_at_k(results_df, k=k)
    print(weekly_performance)
    
    average_performance = weekly_performance.mean()
    print("\n--- Average Season Performance ---")
    print(f"Average Precision@{k}: {average_performance['precision_at_k']:.3f}")
    print(f"Average Recall@{k}:    {average_performance['recall_at_k']:.3f}")
    print(f"Average Successful Picks Per Week: {average_performance['successful_picks']:.1f}")

    # --- 2. Base Model Importance (Stacking Coefficients) ---
    print("\n--- Base Model Importance (Final Estimator Weights) ---")
    final_estimator_coefs = model.final_estimator_.coef_[0]
    base_model_names = [name for name, _ in model.estimators]
    model_importance_df = pd.DataFrame({
        'Base Model': base_model_names,
        'Coefficient (Weight)': final_estimator_coefs
    }).sort_values(by='Coefficient (Weight)', ascending=False)
    print(model_importance_df)

    # --- 3. Overall Feature Importance (Permutation Importance) ---
    print("\n--- Overall Feature Importance (Permutation) ---")
    print("Calculating... (This may take a moment)")
    result = permutation_importance(
        model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_importance_df = pd.DataFrame({
        'feature': features,
        'importance_mean': result.importances_mean,
    }).sort_values('importance_mean', ascending=False)
    print(perm_importance_df.head(30))


# --- 6. Prediction Pipeline ---

def transform_future_odds(df_future_raw, team_map):
    """Transforms a raw future odds CSV into the format needed for prediction."""
    df_future_raw = df_future_raw[['home_team', 'away_team', 'point_1', 'over/under']].copy()
    df_future_raw['home_team'] = df_future_raw['home_team'].map(team_map)
    df_future_raw['away_team'] = df_future_raw['away_team'].map(team_map)
    games_df = df_future_raw.groupby(['home_team', 'away_team']).agg(spread_line=('point_1', 'first'), total_line=('over/under', 'first')).reset_index()
    home_teams = games_df.rename(columns={'home_team': 'team', 'away_team': 'opponent'})
    away_teams = games_df.rename(columns={'away_team': 'team', 'home_team': 'opponent'})
    away_teams['spread_line'] = -away_teams['spread_line']
    df_final = pd.concat([home_teams, away_teams]).reset_index(drop=True)
    df_final['implied_total'] = (df_final['total_line'] / 2) - (df_final['spread_line'] / 2)
    return df_final

def predict_touchdown_scorers(feature_df, rb_model, wr_te_model, qb_model, rb_features, wr_te_features, qb_features, opponent_le, year, week, future_odds_df):
    """
    Predicts touchdown scorers using position-specific models.
    (CORRECTED VERSION)
    """
    # --- Step 1: Isolate historical data ---
    history_df = feature_df[(feature_df['season'] < year) | ((feature_df['season'] == year) & (feature_df['week'] < week))].copy()
    latest_player_stats = history_df.groupby('player_id').last()

    # --- Step 2: Get schedule and roster for the prediction week ---
    schedule = nfl.import_schedules([year])
    week_schedule = schedule[schedule['week'] == week]
    rosters = nfl.import_seasonal_rosters([year])
    opponent_map = {row['home_team']: row['away_team'] for _, row in week_schedule.iterrows()}
    opponent_map.update({row['away_team']: row['home_team'] for _, row in week_schedule.iterrows()})
    teams_playing = list(opponent_map.keys())
    week_rosters = rosters[rosters['team'].isin(teams_playing) & rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()

    # --- Step 3: Build the prediction DataFrame ---
    prediction_df = week_rosters[['player_id', 'player_name', 'position', 'team']].copy()
    prediction_df['opponent_team'] = prediction_df['team'].map(opponent_map)

    # --- Step 4: Merge historical and game-specific features correctly ---

    # CORRECTED LOGIC:
    # 1. Define features that are truly historical and player-specific.
    player_history_features = set(rb_features + wr_te_features + qb_features)
    # 2. REMOVE game-specific features that should NOT come from a player's history.
    game_specific_features = {'implied_total', 'opponent_encoded'}
    player_history_features = list(player_history_features - game_specific_features)
    
    # 3. Merge ONLY the historical player stats.
    prediction_df = pd.merge(prediction_df, latest_player_stats[player_history_features], on='player_id', how='left')

    # 4. NOW, add the fresh, game-specific features for the prediction week.
    prediction_df = pd.merge(prediction_df, future_odds_df[['team', 'implied_total']], on='team', how='left')
    prediction_df['opponent_encoded'] = opponent_le.transform(prediction_df['opponent_team'])

    # Fill NaNs for rookies or players with no history AFTER all merges are done.
    prediction_df.fillna(0, inplace=True)

    # --- Step 5: Split Prediction Data and Apply Correct Model ---
    pred_df_rb = prediction_df[prediction_df['position'] == 'RB'].copy()
    pred_df_wr_te = prediction_df[prediction_df['position'].isin(['WR', 'TE'])].copy()
    pred_df_qb = prediction_df[prediction_df['position'] == 'QB'].copy()

    if not pred_df_rb.empty: pred_df_rb['predicted_touchdown_probability'] = rb_model.predict_proba(pred_df_rb[rb_features])[:, 1]
    if not pred_df_wr_te.empty: pred_df_wr_te['predicted_touchdown_probability'] = wr_te_model.predict_proba(pred_df_wr_te[wr_te_features])[:, 1]
    if not pred_df_qb.empty: pred_df_qb['predicted_touchdown_probability'] = qb_model.predict_proba(pred_df_qb[qb_features])[:, 1]

    # --- Step 6: Combine results and find market edge ---
    final_predictions = pd.concat([pred_df_rb, pred_df_wr_te, pred_df_qb])
    
    td_odds = pd.read_csv('week_1_td_odds.csv')
    td_odds['merge_name'] = td_odds['description'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()
    final_predictions['merge_name'] = final_predictions['player_name'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()
    
    prob_if_pos = 100 / (td_odds['price'] + 100)
    prob_if_neg = abs(td_odds['price']) / (abs(td_odds['price']) + 100)
    td_odds['market_implied_prob'] = np.where(td_odds['price'] > 0, prob_if_pos, prob_if_neg)
    
    final_predictions = pd.merge(final_predictions, td_odds[['merge_name', 'price', 'market_implied_prob']], on='merge_name', how='left')
    final_predictions['model_edge'] = final_predictions['predicted_touchdown_probability'] - final_predictions['market_implied_prob']
    
    display_cols = ['player_name', 'team', 'position', 'predicted_touchdown_probability', 'price', 'market_implied_prob', 'model_edge']
    print('\n--- Top 25 Predicted Touchdown Scorers (Position-Specific Models) ---')
    # Filter out players with no market odds for a cleaner final list
    final_predictions.dropna(subset=['price'], inplace=True)
    print(final_predictions[display_cols].sort_values(by='predicted_touchdown_probability', ascending=False).head(25))
    final_predictions[display_cols].to_csv('final_predictions_week_1.csv', index=False)
    
    return final_predictions

# --- 7. Main Execution Block ---
if __name__ == '__main__':
    # -- Configuration --
    all_years_to_load = range(2021, 2025)
    prediction_year = 2025
    prediction_week = 1

    # -- Data Loading --
    print("Loading data...")
    pbp = nfl.import_pbp_data(all_years_to_load, downcast=True)
    pbp = pbp[pbp['week'] <= 18]
    rosters = nfl.import_seasonal_rosters(all_years_to_load)
    nfl_teams = pd.read_csv('nfl_teams.csv')
    team_map = dict(zip(nfl_teams['team_name'], nfl_teams['team_id']))
    
    nfl_df = get_nfl_data(all_years_to_load)
    redzone_df = get_redzone_data(pbp)
    redzone_td_df = get_redzone_td_rate(pbp)
    ez_target_df = get_endzone_target_data(pbp)
    odds_df = get_odds_data(all_years_to_load, team_map)
    goal_line_df = get_goal_line_data(pbp)
    positional_defense_df = get_opponent_positional_data(pbp, rosters)
    
    # -- Feature Engineering --
    print("Engineering features...")
    feature_df = feature_engineering(nfl_df, redzone_df, redzone_td_df, ez_target_df, odds_df, goal_line_df, positional_defense_df)

    # -- Position-Specific Feature Lists --
    rb_features = [
        'avg_carries', 'avg_rushing_yards', 'avg_receptions', 'avg_receiving_yards', 'avg_wopr', 'avg_rushing_epa', 'avg_receiving_epa', 
        'target_share', 'avg_receiving_air_yards', 'avg_air_yards_share', 'avg_racr', 'avg_scored_touchdown', 'avg_redzone_carry_share', 
        'avg_redzone_target_share', 'avg_endzone_targets', 'avg_endzone_target_share', 'avg_inside_5_carry_share', 'avg_inside_5_target_share',
        'rush_matchup_value', 'pass_matchup_value', 'redzone_td_rate', 'rushing_tds_allowed_to_RB', 'passing_tds_allowed_to_RB', 'implied_total', 'opponent_encoded']
    wr_te_features = [
        'avg_receptions', 'avg_receiving_yards', 'avg_wopr', 'avg_receiving_epa', 'target_share', 'avg_receiving_air_yards', 
        'avg_air_yards_share', 'avg_racr', 'avg_scored_touchdown', 'avg_redzone_target_share', 'avg_endzone_targets', 'avg_endzone_target_share', 
        'avg_inside_5_target_share', 'pass_matchup_value', 'redzone_td_rate', 'passing_tds_allowed_to_WR', 'passing_tds_allowed_to_TE', 'implied_total', 'opponent_encoded']
    qb_features = [
        'avg_carries', 'avg_rushing_yards', 'avg_rushing_epa', 'avg_scored_touchdown', 'avg_redzone_carry_share', 'avg_inside_5_carry_share', 
        'rush_matchup_value', 'redzone_td_rate', 'rushing_tds_allowed_to_QB', 'implied_total', 'opponent_encoded']

    # -- Model Training --
    
    # -- Hyperparameter Distributions --
    rf_param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    lgbm_param_dist = {
        'n_estimators': [100, 200, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40, 50],
        'max_depth': [-1, 10, 20],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }


    df_rb = feature_df[feature_df['position'] == 'RB'].copy()
    df_wr_te = feature_df[feature_df['position'].isin(['WR', 'TE'])].copy()
    df_qb = feature_df[feature_df['position'] == 'QB'].copy()

    rb_model = tune_and_train_specialist_model(df_rb, rb_features, rf_param_dist, lgbm_param_dist)
    wr_te_model = tune_and_train_specialist_model(df_wr_te, wr_te_features, rf_param_dist, lgbm_param_dist)
    qb_model = tune_and_train_specialist_model(df_qb, qb_features, rf_param_dist, lgbm_param_dist)



    # --- MODEL EVALUATION ON 2024 SEASON ---
    validation_year = 2024
    
    # Create validation dataframes for the 2024 season for each position
    validation_df = feature_df[feature_df['season'] == validation_year]
    val_df_rb = validation_df[validation_df['position'] == 'RB'].copy()
    val_df_wr_te = validation_df[validation_df['position'].isin(['WR', 'TE'])].copy()
    val_df_qb = validation_df[validation_df['position'] == 'QB'].copy()

    # Evaluate the RB Model
    evaluate_specialist_model(rb_model, "RB Model", val_df_rb, rb_features)

    # Evaluate the WR/TE Model
    evaluate_specialist_model(wr_te_model, "WR/TE Model", val_df_wr_te, wr_te_features)
    
    # Evaluate the QB Model
    evaluate_specialist_model(qb_model, "QB Model", val_df_qb, qb_features)


    # -- Prediction for Future Week --
    print("\n--- Generating Predictions for Week 1, 2025 ---")
    future_odds_raw = pd.read_csv('week_1_lines.csv')
    future_odds_df = transform_future_odds(future_odds_raw, team_map)
    opponent_le = LabelEncoder().fit(feature_df['opponent_team'].unique())

    predict_touchdown_scorers(feature_df, rb_model, wr_te_model, qb_model, rb_features, wr_te_features, qb_features, opponent_le, prediction_year, prediction_week, future_odds_df)