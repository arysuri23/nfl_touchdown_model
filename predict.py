import json
import sys
import pandas as pd
import numpy as np
import joblib
import nflreadpy as nfl
import data_collection as data # Assuming data_collection.py is in the same deployment package


### CONSTANTS ###

# Feature lists must match those used during training

# Updated, Streamlined Feature Lists
# These lists are curated to reduce multicollinearity and noise, focusing on the strongest predictors.


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
    #'avg_rush_yards_over_expected_per_att', # Correlated with avg_rushing_epa.
    #'avg_rush_pct_over_expected',
    #'avg_avg_time_to_los',
    #'avg_percent_attempts_gte_eight_defenders', # Measures ability against stacked boxes.

    # --- High-Value Touches ---
    # These features are crucial but can be correlated. For example, a high redzone carry share
    # often leads to a high inside_5 carry share. The model should handle this, but it's worth noting.
    'avg_redzone_carry_share',
    'avg_redzone_target_share',
    #'avg_endzone_targets',  # 1.2% importance - lowest RB feature, removed
    #'avg_endzone_target_share',
    'avg_inside_5_carry_share',
    #'avg_inside_5_target_share',

    # --- Contextual & Matchup Features ---
    'rush_matchup_value',
    #'pass_matchup_value',  # 2.5% importance - low value for RBs, removed
    #'pass_rate',
    #'rush_rate',
    'redzone_td_rate', # Team-level efficiency in the red zone.

    'rushing_tds_allowed_to_RB', # Opponent tendency.
    'passing_tds_allowed_to_RB', # Opponent tendency.
    'implied_total', # Game script proxy.
    'spread_line',
    'depth_chart_rank',
    
    # --- Game Context Features ---
    # Removed: All schedule features had <0.05% importance (is_home_game, is_thursday, is_dome, is_grass, is_cold, is_windy)
    # Their effects are already captured by implied_total and spread_line

    # --- Target Variable Lag ---
    # This is a lagged version of the target. Highly predictive, but ensure no data leakage.
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
    'avg_target_share', # Aligned with training (lagged EWM of target_share)

    # --- Efficiency & Route-Running Metrics ---
    'avg_receiving_epa', # Advanced efficiency metric.
    'avg_racr', # Efficiency in converting air yards to real yards.

    # avg_avg_cushion and avg_avg_separation both measure a receiver's ability to get open.
    # They are likely correlated. Test with one or both.
    #'avg_avg_cushion',
    #'avg_avg_separation',  # 1.7% importance - NGS metric, minimal value, removed

    # These metrics describe how a player is used and how well they perform on their targets.
    # They are generally complementary.
    #'avg_percent_share_of_intended_air_yards', # Correlated with avg_wopr.
    #'avg_catch_percentage',
   #'avg_avg_expected_yac',
    #'avg_avg_yac_above_expectation',

    # --- High-Value Touches ---
    # Crucial for touchdown prediction.
    #'avg_redzone_target_share',  # 1.0% importance - surprisingly useless for WR/TE, removed
    'avg_endzone_targets',
    'avg_endzone_target_share',
    #'avg_inside_5_target_share',

    # --- Contextual & Matchup Features ---
    #'pass_matchup_value',  # 0.64% importance - LOWEST feature, engineered feature fails, removed
   
    'redzone_td_rate', # Team-level efficiency.
    #'pass_rate',
    #'rush_rate',

    'passing_tds_allowed_to_WR', # Opponent tendency.
    'passing_tds_allowed_to_TE', # Opponent tendency.
    'implied_total', # Game script proxy.
    'spread_line',
    'depth_chart_rank',
    
    # --- Game Context Features ---
    # Removed: All schedule features had <0.06% importance (is_home_game, is_thursday, is_primetime, is_dome, is_grass, is_cold, is_windy)
    # Their effects are already captured by implied_total and spread_line

    # --- Target Variable Lag ---
    # Highly predictive, but ensure no data leakage.
    'avg_scored_touchdown',

    # --- Volume & Air Yard Stats ---
    # IMPORTANT: These volume stats are TOP 5 features! Don't remove them!
    'avg_receptions',  # 8.5% importance - 4th most important!
    'avg_receiving_yards',  # 8.6% importance - 3rd most important!
    'avg_receiving_air_yards',  # 6.1% importance - 6th most important!
    #'avg_avg_intended_air_yards',  # 1.5% importance - minimal value, removed

    'avg_receiving_yards_allowed',
    'avg_receiving_epa_allowed',
    'avg_receiving_air_yards_allowed',

    'avg_explosive_receiving_plays',
    'avg_explosive_receiving_plays_allowed',
]
QB_FEATURES = [
    'avg_offense_snap_share', 
    #'team_continuity',
    'avg_carries', 'avg_rushing_yards', 'avg_rushing_epa', 
    'avg_scored_touchdown', 
    'avg_redzone_carry_share', 'avg_inside_5_carry_share',
    'rush_matchup_value', 
    #'redzone_td_rate', 
    'rushing_tds_allowed_to_QB', 'implied_total', 'spread_line', 'depth_chart_rank',
    #'avg_rush_yards_over_expected_per_att', 'avg_rush_pct_over_expected', 'avg_avg_time_to_los',
    'avg_explosive_rushing_plays',
    
    # --- Game Context Features ---
   # 1.23% importance - keeping this one for QB as it has marginal value
    # Removed: is_thursday (0.39%), is_dome (1.04%), is_grass (0.98%) - low importance
    ]



### LOCAL FILE LOADING FUNCTIONS ###
def load_joblib_locally(file_path):
    """Loads a joblib file from local filesystem."""
    print(f"Loading model artifact from '{file_path}'...")
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise



def transform_features(df):
    # Ensure chronological order before EWM calculations
    df = df.sort_values(['player_id', 'season', 'week']).copy()

    player_stats = ['carries', 'rushing_yards', 'receptions', 'receiving_yards', 'wopr', 'rushing_epa', 'receiving_epa', 'target_share',
                      'receiving_air_yards', 'racr', 'scored_touchdown', 'redzone_carry_share', 'redzone_target_share',
                      'endzone_targets', 'endzone_target_share', 'inside_5_carry_share', 'inside_5_target_share', 'offense_snap_share', 
                      'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 'avg_time_to_los', 'percent_attempts_gte_eight_defenders',
                      'avg_cushion', 'avg_separation', 'avg_intended_air_yards', 'percent_share_of_intended_air_yards', 
                    'catch_percentage', 'avg_expected_yac', 'avg_yac_above_expectation', 'explosive_rushing_plays', 'explosive_receiving_plays']
    
    
    # In inference, EWMs need not be shifted as we only use historical rows (< target week)
    for stat in player_stats:
        df[f'avg_{stat}'] = df.groupby('player_id')[stat].transform(lambda x: x.ewm(alpha=0.3, min_periods=1).mean())

    pos_defense_cols = [col for col in df.columns if 'tds_allowed_to' in col] + ['rushing_yards_allowed', 'receiving_yards_allowed', 'rushing_epa_allowed', 'receiving_epa_allowed', 'receiving_air_yards_allowed', 'explosive_rushing_plays_allowed', 'explosive_receiving_plays_allowed']

    opponent_stats_df = (
        df[['season', 'week', 'opponent_team'] + pos_defense_cols]
          .groupby(['season', 'week', 'opponent_team'], as_index=False)[pos_defense_cols]
          .mean()
          .sort_values(['opponent_team', 'season', 'week'])
    )
    
    for col in pos_defense_cols:
        if 'tds_allowed_to' in col:
            opponent_stats_df[col] = opponent_stats_df.groupby('opponent_team')[col].transform(lambda x: x.ewm(alpha = 0.3, min_periods=1).mean())
        else:
            opponent_stats_df[f'avg_{col}'] = opponent_stats_df.groupby('opponent_team')[col].transform(lambda x: x.ewm(alpha = 0.3, min_periods=1).mean())


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
    return df

### PREDICTION LOGIC ###
def predict_stacked_proba(X, base_models, meta_model):
    """Generates final probabilities from a manually stacked model."""
    if X.empty:
        return np.array([])
    meta_features = np.column_stack([model.predict_proba(X)[:, 1] for model in base_models])
    final_predictions = meta_model.predict_proba(meta_features)[:, 1]
    return final_predictions

def predict_touchdown_scorers(feature_df, models, calibrators, year, week, future_odds_df, td_odds_df):
    """Predicts touchdown scorers using loaded, position-specific models."""
    print("Assembling features for prediction...")

    ### Filter feature-df to only include data up to the week before the prediction week
    feature_df = feature_df[(feature_df['season'] < year) | ((feature_df['season'] == year) & (feature_df['week'] < week))]
    # Compute lagged EWMs on the filtered historical subset to avoid leakage
    feature_df = transform_features(feature_df)
    # transform_features now called inside predict_touchdown_scorers after filtering; avoid double transform

    # Get schedule and roster for the prediction week
    schedule = nfl.load_schedules([year]).to_pandas()
    #schedule['home_team'] = schedule['home_team'].replace({'LA': 'LAR', 'LV': 'LVR'})
    #schedule['away_team'] = schedule['away_team'].replace({'LA': 'LAR', 'LV': 'LVR'})
    
    week_schedule = schedule[schedule['week'] == week]
    
    rosters = nfl.load_rosters_weekly([year]).to_pandas()
    rosters = rosters[rosters['status']=='ACT']
    
    #rosters['team'] = rosters['team'].replace({'LA': 'LAR', 'LV': 'LVR'})


    
    opponent_map = {row['home_team']: row['away_team'] for _, row in week_schedule.iterrows()}
    opponent_map.update({row['away_team']: row['home_team'] for _, row in week_schedule.iterrows()})
    
    teams_playing = list(opponent_map.keys())
    
    week_rosters = rosters[rosters['team'].isin(teams_playing) & rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])].drop_duplicates(subset=['gsis_id'], keep='last')


    prediction_df = week_rosters[['gsis_id', 'full_name', 'position', 'team']]
    
    prediction_df.rename(columns={'gsis_id': 'player_id', 'full_name': 'player_display_name'}, inplace=True)
    prediction_df['opponent_team'] = prediction_df['team'].map(opponent_map)

    # Assemble features using historical data from the pre-engineered feature_df
    all_features = set(RB_FEATURES + WR_TE_FEATURES + QB_FEATURES)
    
    # Identify features that are NOT player-specific (need to be merged separately)
    non_player_features = set()
    for f in all_features:
        # Opponent defensive stats (all contain 'allowed')
        if 'allowed' in f:
            non_player_features.add(f)
        # Matchup values (calculated from player + opponent stats)
        elif f in ['rush_matchup_value', 'pass_matchup_value']:
            non_player_features.add(f)
        # Game context features (week-specific, not player-specific)
        elif f in ['implied_total', 'spread_line', 'is_home_game', 'div_game', 'redzone_td_rate']:
            non_player_features.add(f)
    
    player_history_features = sorted(list(all_features - non_player_features))
    opponent_history_features = [f for f in all_features if 'allowed' in f]  # All opponent defensive stats
    team_history_features = ['redzone_td_rate'] if 'redzone_td_rate' in all_features else []
    
    features_from_player_history = player_history_features
    # Ensure chronological order so groupby().last() picks the most recent row per player
    feature_df = feature_df.sort_values(['player_id', 'season', 'week']).copy()
    latest_player_data = feature_df.groupby('player_id')[features_from_player_history].last().reset_index()

    prediction_df = pd.merge(prediction_df, latest_player_data, on='player_id', how='left')

    # Get the latest opponent defensive stats for each opponent team
    if opponent_history_features:
        opponent_stats_df = feature_df[['season', 'week', 'opponent_team'] + opponent_history_features]
        opponent_stats_df = opponent_stats_df.sort_values(by=['opponent_team', 'season', 'week']).copy()
        latest_opponent_data = opponent_stats_df.groupby('opponent_team')[opponent_history_features].last().reset_index()
        prediction_df = pd.merge(prediction_df, latest_opponent_data, on='opponent_team', how='left')
    
    # Get the latest team-level stats (redzone_td_rate) for each team
    if team_history_features:
        team_stats_df = feature_df[['season', 'week', 'team'] + team_history_features]
        team_stats_df = team_stats_df.sort_values(by=['team', 'season', 'week']).copy()
        latest_team_data = team_stats_df.groupby('team')[team_history_features].last().reset_index()
        prediction_df = pd.merge(prediction_df, latest_team_data, on='team', how='left', suffixes=('', '_team'))
        
    
    prediction_df = pd.merge(prediction_df, future_odds_df[['team', 'implied_total', 'spread_line']], on='team', how='left')
    
    # Opponent encoding removed; rely on engineered opponent-week features instead

    prediction_df['rush_matchup_value'] = np.select(
        [prediction_df['position'] == 'RB', prediction_df['position'] == 'QB'],
        [prediction_df['avg_redzone_carry_share'] * prediction_df['rushing_tds_allowed_to_RB'], prediction_df['avg_redzone_carry_share'] * prediction_df['rushing_tds_allowed_to_QB']],
        default=0)
    prediction_df['pass_matchup_value'] = np.select(
        [prediction_df['position'] == 'RB', prediction_df['position'] == 'WR', prediction_df['position'] == 'TE'],
        [prediction_df['avg_redzone_target_share'] * prediction_df['passing_tds_allowed_to_RB'], prediction_df['avg_redzone_target_share'] * prediction_df['passing_tds_allowed_to_WR'], prediction_df['avg_redzone_target_share'] * prediction_df['passing_tds_allowed_to_TE']],
        default=0)
    
    depth_chart_2025_data = data.get_2025_depth_chart_data()
    depth_chart_2025_data = depth_chart_2025_data[depth_chart_2025_data['week'] == week]
    prediction_df.drop(columns=['depth_chart_rank'], inplace=True)  # Ensure no duplicate column
    #merge 2025 depth chart data on player_id and use depth_chart_rank from depth_chart_2025_data
    prediction_df = pd.merge(prediction_df, depth_chart_2025_data, on='player_id', how='left')
    #fill depth_chart_rank with 0 if NaN
    prediction_df['depth_chart_rank'] = prediction_df['depth_chart_rank'].fillna(4).astype(int)
    
    
    prediction_df.fillna(0, inplace=True)
    print("Feature assembly complete.")


    prediction_df.to_csv('unscaled.csv', index=False)
      # Save unscaled features for debugging

    # Split data and apply correct model
    pred_df_rb = prediction_df[prediction_df['position'] == 'RB'].copy()
    pred_df_wr_te = prediction_df[prediction_df['position'].isin(['WR', 'TE'])].copy()
    pred_df_qb = prediction_df[prediction_df['position'] == 'QB'].copy()

    # No scaling required; tree-based models trained without scaling

    pred_df_rb['predicted_touchdown_probability'] = predict_stacked_proba(pred_df_rb[RB_FEATURES], models['rb_base'], models['rb_meta'])
    pred_df_wr_te['predicted_touchdown_probability'] = predict_stacked_proba(pred_df_wr_te[WR_TE_FEATURES], models['wr_te_base'], models['wr_te_meta'])
    pred_df_qb['predicted_touchdown_probability'] = predict_stacked_proba(pred_df_qb[QB_FEATURES], models['qb_base'], models['qb_meta'])

    # Apply Platt calibration per position if calibrators are available
    if not pred_df_rb.empty and 'rb' in calibrators and calibrators['rb'] is not None:
        rb_raw = pred_df_rb['predicted_touchdown_probability'].values.reshape(-1, 1)
        pred_df_rb['predicted_touchdown_probability'] = calibrators['rb'].predict_proba(rb_raw)[:, 1]
    if not pred_df_wr_te.empty and 'wr_te' in calibrators and calibrators['wr_te'] is not None:
        wrte_raw = pred_df_wr_te['predicted_touchdown_probability'].values.reshape(-1, 1)
        pred_df_wr_te['predicted_touchdown_probability'] = calibrators['wr_te'].predict_proba(wrte_raw)[:, 1]
    if not pred_df_qb.empty and 'qb' in calibrators and calibrators['qb'] is not None:
        qb_raw = pred_df_qb['predicted_touchdown_probability'].values.reshape(-1, 1)
        pred_df_qb['predicted_touchdown_probability'] = calibrators['qb'].predict_proba(qb_raw)[:, 1]

    # Combine results and find market edge
    final_predictions = pd.concat([pred_df_rb, pred_df_wr_te, pred_df_qb])
    
    td_odds_df['merge_name'] = td_odds_df['description'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()
    final_predictions['merge_name'] = final_predictions['player_display_name'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()
    
    prob_if_pos = 100 / (td_odds_df['price'] + 100)
    prob_if_neg = abs(td_odds_df['price']) / (abs(td_odds_df['price']) + 100)
    td_odds_df['market_implied_prob'] = np.where(td_odds_df['price'] > 0, prob_if_pos, prob_if_neg)
    
    final_predictions = pd.merge(final_predictions, td_odds_df[['merge_name', 'price', 'market_implied_prob']], on='merge_name', how='left')
    
    final_predictions['model_edge'] = final_predictions['predicted_touchdown_probability'] - final_predictions['market_implied_prob']
    
    display_cols = ['player_display_name', 'team', 'position', 'predicted_touchdown_probability', 'price', 'market_implied_prob', 'model_edge']
    final_predictions.dropna(subset=['price'], inplace=True)
    
    return final_predictions.sort_values(by='predicted_touchdown_probability', ascending=False)


### main
if __name__ == '__main__':
    """
    Main entry point for the AWS Lambda function.
    """
    print("Lambda function initiated.")
    prediction_year = 2025
    prediction_week = 6
    
    # --- 1. Load Models and Encoders from Local Files ---
    print("Loading model artifacts from local files...")
    models = {
        'rb_base': load_joblib_locally('models/rb_base_final.pkl'),
        'rb_meta': load_joblib_locally('models/rb_meta_final.pkl'),
        'wr_te_base': load_joblib_locally('models/wr_te_base_final.pkl'),
        'wr_te_meta': load_joblib_locally('models/wr_te_meta_final.pkl'),
        'qb_base': load_joblib_locally('models/qb_base_final.pkl'),
        'qb_meta': load_joblib_locally('models/qb_meta_final.pkl')
    }
    calibrators = {
        'rb': load_joblib_locally('models/rb_calibrator.pkl'),
        'wr_te': load_joblib_locally('models/wr_te_calibrator.pkl'),
        'qb': load_joblib_locally('models/qb_calibrator.pkl')
    }
    # No scalers or opponent encoders
    print("Model loading complete.")

    # --- 2. Load Data Files from Local Files ---
    print("Loading data files from local files...")
    nfl_teams_df = pd.read_csv('nfl_teams.csv')
    future_odds_raw = pd.read_csv(f'data/week_{prediction_week}_lines.csv')
    td_odds_df = pd.read_csv(f'data/week_{prediction_week}_td_odds.csv')
    feature_df = pd.read_csv('raw_nfl_data.csv')
    team_map = dict(zip(nfl_teams_df['team_name'], nfl_teams_df['team_id']))

    
    #feature_df = transform_features(feature_df)
    
   
    future_odds_df = data.transform_future_odds(future_odds_raw, team_map)
    print(future_odds_df)
    print("Data loading complete.")

    # --- 3. Run Prediction ---
    # In a real app, you would get the year/week from the API Gateway event
   
    print(f"Generating predictions for {prediction_year}, Week {prediction_week}...")
    
    final_predictions_df = predict_touchdown_scorers(feature_df, models, calibrators, prediction_year, prediction_week, future_odds_df, td_odds_df)
    #save predictions to csv with prediction week
    final_predictions_df.to_csv(f'data/predictions_week_{prediction_week}.csv', index=False)  # Save predictions to a CSV file for review
    print(final_predictions_df.head(10))
   