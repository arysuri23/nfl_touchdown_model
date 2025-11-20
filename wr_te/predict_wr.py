import json
import sys
import pandas as pd
import numpy as np
import joblib
import nflreadpy as nfl
import data_collection as data


### CONSTANTS ###

# Feature lists must match those used during training

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
    #'avg_total_tds',
    'avg_scored_touchdown',
    'avg_receptions',
    'avg_receiving_yards',
    'avg_receiving_air_yards',
    'avg_receiving_yards_allowed',
    'avg_receiving_epa_allowed',
    'avg_receiving_air_yards_allowed',
    'avg_explosive_receiving_plays',
    'avg_explosive_receiving_plays_allowed'
    #'avg_rec_touchdown_exp', 
    #'avg_rec_touchdown_exp_team'
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

    player_stats = ['receptions', 'receiving_yards', 'wopr', 'receiving_epa', 'target_share',
                      'receiving_air_yards', 'racr', 'scored_touchdown', 'redzone_target_share', 'total_tds',
                      'endzone_targets', 'endzone_target_share', 'inside_5_target_share', 'offense_snap_share',
                      'avg_cushion', 'avg_separation', 'avg_intended_air_yards', 'percent_share_of_intended_air_yards',
                    'catch_percentage', 'avg_expected_yac', 'avg_yac_above_expectation', 'explosive_receiving_plays', 'rec_touchdown_exp', 'rec_touchdown_exp_team']
    
    # In inference, EWMs need not be shifted as we only use historical rows (< target week)
    for stat in player_stats:
        df[f'avg_{stat}'] = df.groupby('player_id')[stat].transform(lambda x: x.ewm(alpha=0.3, min_periods=1).mean())

    pos_defense_cols = [col for col in df.columns if 'tds_allowed_to' in col] + ['receiving_yards_allowed', 'receiving_epa_allowed', 'receiving_air_yards_allowed', 'explosive_receiving_plays_allowed']

    opponent_stats_df = (
        df[['season', 'week', 'opponent_team'] + pos_defense_cols]
          .groupby(['season', 'week', 'opponent_team'], as_index=False)[pos_defense_cols]
          .mean()
          .sort_values(['opponent_team', 'season', 'week'])
    )
    
    # Apply EWM and add avg_ prefix to match feature list expectations
    for col in pos_defense_cols:
         if 'tds_allowed_to' in col:
             opponent_stats_df[col] = opponent_stats_df.groupby('opponent_team')[col].transform(lambda x: x.ewm(alpha = 0.3, min_periods=1).mean())
         else:
             opponent_stats_df[f'avg_{col}'] = opponent_stats_df.groupby('opponent_team')[col].transform(lambda x: x.ewm(alpha = 0.3, min_periods=1).mean())

    df.drop(columns=pos_defense_cols, inplace=True)
    df = pd.merge(df, opponent_stats_df, on=['season', 'week', 'opponent_team'], how='left')
    
    df.fillna(0, inplace=True)
    return df


### PREDICTION LOGIC ###
def predict_touchdown_scorers(feature_df, model, calibrator, year, week, future_odds_df, td_odds_df):
    """Predicts touchdown scorers for WR/TE using loaded RF model."""
    print("Assembling features for prediction...")

    ### Filter feature-df to only include data up to the week before the prediction week
    feature_df = feature_df[(feature_df['season'] < year) | ((feature_df['season'] == year) & (feature_df['week'] < week))]
    # Compute lagged EWMs on the filtered historical subset to avoid leakage
    print(feature_df.tail().week)
    feature_df = transform_features(feature_df)

    # Get schedule and roster for the prediction week
    schedule = nfl.load_schedules([year]).to_pandas()
    
    week_schedule = schedule[schedule['week'] == week]
    
    rosters = nfl.load_rosters_weekly([year]).to_pandas()
    rosters = rosters[rosters['status']=='ACT']
    
    opponent_map = {row['home_team']: row['away_team'] for _, row in week_schedule.iterrows()}
    opponent_map.update({row['away_team']: row['home_team'] for _, row in week_schedule.iterrows()})
    
    teams_playing = list(opponent_map.keys())
    
    week_rosters = rosters[rosters['team'].isin(teams_playing) & rosters['position'].isin(['WR', 'TE'])].drop_duplicates(subset=['gsis_id'], keep='last')

    prediction_df = week_rosters[['gsis_id', 'full_name', 'position', 'team']].copy()
    
    prediction_df.rename(columns={'gsis_id': 'player_id', 'full_name': 'player_display_name'}, inplace=True)
    prediction_df['opponent_team'] = prediction_df['team'].map(opponent_map)

    # Assemble features using historical data from the pre-engineered feature_df
    all_features = set(WR_TE_FEATURES)
    
    # Identify features that are NOT player-specific (need to be merged separately)
    non_player_features = set()
    for f in all_features:
        # Opponent defensive stats (all contain 'allowed')
        if 'allowed' in f:
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
    
    depth_chart_2025_data = data.get_2025_depth_chart_data()
    depth_chart_2025_data = depth_chart_2025_data[depth_chart_2025_data['week'] == week]
    prediction_df.drop(columns=['depth_chart_rank'], inplace=True, errors='ignore')  # Ensure no duplicate column
    #merge 2025 depth chart data on player_id and use depth_chart_rank from depth_chart_2025_data
    prediction_df = pd.merge(prediction_df, depth_chart_2025_data, on='player_id', how='left')
    #fill depth_chart_rank with 0 if NaN
    prediction_df['depth_chart_rank'] = prediction_df['depth_chart_rank'].fillna(4).astype(int)
    
    
    prediction_df.fillna(0, inplace=True)
    print("Feature assembly complete.")

    # Filter for WR/TE only
    pred_df_wr_te = prediction_df[prediction_df['position'].isin(['WR', 'TE'])].copy()

    # No scaling required; tree-based models trained without scaling

    # Make predictions
    if not pred_df_wr_te.empty:
        pred_df_wr_te['predicted_touchdown_probability'] = model.predict_proba(pred_df_wr_te[WR_TE_FEATURES])[:, 1]
        
        # Apply Platt calibration if calibrator is available
        if calibrator is not None:
            wrte_raw = pred_df_wr_te['predicted_touchdown_probability'].values.reshape(-1, 1)
            pred_df_wr_te['predicted_touchdown_probability'] = calibrator.predict_proba(wrte_raw)[:, 1]

    final_predictions = pred_df_wr_te
    
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
    Main entry point for WR/TE predictions.
    """
    print("="*60)
    print("WR/TE TD PREDICTION")
    print("="*60)
    
    prediction_year = 2025
    prediction_week = 11 # Update this for each week
    
    # --- 1. Load WR/TE Model ---
    print("\nLoading WR/TE model artifacts from local files...")
    wr_te_model = load_joblib_locally('models/wr_te_rf_final.pkl')
    wr_te_calibrator = load_joblib_locally('models/wr_te_rf_calibrator.pkl')
    print("✓ Model loading complete")

    # --- 2. Load Data Files ---
    print("\nLoading data files from local files...")
    nfl_teams_df = pd.read_csv('data/nfl_teams.csv')
    future_odds_raw = pd.read_csv(f'vegas/week_{prediction_week}_lines.csv')
    td_odds_df = pd.read_csv(f'vegas/week_{prediction_week}_td_odds.csv')
    feature_df = pd.read_csv('data/raw_nfl_data.csv')
    team_map = dict(zip(nfl_teams_df['team_name'], nfl_teams_df['team_id']))
    
    future_odds_df = data.transform_future_odds(future_odds_raw, team_map)
    print("✓ Data loading complete")

    # --- 3. Run Prediction ---
    print(f"\nGenerating WR/TE predictions for {prediction_year}, Week {prediction_week}...")
    
    final_predictions_df = predict_touchdown_scorers(feature_df, wr_te_model, wr_te_calibrator, prediction_year, prediction_week, future_odds_df, td_odds_df)
    
    # Save predictions to csv
    #final_predictions_df.to_csv(f'predictions/predictions_week_{prediction_week}_new_features.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"TOP 20 WR/TE TD PREDICTIONS - WEEK {prediction_week}")
    print("="*60)
    print(final_predictions_df[['player_display_name', 'team', 'position', 'predicted_touchdown_probability', 'model_edge']].head(20).to_string(index=False))
    
    print(f"\n✓ Predictions saved to predictions/predictions_week_{prediction_week}.csv")
    print("\n🎯 Use these predictions for your analysis!")

