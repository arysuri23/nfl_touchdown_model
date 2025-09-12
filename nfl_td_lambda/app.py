import json
import boto3
import pandas as pd
import numpy as np
import joblib
import nfl_data_py as nfl
from io import BytesIO, StringIO
import data_collection as data # data_collection.py is in the same directory


### CONSTANTS ###
S3_BUCKET_NAME = "nfl-touchdown-model-data"

# Feature lists must match those used during training

CATEGORICAL_FEATURES = ['opponent_encoded']
# Updated, Streamlined Feature Lists
# These lists are curated to reduce multicollinearity and noise, focusing on the strongest predictors.

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



### S3 HELPER FUNCTIONS ###
def read_joblib_from_s3(s3_key):
    """Loads a joblib file from S3."""
    s3_client = boto3.client('s3')
    with BytesIO() as buffer:
        s3_client.download_fileobj(Bucket=S3_BUCKET_NAME, Key=s3_key, Fileobj=buffer)
        buffer.seek(0)
        return joblib.load(buffer)

def read_csv_from_s3(s3_key):
    """Loads a CSV file from S3 into a pandas DataFrame."""
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
    csv_string = response['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(csv_string))

def transform_features(df):
    player_stats = ['carries', 'rushing_yards', 'receptions', 'receiving_yards', 'wopr', 'rushing_epa', 'receiving_epa', 'target_share',
                      'receiving_air_yards', 'racr', 'scored_touchdown', 'redzone_carry_share', 'redzone_target_share',
                      'endzone_targets', 'endzone_target_share', 'inside_5_carry_share', 'inside_5_target_share', 'offense_snap_share', 
                      'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 'avg_time_to_los', 'percent_attempts_gte_eight_defenders',
                      'avg_cushion', 'avg_separation', 'avg_intended_air_yards', 'percent_share_of_intended_air_yards', 
                    'catch_percentage', 'avg_expected_yac', 'avg_yac_above_expectation']
    
    for stat in player_stats:
        df[f'avg_{stat}'] = df.groupby('player_id')[stat].transform(lambda x: x.ewm(span=6, min_periods=1).mean())

    pos_defense_cols = [col for col in df.columns if 'tds_allowed_to' in col]

    opponent_stats_df = df[['season', 'week', 'opponent_team'] + pos_defense_cols].drop_duplicates()
    opponent_stats_df.sort_values(by=['season', 'week'], inplace=True)
    
    for col in pos_defense_cols:
         opponent_stats_df[col] = opponent_stats_df.groupby('opponent_team')[col].transform(lambda x: x.ewm(span=6, min_periods=1).mean())


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

def predict_touchdown_scorers(feature_df, models, scalers, opponent_le, year, week, future_odds_df, td_odds_df):
    """Predicts touchdown scorers using loaded, position-specific models."""
    print("Assembling features for prediction...")

    feature_df = feature_df[(feature_df['season'] < year) | ((feature_df['season'] == year) & (feature_df['week'] < week))]

    # Get schedule and roster for the prediction week
    schedule = nfl.import_schedules([year])
    #schedule['home_team'] = schedule['home_team'].replace({'LA': 'LAR', 'LV': 'LVR'})
    #schedule['away_team'] = schedule['away_team'].replace({'LA': 'LAR', 'LV': 'LVR'})
    
    week_schedule = schedule[schedule['week'] == week]
    
    rosters = nfl.import_weekly_rosters([year])
    rosters = rosters[rosters['status']=='ACT']
    
    #rosters['team'] = rosters['team'].replace({'LA': 'LAR', 'LV': 'LVR'})
    
    opponent_map = {row['home_team']: row['away_team'] for _, row in week_schedule.iterrows()}
    opponent_map.update({row['away_team']: row['home_team'] for _, row in week_schedule.iterrows()})
    
    teams_playing = list(opponent_map.keys())
    week_rosters = rosters[rosters['team'].isin(teams_playing) & rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()
    
    prediction_df = week_rosters[['player_id', 'player_name', 'position', 'team']].copy()
    prediction_df.rename(columns={'player_name': 'player_display_name'}, inplace=True)
    prediction_df['opponent_team'] = prediction_df['team'].map(opponent_map)

    # Assemble features using historical data from the pre-engineered feature_df
    all_features = set(RB_FEATURES + WR_TE_FEATURES + QB_FEATURES)
    non_player_features = set()
    for f in all_features:
        if 'allowed_to' in f or f in ['rush_matchup_value', 'pass_matchup_value', 
                                      #'redzone_td_rate', 
                                      'implied_total', 'opponent_encoded']:
            non_player_features.add(f)
    
    player_history_features = sorted(list(all_features - non_player_features))
    #team_history_features = ['redzone_td_rate']
    opponent_history_features = [f for f in all_features if 'allowed_to' in f]
    
    features_from_player_history = player_history_features #+ team_history_features
    latest_player_data = feature_df.groupby('player_id')[features_from_player_history + ['recent_team']].last().reset_index()
    prediction_df = pd.merge(prediction_df, latest_player_data, on='player_id', how='left')


     # NEW: Calculate team_continuity for the prediction week
    # A player has continuity if their team for the upcoming week is the same as their last known team from history.
   # prediction_df['team_continuity'] = (prediction_df['team'] == prediction_df['recent_team']).astype(int) # Note: pandas may add a suffix like _y
    
    # Get the latest opponent data for each team
    opponent_stats_df = feature_df[['season', 'week', 'opponent_team'] + opponent_history_features]
    #opponent_stats_df = opponent_stats_df[(opponent_stats_df['opponent_team'] == "NO") & (opponent_stats_df['season'] == 2024) & (opponent_stats_df['week'] == 18)]
    opponent_stats_df.sort_values(by=['season', 'week'], inplace=True)
    print(opponent_stats_df)
   

     # Debugging output
    latest_opponent_data = opponent_stats_df.groupby('opponent_team')[opponent_history_features].last().reset_index()

    
    
   
    #rename recent_team to opponent_team
    #latest_opponent_data.rename(columns={'recent_team': 'opponent_team'}, inplace=True)
    prediction_df = pd.merge(prediction_df, latest_opponent_data, on='opponent_team', how='left')
    
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
    
    depth_chart_2025_data = data.get_2025_depth_chart_data()
    prediction_df.drop(columns=['depth_chart_rank'], inplace=True)  # Ensure no duplicate column
    #merge 2025 depth chart data on player_id and use depth_chart_rank from depth_chart_2025_data
    prediction_df = pd.merge(prediction_df, depth_chart_2025_data, on='player_id', how='left')
    #fill depth_chart_rank with 0 if NaN
    prediction_df['depth_chart_rank'] = prediction_df['depth_chart_rank'].fillna(4).astype(int)
    
    
    prediction_df.fillna(0, inplace=True)
    print("Feature assembly complete.")



    # Split data and apply correct model
    pred_df_rb = prediction_df[prediction_df['position'] == 'RB'].copy()
    pred_df_wr_te = prediction_df[prediction_df['position'].isin(['WR', 'TE'])].copy()
    pred_df_qb = prediction_df[prediction_df['position'] == 'QB'].copy()

    #Scale each position's data with its specific scaler ---
    if not pred_df_rb.empty:
        numerical_features_rb = [f for f in RB_FEATURES if f not in CATEGORICAL_FEATURES]
        pred_df_rb.loc[:, numerical_features_rb] = scalers['rb'].transform(pred_df_rb[numerical_features_rb])

    if not pred_df_wr_te.empty:
        numerical_features_wr_te = [f for f in WR_TE_FEATURES if f not in CATEGORICAL_FEATURES]
        pred_df_wr_te.loc[:, numerical_features_wr_te] = scalers['wr_te'].transform(pred_df_wr_te[numerical_features_wr_te])

    if not pred_df_qb.empty:
        numerical_features_qb = [f for f in QB_FEATURES if f not in CATEGORICAL_FEATURES]
        pred_df_qb.loc[:, numerical_features_qb] = scalers['qb'].transform(pred_df_qb[numerical_features_qb])

    pred_df_rb['predicted_touchdown_probability'] = predict_stacked_proba(pred_df_rb[RB_FEATURES], models['rb_base'], models['rb_meta'])
    pred_df_wr_te['predicted_touchdown_probability'] = predict_stacked_proba(pred_df_wr_te[WR_TE_FEATURES], models['wr_te_base'], models['wr_te_meta'])
    pred_df_qb['predicted_touchdown_probability'] = predict_stacked_proba(pred_df_qb[QB_FEATURES], models['qb_base'], models['qb_meta'])

    # Combine results and find market edge
    final_predictions = pd.concat([pred_df_rb, pred_df_wr_te, pred_df_qb])
    
    td_odds_df['merge_name'] = td_odds_df['description'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()
    final_predictions['merge_name'] = final_predictions['player_display_name'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()
    
    prob_if_pos = 100 / (td_odds_df['price'] + 100)
    prob_if_neg = abs(td_odds_df['price']) / (abs(td_odds_df['price']) + 100)
    td_odds_df['market_implied_prob'] = np.where(td_odds_df['price'] > 0, prob_if_pos, prob_if_neg)
    
    final_predictions = pd.merge(final_predictions, td_odds_df[['merge_name', 'price', 'market_implied_prob']], on='merge_name', how='left')
    final_predictions[['price', 'market_implied_prob']] = final_predictions[['price', 'market_implied_prob']]
    final_predictions['model_edge'] = final_predictions['predicted_touchdown_probability'] - final_predictions['market_implied_prob']
    final_predictions.drop_duplicates(subset=['player_id'], keep='last', inplace=True)

    display_cols = ['player_display_name', 'team', 'position', 'predicted_touchdown_probability', 'price', 'market_implied_prob', 'model_edge']
    final_predictions.dropna(subset=['price'], inplace=True)
    
    
    return final_predictions.sort_values(by='predicted_touchdown_probability', ascending=False)


### LAMBDA HANDLER ###
def lambda_handler(event, context):
    """
    Main entry point for the AWS Lambda function.
    """
    print("Lambda function initiated.")
    prediction_year = 2025
    prediction_week = 2
    try:
        # --- 1. Load Models and Encoders from S3 ---
        print("Loading model artifacts from S3...")
        models = {
            'rb_base': read_joblib_from_s3('models/rb_base_final.pkl'),
            'rb_meta': read_joblib_from_s3('models/rb_meta_final.pkl'),
            'wr_te_base': read_joblib_from_s3('models/wr_te_base_final.pkl'),
            'wr_te_meta': read_joblib_from_s3('models/wr_te_meta_final.pkl'),
            'qb_base': read_joblib_from_s3('models/qb_base_final.pkl'),
            'qb_meta': read_joblib_from_s3('models/qb_meta_final.pkl')
        }
        # --- NEW: Load the scalers ---
        scalers = {
            'rb': read_joblib_from_s3('models/rb_scaler.pkl'),
            'wr_te': read_joblib_from_s3('models/wr_te_scaler.pkl'),
            'qb': read_joblib_from_s3('models/qb_scaler.pkl')
        }
        opponent_le = read_joblib_from_s3('models/opponent_encoder.pkl')
        print("Model loading complete.")

        # --- 2. Load Data Files from S3 ---
        print("Loading data files from S3...")
        nfl_teams_df = read_csv_from_s3('data/nfl_teams.csv')
        future_odds_raw = read_csv_from_s3('data/data/week_2_lines.csv')
        td_odds_df = read_csv_from_s3('data/data/week_2_td_odds.csv')
        feature_df = read_csv_from_s3('data/raw_nfl_data.csv')
        
        team_map = dict(zip(nfl_teams_df['team_name'], nfl_teams_df['team_id']))
    

        #feature_df = data.get_all_historic_data([2020,2021,2022,2023,2024,2025], team_map)
        #feature_df = feature_df[(feature_df['season'] < prediction_year) | ((feature_df['season'] == prediction_year) & (feature_df['week'] < prediction_week))]

        feature_df = transform_features(feature_df)

        future_odds_df = data.transform_future_odds(future_odds_raw, team_map)
        print("Data loading complete.")

        # --- 3. Run Prediction ---
        # In a real app, you would get the year/week from the API Gateway event
        print(f"Generating predictions for {prediction_year}, Week {prediction_week}...")
        
        final_predictions_df = predict_touchdown_scorers(feature_df, models, scalers, opponent_le, prediction_year, prediction_week, future_odds_df, td_odds_df)
        
        # --- 4. Format and Return Response ---
        # Convert DataFrame to a list of dictionaries for JSON serialization
        predictions_json = final_predictions_df.to_dict(orient='records')
        
        print("Prediction complete. Returning results.")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*', # Allows any website to call this API
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps(predictions_json)
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'error': str(e)})
        }
