# NFL Touchdown Scorer Prediction Model

# --- 1. Introduction ---
# This script builds a machine learning model to predict which players are likely to score a touchdown in an NFL game.
# We will use the 'nfl_data_py' library to acquire historical player and game data.
# The model will be a Random Forest Classifier, a powerful and popular choice for this type of prediction task.

# --- 2. Installation of Necessary Libraries ---
# Make sure you have the following libraries installed. You can install them using pip:
# pip install nfl_data_py
# pip install pandas
# pip install scikit-learn

# --- 3. Importing Libraries ---
import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- 4. Data Acquisition and Preprocessing ---

def get_nfl_data(years):
    """
    Fetches and preprocesses NFL data for a given list of years.
    """
    # Import weekly data for the specified years
    df = nfl.import_weekly_data(years, downcast=True)

    # Select relevant columns for our model
    df = df[['player_id', 'player_display_name', 'position', 'recent_team', 'season', 'week',
             'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
             'receiving_yards', 'receiving_tds', 'opponent_team', 'wopr', 'rushing_epa', 'receiving_epa', 'target_share' ]]
    # Drop players that are not QBs, RBs, WRs or TEs
    df = df[df['position'].isin(['QB','RB','TE', 'WR'])]

    # Create a binary target variable: 1 if the player scored a touchdown, 0 otherwise
    df['scored_touchdown'] = ((df['rushing_tds'] > 0) | (df['receiving_tds'] > 0)).astype(int)

    # Fill missing values with 0
    df.fillna(0, inplace=True)


    return df

def get_redzone_data(years, pbp):
    """
    Fetches red zone data for a given list of years.
    This function is currently not used but can be implemented for additional features.
    """

    # Filter for red zone plays
    redZone_df = pbp[pbp['yardline_100'] <= 20]


    # Filter for rush and pass attempts
    rushes = redZone_df[redZone_df['rush_attempt'] == 1]
    redzone_rush_usage = rushes.groupby(['rusher_player_id', 'week', 'season']).agg(
        rushes=('rush_attempt', 'sum')
    ).reset_index()


    receiving = redZone_df[redZone_df['pass_attempt'] == 1]
    redzone_receiving_usage = receiving.groupby(['receiver_player_id', 'week', 'season']).agg(
        receptions=('pass_attempt', 'sum')
    ).reset_index()


    # Rename columns for consistency
    redzone_rush_usage.rename(columns={'rusher_player_id': 'player_id'}, inplace=True)
    redzone_receiving_usage.rename(columns={'receiver_player_id': 'player_id'}, inplace=True)

    # Merge into single dataframe
    redzone_usage = pd.merge(redzone_rush_usage, redzone_receiving_usage, on=['player_id', 'week', 'season'], how='outer').fillna(0)

    redzone_usage.rename(columns={'rushes': 'redzone_rushes', 'receptions': 'redzone_receptions'}, inplace=True)
    return redzone_usage

def get_redzone_td_rate(years, pbp):

    redzone = pbp[pbp['yardline_100'] <= 20]
    
    # Define red zone trips: unique (posteam, drive) pairs in red zone
    redzone_trips = redzone.groupby(['posteam', 'season', 'week', 'drive']).size().reset_index()
    redzone_trips['is_trip'] = 1

    # Red zone TDs: any play in the red zone with touchdown = 1
    redzone_tds = redzone[redzone['touchdown'] == 1]
    redzone_td_drives = redzone_tds.groupby(['posteam', 'season', 'week', 'drive']).size().reset_index()
    redzone_td_drives['is_td'] = 1

    # Merge trips and TDs
    rz = pd.merge(redzone_trips[['posteam', 'season', 'week', 'drive', 'is_trip']],
                  redzone_td_drives[['posteam', 'season', 'week', 'drive', 'is_td']],
                  on=['posteam', 'season', 'week', 'drive'], how='left')
    
    rz['is_td'] = rz['is_td'].fillna(0)

    redzone_summary = rz.groupby(['posteam', 'season', 'week']).agg(
        redzone_trips=('is_trip', 'sum'),
        redzone_tds=('is_td', 'sum')
    ).reset_index()

    redzone_summary['redzone_td_rate'] = redzone_summary['redzone_tds'] / redzone_summary['redzone_trips']
    redzone_summary.rename(columns={'posteam': 'recent_team'}, inplace=True)
    
    return redzone_summary

def get_opponent_data(years, pbp):

    pbp = pbp.dropna(subset=['posteam', 'defteam'])

    # Calculate defensive stats by team and week
    defense_stats = pbp.groupby(['defteam', 'season', 'week']).agg(
        passing_tds_allowed = ('pass_touchdown', 'sum'),
        rushing_tds_allowed = ('rush_touchdown', 'sum'),
        total_tds_allowed = ('touchdown', 'sum'),
        epa_allowed = ('epa', 'sum'),
    ).reset_index()

    defense_stats.rename(columns={'defteam': 'opponent_team'}, inplace=True)
    return defense_stats


# --- 5. Feature Engineering ---

def feature_engineering(df, redzone_df, defense_df, redzone_td_rate):
    """
    Engineers features from the raw data to improve model performance.
    """

    # Merge red zone data in df
    df = pd.merge(df, redzone_df, on=['player_id', 'week', 'season'], how='left').fillna(0)
    # Merge opponent data
    df = pd.merge(df, defense_df, on=['opponent_team', 'season', 'week'], how='left').fillna(0)
    # Merge redzone TD rate

    df = pd.merge(df, redzone_td_rate, on=['recent_team', 'season', 'week'], how='left').fillna(0)
    # Create rolling averages of key stats to capture recent performance
    df['avg_carries'] = df.groupby('player_display_name')['carries'].transform(lambda x: x.shift(1).rolling(3, 1).mean())
    df['avg_rushing_yards'] = df.groupby('player_display_name')['rushing_yards'].transform(lambda x: x.shift(1).rolling(3, 1).mean())
    df['avg_receptions'] = df.groupby('player_display_name')['receptions'].transform(lambda x: x.shift(1).rolling(3, 1).mean())
    df['avg_receiving_yards'] = df.groupby('player_display_name')['receiving_yards'].transform(lambda x: x.shift(1).rolling(3, 1).mean())
    df['avg_wopr'] = df.groupby('player_display_name')['wopr'].transform(lambda x: x.shift(1).rolling(3, 1).mean())
    df['avg_rushing_epa'] = df.groupby('player_display_name')['rushing_epa'].transform(lambda x: x.shift(1).rolling(3, 1).mean())
    df['avg_receiving_epa'] = df.groupby('player_display_name')['receiving_epa'].transform(lambda x: x.shift(1).rolling(3, 1).mean())
    df['target_share'] = df.groupby('player_display_name')['target_share'].transform(lambda x: x.shift(1).rolling(3, 1).mean())
    df['avg_touchdowns'] = df.groupby('player_display_name')['scored_touchdown'].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['avg_redzone_rushes'] = df.groupby('player_display_name')['redzone_rushes'].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['avg_redzone_receptions'] = df.groupby('player_display_name')['redzone_receptions'].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['opponent_team'] = df['opponent_team'].astype('category')
    df['position'] = df['position'].astype('category')
    df['passing_tds_allowed'] = df['passing_tds_allowed'].groupby(df['opponent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['rushing_tds_allowed'] = df['rushing_tds_allowed'].groupby(df['opponent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['total_tds_allowed'] = df['total_tds_allowed'].groupby(df['opponent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['epa_allowed'] = df['epa_allowed'].groupby(df['opponent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['redzone_td_rate'] = df['redzone_td_rate'].groupby(df['recent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    # Fill NaN values resulting from rolling means
    df.fillna(0, inplace=True)


    df.to_csv('feature_df.csv', index=False)
    # Encode categorical variables
    le = LabelEncoder()
    df['position_encoded'] = le.fit_transform(df['position'])
    df['opponent_encoded'] = le.fit_transform(df['opponent_team'])

    return df

# --- 6. Model Training and Evaluation ---

from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate_model(df, validation_year=2024):
    """
    Trains a model on data before the validation year and evaluates it
    on the validation year data to simulate a real-world prediction task.
    """
    # Define features (X) and target (y)
    # Corrected: Removed the duplicate 'position_encoded' feature
    features = ['avg_carries', 'avg_rushing_yards', 'avg_receptions', 'avg_receiving_yards',
                'position_encoded', 'opponent_encoded', 'avg_wopr', 'avg_rushing_epa', 'avg_receiving_epa',
                'target_share', 'avg_touchdowns', 'redzone_rushes', 'redzone_receptions', 'passing_tds_allowed',
                'rushing_tds_allowed', 'total_tds_allowed', 'epa_allowed', 'redzone_td_rate']
    target = 'scored_touchdown'

    # --- Key Change: Chronological Split ---
    # Train on all data BEFORE the validation year
    train_df = df[df['season'] < validation_year]
    # Test on the data FROM the validation year
    test_df = df[df['season'] == validation_year]

    # Ensure there is data to test on
    if test_df.empty:
        print(f"Error: No data found for the validation year {validation_year}. Cannot evaluate model.")
        return None, None

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Make predictions on the validation/test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance on the unseen validation year
    print(f"--- Model Evaluation on {validation_year} Season ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No TD', 'Scored TD']))

    return model, features

def predict_touchdown_scorers(feature_df, model, features, year, week):
    """
    Predicts touchdown scorers for a given future week by assembling features
    based on historical data and the specific matchups for that week.
    """
    # --- 1. Isolate all data strictly BEFORE the prediction week ---
    # This prevents any data leakage from the week you are trying to predict.
    history_df = feature_df[
        (feature_df['season'] < year) | 
        ((feature_df['season'] == year) & (feature_df['week'] < week))
    ].copy()

    # --- 2. Get the most recent stats for each player and team from history ---
    # .last() gets the final row for each player/team, which holds their latest rolling averages.
    latest_player_stats = history_df.groupby('player_id').last()
    latest_team_stats = history_df.groupby('recent_team').last()

    # --- 3. Get the schedule and active rosters for the prediction week ---
    try:
        schedule = nfl.import_schedules([year])
        week_schedule = schedule[schedule['week'] == week]
        rosters = nfl.import_seasonal_rosters([year])
    except Exception as e:
        print(f"Could not fetch schedule or roster for {year} Week {week}. Error: {e}")
        return

    if week_schedule.empty:
        print(f"No schedule found for {year} Week {week}. Cannot make predictions.")
        return

    # Create a dictionary to easily find a team's opponent for the week
    opponent_map = {}
    for _, row in week_schedule.iterrows():
        opponent_map[row['home_team']] = row['away_team']
        opponent_map[row['away_team']] = row['home_team']

    teams_playing = list(opponent_map.keys())
    week_rosters = rosters[
        rosters['team'].isin(teams_playing) & 
        rosters['position'].isin(['QB', 'RB', 'WR', 'TE'])
    ].copy()

    # --- 4. Build the prediction DataFrame step-by-step ---
    # Start with the players who are active for the week
    prediction_df = week_rosters[['player_id', 'player_name', 'position', 'team']].copy()

    # Add their opponent for the upcoming game
    prediction_df['opponent_team'] = prediction_df['team'].map(opponent_map)

    # Merge the player's historical stats (e.g., avg_carries, avg_wopr)
    player_history_features = [
        'avg_carries', 'avg_rushing_yards', 'avg_receptions', 'avg_receiving_yards', 
        'avg_wopr', 'avg_rushing_epa', 'avg_receiving_epa', 'target_share', 
        'avg_touchdowns', 'avg_redzone_rushes', 'avg_redzone_receptions',
        'position_encoded' # A player's position is part of their history.
    ]

    # Merge ONLY the selected columns from the player's history.
    prediction_df = pd.merge(
        prediction_df,
        latest_player_stats[player_history_features],
        on='player_id',
        how='left'
    )
    # Merge the team's offensive stats (e.g., redzone_td_rate)
    team_feature_cols = ['redzone_td_rate']
    prediction_df = pd.merge(prediction_df, latest_team_stats[team_feature_cols], 
                             left_on='team', right_index=True, how='left')

    # Merge the OPPONENT's defensive stats (e.g., tds_allowed)
    opponent_feature_cols = [
        'passing_tds_allowed', 'rushing_tds_allowed', 'total_tds_allowed', 
        'epa_allowed', 'opponent_encoded'
    ]
    prediction_df = pd.merge(prediction_df, latest_team_stats[opponent_feature_cols], 
                             left_on='opponent_team', right_index=True, how='left',
                             suffixes=('', '_opponent'))

    # --- 5. Final data prep ---
    # Fill NaNs with 0 for rookies or players with no historical data
    prediction_df.fillna(0, inplace=True)

    # Ensure all feature columns are present before predicting
    for col in features:
        if col not in prediction_df.columns:
            prediction_df[col] = 0

    # Make sure data types are correct (especially for encoded columns)
    prediction_df['position_encoded'] = prediction_df['position_encoded'].astype(int)
    prediction_df['opponent_encoded'] = prediction_df['opponent_encoded'].astype(int)

    # --- 6. Make and Display Predictions ---
    X_pred = prediction_df[features]
    
    print(f'--- Making TD Predictions for Week {week}, {year} ---')
    week_probabilities = model.predict_proba(X_pred)[:, 1]

    prediction_df['predicted_touchdown_probability'] = week_probabilities

    prediction_df.to_csv(f'predictions_week_{week}_{year}.csv', index=False)
    
    output_cols = ['player_name', 'team', 'opponent_team', 'position', 'predicted_touchdown_probability']
    final_predictions = prediction_df[output_cols].sort_values(by='predicted_touchdown_probability', ascending=False)

    print('\n--- Top 25 Predicted Touchdown Scorers ---')
    print(final_predictions.head(25))
    return


# if __name__ == '__main__':
#     # To validate on 2024, the range MUST go to 2025
#     all_years_to_load = range(2021, 2025) 
#     validation_year = 2024

#     pbp = nfl.import_pbp_data(all_years_to_load, downcast=True)
#     # Get and engineer features for all years
#     print(f"Loading data for seasons: {list(all_years_to_load)}...")
#     nfl_df = get_nfl_data(all_years_to_load)
#     redzone_df = get_redzone_data(all_years_to_load, pbp)
#     defense_df = get_opponent_data(all_years_to_load, pbp)
#     redzone_td_df = get_redzone_td_rate(all_years_to_load, pbp)
#     feature_df = feature_engineering(nfl_df, redzone_df, defense_df, redzone_td_df)

#     # Train model on data before 2024 and evaluate its performance on the 2024 season
#     touchdown_model, model_features = train_and_evaluate_model(
#         feature_df, 
#         validation_year=validation_year
#     )

#     # Only attempt to make predictions if the model was successfully trained
#     if touchdown_model:
#         print("\nModel trained successfully. Proceeding to make predictions.")
#         predict_touchdown_scorers(feature_df, touchdown_model, model_features,
#                                   year=2024, week=1)
#     else:
#         print("\nSkipping prediction because model training failed.")

if __name__ == '__main__':
    # 1. Load all available historical data (2021 through the 2024 season)
    all_years_to_load = range(2021, 2025) 
    print(f"Loading all historical data for seasons: {list(all_years_to_load)}...")

    pbp = nfl.import_pbp_data(all_years_to_load, downcast=True)
    nfl_df = get_nfl_data(all_years_to_load)
    redzone_df = get_redzone_data(all_years_to_load, pbp)
    defense_df = get_opponent_data(all_years_to_load, pbp)
    redzone_td_df = get_redzone_td_rate(all_years_to_load, pbp)
    feature_df = feature_engineering(nfl_df, redzone_df, defense_df, redzone_td_df)
    print("Feature engineering complete.")

    # 2. Train the final model on the ENTIRE dataset
    # We no longer need the train_and_evaluate_model function for this
    print("Training final model on all historical data...")
    features = ['avg_carries', 'avg_rushing_yards', 'avg_receptions', 'avg_receiving_yards',
                'position_encoded', 'opponent_encoded', 'avg_wopr', 'avg_rushing_epa', 'avg_receiving_epa',
                'target_share', 'avg_touchdowns', 'redzone_rushes', 'redzone_receptions', 'passing_tds_allowed',
                'rushing_tds_allowed', 'total_tds_allowed', 'epa_allowed', 'redzone_td_rate']
    target = 'scored_touchdown'

    X_full = feature_df[features]
    y_full = feature_df[target]

    # Initialize and train the final model
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    final_model.fit(X_full, y_full)
    print("Model training complete.")

    # 3. Predict for Week 1 of the 2025 season
    predict_touchdown_scorers(feature_df, final_model, features,
                              year=2025, week=1)