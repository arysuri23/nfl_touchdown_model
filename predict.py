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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
# Suppress specific LightGBM warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')


# --- 4. Data Acquisition and Preprocessing ---

def get_nfl_data(years):
    """
    Fetches and preprocesses NFL data for a given list of years.
    """
    # Import weekly data for the specified years
    df = nfl.import_weekly_data(years, downcast=True)

    df = df[df['week'] <= 18]

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
def get_odds_data(years, team_map):
    """
    Loads and processes betting odds data from the local spreadspoke_scores.csv file,
    using the provided initial loading method.
    """
    # --- User's initial data loading ---
    print("Loading odds data from local CSV...")

    df_odds = pd.read_csv('spreadspoke_scores.csv', low_memory=False)
    df_odds = df_odds[['schedule_season', 'schedule_week', 'team_home', 'team_away',
                       'team_favorite_id', 'spread_favorite', 'over_under_line', 'schedule_playoff']]

    # --- Step 1: Data Cleaning and Filtering ---
    # Rename columns to match what our script expects
    df_odds.rename(columns={'schedule_season': 'season', 'schedule_week': 'week', 'over_under_line': 'total_line'}, inplace=True)

    # Filter for the years we need and for regular season games only
    df_odds = df_odds[df_odds['season'].isin(years)]
    df_odds = df_odds[df_odds['schedule_playoff'] == False]
    
    # Convert week to a number and drop rows with missing essential data
    df_odds['total_line'] = pd.to_numeric(df_odds['total_line'], errors='coerce')
    df_odds['spread_favorite'] = pd.to_numeric(df_odds['spread_favorite'], errors='coerce')
    df_odds['season'] = pd.to_numeric(df_odds['season'], errors='coerce')
    df_odds['week'] = pd.to_numeric(df_odds['week'], errors='coerce')
    df_odds.dropna(subset=['week', 'total_line', 'spread_favorite'], inplace=True)
    df_odds['week'] = df_odds['week'].astype(int)

    # --- Step 2: Reshape to Per-Team Data ---
    # Use the team_map to get team abbreviations for merging
    # Note: This assumes 'team_id' in your nfl_teams.csv is the abbreviation (e.g., 'KC')
    df_odds['home_team_abbr'] = df_odds['team_home'].map(team_map)

    # Determine the correct spread for the HOME team based on who the favorite is
    df_odds['home_spread'] = np.where(
        df_odds['team_favorite_id'] == df_odds['home_team_abbr'],
        df_odds['spread_favorite'],
        -df_odds['spread_favorite']
    )

    # Create home and away dataframes
    df_home = df_odds[['season', 'week', 'home_team_abbr', 'home_spread', 'total_line']].rename(
        columns={'home_team_abbr': 'team', 'home_spread': 'spread_line'})

    df_away = df_odds[['season', 'week', 'team_away', 'home_spread', 'total_line']].rename(
        columns={'team_away': 'team_full_name'})
    df_away['team'] = df_away['team_full_name'].map(team_map) # Map away team names to abbreviations

    # The away team's spread is the inverse of the home team's spread
    df_away['spread_line'] = -df_away['home_spread']
    df_away.drop(columns=['home_spread', 'team_full_name'], inplace=True)

    # Combine into a single dataframe with one row per team per game
    df_processed_odds = pd.concat([df_home, df_away]).dropna(subset=['team'])

    # --- Step 3: Final Feature Creation ---
    # Calculate the team's implied score total
    df_processed_odds['implied_total'] = (df_processed_odds['total_line'] / 2) - (df_processed_odds['spread_line'] / 2)

    return df_processed_odds

def get_redzone_data(years, pbp):
    """
    Calculates each player's share of their team's red zone carries and targets.
    """
    # Filter for red zone plays (inside the 20-yard line)
    redzone_df = pbp[pbp['yardline_100'] <= 20].copy()

    # --- Calculate Team-Level Red Zone Plays ---
    # Group by team, season, and week to get total red zone plays
    team_rz_plays = redzone_df.groupby(['posteam', 'season', 'week']).agg(
        team_rz_rushes=('rush_attempt', 'sum'),
        team_rz_targets=('pass_attempt', 'sum')
    ).reset_index()

    # --- Calculate Player-Level Red Zone Plays ---
    # Player rushes in the red zone
    player_rz_rushes = redzone_df.groupby(['rusher_player_id', 'posteam', 'season', 'week']).agg(
        player_rz_rushes=('rush_attempt', 'sum')
    ).reset_index().rename(columns={'rusher_player_id': 'player_id'})

    # Player targets in the red zone
    player_rz_targets = redzone_df.groupby(['receiver_player_id', 'posteam', 'season', 'week']).agg(
        player_rz_targets=('pass_attempt', 'sum')
    ).reset_index().rename(columns={'receiver_player_id': 'player_id'})

    # --- Merge and Calculate Shares ---
    # Merge player rushes and targets
    player_usage = pd.merge(
        player_rz_rushes,
        player_rz_targets,
        on=['player_id', 'posteam', 'season', 'week'],
        how='outer'
    )

    # Merge player usage with team totals
    final_rz_df = pd.merge(
        player_usage,
        team_rz_plays,
        on=['posteam', 'season', 'week'],
        how='left'
    )

    # Calculate the share, handling division by zero for teams with 0 RZ plays
    final_rz_df['redzone_carry_share'] = (final_rz_df['player_rz_rushes'] / final_rz_df['team_rz_rushes']).fillna(0)
    final_rz_df['redzone_target_share'] = (final_rz_df['player_rz_targets'] / final_rz_df['team_rz_targets']).fillna(0)

    # Select and return the final columns
    return final_rz_df[['player_id', 'season', 'week', 'redzone_carry_share', 'redzone_target_share']]

def get_goal_line_data(pbp):
    """
    Calculates each player's share of their team's carries and targets
    inside the 5-yard line (goal-to-go situations).
    """
    # Filter for plays inside the 5-yard line
    goal_line_df = pbp[pbp['yardline_100'] <= 5].copy()

    # --- Calculate Team-Level Goal-Line Plays ---
    team_gl_plays = goal_line_df.groupby(['posteam', 'season', 'week']).agg(
        team_gl_rushes=('rush_attempt', 'sum'),
        team_gl_targets=('pass_attempt', 'sum')
    ).reset_index()

    # --- Calculate Player-Level Goal-Line Plays ---
    player_gl_rushes = goal_line_df.groupby(['rusher_player_id', 'posteam', 'season', 'week']).agg(
        player_gl_rushes=('rush_attempt', 'sum')
    ).reset_index().rename(columns={'rusher_player_id': 'player_id'})

    player_gl_targets = goal_line_df.groupby(['receiver_player_id', 'posteam', 'season', 'week']).agg(
        player_gl_targets=('pass_attempt', 'sum')
    ).reset_index().rename(columns={'receiver_player_id': 'player_id'})

    # --- Merge and Calculate Shares ---
    player_usage = pd.merge(player_gl_rushes, player_gl_targets,
                            on=['player_id', 'posteam', 'season', 'week'], how='outer')

    final_gl_df = pd.merge(player_usage, team_gl_plays,
                           on=['posteam', 'season', 'week'], how='left')

    # Calculate the share, handling division by zero
    final_gl_df['inside_5_carry_share'] = (final_gl_df['player_gl_rushes'] / final_gl_df['team_gl_rushes']).fillna(0)
    final_gl_df['inside_5_target_share'] = (final_gl_df['player_gl_targets'] / final_gl_df['team_gl_targets']).fillna(0)

    return final_gl_df[['player_id', 'season', 'week', 'inside_5_carry_share', 'inside_5_target_share']]

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

def get_opponent_positional_data(pbp, rosters):
    """
    Calculates how many TDs each defense allows to specific offensive positions.
    """
    print("Calculating positional defensive vulnerabilities...")
    # We need player positions, so merge rosters into PBP data
    # We only need the position of the rusher and receiver
    roster_positions = rosters[['player_id', 'position']].drop_duplicates()

    # --- Rushing TDs Allowed by Position ---
    rush_tds = pbp[pbp['rush_touchdown'] == 1][['defteam', 'season', 'week', 'rusher_player_id']]
    rush_tds = pd.merge(rush_tds, roster_positions, left_on='rusher_player_id', right_on='player_id', how='left')
    
    # Group by defense, season, week, and the position of the player who scored
    rush_tds_allowed = rush_tds.groupby(['defteam', 'season', 'week', 'position']).size().unstack(fill_value=0)
    # Rename columns for clarity
    rush_tds_allowed.columns = [f'rushing_tds_allowed_to_{col}' for col in rush_tds_allowed.columns]
    
    # --- Passing TDs Allowed by Position ---
    pass_tds = pbp[pbp['pass_touchdown'] == 1][['defteam', 'season', 'week', 'receiver_player_id']]
    pass_tds = pd.merge(pass_tds, roster_positions, left_on='receiver_player_id', right_on='player_id', how='left')
    
    # Group and pivot
    pass_tds_allowed = pass_tds.groupby(['defteam', 'season', 'week', 'position']).size().unstack(fill_value=0)
    pass_tds_allowed.columns = [f'passing_tds_allowed_to_{col}' for col in pass_tds_allowed.columns]

    # --- Merge Rushing and Passing Data ---
    positional_defense_df = pd.merge(
        rush_tds_allowed,
        pass_tds_allowed,
        on=['defteam', 'season', 'week'],
        how='outer'
    ).fillna(0).reset_index()

    # Ensure all relevant columns exist, even if a position never scored
    for pos in ['RB', 'WR', 'TE', 'QB']:
        if f'rushing_tds_allowed_to_{pos}' not in positional_defense_df.columns:
            positional_defense_df[f'rushing_tds_allowed_to_{pos}'] = 0
        if f'passing_tds_allowed_to_{pos}' not in positional_defense_df.columns:
            positional_defense_df[f'passing_tds_allowed_to_{pos}'] = 0
            
    positional_defense_df.rename(columns={'defteam': 'opponent_team'}, inplace=True)
    
    # Select final columns to avoid clutter
    final_cols = ['opponent_team', 'season', 'week',
                  'rushing_tds_allowed_to_RB', 'rushing_tds_allowed_to_WR', 'rushing_tds_allowed_to_QB',
                  'passing_tds_allowed_to_RB', 'passing_tds_allowed_to_WR', 'passing_tds_allowed_to_TE']
                  
    return positional_defense_df[final_cols]
# --- 5. Feature Engineering ---

def feature_engineering(df, redzone_df, defense_df, redzone_td_rate, odds_df, goal_line_df, positional_defense_df):
    """
    Engineers features from the raw data to improve model performance.
    """

    # Merge red zone data in df
    df = pd.merge(df, redzone_df, on=['player_id', 'week', 'season'], how='left')
    # Merge opponent data
    df = pd.merge(df, defense_df, on=['opponent_team', 'season', 'week'], how='left')
    # Merge redzone TD rate
    df = pd.merge(df, redzone_td_rate, on=['recent_team', 'season', 'week'], how='left')
    # Merge odds data
    df = pd.merge(df, odds_df, left_on=['recent_team', 'season', 'week'],
                  right_on=['team', 'season', 'week'], how='left')
    # Merge goal line df
    df = pd.merge(df, goal_line_df, on=['player_id', 'week', 'season'], how='left')

    # Merge Positional defense df
    df = pd.merge(df, positional_defense_df, on=['opponent_team', 'season', 'week'], how='left')
    df.fillna(0, inplace=True)
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
    df['avg_redzone_carry_share'] = df.groupby('player_display_name')['redzone_carry_share'].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['avg_redzone_target_share'] = df.groupby('player_display_name')['redzone_target_share'].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['avg_inside_5_carry_share'] = df.groupby('player_display_name')['inside_5_carry_share'].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['avg_inside_5_target_share'] = df.groupby('player_display_name')['inside_5_target_share'].transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['opponent_team'] = df['opponent_team'].astype('category')
    df['position'] = df['position'].astype('category')
    # df['passing_tds_allowed'] = df['passing_tds_allowed'].groupby(df['opponent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    # df['rushing_tds_allowed'] = df['rushing_tds_allowed'].groupby(df['opponent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    # df['total_tds_allowed'] = df['total_tds_allowed'].groupby(df['opponent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['epa_allowed'] = df['epa_allowed'].groupby(df['opponent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    df['redzone_td_rate'] = df['redzone_td_rate'].groupby(df['recent_team']).transform(lambda x: x.shift(1).rolling(5, 1).mean())
    #df['rush_interaction'] = df['avg_redzone_carry_share'] * df['rushing_tds_allowed']
    #df['pass_interaction'] = df['avg_redzone_target_share'] * df['passing_tds_allowed']

     # --- NEW: Rolling averages for positional defense ---
    pos_defense_cols = [
        'rushing_tds_allowed_to_RB', 'rushing_tds_allowed_to_WR', 'rushing_tds_allowed_to_QB',
        'passing_tds_allowed_to_RB', 'passing_tds_allowed_to_WR', 'passing_tds_allowed_to_TE'
    ]
    for col in pos_defense_cols:
        df[col] = df.groupby('opponent_team')[col].transform(lambda x: x.shift(1).rolling(5, 1).mean())

    # --- NEW: Advanced Positional Interaction Feature ---
    # This feature combines a player's usage with the opponent's specific weakness against that player's position.
    
    # RBs get credit for both rushing and receiving matchup advantages
    rb_interaction = (df['avg_redzone_carry_share'] * df['rushing_tds_allowed_to_RB']) + \
                     (df['avg_redzone_target_share'] * df['passing_tds_allowed_to_RB'])
    
    # WRs and TEs are based on their receiving matchup
    wr_interaction = df['avg_redzone_target_share'] * df['passing_tds_allowed_to_WR']
    te_interaction = df['avg_redzone_target_share'] * df['passing_tds_allowed_to_TE']
    
    # QBs are mostly based on their rushing opportunity
    qb_interaction = df['avg_redzone_carry_share'] * df['rushing_tds_allowed_to_QB']
    
    # Combine into a single feature based on the player's position
    df['positional_matchup_interaction'] = np.select(
        [
            df['position'] == 'RB',
            df['position'] == 'WR',
            df['position'] == 'TE',
            df['position'] == 'QB'
        ],
        [
            rb_interaction,
            wr_interaction,
            te_interaction,
            qb_interaction
        ],
        default=0  # Default value if position is none of the above
    )

    # Fill NaN values resulting from rolling means
    df.fillna(0, inplace=True)


    # Encode categorical variables
    le = LabelEncoder()
    df['position_encoded'] = le.fit_transform(df['position'])
    df['opponent_encoded'] = le.fit_transform(df['opponent_team'])

    return df


# Add 'model' to the function signature
def train_and_evaluate_model(model, param_dist, df, features, validation_year=2024):
    """
    Trains a given model and evaluates its performance on the validation year.
    """
    model_name = model.__class__.__name__
    print(f"\n--- Tuning and Evaluating Model: {model_name} on {validation_year} Season ---")

    target = 'scored_touchdown'
    train_df = df[df['season'] < validation_year]
    test_df = df[df['season'] == validation_year]

    if test_df.empty:
        print(f"Error: No data found for the validation year {validation_year}.")
        return None

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # --- NEW: Hyperparameter Tuning with TimeSeriesSplit ---
    print("Performing hyperparameter tuning...")
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=25,  # Number of parameter settings that are sampled
        cv=tscv,
        scoring='f1',
        n_jobs=-1,  # Use all available cores
        random_state=42,
        verbose=1 # Set to 2 for more details
    )
    random_search.fit(X_train, y_train)

    print(f"\nBest Hyperparameters found for {model_name}:")
    print(random_search.best_params_)
    
    best_model = random_search.best_estimator_
    # --- End of New Tuning Section ---

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Find the optimal threshold for F1-score on the validation set
    best_threshold, best_f1 = 0, 0
    for threshold in np.arange(0.2, 0.7, 0.01):
        y_pred_loop = (y_pred_proba >= threshold).astype(int)
        current_f1 = f1_score(y_test, y_pred_loop, pos_label=1)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
    
    print(f"\nOptimal Threshold for {model_name}: {best_threshold:.2f} (Achieved F1-Score: {best_f1:.4f})")
    
    # Apply the best threshold to get the final classification report
    y_pred_final = (y_pred_proba >= best_threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final, target_names=['No TD', 'Scored TD']))

    print("\n" + "="*50)
    print("PART 3: FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Use the feature importances from the best model found during the search
    importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(importance.head(20)) # Print top 20 features

    # Return the best parameters to be used for the final model
    return random_search.best_params_

def predict_touchdown_scorers(feature_df, model, features, year, week, future_odds_df=None):
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
        'avg_touchdowns', 'avg_redzone_carry_share', 'avg_redzone_target_share',
        #'rush_interaction', 'pass_interaction',
        'position_encoded', 'avg_inside_5_carry_share', 'avg_inside_5_target_share', # A player's position is part of their history.
        # NEW: Advanced interaction feature
        'positional_matchup_interaction',
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
        #'passing_tds_allowed', 'rushing_tds_allowed', 
        'rushing_tds_allowed_to_RB', 'rushing_tds_allowed_to_WR', 'rushing_tds_allowed_to_QB',
        'passing_tds_allowed_to_RB', 'passing_tds_allowed_to_WR', 'passing_tds_allowed_to_TE',

        #'total_tds_allowed', 
        'epa_allowed', 'opponent_encoded'
    ]
    prediction_df = pd.merge(prediction_df, latest_team_stats[opponent_feature_cols], 
                             left_on='opponent_team', right_index=True, how='left',
                             suffixes=('', '_opponent'))



    if future_odds_df is not None:
        print("Using provided future odds to update prediction data...")
        # Drop the old, incorrect odds columns that came from the historical data
        prediction_df.drop(columns=['spread_line', 'total_line', 'implied_total'], inplace=True, errors='ignore')

        # Merge the new, correct odds for the upcoming games
        prediction_df = pd.merge(prediction_df, future_odds_df, on='team', how='left')

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
    
    # Get the model name for clear output
    model_name = model.__class__.__name__
    
    print(f'\n--- Making TD Predictions with {model_name} for Week {week}, {year} ---')
    week_probabilities = model.predict_proba(X_pred)[:, 1]

    prediction_df['predicted_touchdown_probability'] = week_probabilities

     # --- NEW: Add usage filter ---
    # Define minimum usage thresholds to be considered a relevant player.
    min_carries = 2.0
    min_receptions = 1.0
    
    # Keep players who meet at least one of the usage criteria.
    usage_filter = (prediction_df['avg_carries'] >= min_carries) | (prediction_df['avg_receptions'] >= min_receptions)
    filtered_predictions_df = prediction_df[usage_filter].copy()

    # --- Add model_name to the output filename ---
    
    output_cols = ['player_name', 'team', 'opponent_team', 'position', 'predicted_touchdown_probability']
    final_predictions = prediction_df[output_cols].sort_values(by='predicted_touchdown_probability', ascending=False)

    td_odds = pd.read_csv('week_1_td_odds.csv')


    td_odds['merge_name'] = td_odds['description'].str.lower() \
        .str.replace(r'[^a-z0-9\s]', '', regex=True) \
        .str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()

    # 2. Clean the predictions data
    final_predictions['merge_name'] = final_predictions['player_name'].str.lower() \
        .str.replace(r'[^a-z0-9\s]', '', regex=True) \
        .str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()

    # --- Convert odds to probability using the 'price' column ---
    # Using np.where for efficient conditional logic
    prob_if_pos = 100 / (td_odds['price'] + 100)
    prob_if_neg = abs(td_odds['price']) / (abs(td_odds['price']) + 100)
    td_odds['market_implied_prob'] = np.where(td_odds['price'] > 0, prob_if_pos, prob_if_neg)
    td_odds['market_implied_prob'] = np.where(td_odds['price'] == 0, 0, td_odds['market_implied_prob'])

    # --- Merge using a 'left' join to keep all model predictions ---
    td_odds_to_merge = td_odds[['merge_name', 'price', 'market_implied_prob']]
    final_predictions = pd.merge(final_predictions, td_odds_to_merge, on='merge_name', how='left')

    # --- Calculate the difference (edge) between your model and the market ---
    final_predictions['model_edge'] = final_predictions['predicted_touchdown_probability'] - final_predictions['market_implied_prob']

    # Clean up and reorder columns
    final_predictions.drop(columns=['merge_name'], inplace=True)

    ### drop rows with Na entries:
    final_predictions.dropna(inplace=True)
        
    #display_cols = ['player_name', 'team', 'position', 'predicted_touchdown_probability', 'market_implied_prob', 'model_edge', 'price']
    
    


    final_predictions.to_csv("final_predicitons_week_{week}.csv", index=False)
    print(f'\n--- Top 25 Predicted Touchdown Scorers ({model_name}) ---')
    print(final_predictions.head(25))
    return final_predictions


def transform_future_odds(df_future_raw, team_map):
    """
    Transforms a raw future odds CSV into the format needed for prediction.

    Args:
        df_future_raw (pd.DataFrame): The raw DataFrame read from your CSV.
        team_map (dict): A dictionary mapping full team names to abbreviations.

    Returns:
        pd.DataFrame: A cleaned DataFrame ready for the prediction function.
    """
    # Assuming column names based on your example
    
    df_future_raw = df_future_raw[['home_team', 'away_team', 'market',
                                   'point_1', 'label_2', 'odd_2', 'point_2', 'over/under']].copy()
    
    # Map full team names to the abbreviations used by your model
    df_future_raw['home_team'] = df_future_raw['home_team'].map(team_map)
    df_future_raw['away_team'] = df_future_raw['away_team'].map(team_map)

    # Separate the spread and total (over/under) data
    df_future_raw['point_1'] = pd.to_numeric(df_future_raw['point_1'], errors='coerce')
    df_future_raw['over_under'] = pd.to_numeric(df_future_raw['over/under'], errors='coerce')

    # --- Step 2: Consolidate Game Data using GroupBy ---
    # This is the new core logic. We group by each unique game and aggregate.
    # .first() will grab the valid value for spread and total from their respective rows.
    games_df = df_future_raw.groupby(['home_team', 'away_team']).agg(
        spread_line=('point_1', 'first'),
        total_line=('over_under', 'first')
    ).reset_index()


    home_teams = games_df[['home_team', 'away_team', 'spread_line', 'total_line']].rename(
        columns={'home_team': 'team', 'away_team': 'opponent'})

    away_teams = games_df[['away_team', 'home_team', 'spread_line', 'total_line']].rename(
        columns={'away_team': 'team', 'home_team': 'opponent'})
    
    # The away team's spread is the inverse of the home team's spread
    away_teams['spread_line'] = -away_teams['spread_line']

    # Combine into a single DataFrame
    df_final = pd.concat([home_teams, away_teams]).reset_index(drop=True)
    df_final.dropna(inplace=True)

    df_final['implied_total'] = (df_final['total_line'] / 2) - (df_final['spread_line'] / 2)
    
    return df_final

if __name__ == '__main__':
    # --- Data Preparation (For both Backtesting and Final Prediction) ---
    all_years_to_load = range(2021, 2025) # Includes 2021, 2022, 2023, 2024
    validation_year = 2024
    prediction_year = 2025
    prediction_week = 1

    nfl_teams = pd.read_csv('nfl_teams.csv')
    team_map = dict(zip(nfl_teams['team_name'], nfl_teams['team_id']))

    print(f"Loading data for seasons: {list(all_years_to_load)}...")
    pbp = nfl.import_pbp_data(all_years_to_load, downcast=True)
    pbp = pbp[pbp['week'] <= 18] # Filter PBP data for regular season

    rosters = nfl.import_seasonal_rosters(all_years_to_load)

    nfl_df = get_nfl_data(all_years_to_load) # Ensure your get_nfl_data also filters for week <= 18
    redzone_df = get_redzone_data(all_years_to_load, pbp)
    defense_df = get_opponent_data(all_years_to_load, pbp)
    redzone_td_df = get_redzone_td_rate(all_years_to_load, pbp)
    odds_df = get_odds_data(all_years_to_load, team_map)
    goal_line_df = get_goal_line_data(pbp)
    positional_defense_df = get_opponent_positional_data(pbp, rosters)

    feature_df = feature_engineering(nfl_df, redzone_df, defense_df, redzone_td_df, odds_df, goal_line_df, positional_defense_df)

    features = [
    # Player rolling averages (Correctly shifted)
    'avg_carries', 'avg_rushing_yards', 'avg_receptions', 'avg_receiving_yards',
    'avg_wopr', 'avg_rushing_epa', 'avg_receiving_epa', 'target_share', 
    'avg_touchdowns', 'avg_redzone_carry_share', 'avg_redzone_target_share',
    #'rush_interaction', 'pass_interaction', 
    'avg_inside_5_carry_share', 'avg_inside_5_target_share',
    
    # Opponent/Team rolling averages (Correctly shifted)
    #'passing_tds_allowed', 'rushing_tds_allowed', ''
    #'total_tds_allowed', 
    'epa_allowed', 'redzone_td_rate',
    
    'rushing_tds_allowed_to_RB', 'rushing_tds_allowed_to_WR', 'rushing_tds_allowed_to_QB',
    'passing_tds_allowed_to_RB', 'passing_tds_allowed_to_WR', 'passing_tds_allowed_to_TE',

    # NEW: Advanced interaction feature
    'positional_matchup_interaction',

    # Betting odds & Encoded features
    'implied_total', 
    #'spread_line', 'total_line',
    'position_encoded', 'opponent_encoded'
]

    
    print("Feature engineering complete.")

   # --- Part 1: Backtest and Tune Models on the 2024 Season ---
    print("\n" + "="*50)
    print(f"PART 1: TUNING AND BACKTESTING MODELS ON {validation_year} SEASON")
    print("="*50)
    
    # --- NEW: Define Hyperparameter Distributions for Randomized Search ---
    rf_param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 10, 15, 20, None],
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

    # Tune and Evaluate Random Forest
    rf_model_config = RandomForestClassifier(random_state=42, class_weight='balanced')
    best_rf_params = train_and_evaluate_model(rf_model_config, rf_param_dist, feature_df, features, validation_year)

    # Tune and Evaluate LightGBM
    lgbm_model_config = lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1)
    best_lgbm_params = train_and_evaluate_model(lgbm_model_config, lgbm_param_dist, feature_df, features, validation_year)

    # --- NEW: Define and Evaluate the Stacking Classifier on the Validation Set ---
    print("\n" + "="*50)
    print(f"PART 2: EVALUATING STACKING ENSEMBLE ON {validation_year} SEASON")
    print("="*50)

    # Define the base models with their best found parameters
    estimators = []
    if best_rf_params:
        estimators.append(('rf', RandomForestClassifier(random_state=42, class_weight='balanced', **best_rf_params)))
    if best_lgbm_params:
        estimators.append(('lgbm', lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1, **best_lgbm_params)))

    if estimators:
        # Define the Stacking model that combines them
        # The 'cv' parameter is removed to use the default 5-fold partition
        stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

        # Train and evaluate the stacking model on the validation set
        train_df = feature_df[feature_df['season'] < validation_year]
        test_df = feature_df[feature_df['season'] == validation_year]
        X_train_stack = train_df[features]
        y_train_stack = train_df['scored_touchdown']
        X_test_stack = test_df[features]
        y_test_stack = test_df['scored_touchdown']

        print("Training Stacking Ensemble on training data...")
        stacking_model.fit(X_train_stack, y_train_stack)
        print("Stacking Ensemble Training Complete.")
        
        y_pred_proba_stack = stacking_model.predict_proba(X_test_stack)[:, 1]
        best_threshold, best_f1 = 0, 0
        for threshold in np.arange(0.2, 0.7, 0.01):
            y_pred_loop = (y_pred_proba_stack >= threshold).astype(int)
            current_f1 = f1_score(y_test_stack, y_pred_loop, pos_label=1)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        
        print(f"\nOptimal Threshold for Stacking Ensemble: {best_threshold:.2f} (Achieved F1-Score: {best_f1:.4f})")
        y_pred_final_stack = (y_pred_proba_stack >= best_threshold).astype(int)
        print("\nStacking Ensemble Classification Report:")
        print(classification_report(y_test_stack, y_pred_final_stack, target_names=['No TD', 'Scored TD']))

    # --- Part 3: Train Final Models with Best Parameters and Predict 2025 ---
    print("\n" + "="*50)
    print("PART 3: TRAINING FINAL MODELS & PREDICTING WEEK 1, 2025")
    print("="*50)

    target = 'scored_touchdown'
    X_full = feature_df[features]
    y_full = feature_df[target]

    future_odds_raw = pd.read_csv('week_1_lines.csv')
    future_odds_df = transform_future_odds(future_odds_raw, team_map)
    print("Future odds data loaded and transformed successfully.")

    # --- Train and Predict with Final STACKING Model ---
    if estimators:
        print("\n--- Training Final Stacking Ensemble on All Data ---")
        # The 'cv' parameter is also removed here
        final_stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        final_stacking_model.fit(X_full, y_full)
        print("Stacking Model Training Complete.")
        predict_touchdown_scorers(feature_df, final_stacking_model, features,
                                  year=prediction_year, week=prediction_week, future_odds_df=future_odds_df)