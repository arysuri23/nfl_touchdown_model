# NFL Touchdown Scorer Prediction Model
# Final Version with Position-Specific Models & Manual Time-Series Stacking


# --- 1. Importing Libraries ---
import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
 
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss
import data_collection as data
import joblib
import os
import sys
import json



### CONSTANTS ###

# -- Position-Specific Feature Lists --

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
    'avg_target_share', # Consider testing this uncommented. It's a fundamental metric and might offer value alongside wopr.

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
    'avg_scored_touchdown', 
    'avg_redzone_carry_share', 'avg_inside_5_carry_share',
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
    'reg_lambda': [0, 0.1, 0.5],
    # Expanded search space for better generalization on imbalanced data
    'min_child_samples': [20, 50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'subsample_freq': [1],
    'colsample_bytree': [0.6, 0.8, 1.0]
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

    # Ensure strict chronological ordering per player before lag/EWM to avoid leakage
    df.sort_values(by=['player_id', 'season', 'week'], inplace=True, ignore_index=True)
    
    
 

    player_stats = ['carries', 'rushing_yards', 'receptions', 'receiving_yards', 'wopr', 'rushing_epa', 'receiving_epa', 'target_share',
                      'receiving_air_yards', 'racr', 'scored_touchdown', 'redzone_carry_share', 'redzone_target_share',
                      'endzone_targets', 'endzone_target_share', 'inside_5_carry_share', 'inside_5_target_share', 'offense_snap_share', 
                      'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 'avg_time_to_los', 'percent_attempts_gte_eight_defenders',
                      'avg_cushion', 'avg_separation', 'avg_intended_air_yards', 'percent_share_of_intended_air_yards', 
                    'catch_percentage', 'avg_expected_yac', 'avg_yac_above_expectation']
    
    for stat in player_stats:
        df[f'avg_{stat}'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())
   
    pos_defense_cols = [col for col in df.columns if 'tds_allowed_to' in col]

    opponent_stats_df = (
        df[['season', 'week', 'opponent_team'] + pos_defense_cols]
          .groupby(['season', 'week', 'opponent_team'], as_index=False)[pos_defense_cols]
          .mean()
    )
    # Ensure chronological order within each opponent for lag/EWM
    opponent_stats_df.sort_values(by=['opponent_team', 'season', 'week'], inplace=True)
    
    for col in pos_defense_cols:
         opponent_stats_df[col] = opponent_stats_df.groupby('opponent_team')[col].transform(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())


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
    



# --- Week-Grouped Expanding Time-Series CV Utilities ---
def _get_ordered_unique_weeks(df_like: pd.DataFrame):
    """Return a sorted list of unique (season, week) tuples present in df_like."""
    unique_weeks = (
        df_like[['season', 'week']]
        .drop_duplicates()
        .sort_values(by=['season', 'week'])
    )
    return list(unique_weeks.itertuples(index=False, name=None))


def _map_week_to_row_positions(df_like: pd.DataFrame):
    """Map each (season, week) tuple to the row positions (0..N-1) belonging to that week."""
    df_reset = df_like.reset_index(drop=True)
    week_to_positions = {}
    for pos, (season, week) in enumerate(zip(df_reset['season'].values, df_reset['week'].values)):
        week_key = (int(season), int(week))
        if week_key not in week_to_positions:
            week_to_positions[week_key] = []
        week_to_positions[week_key].append(pos)
    return week_to_positions


def build_week_splits_from_df(
    df_like: pd.DataFrame,
    n_splits: int = 5,
    test_weeks: int = 1,
    embargo_weeks: int = 0,
    min_train_weeks: int = 8,
):
    """
    Build week-grouped expanding CV splits over df_like rows (assumes columns 'season' and 'week').

    - Keeps entire (season, week) blocks together in either train or test.
    - Train grows from the start; test is the next contiguous block of size `test_weeks`.
    - Optional `embargo_weeks` gap between the end of train and start of test.
    - Ensures at least `min_train_weeks` weeks in train.

    Returns a list of (train_idx, test_idx) tuples where indices are row positions (0..N-1).
    """
    df_reset = df_like[['season', 'week']].reset_index(drop=True).copy()
    week_to_positions = _map_week_to_row_positions(df_reset)
    ordered_weeks = _get_ordered_unique_weeks(df_reset)

    num_weeks = len(ordered_weeks)
    if num_weeks < (min_train_weeks + test_weeks):
        raise ValueError("Not enough weeks to construct the requested CV splits.")

    first_valid_test_start = min_train_weeks + embargo_weeks
    last_valid_test_start = num_weeks - test_weeks
    if first_valid_test_start > last_valid_test_start:
        raise ValueError("Embargo/min_train/test_weeks settings leave no valid test window.")

    candidate_test_starts = list(range(first_valid_test_start, last_valid_test_start + 1))
    if n_splits > len(candidate_test_starts):
        n_splits = len(candidate_test_starts)
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")

    if n_splits == len(candidate_test_starts):
        selected_indices = list(range(len(candidate_test_starts)))
    else:
        selected_indices = sorted(set(np.linspace(0, len(candidate_test_starts) - 1, num=n_splits).round().astype(int).tolist()))
        while len(selected_indices) > n_splits:
            selected_indices.pop(-1)
        while len(selected_indices) < n_splits:
            selected_indices.append(selected_indices[-1])

    splits = []
    for idx in selected_indices:
        test_start_week_idx = candidate_test_starts[idx]
        train_end_week_idx_exclusive = test_start_week_idx - embargo_weeks
        test_end_week_idx_exclusive = test_start_week_idx + test_weeks

        train_weeks = ordered_weeks[:train_end_week_idx_exclusive]
        test_weeks_block = ordered_weeks[test_start_week_idx:test_end_week_idx_exclusive]

        train_positions = []
        for wk in train_weeks:
            train_positions.extend(week_to_positions[wk])

        test_positions = []
        for wk in test_weeks_block:
            test_positions.extend(week_to_positions[wk])

        splits.append((np.array(train_positions, dtype=int), np.array(test_positions, dtype=int)))

    return splits


def print_week_split_diagnostics(df_like: pd.DataFrame, splits):
    """Print summary of week boundaries for each split to visually verify no leakage."""
    df_reset = df_like[['season', 'week']].reset_index(drop=True).copy()

    def idx_to_week_bounds(idxs: np.ndarray):
        if idxs.size == 0:
            return None, None
        weeks_present = df_reset.loc[idxs, ['season', 'week']].drop_duplicates().apply(tuple, axis=1).tolist()
        weeks_present_sorted = sorted(weeks_present)
        return weeks_present_sorted[0], weeks_present_sorted[-1]

    print("\nCV fold diagnostics (train_end_week -> test_range):")
    for i, (tr, te) in enumerate(splits, start=1):
        _, train_end = idx_to_week_bounds(tr)
        test_start, test_end = idx_to_week_bounds(te)
        print(f"  Fold {i}: train_end={train_end}  |  test={test_start}..{test_end}")


def build_week_splits_covering_all_weeks(
    df_like: pd.DataFrame,
    test_weeks: int = 1,
    embargo_weeks: int = 0,
    min_train_weeks: int = 8,
):
    """
    Build week-grouped expanding CV splits that COVER ALL valid test weeks.

    Returns a list of (train_idx, test_idx) per contiguous test window so that
    every week from the first valid test start to the last possible test end
    appears in exactly one test fold (given test_weeks windowing).
    """
    df_reset = df_like[['season', 'week']].reset_index(drop=True).copy()
    week_to_positions = _map_week_to_row_positions(df_reset)
    ordered_weeks = _get_ordered_unique_weeks(df_reset)

    num_weeks = len(ordered_weeks)
    if num_weeks < (min_train_weeks + test_weeks):
        raise ValueError("Not enough weeks to construct the requested CV splits.")

    first_valid_test_start = min_train_weeks + embargo_weeks
    last_valid_test_start = num_weeks - test_weeks
    if first_valid_test_start > last_valid_test_start:
        raise ValueError("Embargo/min_train/test_weeks settings leave no valid test window.")

    splits = []
    for test_start_week_idx in range(first_valid_test_start, last_valid_test_start + 1):
        train_end_week_idx_exclusive = test_start_week_idx - embargo_weeks
        test_end_week_idx_exclusive = test_start_week_idx + test_weeks

        train_weeks = ordered_weeks[:train_end_week_idx_exclusive]
        test_weeks_block = ordered_weeks[test_start_week_idx:test_end_week_idx_exclusive]

        train_positions = []
        for wk in train_weeks:
            train_positions.extend(week_to_positions[wk])

        test_positions = []
        for wk in test_weeks_block:
            test_positions.extend(week_to_positions[wk])

        splits.append((np.array(train_positions, dtype=int), np.array(test_positions, dtype=int)))

    return splits


def assert_week_splits_valid(df_like: pd.DataFrame, splits):
    """Raise AssertionError if any split leaks: overlapping weeks or train not strictly before test."""
    df_reset = df_like[['season', 'week']].reset_index(drop=True).copy()

    def weeks_of(idxs: np.ndarray):
        if idxs.size == 0:
            return []
        return df_reset.loc[idxs, ['season', 'week']].drop_duplicates().apply(tuple, axis=1).tolist()

    def week_key(w):
        # (season, week) -> comparable key
        return (int(w[0]), int(w[1]))

    for i, (tr, te) in enumerate(splits, start=1):
        train_weeks = weeks_of(tr)
        test_weeks = weeks_of(te)
        # No overlap
        assert set(train_weeks).isdisjoint(set(test_weeks)), f"Overlap in fold {i}: {set(train_weeks) & set(test_weeks)}"
        if train_weeks and test_weeks:
            max_train = max(train_weeks, key=week_key)
            min_test = min(test_weeks, key=week_key)
            # Strictly earlier
            assert week_key(max_train) < week_key(min_test), f"Temporal order violated in fold {i}: train_end {max_train} !< test_start {min_test}"


# --- Hyperparameter Persistence Helpers ---
def save_best_params(key: str, rf_params: dict, lgbm_params: dict, feature_names: list):
    """Save tuned params and feature names to models/{key}_best_params.json."""
    os.makedirs('models', exist_ok=True)
    payload = {
        'rf_params': rf_params,
        'lgbm_params': lgbm_params,
        'feature_names': list(feature_names),
    }
    with open(f'models/{key}_best_params.json', 'w') as f:
        json.dump(payload, f, indent=2)


def load_best_params(key: str):
    """Load tuned params and feature names from models/{key}_best_params.json."""
    with open(f'models/{key}_best_params.json', 'r') as f:
        data = json.load(f)
    return data['rf_params'], data['lgbm_params'], data['feature_names']

# --- 4. Position-Specific Model Training ---


###
### NEW: MANUAL TIME-SERIES STACKING IMPLEMENTATION
###
def train_stacked_model_timeseries(X, y, base_estimators, meta_estimator, n_splits=5, cv_splits=None):
    """
    Trains a stacked model using time-series cross-validation to generate meta-features.


    Returns:
        - A list of base estimators trained on the full dataset.
        - The meta-estimator trained on the out-of-fold predictions.
    """
    print("Generating out-of-fold predictions for meta-model training...")
    # Initialize an array for meta-features, with one column per base estimator
    meta_features = np.full((len(X), len(base_estimators)), np.nan)
    
    # Use provided week-grouped CV splits to respect chronological order
    if cv_splits is None:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        split_iter = tscv.split(X)
    else:
        split_iter = cv_splits
    
    # Track which rows received OOF predictions
    has_prediction_mask = np.zeros(len(X), dtype=bool)
    for i, (train_index, test_index) in enumerate(split_iter):
            
        # For each base model, fit on past data and predict on future data
        for j, estimator in enumerate(base_estimators):
            # Clone the estimator to ensure it's fresh for each fold
            model = clone(estimator)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            predictions = model.predict_proba(X.iloc[test_index])[:, 1]
            meta_features[test_index, j] = predictions
        has_prediction_mask[test_index] = True


    # Keep only rows where all base models produced an OOF prediction (no NaNs)
    row_has_all_models = ~np.any(np.isnan(meta_features), axis=1)
    valid_mask = has_prediction_mask & row_has_all_models
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        raise ValueError("No valid OOF meta-features were generated. Check CV splits.")
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


    # Ensure global chronological order across players for proper time-based splits
    df_position = df_position.sort_values(['season', 'week', 'player_id']).copy()

    # --- 1. Split data for hyperparameter tuning ---
    train_df = df_position[df_position['season'] < validation_year]
    train_df = train_df.sort_values(['season', 'week', 'player_id']).copy()
    X_train = train_df[features]
    y_train = train_df['scored_touchdown']

    # No feature scaling for tree-based base models

    # Build week-grouped CV splits for tuning on training data only
    cv_splits = build_week_splits_from_df(train_df, n_splits=5, test_weeks=1, embargo_weeks=0, min_train_weeks=8)
    print_week_split_diagnostics(train_df, cv_splits)
    assert_week_splits_valid(train_df, cv_splits)


    # --- 2. Tune RandomForest ---
    print(f"\nTuning RandomForest for {model_type}s...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_dist, n_iter=25, cv=cv_splits, scoring='average_precision', n_jobs=-1, random_state=42)
    rf_search.fit(X_train, y_train)
    best_rf_params = rf_search.best_params_
    print(f"Best RF Params: {best_rf_params}")


    # --- 3. Tune LightGBM ---
    print(f"\nTuning LightGBM for {model_type}s...")
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1)
    lgbm_search = RandomizedSearchCV(estimator=lgbm, param_distributions=lgbm_param_dist, n_iter=25, cv=cv_splits, scoring='average_precision', n_jobs=-1, random_state=42)
    lgbm_search.fit(X_train, y_train)
    best_lgbm_params = lgbm_search.best_params_
    print(f"Best LGBM Params: {best_lgbm_params}")


    # --- 4. Train Final Stacked Model using Manual Time-Series Logic ---
    print(f"\nTraining {model_type} Stacked Model for validation...")
    base_estimators = [
        RandomForestClassifier(random_state=42, class_weight='balanced', **best_rf_params),
        lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1, **best_lgbm_params)
    ]
    meta_estimator = LogisticRegression(class_weight='balanced', penalty='l2', C=0.5)
    
    # This trains the models needed for evaluation on the validation set
    # Use denser week-coverage for OOF stacking to maximize meta-model training data
    oof_splits = build_week_splits_covering_all_weeks(train_df, test_weeks=1, embargo_weeks=0, min_train_weeks=8)
    assert_week_splits_valid(train_df, oof_splits)
    base_models, meta_model = train_stacked_model_timeseries(X_train, y_train, base_estimators, meta_estimator, n_splits=5, cv_splits=oof_splits)
        
    # Persist tuned params and features per position key
    key = 'rb'
    if model_type == 'WR/TE':
        key = 'wr_te'
    elif model_type == 'QB':
        key = 'qb'
    try:
        save_best_params(key, best_rf_params, best_lgbm_params, features)
        print(f"Saved best hyperparameters to models/{key}_best_params.json")
    except Exception as e:
        print(f"Warning: could not save best params for {model_type}: {e}")

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


def fit_platt_calibrator(y_true: pd.Series, y_proba: np.ndarray) -> LogisticRegression:
    """Fits a simple Platt (logistic) calibrator on validation probabilities."""
    calibrator = LogisticRegression(penalty='l2', C=1.0)
    calibrator.fit(y_proba.reshape(-1, 1), y_true.values)
    return calibrator


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


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Compute ECE (Expected Calibration Error) with equal-width bins in [0,1]."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    calib_table = []
    for b in range(n_bins):
        in_bin = bin_ids == b
        if not np.any(in_bin):
            calib_table.append({'bin': b, 'count': 0, 'avg_pred': np.nan, 'emp_rate': np.nan, 'abs_gap': np.nan})
            continue
        avg_pred = y_prob[in_bin].mean()
        emp_rate = y_true[in_bin].mean()
        weight = in_bin.mean()
        ece += weight * abs(emp_rate - avg_pred)
        calib_table.append({'bin': b, 'count': int(in_bin.sum()), 'avg_pred': avg_pred, 'emp_rate': emp_rate, 'abs_gap': abs(emp_rate - avg_pred)})
    calib_df = pd.DataFrame(calib_table)
    return ece, calib_df


def evaluate_probability_quality(y_true: pd.Series, y_prob: np.ndarray, label: str = ""):
    """Compute Brier score, log loss, ECE and return a dict plus calibration table."""
    metrics = {}
    try:
        metrics['brier'] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        metrics['brier'] = np.nan
    try:
        # add a small epsilon clamp to avoid log(0)
        eps = 1e-15
        metrics['log_loss'] = float(log_loss(y_true, np.clip(y_prob, eps, 1 - eps)))
    except Exception:
        metrics['log_loss'] = np.nan
    ece, calib_df = expected_calibration_error(y_true.values, y_prob, n_bins=10)
    metrics['ece'] = float(ece)
    if label:
        print(f"\n--- Probability Quality ({label}) ---")
    else:
        print("\n--- Probability Quality ---")
    print({k: round(v, 4) if v == v else v for k, v in metrics.items()})
    print("Calibration table (first 10 bins):")
    print(calib_df.round(3))
    return metrics, calib_df


def evaluate_model_at_50_threshold(predictions_df: pd.DataFrame):
    """
    Evaluates model performance based on a fixed 50% probability threshold.
    Any player with a predicted probability > 0.5 is considered a 'positive' prediction.
    Calculates precision, recall, and F1-score on a weekly and seasonal basis.
    """
    print("\n" + "="*60 + "\nEVALUATION AT 50% PROBABILITY THRESHOLD\n" + "="*60)

    weekly_results = []
    
    # Ensure prediction column exists
    if 'predicted_prob' not in predictions_df.columns:
        print("Error: 'predicted_prob' column not found.")
        return

    for week in sorted(predictions_df['week'].unique()):
        week_df = predictions_df[predictions_df['week'] == week].copy()
        
        # Define predictions and actuals based on the threshold
        predicted_scorers = week_df['predicted_prob'] > 0.5
        actual_scorers = week_df['scored_touchdown'] == 1
        
        # Calculate confusion matrix components
        tp = (predicted_scorers & actual_scorers).sum()
        fp = (predicted_scorers & ~actual_scorers).sum()
        fn = (~predicted_scorers & actual_scorers).sum()
        
        # Calculate metrics, handling division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        weekly_results.append({
            'week': week,
            'scorers_predicted': tp + fp,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })

    # Create and display results
    results_summary = pd.DataFrame(weekly_results)
    print("\n--- Weekly Performance Summary ---")
    print(results_summary.round(3))
    
    # Calculate and display overall season averages
    average_performance = results_summary.drop(columns=['week']).mean()
    print("\n--- Average Season Performance ---")
    print(f"Average Scorers Predicted per Week: {average_performance['scorers_predicted']:.2f}")
    print(f"Average True Positives per Week:    {average_performance['true_positives']:.2f}")
    print(f"Average Precision:                  {average_performance['precision']:.3f}")
    print(f"Average Recall:                     {average_performance['recall']:.3f}")
    print(f"Average F1-Score:                   {average_performance['f1_score']:.3f}")
    
    return results_summary


def precision_recall_at_k_sweep(predictions_df: pd.DataFrame, k_values):
    """Return a dataframe with precision/recall@k for a list of k's (weekly averaged)."""
    rows = []
    for k in k_values:
        wk = evaluate_model_at_k(predictions_df, k=k)
        avg = wk.mean()
        rows.append({'k': k, 'precision_at_k': avg['precision_at_k'], 'recall_at_k': avg['recall_at_k'], 'successful_picks': avg['successful_picks']})
    sweep_df = pd.DataFrame(rows)
    print("\n--- Precision/Recall@K Sweep ---")
    print(sweep_df.round(3))
    return sweep_df


def evaluate_specialist_model(base_models, meta_model, model_name, validation_df, features, k=25):
    """Calculates performance metrics for a manually stacked model."""
    print("\n" + "="*60 + f"\nEVALUATION FOR: {model_name}\n" + "="*60)
    if validation_df.empty:
        print("Validation data is empty. Skipping evaluation.")
        return

    # Prepare X_val from the validation dataframe
    X_val = validation_df[features].copy() # Use .copy() to avoid SettingWithCopyWarning

    # No scaling required for tree-based base models

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

    # --- 3. Probability Quality ---
    evaluate_probability_quality(validation_df['scored_touchdown'], y_pred_proba, label=f"{model_name}")


###
### UPDATED: RETRAINING FUNCTION FOR MANUAL STACKING
###
def train_model_on_all_data(df_position, features, best_rf_params, best_lgbm_params):
    """Retrains a final stacked model on all data using the best hyperparameters."""
    model_type = df_position['position'].unique()[0]
    if len(df_position['position'].unique()) > 1: model_type = "WR/TE"
    print(f"\nRetraining final {model_type} model on all data (2020-2024)...")


    # Ensure global chronological order across players before final training
    df_position = df_position.sort_values(['season', 'week', 'player_id']).copy()

    X_full = df_position[features]
    y_full = df_position['scored_touchdown']

    # No feature scaling for tree-based base models
    


    base_estimators = [
        RandomForestClassifier(random_state=42, class_weight='balanced', **best_rf_params),
        lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1, **best_lgbm_params)
    ]
    meta_estimator = LogisticRegression(class_weight='balanced', penalty='l2', C=0.5)


    # Use the same robust training logic on the full dataset
    final_base_models, final_meta_model = train_stacked_model_timeseries(X_full, y_full, base_estimators, meta_estimator)


    print(f"{model_type} retraining complete.")
    return final_base_models, final_meta_model



 


# --- Local File Saving Helper Functions ---
def save_joblib_locally(python_object, file_path):
    """
    Serializes a Python object with joblib and saves it locally.
    """
    print(f"Saving model artifact to '{file_path}'...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(python_object, file_path)
        print("Save successful.")
        return True
    except Exception as e:
        print(f"An error occurred during local save: {e}")
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
    nfl_df = nfl_df[nfl_df['week'] <= 18]
    nfl_df = nfl_df[nfl_df['season'] < 2025]
    nfl_df.to_csv("raw_nfl_data.csv", index=False)
    feature_df = feature_engineering(nfl_df)
    #feature_df = feature_engineering(nfl_df, redzone_df, redzone_td_df, ez_target_df, odds_df, goal_line_df, positional_defense_df, depth_chart_df, snap_counts_df, ngs_rushing_df, ngs_receiving_df)
    feature_df.to_csv("feature_df.csv", index=False)

    

    


    df_rb = feature_df[feature_df['position'] == 'RB'].copy()
    df_wr_te = feature_df[feature_df['position'].isin(['WR', 'TE'])].copy()
    df_qb = feature_df[feature_df['position'] == 'QB'].copy()


    # Toggle: reuse saved best params (weekly retrains) vs. re-tune
    USE_SAVED_PARAMS = True

    if not USE_SAVED_PARAMS:
        # --- Phase 1: Tune, Train, and Evaluate on 2024 Season ---
        # The function now returns the trained base/meta models and the best params
        rb_models, rb_meta_model, rb_rf_params, rb_lgbm_params = tune_and_train_specialist_model(df_rb, RB_FEATURES, RF_PARAM_DIST, LGBM_PARAM_DIST)
        wr_te_models, wr_te_meta_model, wr_te_rf_params, wr_te_lgbm_params = tune_and_train_specialist_model(df_wr_te, WR_TE_FEATURES, RF_PARAM_DIST, LGBM_PARAM_DIST)
        qb_models, qb_meta_model, qb_rf_params, qb_lgbm_params = tune_and_train_specialist_model(df_qb, QB_FEATURES, RF_PARAM_DIST, LGBM_PARAM_DIST)
    else:
        print("\n" + "="*60 + "\nLOADING SAVED BEST HYPERPARAMETERS\n" + "="*60)
        rb_rf_params, rb_lgbm_params, _ = load_best_params('rb')
        wr_te_rf_params, wr_te_lgbm_params, _ = load_best_params('wr_te')
        qb_rf_params, qb_lgbm_params, _ = load_best_params('qb')

        # Train models for validation using loaded params (no re-tuning)
        validation_year = 2024

        def train_for_validation(df_pos, features, rf_params, lgbm_params):
            df_pos = df_pos.sort_values(['season', 'week', 'player_id']).copy()
            train_df = df_pos[df_pos['season'] < validation_year].sort_values(['season', 'week', 'player_id']).copy()
            X_train = train_df[features]
            y_train = train_df['scored_touchdown']
            base_estimators = [
                RandomForestClassifier(random_state=42, class_weight='balanced', **rf_params),
                lgb.LGBMClassifier(objective='binary', random_state=42, is_unbalance=True, verbosity=-1, **lgbm_params)
            ]
            meta_estimator = LogisticRegression(class_weight='balanced', penalty='l2', C=0.5)
            oof_splits = build_week_splits_covering_all_weeks(train_df, test_weeks=1, embargo_weeks=0, min_train_weeks=8)
            assert_week_splits_valid(train_df, oof_splits)
            return train_stacked_model_timeseries(X_train, y_train, base_estimators, meta_estimator, cv_splits=oof_splits)

        rb_models, rb_meta_model = train_for_validation(df_rb, RB_FEATURES, rb_rf_params, rb_lgbm_params)
        wr_te_models, wr_te_meta_model = train_for_validation(df_wr_te, WR_TE_FEATURES, wr_te_rf_params, wr_te_lgbm_params)
        qb_models, qb_meta_model = train_for_validation(df_qb, QB_FEATURES, qb_rf_params, qb_lgbm_params)


    # --- MODEL EVALUATION ON 2024 SEASON ---
    validation_year = 2024
    validation_df = feature_df[feature_df['season'] == validation_year]
    val_df_rb = validation_df[validation_df['position'] == 'RB'].copy()
    val_df_wr_te = validation_df[validation_df['position'].isin(['WR', 'TE'])].copy()
    val_df_qb = validation_df[validation_df['position'] == 'QB'].copy()

    # Evaluate and collect validation probabilities for calibration
    print("\nCollecting validation probabilities for calibration...")
    rb_val_proba = predict_stacked_proba(val_df_rb[RB_FEATURES], rb_models, rb_meta_model)
    wr_te_val_proba = predict_stacked_proba(val_df_wr_te[WR_TE_FEATURES], wr_te_models, wr_te_meta_model)
    qb_val_proba = predict_stacked_proba(val_df_qb[QB_FEATURES], qb_models, qb_meta_model)

    evaluate_specialist_model(rb_models, rb_meta_model, "RB Model", val_df_rb, RB_FEATURES, k=15)
    evaluate_specialist_model(wr_te_models, wr_te_meta_model, "WR/TE Model", val_df_wr_te, WR_TE_FEATURES)
    evaluate_specialist_model(qb_models, qb_meta_model, "QB Model", val_df_qb, QB_FEATURES, k=5)

    



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
    unified_weekly_performance = evaluate_model_at_k(combined_results_df, k=16)
    print("\n--- Weekly Performance @ K=16 (All Positions) ---")
    print(unified_weekly_performance)

    average_performance = unified_weekly_performance.mean()
    print("\n--- Average Season Performance (All Positions) ---")
    print(f"Average Precision@16: {average_performance['precision_at_k']:.3f}")
    print(f"Average Recall@16:    {average_performance['recall_at_k']:.3f}")
    print(f"Average Successful Picks Per Week: {average_performance['successful_picks']:.1f}")

    # Probability quality for unified predictions
    evaluate_probability_quality(combined_results_df['scored_touchdown'], combined_results_df['predicted_prob'].values, label="Unified (All Positions)")


    #evaluate_model_at_50_threshold(combined_results_df)
    
  
   
    # --- Phase 2: Fit Platt calibrators on 2024 validation and retrain final models ---
    print("\n" + "="*60 + "\nRETRAINING FINAL MODELS ON ALL HISTORICAL DATA FOR PREDICTION\n" + "="*60)
    rb_base_final, rb_meta_final = train_model_on_all_data(df_rb, RB_FEATURES, rb_rf_params, rb_lgbm_params)
    wr_te_base_final, wr_te_meta_final = train_model_on_all_data(df_wr_te, WR_TE_FEATURES, wr_te_rf_params, wr_te_lgbm_params)
    qb_base_final, qb_meta_final = train_model_on_all_data(df_qb, QB_FEATURES, qb_rf_params, qb_lgbm_params)

    print("\n" + "="*60 + "\nFITTING CALIBRATORS ON 2024 VALIDATION\n" + "="*60)
    # Fit calibrators directly on validation probabilities as before
    rb_calibrator = fit_platt_calibrator(val_df_rb['scored_touchdown'], rb_val_proba)
    wr_te_calibrator = fit_platt_calibrator(val_df_wr_te['scored_touchdown'], wr_te_val_proba)
    qb_calibrator = fit_platt_calibrator(val_df_qb['scored_touchdown'], qb_val_proba)

    print("\n" + "="*60 + "\nSAVING MODEL ARTIFACTS LOCALLY\n" + "="*60)
    # Save RB models
    save_joblib_locally(rb_base_final, 'models/rb_base_final.pkl')
    save_joblib_locally(rb_meta_final, 'models/rb_meta_final.pkl')
    # Save WR/TE models
    save_joblib_locally(wr_te_base_final, 'models/wr_te_base_final.pkl')
    save_joblib_locally(wr_te_meta_final, 'models/wr_te_meta_final.pkl')
    # Save QB models
    save_joblib_locally(qb_base_final, 'models/qb_base_final.pkl')
    save_joblib_locally(qb_meta_final, 'models/qb_meta_final.pkl')
    # Save calibrators
    save_joblib_locally(rb_calibrator, 'models/rb_calibrator.pkl')
    save_joblib_locally(wr_te_calibrator, 'models/wr_te_calibrator.pkl')
    save_joblib_locally(qb_calibrator, 'models/qb_calibrator.pkl')

    # No scaler artifacts to save

    print("\n" + "="*60 + "\nOFFLINE TRAINING AND DEPLOYMENT COMPLETE.\n" + "="*60)
    

