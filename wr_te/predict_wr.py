import pandas as pd
import numpy as np
import joblib
import nflreadpy as nfl
import data_collection as data
import odds_match
import config

from features import WR_TE_FEATURES, PLAYER_EWM_STATS


### CONSTANTS ###

# Feature lists must match those used during training (imported from features.py)


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

    player_stats = PLAYER_EWM_STATS

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


### ROSTER LOADING ###
def load_week_roster(season, week, load_rosters_weekly=nfl.load_rosters_weekly, load_rosters=nfl.load_rosters):
    """Returns the active WR/TE roster for a single season/week.

    Prefers week-scoped `load_rosters_weekly`, filtered to `week == week`.
    Falls back to the season-level `load_rosters` snapshot when weekly data
    isn't available yet -- `load_rosters_weekly` raises `ValueError` before
    the season starts, or may return no rows for the requested week.
    """
    try:
        weekly = load_rosters_weekly([season])
        if hasattr(weekly, 'to_pandas'):
            weekly = weekly.to_pandas()
        weekly = weekly[weekly['week'] == week]
    except ValueError:
        weekly = pd.DataFrame()

    if not weekly.empty:
        df = weekly
    else:
        df = load_rosters([season])
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()

    df = df[(df['status'] == 'ACT') & (df['position'].isin(['WR', 'TE']))].copy()
    df = df.rename(columns={'gsis_id': 'player_id', 'full_name': 'player_display_name'})
    df = df[['player_id', 'player_display_name', 'position', 'team']]
    df = df.drop_duplicates(subset=['player_id'], keep='last')
    return df.reset_index(drop=True)


### ODDS JOIN ###
merge_name = odds_match.merge_name


def join_odds(predictions: pd.DataFrame, odds: pd.DataFrame, team_map: dict) -> pd.DataFrame:
    """Thin wrapper over `odds_match.match_odds_to_players` that adds
    `market_implied_prob` and `model_edge` on top of the matched `price`.
    """
    matched = odds_match.match_odds_to_players(predictions, odds, team_map)

    prob_if_pos = 100 / (matched['price'] + 100)
    prob_if_neg = matched['price'].abs() / (matched['price'].abs() + 100)
    matched['market_implied_prob'] = np.where(matched['price'] > 0, prob_if_pos, prob_if_neg)

    matched['model_edge'] = matched['predicted_touchdown_probability'] - matched['market_implied_prob']

    return matched


### PREDICTION LOGIC ###
def predict_touchdown_scorers(feature_df, model, calibrator, season, week, lines_df, depth_df, roster_df):
    """Predicts touchdown scorers for WR/TE using a loaded model.

    `roster_df` (from `load_week_roster`) supplies the players to predict
    for; `lines_df` (from `data.get_week_lines`) supplies opponent/game
    context; `depth_df` (from `data.get_depth_chart_for_week`) supplies
    current depth chart rank. `feature_df` is the full historical
    per-player-week feature table -- only rows strictly before `season`/
    `week` are used, to avoid leakage.
    """
    print("Assembling features for prediction...")

    # Filter feature_df to only include data up to the week before the prediction week
    hist_df = feature_df[(feature_df['season'] < season) | ((feature_df['season'] == season) & (feature_df['week'] < week))].copy()
    # Compute lagged EWMs on the filtered historical subset to avoid leakage
    hist_df = transform_features(hist_df)

    prediction_df = roster_df.copy()

    # Attach opponent + game context (implied_total, spread_line) for the target week
    game_context = lines_df[['team', 'opponent', 'implied_total', 'spread_line']].rename(columns={'opponent': 'opponent_team'})
    prediction_df = pd.merge(prediction_df, game_context, on='team', how='left')

    # Assemble features using historical data from the pre-engineered feature_df
    all_features = set(WR_TE_FEATURES)

    # Identify features that are NOT player-specific (need to be merged separately)
    non_player_features = set()
    for f in all_features:
        # Opponent defensive stats (all contain 'allowed')
        if 'allowed' in f:
            non_player_features.add(f)
        # Game context features (week-specific, not player-specific)
        elif f in ['implied_total', 'spread_line', 'is_home_game', 'div_game']:
            non_player_features.add(f)
        # depth_chart_rank always comes fresh from `depth_df` below, never
        # from historical feature rows.
        elif f == 'depth_chart_rank':
            non_player_features.add(f)

    player_history_features = sorted(list(all_features - non_player_features))
    opponent_history_features = [f for f in all_features if 'allowed' in f]  # All opponent defensive stats

    # Ensure chronological order so groupby().last() picks the most recent row per player
    hist_df = hist_df.sort_values(['player_id', 'season', 'week']).copy()
    latest_player_data = hist_df.groupby('player_id')[player_history_features].last().reset_index()

    prediction_df = pd.merge(prediction_df, latest_player_data, on='player_id', how='left')

    # Get the latest opponent defensive stats for each opponent team
    if opponent_history_features:
        opponent_stats_df = hist_df[['season', 'week', 'opponent_team'] + opponent_history_features]
        opponent_stats_df = opponent_stats_df.sort_values(by=['opponent_team', 'season', 'week']).copy()
        latest_opponent_data = opponent_stats_df.groupby('opponent_team')[opponent_history_features].last().reset_index()
        prediction_df = pd.merge(prediction_df, latest_opponent_data, on='opponent_team', how='left')

    # Current depth chart rank replaces any stale historical value pulled in above
    prediction_df.drop(columns=['depth_chart_rank'], inplace=True, errors='ignore')
    prediction_df = pd.merge(prediction_df, depth_df[['player_id', 'depth_chart_rank']], on='player_id', how='left')
    prediction_df['depth_chart_rank'] = prediction_df['depth_chart_rank'].fillna(4).astype(int)

    prediction_df.fillna(0, inplace=True)
    print("Feature assembly complete.")

    # Filter for WR/TE only (roster_df should already be scoped, but stay defensive)
    pred_df_wr_te = prediction_df[prediction_df['position'].isin(['WR', 'TE'])].copy()

    # No scaling required; tree-based models trained without scaling

    # Make predictions
    if not pred_df_wr_te.empty:
        pred_df_wr_te['predicted_touchdown_probability'] = model.predict_proba(pred_df_wr_te[WR_TE_FEATURES])[:, 1]

        # Apply Platt calibration if calibrator is available
        if calibrator is not None:
            wrte_raw = pred_df_wr_te['predicted_touchdown_probability'].values.reshape(-1, 1)
            pred_df_wr_te['predicted_touchdown_probability'] = calibrator.predict_proba(wrte_raw)[:, 1]

    pred_df_wr_te['season'] = season
    pred_df_wr_te['week'] = week

    return pred_df_wr_te.sort_values(by='predicted_touchdown_probability', ascending=False).reset_index(drop=True)


### main
if __name__ == '__main__':
    """
    Main entry point for WR/TE predictions.
    """
    print("="*60)
    print("WR/TE TD PREDICTION")
    print("="*60)

    season, week = config.SEASON, config.WEEK

    # --- 1. Load WR/TE Model ---
    print("\nLoading WR/TE model artifacts from local files...")
    wr_te_model = load_joblib_locally(config.MODELS_DIR / 'wr_te_rf_final.pkl')
    wr_te_calibrator = load_joblib_locally(config.MODELS_DIR / 'wr_te_rf_calibrator.pkl')
    print("Model loading complete")

    # --- 2. Load Data Files ---
    print("\nLoading data files from local files...")
    nfl_teams_df = pd.read_csv(config.DATA_DIR / 'nfl_teams.csv')
    team_map = dict(zip(nfl_teams_df['team_name'], nfl_teams_df['team_id']))

    odds_path = config.odds_snapshot_path(season, week, 'open')
    if not odds_path.exists():
        raise FileNotFoundError(
            f"Odds snapshot not found: {odds_path}. Run fetch_live_odds.py to generate it."
        )
    td_odds_df = pd.read_csv(odds_path)

    feature_df = pd.read_csv(config.DATA_DIR / 'raw_nfl_data.csv')

    lines_df = data.get_week_lines(season, week)
    depth_df = data.get_depth_chart_for_week(season, week)
    roster_df = load_week_roster(season, week)
    print("Data loading complete")

    # --- 3. Run Prediction ---
    print(f"\nGenerating WR/TE predictions for {season}, Week {week}...")

    predictions_df = predict_touchdown_scorers(
        feature_df, wr_te_model, wr_te_calibrator, season, week, lines_df, depth_df, roster_df
    )

    # --- 4. Join Odds ---
    predictions_df = join_odds(predictions_df, td_odds_df, team_map)
    predictions_df['odds_snapshot'] = odds_path.name

    output_cols = [
        'season', 'week', 'player_id', 'player_display_name', 'team', 'opponent_team',
        'position', 'predicted_touchdown_probability', 'price', 'market_implied_prob',
        'model_edge', 'bookmaker', 'odds_snapshot',
    ]
    predictions_df = predictions_df[output_cols + [c for c in predictions_df.columns if c not in output_cols]]

    # Save predictions to csv
    output_path = config.predictions_path(season, week)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"TOP 20 WR/TE TD PREDICTIONS - WEEK {week}")
    print("="*60)
    print(predictions_df[['player_display_name', 'team', 'position', 'predicted_touchdown_probability', 'model_edge']].head(20).to_string(index=False))

    print(f"\n{'='*60}")
    print("TOP 10 BY MODEL EDGE (price <= 400) -- informational, edge ranking, not yet validated")
    print("="*60)
    edge_candidates = predictions_df[predictions_df['price'] <= 400]
    print(edge_candidates.sort_values(by='model_edge', ascending=False)[['player_display_name', 'team', 'position', 'predicted_touchdown_probability', 'price', 'model_edge']].head(10).to_string(index=False))

    print(f"\nPredictions saved to {output_path}")
