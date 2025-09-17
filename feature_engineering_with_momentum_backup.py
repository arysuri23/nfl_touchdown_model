import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def create_rolling_features(df, features_to_roll, window=3):
    """
    Create rolling average features with proper data leakage prevention.
    
    For week N predictions, only uses data from weeks 1 through N-1.
    This prevents any future information from leaking into the model.
    
    Args:
        df: DataFrame with player data, must have 'player_id', 'season', 'week' columns
        features_to_roll: List of column names to create rolling averages for
        window: Rolling window size (default 3)
    
    Returns:
        DataFrame with additional rolling average columns prefixed with 'avg_'
    """
    # Ensure data is sorted properly (chronological order)
    df = df.sort_values(['season', 'week', 'player_id']).reset_index(drop=True)
    
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Create rolling averages for each feature
    for feature in features_to_roll:
        if feature in df.columns:
            # Use shift(1) to prevent data leakage - only use past data
            df[f'avg_{feature}'] = df.groupby('player_id')[feature].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        else:
            print(f"Warning: Feature '{feature}' not found in data")
    
    print(f"✓ Created rolling averages (window={window}) with data leakage prevention")
    return df


def create_ewm_features(df, features_to_roll, alpha=0.3):
    """
    Create exponentially weighted moving average features with proper data leakage prevention.
    
    For week N predictions, only uses data from weeks 1 through N-1.
    EWM gives more weight to recent games while incorporating all historical data.
    
    Args:
        df: DataFrame with player data, must have 'player_id', 'season', 'week' columns
        features_to_roll: List of column names to create EWM averages for
        alpha: Decay factor (0 < alpha <= 1). Higher = more weight to recent games
    
    Returns:
        DataFrame with additional EWM average columns prefixed with 'avg_'
    """
    # Ensure data is sorted properly (chronological order)
    df = df.sort_values(['season', 'week', 'player_id']).reset_index(drop=True)
    
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Create EWM averages for each feature
    for feature in features_to_roll:
        if feature in df.columns:
            # Use shift(1) to prevent data leakage - only use past data
            df[f'avg_{feature}'] = df.groupby('player_id')[feature].transform(
                lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1).mean()
            )
        else:
            print(f"Warning: Feature '{feature}' not found in data")
    
    return df


def create_matchup_features(df):
    """
    Create interactive matchup features that combine player usage with opponent defensive tendencies.
    
    These features capture the interaction between:
    - Player's opportunity/usage metrics
    - Opponent's defensive weaknesses
    
    Args:
        df: DataFrame with player data and opponent defensive stats
    
    Returns:
        DataFrame with additional matchup value columns
    """
    df = df.copy()
    
    # Initialize matchup features
    df['rush_matchup_value'] = 0.0
    df['pass_matchup_value'] = 0.0
    
    # Rush Matchup Value - combines player rushing usage with opponent rush defense weakness
    # For RBs: red zone carry opportunities × opponent's tendency to allow rushing TDs to RBs (lagged avg)
    if 'avg_redzone_carry_share' in df.columns and 'avg_rushing_tds_allowed_to_RB' in df.columns:
        rb_mask = df['position'] == 'RB'
        df.loc[rb_mask, 'rush_matchup_value'] = (
            df.loc[rb_mask, 'avg_redzone_carry_share'] * 
            df.loc[rb_mask, 'avg_rushing_tds_allowed_to_RB']
        )
    
    # For QBs: red zone carry opportunities × opponent's tendency to allow rushing TDs to QBs (lagged avg)
    if 'avg_redzone_carry_share' in df.columns and 'avg_rushing_tds_allowed_to_QB' in df.columns:
        qb_mask = df['position'] == 'QB'
        df.loc[qb_mask, 'rush_matchup_value'] = (
            df.loc[qb_mask, 'avg_redzone_carry_share'] * 
            df.loc[qb_mask, 'avg_rushing_tds_allowed_to_QB']
        )
    
    # Pass Matchup Value - combines player receiving usage with opponent pass defense weakness
    # For RBs: red zone target opportunities × opponent's tendency to allow passing TDs to RBs (lagged avg)
    if 'avg_redzone_target_share' in df.columns and 'avg_passing_tds_allowed_to_RB' in df.columns:
        rb_mask = df['position'] == 'RB'
        df.loc[rb_mask, 'pass_matchup_value'] = (
            df.loc[rb_mask, 'avg_redzone_target_share'] * 
            df.loc[rb_mask, 'avg_passing_tds_allowed_to_RB']
        )
    
    # For WRs: red zone target opportunities × opponent's tendency to allow passing TDs to WRs (lagged avg)
    if 'avg_redzone_target_share' in df.columns and 'avg_passing_tds_allowed_to_WR' in df.columns:
        wr_mask = df['position'] == 'WR'
        df.loc[wr_mask, 'pass_matchup_value'] = (
            df.loc[wr_mask, 'avg_redzone_target_share'] * 
            df.loc[wr_mask, 'avg_passing_tds_allowed_to_WR']
        )
    
    # For TEs: red zone target opportunities × opponent's tendency to allow passing TDs to TEs (lagged avg)
    if 'avg_redzone_target_share' in df.columns and 'avg_passing_tds_allowed_to_TE' in df.columns:
        te_mask = df['position'] == 'TE'
        df.loc[te_mask, 'pass_matchup_value'] = (
            df.loc[te_mask, 'avg_redzone_target_share'] * 
            df.loc[te_mask, 'avg_passing_tds_allowed_to_TE']
        )
    
    # Fill any remaining NaN values
    df['rush_matchup_value'] = df['rush_matchup_value'].fillna(0)
    df['pass_matchup_value'] = df['pass_matchup_value'].fillna(0)
    
    print("✓ Created position-specific matchup features")
    return df


def create_team_ewm_features(df, team_features_to_roll, alpha=0.3):
    """
    Create team-level EWM averages for opponent defensive stats with proper data leakage prevention.
    
    This ensures all players facing the same opponent in the same week have identical opponent stats,
    and that we only use historical data (no data leakage).
    
    Args:
        df: DataFrame with player data
        team_features_to_roll: List of team defensive feature names
        alpha: EWM alpha parameter (decay factor, 0 < alpha <= 1)
    
    Returns:
        DataFrame with team-level EWM averages added
    """
    # Create unique team-week combinations for team-level stats
    team_stats = df.groupby(['opponent_team', 'season', 'week']).agg({
        feature: 'first' for feature in team_features_to_roll if feature in df.columns
    }).reset_index()
    
    # Sort by team and chronological order for proper EWM calculation
    team_stats = team_stats.sort_values(['opponent_team', 'season', 'week']).reset_index(drop=True)
    
    # Create EWM averages for team defensive stats
    for feature in team_features_to_roll:
        if feature in team_stats.columns:
            team_stats[f'avg_{feature}'] = team_stats.groupby('opponent_team')[feature].transform(
                lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1).mean()
            )
    
    # Merge back to main dataframe
    merge_cols = ['opponent_team', 'season', 'week'] + [f'avg_{f}' for f in team_features_to_roll if f in df.columns]
    df = df.merge(team_stats[merge_cols], on=['opponent_team', 'season', 'week'], how='left')
    
    print(f"✓ Created team-level EWM averages (alpha={alpha}) with data leakage prevention")
    print(f"✓ All players facing same opponent now have identical defensive stats")
    return df


def create_team_rolling_features(df, team_features_to_roll, window=3):
    """
    Create rolling average features for TEAM-LEVEL statistics with proper data leakage prevention.
    
    These are opponent defensive stats that are grouped by team, not player.
    For week N predictions, only uses opponent team data from weeks 1 through N-1.
    
    Args:
        df: DataFrame with team defensive data
        team_features_to_roll: List of team-level column names to create rolling averages for
        window: Rolling window size (default 3)
    
    Returns:
        DataFrame with additional team-level rolling average columns
    """
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # First, create unique team-week combinations with their defensive stats
    team_stats = df[['opponent_team', 'season', 'week'] + team_features_to_roll].drop_duplicates()
    team_stats = team_stats.sort_values(['opponent_team', 'season', 'week']).reset_index(drop=True)
    
    # Create rolling averages for each team-level feature
    for feature in team_features_to_roll:
        if feature in team_stats.columns:
            # Use shift(1) to prevent data leakage - only use past opponent data
            team_stats[f'avg_{feature}'] = team_stats.groupby('opponent_team')[feature].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        else:
            print(f"Warning: Team feature '{feature}' not found in data")
    
    # Now merge the rolling averages back to the main dataframe
    # This ensures all players facing the same opponent get identical rolling stats
    rolling_cols = [f'avg_{feature}' for feature in team_features_to_roll if feature in df.columns]
    merge_cols = ['opponent_team', 'season', 'week'] + rolling_cols
    
    df = pd.merge(
        df.drop(columns=[col for col in rolling_cols if col in df.columns]),  # Remove any existing avg columns
        team_stats[merge_cols],
        on=['opponent_team', 'season', 'week'],
        how='left'
    )
    
    print(f"✓ Created team-level rolling averages (window={window}) with data leakage prevention")
    print(f"✓ All players facing same opponent now have identical defensive stats")
    return df


def engineer_features_for_touchdown_prediction_ewm(years, team_map, alpha=0.3):
    """
    Master feature engineering function that creates all features using EWM for touchdown prediction.
    
    This function orchestrates the complete pipeline:
    1. Load raw data
    2. Create player-level EWM averages 
    3. Create team-level EWM averages (opponent defense)
    4. Create matchup interaction features
    5. Return engineered dataset ready for modeling
    
    Args:
        years: List of years to load data for (e.g., [2023, 2024])
        team_map: Dictionary mapping full team names to abbreviations
        alpha: EWM alpha parameter (decay factor, 0 < alpha <= 1)
    
    Returns:
        DataFrame with all engineered features ready for position-specific modeling
    """
    from data_collection import get_all_historic_data
    
    print("============================================================")
    print("STARTING TOUCHDOWN PREDICTION FEATURE ENGINEERING (EWM)")
    print("============================================================")
    
    print("\nStep 1/4: Loading raw NFL data...")
    df = get_all_historic_data(years, team_map)
    
    # Ensure we only have the requested seasons (critical for proper EWM averages)
    df = df[df['season'].isin(years)].copy()
    
    # Enforce one row per player-week before computing any EWM features
    df = (
        df.sort_values(['season', 'week', 'player_id'])
          .drop_duplicates(subset=['player_id', 'season', 'week'], keep='last')
          .reset_index(drop=True)
    )
    
    print(f"   ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"   ✓ Years: {sorted(df['season'].unique())}")
    print(f"   ✓ Positions: {sorted(df['position'].unique())}")
    
    print(f"\nStep 2/4: Creating player-level EWM averages (alpha={alpha})...")
    
    # Define features to create EWM averages for (player-level stats)
    player_features_to_roll = [
        'offense_snap_share', 'wopr', 'rushing_epa', 'receiving_epa', 'racr',
        'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 
        'avg_time_to_los', 'percent_attempts_gte_eight_defenders',
        'redzone_carry_share', 'redzone_target_share', 'endzone_targets', 
        'endzone_target_share', 'inside_5_carry_share', 'inside_5_target_share',
        'scored_touchdown', 'target_share', 'avg_separation',
        'percent_share_of_intended_air_yards', 'catch_percentage', 
        'avg_expected_yac', 'avg_yac_above_expectation', 'receptions', 
        'receiving_yards', 'receiving_air_yards', 'avg_intended_air_yards',
        'carries', 'rushing_yards'
    ]
    
    df = create_ewm_features(df, player_features_to_roll, alpha=alpha)
    
    print(f"\nStep 3/4: Creating team-level EWM averages (alpha={alpha})...")
    
    # Define team defensive features to create EWM averages for
    team_features_to_roll = [
        'rushing_tds_allowed_to_RB', 'passing_tds_allowed_to_RB',
        'passing_tds_allowed_to_WR', 'passing_tds_allowed_to_TE',
        'rushing_tds_allowed_to_QB'
    ]
    
    df = create_team_ewm_features(df, team_features_to_roll, alpha=alpha)
    
    print("\nStep 4/4: Creating matchup interaction features...")
    df = create_matchup_features(df)
    
    print("\n============================================================")
    print("FEATURE ENGINEERING COMPLETED (EWM)")
    print("============================================================")
    print(f"Final dataset shape: ({len(df):,}, {len(df.columns)})")
    print(f"Total touchdowns: {df['scored_touchdown'].sum():,}")
    print("Data leakage prevention: ✅ Applied to all EWM features")
    
    return df


def engineer_features_for_touchdown_prediction(years, team_map, window=3):
    """
    Master feature engineering function that creates all features needed for touchdown prediction.
    
    This function orchestrates the complete pipeline:
    1. Load raw data
    2. Create player-level rolling averages 
    3. Create team-level rolling averages (opponent defense)
    4. Create matchup interaction features
    5. Return engineered dataset ready for modeling
    
    Args:
        years: List of years to load data for (e.g., [2023, 2024])
        team_map: Dictionary mapping full team names to abbreviations
        window: Rolling window size for averages (default 3)
    
    Returns:
        DataFrame with all engineered features ready for position-specific modeling
    """
    from data_collection import get_all_historic_data
    
    print("=" * 60)
    print("STARTING TOUCHDOWN PREDICTION FEATURE ENGINEERING")
    print("=" * 60)
    print()
    
    # Step 1: Load raw data
    print("Step 1/4: Loading raw NFL data...")
    df = get_all_historic_data(years, team_map)
    
    # Ensure we only have the requested seasons (critical for proper rolling averages)
    df = df[df['season'].isin(years)].copy()

    # Enforce one row per player-week before computing any rolling features
    df = (
        df.sort_values(['season', 'week', 'player_id'])
          .drop_duplicates(subset=['player_id', 'season', 'week'], keep='last')
          .reset_index(drop=True)
    )
    
    print(f"   ✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   ✓ Years: {sorted(df['season'].unique())}")
    print(f"   ✓ Positions: {sorted(df['position'].unique())}")
    print()
    
    # Step 2: Create player-level rolling averages
    print("Step 2/4: Creating player-level rolling averages...")
    player_features_to_roll = [
        # Basic stats
        'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
        'receiving_yards', 'receiving_tds', 'wopr', 'target_share',
        
        # Advanced metrics
        'rushing_epa', 'receiving_epa', 'racr', 'receiving_air_yards',
        
        # NGS metrics
        'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 'avg_time_to_los',
        'percent_attempts_gte_eight_defenders', 'avg_separation', 'avg_cushion',
        'percent_share_of_intended_air_yards', 'catch_percentage', 
        'avg_expected_yac', 'avg_yac_above_expectation', 'avg_intended_air_yards',
        
        # Usage metrics
        'offense_snap_share', 'redzone_carry_share', 'redzone_target_share',
        'inside_5_carry_share', 'inside_5_target_share', 'endzone_targets', 'endzone_target_share',
        
        # Target variable (lagged)
        'scored_touchdown'
    ]
    
    df = create_rolling_features(df, player_features_to_roll, window)
    print(f"   ✓ Created rolling averages for {len(player_features_to_roll)} player features")
    print()
    
    # Step 3: Create team-level rolling averages (opponent defense)
    print("Step 3/4: Creating team-level rolling averages...")
    team_features_to_roll = [
        'rushing_tds_allowed_to_RB', 'rushing_tds_allowed_to_QB',
        'passing_tds_allowed_to_RB', 'passing_tds_allowed_to_WR', 'passing_tds_allowed_to_TE'
    ]
    
    df = create_team_rolling_features(df, team_features_to_roll, window)
    print(f"   ✓ Created rolling averages for {len(team_features_to_roll)} team defensive features")
    print()
    
    # Step 4: Create matchup interaction features
    print("Step 4/5: Creating matchup interaction features...")
    df = create_matchup_features(df)
    print()
    
    # Step 5: Create momentum/trend features
    print("Step 5/5: Creating momentum/trend features...")
    df = create_momentum_features(df, window)
    print()
    
    # Final summary
    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)
    print(f"Final dataset shape: {df.shape}")
    print(f"Total touchdowns: {df['scored_touchdown'].sum():,}")
    print(f"Data leakage prevention: ✅ Applied to all rolling features")
    print()
    
    return df


def create_position_specific_datasets(df):
    """
    Filter the engineered dataset into position-specific feature sets ready for modeling.
    
    Each position gets only the features relevant for their touchdown prediction model.
    This follows the feature lists defined at the top of this file.
    
    Args:
        df: Engineered DataFrame from engineer_features_for_touchdown_prediction()
    
    Returns:
        Dictionary with keys 'RB', 'WR_TE', 'QB' containing position-specific datasets
    """
    
    # Define the feature lists for each position
    RB_FEATURES = [
        'avg_offense_snap_share',
        'avg_wopr',
        'avg_rushing_epa', 'avg_receiving_epa', 'avg_racr',
        'avg_rush_yards_over_expected_per_att', 'avg_rush_pct_over_expected', 
        'avg_avg_time_to_los', 'avg_percent_attempts_gte_eight_defenders',
        'avg_redzone_carry_share', 'avg_redzone_target_share', 'avg_endzone_targets',
        'avg_endzone_target_share', 'avg_inside_5_carry_share', 'avg_inside_5_target_share',
        'rush_matchup_value', 'pass_matchup_value',
        'avg_rushing_tds_allowed_to_RB', 'avg_passing_tds_allowed_to_RB',
        'implied_total', 'depth_chart_rank', 'avg_scored_touchdown',
        # Momentum features
        'last_2_games_td_rate', 'last_3_games_td_rate', 'games_since_last_td', 'consecutive_games_with_td',
        'redzone_carry_share_trend_3game', 'redzone_target_share_trend_3game', 'offense_snap_share_trend_3game',
        'redzone_carry_share_recent_vs_season', 'scored_touchdown_recent_vs_season',
        'offense_snap_share_volatility_3game'
    ]
    
    WR_TE_FEATURES = [
        'avg_offense_snap_share',
        'avg_wopr', 'avg_target_share',
        'avg_receiving_epa', 'avg_racr',
        'avg_avg_separation',
        'avg_percent_share_of_intended_air_yards', 'avg_catch_percentage',
        'avg_avg_expected_yac', 'avg_avg_yac_above_expectation',
        'avg_redzone_target_share', 'avg_endzone_targets', 'avg_endzone_target_share',
        'avg_inside_5_target_share',
        'pass_matchup_value',
        'avg_passing_tds_allowed_to_WR', 'avg_passing_tds_allowed_to_TE',
        'implied_total', 'depth_chart_rank', 'avg_scored_touchdown',
        'avg_receptions', 'avg_receiving_yards', 'avg_receiving_air_yards', 
        'avg_avg_intended_air_yards',
        # Momentum features (PRIORITY - WR/TE needs most improvement)
        'last_2_games_td_rate', 'last_3_games_td_rate', 'games_since_last_td', 'consecutive_games_with_td',
        'target_share_trend_3game', 'redzone_target_share_trend_3game', 'catch_percentage_trend_3game',
        'target_share_recent_vs_season', 'redzone_target_share_recent_vs_season', 'scored_touchdown_recent_vs_season',
        'target_share_volatility_3game', 'offense_snap_share_volatility_3game'
    ]
    
    QB_FEATURES = [
        'avg_offense_snap_share',
        'avg_carries', 'avg_rushing_yards', 'avg_rushing_epa',
        'avg_scored_touchdown', 'avg_redzone_carry_share', 'avg_inside_5_carry_share',
        'rush_matchup_value',
        'avg_rushing_tds_allowed_to_QB', 'implied_total', 'depth_chart_rank',
        'avg_rush_yards_over_expected_per_att', 'avg_rush_pct_over_expected', 
        'avg_avg_time_to_los',
        # Momentum features
        'last_2_games_td_rate', 'last_3_games_td_rate', 'games_since_last_td', 'consecutive_games_with_td',
        'redzone_carry_share_trend_3game', 'redzone_carry_share_recent_vs_season', 'scored_touchdown_recent_vs_season',
        'carries_volatility_3game'
    ]
    
    # Base columns to include in all datasets
    base_columns = [
        'player_id', 'player_display_name', 'position', 'recent_team', 
        'opponent_team', 'season', 'week', 'scored_touchdown'
    ]
    
    print("Creating position-specific datasets...")
    print()
    
    # Create RB dataset
    rb_data = df[df['position'] == 'RB'].copy()
    rb_columns = base_columns + [col for col in RB_FEATURES if col in df.columns]
    rb_dataset = rb_data[rb_columns]
    
    print(f"✓ RB Dataset: {rb_dataset.shape[0]:,} rows × {rb_dataset.shape[1]} features")
    print(f"  - Features available: {len([col for col in RB_FEATURES if col in df.columns])}/{len(RB_FEATURES)}")
    print(f"  - Touchdowns: {rb_dataset['scored_touchdown'].sum():,}")
    
    # Create WR/TE dataset  
    wr_te_data = df[df['position'].isin(['WR', 'TE'])].copy()
    wr_te_columns = base_columns + [col for col in WR_TE_FEATURES if col in df.columns]
    wr_te_dataset = wr_te_data[wr_te_columns]
    
    print(f"✓ WR/TE Dataset: {wr_te_dataset.shape[0]:,} rows × {wr_te_dataset.shape[1]} features")
    print(f"  - Features available: {len([col for col in WR_TE_FEATURES if col in df.columns])}/{len(WR_TE_FEATURES)}")
    print(f"  - Touchdowns: {wr_te_dataset['scored_touchdown'].sum():,}")
    
    # Create QB dataset
    qb_data = df[df['position'] == 'QB'].copy()
    qb_columns = base_columns + [col for col in QB_FEATURES if col in df.columns]
    qb_dataset = qb_data[qb_columns]
    
    print(f"✓ QB Dataset: {qb_dataset.shape[0]:,} rows × {qb_dataset.shape[1]} features")
    print(f"  - Features available: {len([col for col in QB_FEATURES if col in df.columns])}/{len(QB_FEATURES)}")
    print(f"  - Touchdowns: {qb_dataset['scored_touchdown'].sum():,}")
    print()
    
    # Check for missing features
    missing_features = {}
    for pos, features in [('RB', RB_FEATURES), ('WR_TE', WR_TE_FEATURES), ('QB', QB_FEATURES)]:
        missing = [col for col in features if col not in df.columns]
        if missing:
            missing_features[pos] = missing
    
    if missing_features:
        print("⚠️  Missing features:")
        for pos, missing in missing_features.items():
            print(f"  {pos}: {missing}")
        print()
    
    return {
        'RB': rb_dataset,
        'WR_TE': wr_te_dataset, 
        'QB': qb_dataset
    }


def create_momentum_features(df, window=3):
    """
    Create momentum and trend features to capture recent player performance patterns.
    
    These features help identify players who are trending up/down, on hot streaks,
    or experiencing role changes. Critical for improving model performance.
    
    Args:
        df: DataFrame with player data, must have 'player_id', 'season', 'week' columns
        window: Window size for momentum calculations (default: 3 games)
    
    Returns:
        DataFrame with additional momentum features
    """
    print(f"✓ Creating momentum/trend features (window={window})...")
    
    # Ensure data is sorted properly (chronological order)
    df = df.sort_values(['season', 'week', 'player_id']).reset_index(drop=True)
    df = df.copy()
    
    # === HOT HAND FEATURES ===
    print("  → Hot hand features (recent TD performance)...")
    
    # Recent touchdown rates (last 2 and 3 games)
    df['last_2_games_td_rate'] = df.groupby('player_id')['scored_touchdown'].transform(
        lambda x: x.shift(1).rolling(window=2, min_periods=1).mean()
    )
    
    df['last_3_games_td_rate'] = df.groupby('player_id')['scored_touchdown'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    
    # Games since last touchdown (drought tracking)
    def games_since_td(series):
        result = []
        games_since = 0
        for td in series:
            if pd.isna(td):
                result.append(np.nan)
            elif td == 1:
                games_since = 0
                result.append(games_since)
            else:
                games_since += 1
                result.append(games_since)
        return pd.Series(result, index=series.index)
    
    df['games_since_last_td'] = df.groupby('player_id')['scored_touchdown'].transform(
        lambda x: games_since_td(x.shift(1))
    )
    
    # Consecutive games with touchdown (streak tracking)
    def consecutive_td_games(series):
        result = []
        consecutive = 0
        for td in series:
            if pd.isna(td):
                result.append(np.nan)
            elif td == 1:
                consecutive += 1
                result.append(consecutive)
            else:
                consecutive = 0
                result.append(consecutive)
        return pd.Series(result, index=series.index)
    
    df['consecutive_games_with_td'] = df.groupby('player_id')['scored_touchdown'].transform(
        lambda x: consecutive_td_games(x.shift(1))
    )
    
    # === TREND FEATURES ===
    print("  → Trend features (performance direction)...")
    
    # Key metrics for trend analysis
    trend_features = {
        'usage_metrics': ['offense_snap_share', 'target_share', 'redzone_carry_share', 'redzone_target_share'],
        'efficiency_metrics': ['rushing_epa', 'receiving_epa', 'catch_percentage'],
        'opportunity_metrics': ['endzone_targets', 'inside_5_carry_share', 'inside_5_target_share']
    }
    
    # Calculate 3-game trends (slopes) for key metrics
    def calculate_trend(series, window=3):
        """Calculate slope of last N games using linear regression"""
        def trend_slope(x):
            if len(x.dropna()) < 2:
                return 0
            try:
                # Simple slope calculation: (last - first) / length
                clean_x = x.dropna()
                if len(clean_x) < 2:
                    return 0
                return (clean_x.iloc[-1] - clean_x.iloc[0]) / (len(clean_x) - 1)
            except:
                return 0
        
        return series.shift(1).rolling(window=window, min_periods=2).apply(trend_slope)
    
    # Apply trend calculations to key feature groups
    for category, features in trend_features.items():
        for feature in features:
            if feature in df.columns:
                df[f'{feature}_trend_3game'] = df.groupby('player_id')[feature].transform(
                    lambda x: calculate_trend(x, window=3)
                )
    
    # Recent vs season performance ratios
    print("  → Recent vs season comparison features...")
    
    key_comparison_features = ['offense_snap_share', 'target_share', 'redzone_carry_share', 
                              'redzone_target_share', 'scored_touchdown']
    
    for feature in key_comparison_features:
        if feature in df.columns:
            # Season average (excluding current week)
            season_avg = df.groupby(['player_id', 'season'])[feature].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
            
            # Recent 3-game average
            recent_avg = df.groupby('player_id')[feature].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
            )
            
            # Ratio of recent to season (>1 = trending up, <1 = trending down)
            df[f'{feature}_recent_vs_season'] = recent_avg / (season_avg + 0.001)  # Avoid division by zero
    
    # === VOLATILITY FEATURES ===
    print("  → Volatility features (consistency metrics)...")
    
    volatility_features = ['offense_snap_share', 'target_share', 'carries', 'receptions']
    
    for feature in volatility_features:
        if feature in df.columns:
            # 3-game volatility (standard deviation)
            df[f'{feature}_volatility_3game'] = df.groupby('player_id')[feature].transform(
                lambda x: x.shift(1).rolling(window=3, min_periods=2).std()
            )
    
    print(f"✓ Created {len([col for col in df.columns if any(suffix in col for suffix in ['_trend_', '_recent_vs_', '_volatility_', 'last_', 'games_since_', 'consecutive_'])])} momentum features")
    
    return df


def prepare_touchdown_prediction_data(years, team_map, window=3):
    """
    Complete end-to-end pipeline: Raw data → Model-ready position-specific datasets.
    
    This is the main function users should call. It handles the entire feature engineering
    pipeline and returns clean, position-specific datasets ready for model training.
    
    Args:
        years: List of years to load data for (e.g., [2023, 2024])
        team_map: Dictionary mapping full team names to abbreviations
        window: Rolling window size for averages (default 3)
    
    Returns:
        Dictionary with keys 'RB', 'WR_TE', 'QB' containing model-ready datasets
        
    Example:
        team_map = {'Arizona Cardinals': 'ARI', ...}  # Full team mapping
        datasets = prepare_touchdown_prediction_data([2023, 2024], team_map)
        
        rb_data = datasets['RB']
        wr_te_data = datasets['WR_TE'] 
        qb_data = datasets['QB']
    """
    
    print("🏈 TOUCHDOWN PREDICTION DATA PREPARATION PIPELINE")
    print("=" * 70)
    print()
    
    # Step 1: Engineer all features
    print("Phase 1: Feature Engineering")
    print("-" * 30)
    engineered_df = engineer_features_for_touchdown_prediction(years, team_map, window)
    print()
    
    # Step 2: Create position-specific datasets
    print("Phase 2: Position-Specific Dataset Creation")
    print("-" * 30)
    position_datasets = create_position_specific_datasets(engineered_df)
    print()
    
    # Final summary
    print("🎯 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    
    total_samples = sum(dataset.shape[0] for dataset in position_datasets.values())
    total_touchdowns = sum(dataset['scored_touchdown'].sum() for dataset in position_datasets.values())
    
    print("📊 FINAL DATASET SUMMARY:")
    print(f"  • Total samples: {total_samples:,}")
    print(f"  • Total touchdowns: {total_touchdowns:,}")
    print(f"  • TD rate: {total_touchdowns/total_samples:.1%}")
    print()
    
    for pos_name, dataset in position_datasets.items():
        feature_count = dataset.shape[1] - 8  # Subtract base columns
        td_rate = dataset['scored_touchdown'].mean()
        print(f"  • {pos_name}: {dataset.shape[0]:,} samples, {feature_count} features, {td_rate:.1%} TD rate")
    
    print()
    print("✅ Data is ready for position-specific model training!")
    print("✅ All rolling features use proper data leakage prevention")
    print("✅ Matchup features combine player usage × opponent weakness")
    print()
    
    return position_datasets


def prepare_touchdown_prediction_data_ewm(years, team_map, alpha=0.3):
    """
    Complete end-to-end pipeline using EWM: Raw data → Model-ready position-specific datasets.
    
    This is the main function for EWM-based features. It handles the entire feature engineering
    pipeline using exponentially weighted moving averages and returns clean, position-specific 
    datasets ready for model training.
    
    Args:
        years: List of years to load data for (e.g., [2023, 2024])
        team_map: Dictionary mapping full team names to abbreviations
        alpha: EWM alpha parameter (decay factor, 0 < alpha <= 1)
    
    Returns:
        Dictionary with keys 'RB', 'WR_TE', 'QB' containing model-ready datasets
        
    Example:
        team_map = {'Arizona Cardinals': 'ARI', ...}  # Full team mapping
        datasets = prepare_touchdown_prediction_data_ewm([2023, 2024], team_map, alpha=0.3)
        
        rb_data = datasets['RB']
        wr_te_data = datasets['WR_TE']
        qb_data = datasets['QB']
    """
    print("🏈 TOUCHDOWN PREDICTION DATA PREPARATION PIPELINE (EWM)")
    print("======================================================================")
    
    print("\nPhase 1: Feature Engineering")
    print("-" * 30)
    
    # Step 1: Engineer all features using EWM
    df = engineer_features_for_touchdown_prediction_ewm(years, team_map, alpha=alpha)
    
    print("\nPhase 2: Position-Specific Dataset Creation")
    print("-" * 30)
    
    # Step 2: Create position-specific datasets
    datasets = create_position_specific_datasets(df)
    
    print("\n🎯 PIPELINE COMPLETED SUCCESSFULLY! (EWM)")
    print("======================================================================")
    
    print(f"\n📊 FINAL DATASET SUMMARY:")
    print(f"  • Total samples: {len(df):,}")
    print(f"  • Total touchdowns: {df['scored_touchdown'].sum():,}")
    print(f"  • TD rate: {df['scored_touchdown'].mean():.1%}")
    print()
    
    for position, data in datasets.items():
        td_count = data['scored_touchdown'].sum()
        td_rate = data['scored_touchdown'].mean()
        feature_count = len([col for col in data.columns if col.startswith('avg_') or col in ['implied_total', 'depth_chart_rank']])
        
        print(f"  • {position}: {len(data):,} samples, {feature_count} features, {td_rate:.1%} TD rate")
    
    print(f"\n✅ Data is ready for position-specific model training!")
    print(f"✅ All EWM features use proper data leakage prevention (alpha={alpha})")
    print(f"✅ Matchup features combine player usage × opponent weakness")
    
    return datasets
