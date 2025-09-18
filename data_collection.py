import nfl_data_py as nfl
import numpy as np
import pandas as pd


def get_nfl_data(years):
    """Fetches and preprocesses NFL weekly data for a given list of years."""
    df = nfl.import_weekly_data(years, downcast=True)

    df = df[df['week'] <= 18]
    df = df[['player_id', 'player_display_name', 'position', 'recent_team', 'season', 'week',
               'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
               'receiving_yards', 'receiving_tds', 'opponent_team', 'wopr', 'rushing_epa',
               'receiving_epa', 'target_share', 'receiving_air_yards', 'air_yards_share', 'racr']]
    
    df = df[df['position'].isin(['QB', 'RB', 'TE', 'WR'])]

    df['scored_touchdown'] = ((df['rushing_tds'] > 0) | (df['receiving_tds'] > 0)).astype(int)
    df.fillna(0, inplace=True)

    ## rename recent_team values "LA" to "LAR" and "LV" to "LVR"
    df['recent_team'] = df['recent_team'].replace({'LA': 'LAR', 'LV': 'LVR'})
    return df
def get_nfl_2025_weekly_data():
    df = pd.read_csv('data/stats_player_week_2025.csv')
    df = df[df['week'] <= 18]
    df = df[['player_id', 'player_display_name', 'position', 'team', 'season', 'week',
               'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
               'receiving_yards', 'receiving_tds', 'opponent_team', 'wopr', 'rushing_epa',
               'receiving_epa', 'target_share', 'receiving_air_yards', 'air_yards_share', 'racr']]
    
    #rename team to recent team
    df.rename(columns={'team': 'recent_team'}, inplace=True)
    
    df = df[df['position'].isin(['QB', 'RB', 'TE', 'WR'])]

    df['scored_touchdown'] = ((df['rushing_tds'] > 0) | (df['receiving_tds'] > 0)).astype(int)
    df.fillna(0, inplace=True)

    ## rename recent_team values "LA" to "LAR" and "LV" to "LVR"
    df['recent_team'] = df['recent_team'].replace({'LA': 'LAR', 'LV': 'LVR'})
    return df


def get_odds_data(years, team_map):
    """Loads and processes historical betting odds data."""

    df_odds = pd.read_csv('data/historic_lines.csv', low_memory=False)

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

# Add this new function to data_collection.py
def get_depth_chart_data(years):
    """Fetches and cleans weekly depth chart data."""
    depth_df = nfl.import_depth_charts(years)
    
    # Filter for relevant offensive positions using the correct column
    positions_to_keep = ['QB', 'RB', 'HB', 'WR', 'TE']
    depth_df = depth_df[depth_df['depth_position'].isin(positions_to_keep)]
    
    # Rename columns based on the user's correction
    # 'depth_team' is the rank, 'depth_position' is the position
    depth_df.rename(columns={
        'gsis_id': 'player_id', 
        'depth_team': 'depth_chart_rank', 
    }, inplace=True)
    
    return depth_df[['player_id', 'season', 'week', 'depth_chart_rank']]

def transform_future_odds(df, team_map):
    """Transform week_2_lines data to include: team, opponent, spread_line, total_line, implied_total"""
    games = []
    
    for game_id in df['game_id'].unique():
        game_data = df[df['game_id'] == game_id]
        
        # Get home and away teams
        home_team = game_data['home_team'].iloc[0]
        away_team = game_data['away_team'].iloc[0]
        
        # Get the total line (over/under) - should be the same for both teams
        total_line = game_data['over/under'].iloc[0]
        
        # Get spread data for both teams
        home_spread_data = game_data[game_data['label'] == home_team]
        away_spread_data = game_data[game_data['label'] == away_team]
        
        if len(home_spread_data) > 0 and len(away_spread_data) > 0:
            home_spread = home_spread_data['point'].iloc[0]
            away_spread = away_spread_data['point'].iloc[0]
            
            # Map team names to team IDs
            home_team_id = team_map.get(home_team, home_team)
            away_team_id = team_map.get(away_team, away_team)
            
            # Calculate implied totals
            # For home team: implied_total = (total_line / 2) - (spread_line / 2)
            # For away team: implied_total = (total_line / 2) - (spread_line / 2)
            home_implied_total = (total_line / 2) - (home_spread / 2)
            away_implied_total = (total_line / 2) - (away_spread / 2)
            
            # Add home team row
            games.append({
                'team': home_team_id,
                'opponent': away_team_id,
                'spread_line': home_spread,
                'total_line': total_line,
                'implied_total': home_implied_total
            })
            
            # Add away team row
            games.append({
                'team': away_team_id,
                'opponent': home_team_id,
                'spread_line': away_spread,
                'total_line': total_line,
                'implied_total': away_implied_total
            })
    
    # Create final dataframe
    result_df = pd.DataFrame(games)
    result_df['team'] = result_df['team'].replace({'LAR': 'LA', 'LVR': 'LV'})
    # Sort by team for consistency

    
    return result_df


def get_snap_counts(years):
    """Fetches and prepares snap count data for joining."""
    snap_df = nfl.import_snap_counts(years) #
    
    # Select columns needed for joining and rename 'player' to match our main df
    snap_df = snap_df[['pfr_player_id', 'player', 'team', 'season', 'week', 'offense_pct']] #
    snap_df.rename(columns={
        'pfr_player_id': 'pfr_id',
        'offense_pct': 'offense_snap_share',
        'player': 'player_display_name'
    }, inplace=True)

    snap_df['merge_name'] = snap_df['player_display_name'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()

    # ids = nfl.import_ids()
    # # Get player IDs and merge names
    # ids= ids[['pfr_id', 'gsis_id']]
    # #merge player IDs with snap counts
    # snap_df = pd.merge(snap_df, ids, on='pfr_id', how='left')

    # #rename gsis_id to player_id
    # snap_df.rename(columns={'gsis_id': 'player_id'}, inplace=True)
    # #drop pfr_id
    snap_df.drop(columns=['pfr_id', 'player_display_name'], inplace=True)

    # Ensure snap share is a float between 0 and 1
    snap_df['offense_snap_share'] = snap_df['offense_snap_share'].fillna(0)


    
    return snap_df

def get_ngs_data_rushing(years):
    ngs_rushing_df = nfl.import_ngs_data(stat_type="rushing", years=years)

    ngs_rushing_df = ngs_rushing_df[['season', 'week', 'player_gsis_id', 'rush_yards_over_expected_per_att', 
                                    'rush_pct_over_expected', 'avg_time_to_los', 'percent_attempts_gte_eight_defenders']]
    
    ngs_rushing_df.rename(columns={
        'player_gsis_id': 'player_id'
    }, inplace=True)

    return ngs_rushing_df

def get_ngs_data_receiving(years):
    ngs_receiving_df = nfl.import_ngs_data(stat_type="receiving", years=years)

    ngs_receiving_df = ngs_receiving_df[['season', 'week', 'player_gsis_id', 
                                         'avg_cushion', 'avg_separation', 'avg_intended_air_yards', 'percent_share_of_intended_air_yards', 
                                         'catch_percentage', 'avg_expected_yac', 'avg_yac_above_expectation']]
    
    ngs_receiving_df.rename(columns={
        'player_gsis_id': 'player_id'
    }, inplace=True)


    return ngs_receiving_df


def get_2025_depth_chart_data():
    """Fetches and cleans 2025 depth chart data."""
    #create map of dates to weeks. Week 1 is 2025-09-03, Week 2 is 2025-09-10, etc.
    week_map = {
        '2025-09-03': 1,
        '2025-09-10': 2,
        '2025-09-17': 3,
        '2025-09-24': 4,
        '2025-10-01': 5,
        '2025-10-08': 6,
        '2025-10-15': 7,
        '2025-10-22': 8,
        '2025-10-29': 9,
        '2025-11-05': 10,
        '2025-11-12': 11,
        '2025-11-19': 12,
        '2025-11-26': 13,
        '2025-12-03': 14,
        '2025-12-10': 15,
        '2025-12-17': 16,
        '2025-12-24': 17,
        '2025-12-31': 18
    }
    depth_df = nfl.import_depth_charts([2025])
    
    # Filter for relevant offensive positions using the correct column
    positions_to_keep = ['QB', 'RB', 'WR', 'TE']
    depth_df = depth_df[depth_df['pos_abb'].isin(positions_to_keep)]
    # Rename columns based on the user's correction
    # 'depth_team' is the rank, 'depth_position' is the position
    depth_df.rename(columns={
        'gsis_id': 'player_id', 
        'pos_rank': 'depth_chart_rank', 
    }, inplace=True)

    #add season column that is 2025
    depth_df['season'] = 2025
    depth_df['date'] = pd.to_datetime(depth_df['dt'])
    #filter dats to only include YEAR-MONTH-DAY
    depth_df['date'] = depth_df['date'].dt.strftime('%Y-%m-%d')

    depth_df['week'] = depth_df['date'].map(week_map)

    depth_df = depth_df[depth_df['week'] <= 18]
    return depth_df[['player_id', 'season', 'week', 'depth_chart_rank']]



def get_all_historic_data(years, team_map):
    pbp = nfl.import_pbp_data(years, downcast=True)
    
    rosters = nfl.import_seasonal_rosters(years)
    
    
    # Ensure we only load data for weeks 1-18
    pbp = pbp[pbp['week'] <= 18]

    nfl_df = get_nfl_data([2020,2021,2022,2023,2024])
    nfl_2025_df = get_nfl_2025_weekly_data()

    nfl_df = pd.concat([nfl_df, nfl_2025_df], ignore_index=True)

    nfl_df = nfl_df[nfl_df['week'] <= 18]

    nfl_df['merge_name'] = nfl_df['player_display_name'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()

    
    redzone_df = get_redzone_data(pbp)
    redzone_df = redzone_df[redzone_df['week'] <= 18]

    redzone_td_df = get_redzone_td_rate(pbp)
    redzone_td_df = redzone_td_df[redzone_td_df['week'] <= 18]

    ez_target_df = get_endzone_target_data(pbp)
    ez_target_df = ez_target_df[ez_target_df['week'] <= 18]

    odds_df = get_odds_data(years, team_map)
    odds_df = odds_df[odds_df['week'] <= 18]

    goal_line_df = get_goal_line_data(pbp)
    goal_line_df = goal_line_df[goal_line_df['week'] <= 18]

    positional_defense_df = get_opponent_positional_data(pbp, rosters)
    positional_defense_df = positional_defense_df[positional_defense_df['week'] <= 18]

    depth_chart_df = get_depth_chart_data([2020, 2021, 2022, 2023, 2024])
    depth_chart_df_2025 = get_2025_depth_chart_data()

    depth_chart_df = pd.concat([depth_chart_df, depth_chart_df_2025], ignore_index=True)
    depth_chart_df = depth_chart_df[depth_chart_df['week'] <= 18]

    snap_counts_df = get_snap_counts(years)
    snap_counts_df = snap_counts_df[snap_counts_df['week']<=18]

    ngs_rushing_df = get_ngs_data_rushing(years)
    ngs_rushing_df = ngs_rushing_df[ngs_rushing_df['week'] <= 18]

    ngs_receiving_df = get_ngs_data_receiving(years)
    ngs_receiving_df = ngs_receiving_df[ngs_receiving_df['week'] <= 18]

    nfl_df = pd.merge(nfl_df, redzone_df, on=['player_id', 'week', 'season'], how='left')
    #nfl_df = pd.merge(nfl_df, redzone_td_rate, on=['recent_team', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, ez_target_df, on=['player_id', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, odds_df, left_on=['recent_team', 'season', 'week'], right_on=['team', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, goal_line_df, on=['player_id', 'week', 'season'], how='left')
    nfl_df = pd.merge(nfl_df, positional_defense_df, on=['opponent_team', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, depth_chart_df, on=['player_id', 'season', 'week'], how='left')
    nfl_df['depth_chart_rank'] = pd.to_numeric(nfl_df['depth_chart_rank'], errors='coerce').fillna(4).astype(int)

    nfl_df = pd.merge(nfl_df, snap_counts_df, on=['merge_name', 'season', 'week'])
    nfl_df['offense_snap_share'].fillna(0, inplace=True)

    nfl_df = pd.merge(nfl_df, ngs_rushing_df, on=['player_id', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, ngs_receiving_df, on=['player_id', 'season', 'week'], how='left')

    nfl_df.fillna(0, inplace=True)

    nfl_df.sort_values(by=['season', 'week', 'player_id'], inplace=True, ignore_index=True)
    
    return nfl_df
    

