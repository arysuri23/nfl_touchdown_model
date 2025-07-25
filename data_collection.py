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
