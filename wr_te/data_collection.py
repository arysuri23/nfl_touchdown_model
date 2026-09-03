# Data collection script for WR/TE analysis

#import nfl_data_py as nfl
import nflreadpy as nfl
import numpy as np
import pandas as pd
import polars as pl

import config


def get_nfl_data(years):
    """Fetches and preprocesses NFL weekly data for a given list of years."""
    df = nfl.load_player_stats(years)

    df = df.filter(pl.col('position').is_in(['WR', 'TE']))

    df = df.filter(pl.col('week') <= 18)
    df = df.filter(pl.col('season_type') == 'REG')
    df = df.select([
        'player_id', 'player_display_name', 'position', 'team', 'season', 'week',
        'receptions', 'targets', 'receiving_yards', 'rushing_tds','receiving_tds', 'opponent_team', 'wopr', 
        'receiving_epa', 'target_share', 'receiving_air_yards', 'air_yards_share', 'racr'
    ])
    
    df = df.with_columns([
        ((pl.col('rushing_tds') > 0) | (pl.col('receiving_tds') > 0)).cast(pl.Int8).alias('scored_touchdown'),
        (pl.col('rushing_tds') + pl.col('receiving_tds')).alias('total_tds')
    ])

    df = df.fill_null(0)

    return df.to_pandas()
def get_nfl_2025_weekly_data():
    df = pd.read_csv(config.DATA_DIR / 'stats_player_week_2025.csv')
    df = df[df['week'] <= 18]
    df = df[['player_id', 'player_display_name', 'position', 'team', 'season', 'week',
               'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
               'receiving_yards', 'receiving_tds', 'opponent_team', 'wopr', 'rushing_epa',
               'receiving_epa', 'target_share', 'receiving_air_yards', 'air_yards_share', 'racr']]
    
    #rename team to recent team
    df.rename(columns={'team': 'recent_team'}, inplace=True)
    
    df = df[df['position'].isin(['TE', 'WR'])]

    df['scored_touchdown'] = ((df['rushing_tds'] > 0) | (df['receiving_tds'] > 0)).astype(int)
    df.fillna(0, inplace=True)

    ## rename recent_team values "LA" to "LAR" and "LV" to "LVR"
    df['recent_team'] = df['recent_team'].replace({'LA': 'LAR', 'LV': 'LVR'})
    return df


def schedule_to_team_lines(schedule):
    """Derive per-team game lines from an nflverse-style schedule DataFrame.

    Binding convention: `spread_line` is from the team's own perspective,
    negative = that team is favored. nflverse schedules encode
    `spread_line > 0` as home favored, so home `spread_line = -sched.spread_line`
    and away `spread_line = +sched.spread_line`.
    `implied_total = total_line / 2 - spread_line / 2`.

    Rows with a null `spread_line` or `total_line` are dropped (produce no
    output rows for that game). Team codes `LAR`/`LVR` are renamed to `LA`/`LV`.

    Returns columns: season, week, team, opponent, spread_line, total_line,
    implied_total.
    """
    df = schedule
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'REG']
    df = df.dropna(subset=['spread_line', 'total_line']).copy()

    df_home = df[['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line']].rename(
        columns={'home_team': 'team', 'away_team': 'opponent'}
    )
    df_home['spread_line'] = -df_home['spread_line']

    df_away = df[['season', 'week', 'away_team', 'home_team', 'spread_line', 'total_line']].rename(
        columns={'away_team': 'team', 'home_team': 'opponent'}
    )

    df_team_lines = pd.concat([df_home, df_away], ignore_index=True)
    df_team_lines['team'] = df_team_lines['team'].replace({'LAR': 'LA', 'LVR': 'LV'})
    df_team_lines['opponent'] = df_team_lines['opponent'].replace({'LAR': 'LA', 'LVR': 'LV'})
    df_team_lines['implied_total'] = (df_team_lines['total_line'] / 2) - (df_team_lines['spread_line'] / 2)

    return df_team_lines[['season', 'week', 'team', 'opponent', 'spread_line', 'total_line', 'implied_total']]


def get_odds_data(years, team_map=None, load_schedules=nfl.load_schedules):
    """Loads and processes game lines derived from nflverse schedules."""
    schedule = load_schedules(years).to_pandas()
    schedule = schedule[schedule['game_type'] == 'REG']

    df_processed_odds = schedule_to_team_lines(schedule)
    df_processed_odds = df_processed_odds[df_processed_odds['week'] <= 18]

    return df_processed_odds


def get_week_lines(season, week, load_schedules=nfl.load_schedules):
    """Returns game lines for a single season/week."""
    odds_df = get_odds_data([season], load_schedules=load_schedules)
    return odds_df[odds_df['week'] == week]

def get_redzone_data(pbp):
    """Calculates each player's share of their team's red zone carries and targets."""
    redzone_df = pbp.filter(pl.col('yardline_100') <= 20)

    team_rz_plays = (
        redzone_df
        .group_by(['posteam', 'season', 'week'])
        .agg([
            pl.col('pass_attempt').sum().alias('team_rz_targets'),
        ])
    )

    """ 
    player_rz_rushes = (
        redzone_df
        .group_by(['rusher_player_id', 'posteam', 'season', 'week'])
        .agg([
            pl.col('rush_attempt').sum().alias('player_rz_rushes')
        ])
        .rename({'rusher_player_id': 'player_id'})
    ) 
    """


    player_rz_targets = (
        redzone_df
        .group_by(['receiver_player_id', 'posteam', 'season', 'week'])
        .agg([
            pl.col('pass_attempt').sum().alias('player_rz_targets')
        ])
        .rename({'receiver_player_id': 'player_id'})
    )

    """ player_usage = player_rz_rushes.join(
        player_rz_targets,
        on=['player_id', 'posteam', 'season', 'week'],
        how='outer',
    ) """

    final_rz = player_rz_targets.join(
        team_rz_plays,
        on=['posteam', 'season', 'week' ],
        how='left',
    )

    final_rz = final_rz.with_columns([
        pl.when(pl.col('team_rz_targets') > 0)
          .then((pl.col('player_rz_targets').fill_null(0) / pl.col('team_rz_targets')))
          .otherwise(0.0)
          .alias('redzone_target_share'),
    ])

    return final_rz.select([
        'player_id', 'season', 'week','redzone_target_share'
    ]).to_pandas()

def get_goal_line_data(pbp):
    """Calculates shares of carries and targets inside the 5-yard line."""
    goal_line_df = pbp.filter(pl.col('yardline_100') <= 5)
    
    team_gl_plays = goal_line_df.group_by(['posteam', 'season', 'week']).agg([
        pl.col('pass_attempt').sum().alias('team_gl_targets')
    ])
    
    """ player_gl_rushes = goal_line_df.group_by(['rusher_player_id', 'posteam', 'season', 'week']).agg([
        pl.col('rush_attempt').sum().alias('player_gl_rushes')
    ]).rename({'rusher_player_id': 'player_id'}) """
    
    player_gl_targets = goal_line_df.group_by(['receiver_player_id', 'posteam', 'season', 'week']).agg([
        pl.col('pass_attempt').sum().alias('player_gl_targets')
    ]).rename({'receiver_player_id': 'player_id'})
    
    """ player_usage = player_gl_rushes.join(
        player_gl_targets, 
        on=['player_id', 'posteam', 'season', 'week'], 
        how='outer'
    ) """
    
    final_gl = player_gl_targets.join(
        team_gl_plays, 
        on=['posteam', 'season', 'week'], 
        how='left'
    )
    
    final_gl = final_gl.with_columns([
        pl.when(pl.col('team_gl_targets') > 0)
          .then((pl.col('player_gl_targets').fill_null(0) / pl.col('team_gl_targets')))
          .otherwise(0.0)
          .alias('inside_5_target_share'),
    ])
    
    return final_gl.select([
        'player_id', 'season', 'week', 'inside_5_target_share'
    ]).to_pandas()

def get_green_zone_data(pbp):
    """Calculates raw targets inside the 10-yard line (Green Zone)."""
    green_zone_df = pbp.filter(pl.col('yardline_100') <= 10)
    
    player_gz_targets = green_zone_df.group_by(['receiver_player_id', 'posteam', 'season', 'week']).agg([
        pl.col('pass_attempt').sum().alias('inside_10_targets')
    ]).rename({'receiver_player_id': 'player_id'})
    
    return player_gz_targets.select([
        'player_id', 'season', 'week', 'inside_10_targets'
    ]).to_pandas()

def get_endzone_target_data(pbp):
    """Calculates each player's end zone targets and share of team targets."""
    pass_plays = pbp.filter(
        (pl.col('pass_attempt') == 1) & pl.col('air_yards').is_not_null()
    )
    
    pass_plays = pass_plays.with_columns([
        (pl.col('air_yards') >= pl.col('yardline_100')).alias('is_endzone_target')
    ])
    
    team_targets = pass_plays.group_by(['posteam', 'season', 'week']).agg([
        pl.col('pass_attempt').sum().alias('team_total_targets')
    ])
    
    ez_target_df = pass_plays.filter(pl.col('is_endzone_target') == True)
    
    player_ez_targets = ez_target_df.group_by(['receiver_player_id', 'posteam', 'season', 'week']).agg([
        pl.col('is_endzone_target').sum().alias('endzone_targets')
    ]).rename({'receiver_player_id': 'player_id'})
    
    final_ez_df = player_ez_targets.join(
        team_targets,
        on=['posteam', 'season', 'week'],
        how='left'
    )
    
    final_ez_df = final_ez_df.with_columns([
        pl.when(pl.col('team_total_targets') > 0)
          .then((pl.col('endzone_targets') / pl.col('team_total_targets')))
          .otherwise(0.0)
          .alias('endzone_target_share')
    ])
    
    return final_ez_df.select([
        'player_id', 'season', 'week', 'endzone_targets', 'endzone_target_share'
    ]).to_pandas()

def get_redzone_td_rate(pbp):
    """Calculates team-level red zone touchdown conversion rate."""
    redzone = pbp.filter(pl.col('yardline_100') <= 20)
    
    redzone_trips = redzone.group_by(['posteam', 'season', 'week', 'drive']).agg([
        pl.len().alias('plays')
    ])
    
    redzone_tds = redzone.filter(pl.col('touchdown') == 1).group_by(['posteam', 'season', 'week', 'drive']).agg([
        pl.len().alias('tds')
    ])
    
    drive_summary = redzone_trips.select(['posteam', 'season', 'week', 'drive']).join(
        redzone_tds,
        on=['posteam', 'season', 'week', 'drive'],
        how='left'
    ).with_columns([
        pl.col('tds').fill_null(0)
    ])
    
    redzone_summary = drive_summary.group_by(['posteam', 'season', 'week']).agg([
        pl.col('drive').n_unique().alias('redzone_trips'),
        (pl.col('tds') > 0).sum().alias('redzone_tds')
    ])
    
    redzone_summary = redzone_summary.with_columns([
        (pl.col('redzone_tds') / pl.col('redzone_trips')).alias('redzone_td_rate')
    ])
    
    redzone_summary = redzone_summary.rename({'posteam': 'team'})
    
    return redzone_summary.to_pandas()

def get_explosive_receiving_data(pbp):
    """Calculates explosive play data for each player."""
    explosive_plays = pbp.filter(pl.col('yards_gained') >= 20)

    player_explosive_plays = explosive_plays.group_by(['receiver_player_id', 'season', 'week']).agg([
        pl.len().alias('explosive_receiving_plays')
    ])

    ## rename receiver_player_id to player_id
    player_explosive_plays = player_explosive_plays.rename({'receiver_player_id': 'player_id'})
    
    return player_explosive_plays.to_pandas()

""" def get_explosive_rushing_data(pbp):
    Calculates explosive play data for each player.
    explosive_plays = pbp.filter(pl.col('yards_gained') >= 20)

    player_explosive_plays = explosive_plays.group_by(['rusher_player_id', 'season', 'week']).agg([
        pl.len().alias('explosive_rushing_plays')
    ])

    ## rename rusher_player_id to player_id
    player_explosive_plays = player_explosive_plays.rename({'rusher_player_id': 'player_id'})
    
    return player_explosive_plays.to_pandas()
 """
def get_opponent_positional_data(pbp, rosters):
    """Calculates how many TDs each defense allows to specific offensive positions."""
    roster_positions = rosters.select(['gsis_id', 'position']).unique()
    
    # Rush TDs allowed
    """  rush_tds = pbp.filter(pl.col('rush_touchdown') == 1).select(['defteam', 'season', 'week', 'rusher_player_id'])
    rush_tds = rush_tds.join(roster_positions, left_on='rusher_player_id', right_on='gsis_id', how='left')
    
    rush_tds_counted = rush_tds.group_by(['defteam', 'season', 'week', 'position']).agg([
        pl.len().alias('count')
    ])
    
    rush_tds_allowed = rush_tds_counted.pivot(
        values='count',
        index=['defteam', 'season', 'week'],
        columns='position'
    )
    
    # Rename columns to add prefix
    rush_cols_rename = {col: f'rushing_tds_allowed_to_{col}' 
                        for col in rush_tds_allowed.columns 
                        if col not in ['defteam', 'season', 'week']}
    rush_tds_allowed = rush_tds_allowed.rename(rush_cols_rename)
    """
    # Pass TDs allowed
    pass_tds = pbp.filter(pl.col('pass_touchdown') == 1).select(['defteam', 'season', 'week', 'receiver_player_id'])
    pass_tds = pass_tds.join(roster_positions, left_on='receiver_player_id', right_on='gsis_id', how='left')
    
    pass_tds_counted = pass_tds.group_by(['defteam', 'season', 'week', 'position']).agg([
        pl.len().alias('count')
    ])
    
    pass_tds_allowed = pass_tds_counted.pivot(
        values='count',
        index=['defteam', 'season', 'week'],
        columns='position'
    )
    
    # Rename columns to add prefix
    pass_cols_rename = {col: f'passing_tds_allowed_to_{col}' 
                        for col in pass_tds_allowed.columns 
                        if col not in ['defteam', 'season', 'week']}
    positional_defense_df = pass_tds_allowed.rename(pass_cols_rename)
    
    # Merge rush and pass TDs
    """ positional_defense_df = rush_tds_allowed.join(
        pass_tds_allowed,
        on=['defteam', 'season', 'week'],
        how='outer'
    ) """
    
    # Ensure all expected columns exist with fill_null(0)
    for pos in ['WR', 'TE']:
        #rush_col = f'rushing_tds_allowed_to_{pos}'
        pass_col = f'passing_tds_allowed_to_{pos}'
        
        """ if rush_col not in positional_defense_df.columns:
            positional_defense_df = positional_defense_df.with_columns([
                pl.lit(0).alias(rush_col)
            ]) 
        else:
            positional_defense_df = positional_defense_df.with_columns([
                pl.col(rush_col).fill_null(0)
            ]) """
            
        if pass_col not in positional_defense_df.columns:
            positional_defense_df = positional_defense_df.with_columns([
                pl.lit(0).alias(pass_col)
            ])
        else:
            positional_defense_df = positional_defense_df.with_columns([
                pl.col(pass_col).fill_null(0)
            ])
    
    positional_defense_df = positional_defense_df.rename({'defteam': 'opponent_team'})

    #Drop 'passing_tds_allowed_to_RB', 'passing_tds_allowed_to_OL', 'passing_tds_allowed_to_QB', 'passing_tds_allowed_to_DL',
    cols_to_drop = ['passing_tds_allowed_to_RB', 'passing_tds_allowed_to_OL', 'passing_tds_allowed_to_QB', 'passing_tds_allowed_to_DL']
    positional_defense_df = positional_defense_df.drop([c for c in cols_to_drop if c in positional_defense_df.columns])
    return positional_defense_df.to_pandas()

def get_opponent_defensive_data(years):
    """Calculates defensive data for each opponent."""
    weekly_data = nfl.load_player_stats(years)
    ### get rushing and receiving yards allowed by group by opponent and season and week
    defensive_data = weekly_data.group_by(['opponent_team', 'season', 'week']).agg([
        #pl.col('rushing_yards').sum().alias('rushing_yards_allowed'),
        pl.col('receiving_yards').sum().alias('receiving_yards_allowed'),
       # pl.col('rushing_epa').sum().alias('rushing_epa_allowed'),
        pl.col('receiving_epa').sum().alias('receiving_epa_allowed'),
        pl.col('receiving_air_yards').sum().alias('receiving_air_yards_allowed')
    ])
    return defensive_data.to_pandas()

""" def get_opponent_rushing_explosive_play_allowed(pbp):
    Calculates explosive play data for each opponent.
    explosive_plays = pbp.filter(pl.col('yards_gained') >= 20)

    rushing_explosive_plays = explosive_plays.filter(pl.col('play_type') == 'run')

    opponent_explosive_plays = rushing_explosive_plays.group_by(['defteam', 'season', 'week']).agg([
        pl.len().alias('explosive_rushing_plays_allowed')
    ])
 
    opponent_explosive_plays = opponent_explosive_plays.rename({'defteam': 'opponent_team'})

    return opponent_explosive_plays.to_pandas() """

def get_opponent_receiving_explosive_play_allowed(pbp):
    """Calculates explosive play data for each opponent."""
    explosive_plays = pbp.filter(pl.col('yards_gained') >= 20)

    receiving_explosive_plays = explosive_plays.filter(pl.col('play_type') == 'pass')

    opponent_explosive_plays = receiving_explosive_plays.group_by(['defteam', 'season', 'week']).agg([
        pl.len().alias('explosive_receiving_plays_allowed')
    ])
    opponent_explosive_plays = opponent_explosive_plays.rename({'defteam': 'opponent_team'})

    return opponent_explosive_plays.to_pandas()

def get_team_play_data(years):
    """Calculates team play data for each team."""
    team_data = nfl.load_team_stats(years)
    team_data = team_data.select(['season', 'week', 'team', 'attempts', 'carries'])
    
    #new column total_plays = attempts + carries
    team_data = team_data.with_columns([
        (pl.col('attempts') + pl.col('carries')).alias('total_plays')
    ])

    #new column pass_rate = passes / total_plays
    team_data = team_data.with_columns([
        (pl.col('attempts') / pl.col('total_plays')).alias('pass_rate')
    ])

    #new column rush_rate = carries / total_plays
    """  team_data = team_data.with_columns([
        (pl.col('carries') / pl.col('total_plays')).alias('rush_rate')
    ]) """


    team_data = team_data.select(['season', 'week', 'team', 'pass_rate'])
    
    return team_data.to_pandas()

def select_week_snapshots(snapshot_dates, week_first_gameday):
    """For each week, pick the latest snapshot date <= that week's first gameday.

    Returns a dict {week: date or None}.
    """
    sorted_snapshots = sorted(snapshot_dates)
    result = {}
    for week, gameday in week_first_gameday.items():
        candidates = [snap for snap in sorted_snapshots if snap <= gameday]
        result[week] = max(candidates) if candidates else None
    return result


def _depth_chart_week_gamedays(season, load_schedules):
    """Returns {week: date} of the first REG gameday of each week for a season."""
    schedule = load_schedules([season])
    if hasattr(schedule, 'to_pandas'):
        schedule = schedule.to_pandas()
    schedule = schedule[schedule['game_type'] == 'REG'].copy()
    schedule['gameday_date'] = pd.to_datetime(schedule['gameday']).dt.date
    return schedule.groupby('week')['gameday_date'].min().to_dict()


def _new_schema_depth_chart(depth_df, season, load_schedules):
    """New nflreadpy depth-chart schema: gsis_id, pos_abb, pos_rank, dt (ISO datetime str)."""
    df = depth_df[depth_df['pos_abb'].isin(['WR', 'TE'])].copy()
    if df.empty:
        return pd.DataFrame(columns=['player_id', 'season', 'week', 'depth_chart_rank'])

    df['snapshot_date'] = pd.to_datetime(df['dt']).dt.date

    week_first_gameday = _depth_chart_week_gamedays(season, load_schedules)
    week_snapshot = select_week_snapshots(df['snapshot_date'].unique().tolist(), week_first_gameday)

    frames = []
    for week, snap_date in week_snapshot.items():
        if snap_date is None:
            continue
        week_df = df[df['snapshot_date'] == snap_date][['gsis_id', 'pos_rank']].rename(
            columns={'gsis_id': 'player_id', 'pos_rank': 'depth_chart_rank'}
        ).copy()
        week_df['season'] = season
        week_df['week'] = week
        frames.append(week_df)

    if not frames:
        return pd.DataFrame(columns=['player_id', 'season', 'week', 'depth_chart_rank'])

    return pd.concat(frames, ignore_index=True)[['player_id', 'season', 'week', 'depth_chart_rank']]


def _old_schema_depth_chart(depth_df):
    """Old nflreadpy depth-chart schema: gsis_id, depth_position, depth_team, week, season."""
    df = depth_df[depth_df['depth_position'].isin(['WR', 'TE'])].copy()
    df = df.rename(columns={'gsis_id': 'player_id', 'depth_team': 'depth_chart_rank'})
    return df[['player_id', 'season', 'week', 'depth_chart_rank']]


def get_depth_chart_data(seasons, load_depth_charts=nfl.load_depth_charts, load_schedules=nfl.load_schedules):
    """Fetches and cleans weekly depth chart data for any season, handling both the
    old (season/week already present) and new (dt-snapshot based) nflreadpy schemas.
    """
    frames = []
    for season in seasons:
        raw = load_depth_charts([season])
        if hasattr(raw, 'to_pandas'):
            raw = raw.to_pandas()

        if 'dt' in raw.columns:
            frames.append(_new_schema_depth_chart(raw, season, load_schedules))
        else:
            frames.append(_old_schema_depth_chart(raw))

    if not frames:
        return pd.DataFrame(columns=['player_id', 'season', 'week', 'depth_chart_rank'])

    result = pd.concat(frames, ignore_index=True)
    result['depth_chart_rank'] = pd.to_numeric(result['depth_chart_rank'], errors='coerce')
    result = result.dropna(subset=['depth_chart_rank'])
    result = result.groupby(['player_id', 'season', 'week'], as_index=False)['depth_chart_rank'].min()
    result['depth_chart_rank'] = result['depth_chart_rank'].astype(int)
    return result


def get_depth_chart_for_week(season, week, load_depth_charts=nfl.load_depth_charts, load_schedules=nfl.load_schedules):
    """Returns depth chart data for a single season/week."""
    df = get_depth_chart_data([season], load_depth_charts=load_depth_charts, load_schedules=load_schedules)
    return df[df['week'] == week]

def get_snap_counts(years):
    """Fetches and prepares snap count data for joining."""
    snap_df = nfl.load_snap_counts(years)
    
    # Select columns needed for joining and rename 'player' to match our main df
    snap_df = snap_df.select(['pfr_player_id', 'player', 'season', 'week', 'offense_pct'])
    snap_df = snap_df.rename({
        'pfr_player_id': 'pfr_id',
        'offense_pct': 'offense_snap_share',
        'player': 'player_display_name'
    })

    snap_df = snap_df.with_columns([
        pl.col('player_display_name')
          .str.to_lowercase()
          .str.replace_all(r'[^a-z0-9\s]', '')
          .str.replace(r'\s(jr|sr|ii|iii|iv)$', '')
          .str.strip_chars()
          .alias('merge_name')
    ])

    # ids = nfl.import_ids()
    # # Get player IDs and merge names
    # ids= ids[['pfr_id', 'gsis_id']]
    # #merge player IDs with snap counts
    # snap_df = snap_df.join(ids, on='pfr_id', how='left')

    # #rename gsis_id to player_id
    # snap_df = snap_df.rename({'gsis_id': 'player_id'})
    # #drop pfr_id
    snap_df = snap_df.drop(['pfr_id', 'player_display_name'])

    # Ensure snap share is a float between 0 and 1
    snap_df = snap_df.with_columns([
        pl.col('offense_snap_share').fill_null(0)
    ])
    
    return snap_df.to_pandas()

""" def get_ngs_data_rushing(years):
    ngs_rushing_df = nfl.load_nextgen_stats(stat_type="rushing", seasons=years)

    ngs_rushing_df = ngs_rushing_df.select([
        'season', 'week', 'player_gsis_id', 'rush_yards_over_expected_per_att', 
        'rush_pct_over_expected', 'avg_time_to_los', 'percent_attempts_gte_eight_defenders'
    ])
    
    ngs_rushing_df = ngs_rushing_df.rename({
        'player_gsis_id': 'player_id'
    })

    return ngs_rushing_df.to_pandas() """

def get_ngs_data_receiving(years):
    ngs_receiving_df = nfl.load_nextgen_stats(stat_type="receiving", seasons=years)

    ngs_receiving_df = ngs_receiving_df.select([
        'season', 'week', 'player_gsis_id', 
        'avg_cushion', 'avg_separation', 'avg_intended_air_yards', 'percent_share_of_intended_air_yards', 
        'catch_percentage', 'avg_expected_yac', 'avg_yac_above_expectation'
    ])
    
    ngs_receiving_df = ngs_receiving_df.rename({
        'player_gsis_id': 'player_id'
    })

    return ngs_receiving_df.to_pandas()


def get_game_data(years):
    game_df = nfl.load_schedules(years)
    
    # Select relevant columns including both home and away teams, plus game context
    game_df = game_df.select([
        'season', 'week', 'game_id', 'home_team', 'away_team',
        'weekday',      # Day of week (Thursday/Sunday/Monday/Saturday)
        'gametime',     # Game time (for primetime detection)
        'roof',         # dome/outdoors/closed/open
        'surface',      # grass/fieldturf/a_turf/matrixturf
        'temp',         # Temperature (may be null)
        'wind', 
        'div_game'     # Wind speed (may be null)
    ])
    game_df_pd = game_df.to_pandas()
    
    # Create two rows per game: one for home team, one for away team
    # Include game context fields for both
    home_games = game_df_pd[['season', 'week', 'game_id', 'home_team']].rename(columns={'home_team': 'team'})
    home_games['is_home'] = 1
    
    away_games = game_df_pd[['season', 'week', 'game_id', 'away_team']].rename(columns={'away_team': 'team'})
    away_games['is_home'] = 0
    
    # Combine home and away games
    all_games = pd.concat([home_games, away_games], ignore_index=True)
    
    return all_games, game_df.to_pandas()

def get_ff_opportunity_data(years):
    ff_opportunity_df = nfl.load_ff_opportunity(seasons=years, stat_type='weekly')
    positions_to_keep = ['WR', 'TE']
    ff_opportunity_df = ff_opportunity_df.filter(pl.col('position').is_in(positions_to_keep))
    ff_opportunity_df = ff_opportunity_df.select([
        'season', 'week', 'player_id', 'rec_touchdown_exp', 'rec_touchdown_exp_team'
    ])
    ## player_id to integer
    
   
    ff_opportunity_df = ff_opportunity_df.to_pandas()
    
    # Ensure consistent data types for merging

    ff_opportunity_df['season'] = ff_opportunity_df['season'].astype(int)


    return ff_opportunity_df

def get_all_historic_data(years, team_map):
    
    pbp = nfl.load_pbp(years)
    rosters = nfl.load_rosters(years)
    pbp = pbp.filter(pl.col('week') <= 18)
    nfl_df = get_nfl_data(years)
    nfl_df = nfl_df[nfl_df['week'] <= 18]
    nfl_df['merge_name'] = nfl_df['player_display_name'].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()

    
    redzone_df = get_redzone_data(pbp)
    redzone_df = redzone_df[redzone_df['week'] <= 18]

    redzone_td_df = get_redzone_td_rate(pbp)
    redzone_td_df = redzone_td_df[redzone_td_df['week'] <= 18]

    team_play_df = get_team_play_data(years)
    team_play_df = team_play_df[team_play_df['week'] <= 18]

    ez_target_df = get_endzone_target_data(pbp)
    ez_target_df = ez_target_df[ez_target_df['week'] <= 18]

    odds_df = get_odds_data(years, team_map)
    odds_df = odds_df[odds_df['week'] <= 18]

    goal_line_df = get_goal_line_data(pbp)
    goal_line_df = goal_line_df[goal_line_df['week'] <= 18]

    green_zone_df = get_green_zone_data(pbp)
    green_zone_df = green_zone_df[green_zone_df['week'] <= 18]

    explosive_receiving_df = get_explosive_receiving_data(pbp)
    explosive_receiving_df = explosive_receiving_df[explosive_receiving_df['week'] <= 18]

    """ explosive_rushing_df = get_explosive_rushing_data(pbp)
    explosive_rushing_df = explosive_rushing_df[explosive_rushing_df['week'] <= 18] """

    positional_defense_df = get_opponent_positional_data(pbp, rosters)
    positional_defense_df = positional_defense_df[positional_defense_df['week'] <= 18]

    defensive_data_df = get_opponent_defensive_data(years)
    defensive_data_df = defensive_data_df[defensive_data_df['week'] <= 18]

    """ opponent_rushing_explosive_play_allowed_df = get_opponent_rushing_explosive_play_allowed(pbp)
    opponent_rushing_explosive_play_allowed_df = opponent_rushing_explosive_play_allowed_df[opponent_rushing_explosive_play_allowed_df['week'] <= 18]
    """

    opponent_receiving_explosive_play_allowed_df = get_opponent_receiving_explosive_play_allowed(pbp)
    opponent_receiving_explosive_play_allowed_df = opponent_receiving_explosive_play_allowed_df[opponent_receiving_explosive_play_allowed_df['week'] <= 18]

    depth_chart_df = get_depth_chart_data(years)
    depth_chart_df = depth_chart_df[depth_chart_df['week'] <= 18]

    snap_counts_df = get_snap_counts(years)
    snap_counts_df = snap_counts_df[snap_counts_df['week']<=18]

    """ ngs_rushing_df = get_ngs_data_rushing(years)
    ngs_rushing_df = ngs_rushing_df[ngs_rushing_df['week'] <= 18]
    """
    
    ngs_receiving_df = get_ngs_data_receiving(years)
    ngs_receiving_df = ngs_receiving_df[ngs_receiving_df['week'] <= 18]

    game_id_df, game_info_df = get_game_data(years)
    game_id_df = game_id_df[game_id_df['week'] <= 18]
    game_info_df = game_info_df[game_info_df['week'] <= 18]

    ff_opportunity_df = get_ff_opportunity_data(years)
    ff_opportunity_df = ff_opportunity_df[ff_opportunity_df['week'] <= 18]

    nfl_df = pd.merge(nfl_df, redzone_df, on=['player_id','week', 'season'], how='left')
    nfl_df = pd.merge(nfl_df, redzone_td_df, on=['team', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, team_play_df, on=['team', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, ez_target_df, on=['player_id', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, odds_df, on=['team', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, goal_line_df, on=['player_id', 'week', 'season'], how='left')
    nfl_df = pd.merge(nfl_df, green_zone_df, on=['player_id', 'week', 'season'], how='left')
    nfl_df = pd.merge(nfl_df, ff_opportunity_df, on=['player_id', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, explosive_receiving_df, on=['player_id', 'season', 'week'], how='left')
   # nfl_df = pd.merge(nfl_df, explosive_rushing_df, on=['player_id', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, positional_defense_df, on=['opponent_team', 'season', 'week'], how='left')
    #nfl_df = pd.merge(nfl_df, opponent_rushing_explosive_play_allowed_df, on=['opponent_team', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, opponent_receiving_explosive_play_allowed_df, on=['opponent_team', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, defensive_data_df, on=['opponent_team', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, depth_chart_df, on=['player_id', 'season', 'week'], how='left')
    nfl_df['depth_chart_rank'] = pd.to_numeric(nfl_df['depth_chart_rank'], errors='coerce').fillna(4).astype(int)

    nfl_df = pd.merge(nfl_df, snap_counts_df, on=['merge_name', 'season', 'week'])
    nfl_df['offense_snap_share'].fillna(0, inplace=True)

    #nfl_df = pd.merge(nfl_df, ngs_rushing_df, on=['player_id', 'season', 'week'], how='left')
    nfl_df = pd.merge(nfl_df, ngs_receiving_df, on=['player_id', 'season', 'week'], how='left')

    nfl_df = pd.merge(nfl_df, game_id_df, on=['season', 'week', 'team'], how='left')
    nfl_df = pd.merge(nfl_df, game_info_df, on=['season', 'week', 'game_id'], how='left')

    nfl_df.fillna(0, inplace=True)

    nfl_df.sort_values(by=['season', 'week', 'player_id'], inplace=True, ignore_index=True)

    #drop duplicate rows on player_id, season, week 
    nfl_df = nfl_df.drop_duplicates(subset=['player_id', 'season', 'week'])
    
    return nfl_df
    


def get_historical_vegas_data(years):
    """
    Loads and processes historical player touchdown odds from wr_te/vegas/{year}/week_{week}_td_odds.csv.
    Returns a DataFrame with columns: ['merge_name', 'season', 'week', 'market_implied_prob']
    """
    all_odds = []
    
    for year in years:
        # Iterate through weeks 1-18
        for week in range(1, 19):
            file_path = config.VEGAS_DIR / str(year) / f'week_{week}_td_odds.csv'
            
            if not file_path.exists():
                continue
                
            try:
                df = pd.read_csv(file_path)
                
                # Check for required columns
                if 'Player' not in df.columns or 'Odds' not in df.columns:
                    continue
                    
                # Standardize Player Name for merging
                # Logic matches predict_wr.py and get_snap_counts
                df['merge_name'] = df['Player'].astype(str).str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True).str.replace(r'\s(jr|sr|ii|iii|iv)$', '', regex=True).str.strip()
                
                # Calculate Implied Probability from American Odds
                # Positive Odds (+150): 100 / (Odds + 100)
                # Negative Odds (-150): |Odds| / (|Odds| + 100)
                
                # Ensure Odds is numeric
                df['Odds'] = pd.to_numeric(df['Odds'], errors='coerce')
                df = df.dropna(subset=['Odds'])
                
                prob_if_pos = 100 / (df['Odds'] + 100)
                prob_if_neg = df['Odds'].abs() / (df['Odds'].abs() + 100)
                
                df['market_implied_prob'] = np.where(df['Odds'] > 0, prob_if_pos, prob_if_neg)
                
                # Add metadata
                df['season'] = year
                df['week'] = week
                
                # Select relevant columns
                df_subset = df[['merge_name', 'season', 'week', 'market_implied_prob']].copy()
                all_odds.append(df_subset)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
                
    if not all_odds:
        return pd.DataFrame(columns=['merge_name', 'season', 'week', 'market_implied_prob'])
        
    return pd.concat(all_odds, ignore_index=True)
