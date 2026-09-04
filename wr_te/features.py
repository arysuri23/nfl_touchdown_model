"""Shared feature lists for the WR/TE touchdown model.

Single source of truth for `train_wr.py` and `predict_wr.py` so the two
scripts cannot silently drift apart. `redzone_td_rate` is intentionally
excluded from WR_TE_FEATURES: it is computed from the same-game redzone
trips/tds and leaks information not available before the game is played.
"""

# -- Position-Specific Feature List (WR/TE) --
WR_TE_FEATURES = [
    'avg_offense_snap_share',
    'avg_wopr',
    'avg_target_share',
    'avg_receiving_epa',
    'avg_racr',
    'avg_endzone_targets',
    'avg_endzone_target_share',
    'passing_tds_allowed_to_WR',
    'passing_tds_allowed_to_TE',
    'implied_total',
    'spread_line',
    'depth_chart_rank',
    'avg_scored_touchdown',
    'avg_receptions',
    'avg_receiving_yards',
    'avg_receiving_air_yards',
    'avg_receiving_yards_allowed',
    'avg_receiving_epa_allowed',
    'avg_receiving_air_yards_allowed',
    'avg_explosive_receiving_plays',
    'avg_explosive_receiving_plays_allowed',
    'avg_rec_touchdown_exp',
    'avg_rec_touchdown_exp_team',
]

# -- Raw per-player stats that feature_engineering/transform_features
#    turn into `avg_<stat>` EWM features. --
PLAYER_EWM_STATS = [
    'receptions',
    'receiving_yards',
    'wopr',
    'receiving_epa',
    'target_share',
    'receiving_air_yards',
    'racr',
    'scored_touchdown',
    'redzone_target_share',
    'total_tds',
    'endzone_targets',
    'endzone_target_share',
    'inside_5_target_share',
    'inside_10_targets',
    'offense_snap_share',
    'avg_cushion',
    'avg_separation',
    'avg_intended_air_yards',
    'percent_share_of_intended_air_yards',
    'catch_percentage',
    'avg_expected_yac',
    'avg_yac_above_expectation',
    'explosive_receiving_plays',
    'rec_touchdown_exp',
    'rec_touchdown_exp_team',
]
