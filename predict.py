import os
import pandas as pd
from datetime import datetime

from data_collection import (
    get_all_historic_data,
    transform_future_odds,
)
from feature_engineering import engineer_features_for_touchdown_prediction
from model import load_model
import nfl_data_py as nfl


TEAM_MAP = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LVR', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
}

def _normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s(jr|sr|ii|iii|iv)$", "", s).strip()
    return s


def load_week_lines(lines_csv: str) -> pd.DataFrame:
    raw = pd.read_csv(lines_csv)
    lines = transform_future_odds(raw, TEAM_MAP)
    # Ensure abbreviations align with model data conventions
    # Our training uses 'LAR' and 'LVR'; transform_future_odds ends with 'LA'/'LV'
    lines['team'] = lines['team'].replace({'LA': 'LAR', 'LV': 'LVR'})
    lines['opponent'] = lines['opponent'].replace({'LA': 'LAR', 'LV': 'LVR'})
    return lines

def load_td_odds_player_list(td_odds_csv: str) -> pd.DataFrame:
    df = pd.read_csv(td_odds_csv)
    df = df[df['market'] == 'player_anytime_td'].copy()
    df['merge_name'] = df['description'].apply(_normalize_name)
    df = df.drop_duplicates(subset=['merge_name'])
    return df[['merge_name', 'description', 'home_team', 'away_team']]


def build_inference_dataframe_for_week(season: int, week: int, lines_csv: str) -> pd.DataFrame:
    """
    Build a player-week inference dataframe for the given season/week using only past data.
    Returns one row per player with historical avg_* features (up to week-1) and week-level
    opponent/implied_total merged in.
    """
    assert season == 2025, "This helper is currently tailored for 2025 inference."

    # 1) Historic engineered features up to 2024 (no 2025 stats to avoid leakage)
    historic_years = [2020, 2021, 2022, 2023, 2024]
    engineered = engineer_features_for_touchdown_prediction(historic_years, TEAM_MAP, window=5)

    # Latest per-player feature snapshot (through 2024)
    engineered = engineered.sort_values(['player_id', 'season', 'week'])
    latest_player_feats = engineered.groupby('player_id').tail(1).copy()

    # Keep essential identity columns for later joins
    id_cols = ['player_id', 'player_display_name', 'position', 'recent_team']
    base_cols = [c for c in latest_player_feats.columns if c.startswith('avg_')] + [
        'rush_matchup_value', 'pass_matchup_value', 'implied_total', 'depth_chart_rank'
    ]
    latest_player_feats = latest_player_feats[[c for c in id_cols + base_cols if c in latest_player_feats.columns]]

    # 2) Load lines for target week (team, opponent, implied_total)
    lines = load_week_lines(lines_csv)
    lines = lines.rename(columns={'team': 'recent_team', 'opponent': 'opponent_team'})

    # 3) Build a roster for week (from lines opponents, we select all players historically seen on each team)
    # Use latest_player_feats to select players by their most recent team
    week_df = latest_player_feats.merge(lines[['recent_team', 'opponent_team', 'implied_total']],
                                        on='recent_team', how='inner')

    # 4) Recompute matchup features using week opponent defensive avgs as-of end 2024
    # Derive team-level opponent defensive averages (end of 2024) from engineered df
    def_cols = [
        'avg_rushing_tds_allowed_to_RB', 'avg_passing_tds_allowed_to_RB',
        'avg_passing_tds_allowed_to_WR', 'avg_passing_tds_allowed_to_TE',
        'avg_rushing_tds_allowed_to_QB'
    ]
    team_def = engineered.sort_values(['opponent_team', 'season', 'week']).groupby('opponent_team').tail(1)
    team_def = team_def[['opponent_team'] + [c for c in def_cols if c in engineered.columns]].copy()

    week_df = week_df.merge(team_def, on='opponent_team', how='left')

    # Updated matchup values
    if 'avg_redzone_carry_share' in week_df.columns and 'avg_rushing_tds_allowed_to_RB' in week_df.columns:
        rb_mask = week_df['position'] == 'RB'
        week_df.loc[rb_mask, 'rush_matchup_value'] = (
            week_df.loc[rb_mask, 'avg_redzone_carry_share'] * week_df.loc[rb_mask, 'avg_rushing_tds_allowed_to_RB']
        )
    if 'avg_redzone_carry_share' in week_df.columns and 'avg_rushing_tds_allowed_to_QB' in week_df.columns:
        qb_mask = week_df['position'] == 'QB'
        week_df.loc[qb_mask, 'rush_matchup_value'] = (
            week_df.loc[qb_mask, 'avg_redzone_carry_share'] * week_df.loc[qb_mask, 'avg_rushing_tds_allowed_to_QB']
        )
    if 'avg_redzone_target_share' in week_df.columns:
        wr_mask = week_df['position'] == 'WR'
        te_mask = week_df['position'] == 'TE'
        if 'avg_passing_tds_allowed_to_WR' in week_df.columns:
            week_df.loc[wr_mask, 'pass_matchup_value'] = (
                week_df.loc[wr_mask, 'avg_redzone_target_share'] * week_df.loc[wr_mask, 'avg_passing_tds_allowed_to_WR']
            )
        if 'avg_passing_tds_allowed_to_TE' in week_df.columns:
            week_df.loc[te_mask, 'pass_matchup_value'] = (
                week_df.loc[te_mask, 'avg_redzone_target_share'] * week_df.loc[te_mask, 'avg_passing_tds_allowed_to_TE']
            )

    # Ensure required columns exist (fill missing engineered columns with 0)
    week_df = week_df.fillna(0)
    week_df['season'] = season
    week_df['week'] = week

    return week_df


def score_and_rank_all_players(week_df: pd.DataFrame) -> pd.DataFrame:
    """Score players with calibrated models per position and return unified ranked list."""
    positions = ['RB', 'WR_TE', 'QB']
    model_files = {
        'RB': 'models/RB_RandomForest_touchdown_model.joblib',
        'WR_TE': 'models/WR_TE_RandomForest_touchdown_model.joblib',
        'QB': 'models/QB_RandomForest_touchdown_model.joblib',
    }

    scored_frames = []
    for pos in positions:
        model_path = model_files[pos]
        if not os.path.exists(model_path):
            continue
        pkg = load_model(model_path)
        model = pkg['model']
        feature_cols = pkg['feature_columns']

        if pos == 'WR_TE':
            pos_mask = week_df['position'].isin(['WR', 'TE'])
        else:
            pos_mask = week_df['position'] == pos

        df_pos = week_df.loc[pos_mask].copy()
        # Align features to training columns
        for c in feature_cols:
            if c not in df_pos.columns:
                df_pos[c] = 0
        X = df_pos[feature_cols].fillna(0)

        df_pos['prob_td'] = model.predict_proba(X)[:, 1]
        df_pos['position_group'] = pos
        scored_frames.append(df_pos)

    if not scored_frames:
        return pd.DataFrame()

    scored = pd.concat(scored_frames, ignore_index=True)
    scored = scored.sort_values('prob_td', ascending=False)
    # Select output columns
    out_cols = [
        'season', 'week', 'player_id', 'player_display_name', 'position', 'position_group',
        'recent_team', 'opponent_team', 'implied_total', 'prob_td'
    ]
    out_cols = [c for c in out_cols if c in scored.columns]
    return scored[out_cols]


def main():
    season = 2025
    week = 1
    lines_csv = 'data/week_1_lines.csv'
    td_odds_csv = 'data/week_1_td_odds.csv'

    print('🔧 Building inference dataset...')
    week_df = build_inference_dataframe_for_week(season, week, lines_csv)

    # Filter to only players present in TD odds list
    print('🧾 Loading TD odds player list...')
    td_list = load_td_odds_player_list(td_odds_csv)
    week_df['merge_name'] = week_df['player_display_name'].apply(_normalize_name)
    week_df = week_df.merge(td_list[['merge_name']], on='merge_name', how='inner')

    print('🔮 Scoring players...')
    ranked = score_and_rank_all_players(week_df)

    os.makedirs('candidates', exist_ok=True)
    out_path = f'candidates/{season}_week_{week}_candidates.csv'
    ranked.to_csv(out_path, index=False)
    print(f'✅ Saved ranked candidates: {out_path} ({len(ranked)} players)')


if __name__ == '__main__':
    main()


