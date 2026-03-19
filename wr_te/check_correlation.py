
import pandas as pd
import numpy as np
from data_collection import get_all_historic_data
from train_wr import feature_engineering

def check_correlations():
    print("Loading data...")
    nfl_teams = pd.read_csv('data/nfl_teams.csv')
    team_map = dict(zip(nfl_teams['team_name'], nfl_teams['team_id']))
    # Load just a few years to be fast
    df = get_all_historic_data([2022, 2023], team_map)
    
    print("Engineering features...")
    df = feature_engineering(df)
    
    features_to_check = [
        'avg_endzone_target_share',
        'avg_redzone_target_share',
        'avg_inside_10_targets'
    ]
    
    # Filter for rows where players actually played (snap share > 0) to avoid noise from 0s
    mask = df['offense_snap_share'] > 0.1
    subset = df.loc[mask, features_to_check]
    
    print("\nCorrelation Matrix (Active Players):")
    print(subset.corr().round(3))

if __name__ == "__main__":
    check_correlations()
