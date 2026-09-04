import pandas as pd

import features
import train_wr
import predict_wr


def test_wr_te_features_length_and_uniqueness():
    assert len(features.WR_TE_FEATURES) == 23
    assert len(set(features.WR_TE_FEATURES)) == 23


def test_redzone_td_rate_not_in_features():
    assert 'redzone_td_rate' not in features.WR_TE_FEATURES


def test_train_wr_uses_shared_feature_list():
    assert train_wr.WR_TE_FEATURES is features.WR_TE_FEATURES


def test_predict_wr_uses_shared_feature_list():
    assert predict_wr.WR_TE_FEATURES is features.WR_TE_FEATURES


def _build_synthetic_frame():
    """Two players (one WR, one TE) on the same team, four weeks of data,
    including every column feature_engineering reads."""
    players = [
        ('P1', 'WR'),
        ('P2', 'TE'),
    ]
    weeks = [1, 2, 3, 4]

    rows = []
    for player_id, position in players:
        for week in weeks:
            row = {
                'player_id': player_id,
                'season': 2024,
                'week': week,
                'team': 'AAA',
                'opponent_team': 'BBB',
                'position': position,
            }
            # PLAYER_EWM_STATS raw columns
            for stat in features.PLAYER_EWM_STATS:
                row[stat] = 0.1
            # opponent/defense columns read by feature_engineering
            row['passing_tds_allowed_to_WR'] = 0.2
            row['passing_tds_allowed_to_TE'] = 0.1
            row['receiving_yards_allowed'] = 200.0
            row['receiving_epa_allowed'] = 0.05
            row['receiving_air_yards_allowed'] = 150.0
            row['explosive_receiving_plays_allowed'] = 2.0

            rows.append(row)

    return pd.DataFrame(rows)


def test_feature_engineering_drops_leaky_and_matchup_columns():
    df = _build_synthetic_frame()
    result = train_wr.feature_engineering(df)

    assert 'redzone_td_rate' not in result.columns
    assert 'pass_rate' not in result.columns
    assert 'pass_matchup_value' not in result.columns


def test_feature_engineering_produces_expected_feature_columns():
    df = _build_synthetic_frame()
    result = train_wr.feature_engineering(df)

    excluded = {'implied_total', 'spread_line', 'depth_chart_rank'}
    expected_columns = [f for f in features.WR_TE_FEATURES if f not in excluded]

    missing = [col for col in expected_columns if col not in result.columns]
    assert not missing, f"Missing expected feature columns: {missing}"


def test_feature_engineering_fills_numeric_missing_values_without_coercing_strings():
    df = _build_synthetic_frame()
    df["surface"] = pd.Series(["grass"] + [pd.NA] * (len(df) - 1), dtype="string")
    df.loc[0, "receptions"] = pd.NA

    result = train_wr.feature_engineering(df)

    assert result["surface"].isna().any()
    first_player_week = result.query("player_id == 'P1' and week == 1").iloc[0]
    assert first_player_week["avg_receptions"] == 0
    assert result["avg_receptions"].notna().all()
