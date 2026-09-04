import numpy as np
import pandas as pd
import pytest

import data_collection
import predict_wr
from features import WR_TE_FEATURES, PLAYER_EWM_STATS


# --- load_week_roster ---

def _season_roster_fake(seasons):
    return pd.DataFrame([
        {"gsis_id": "P1", "full_name": "Player One", "position": "WR", "team": "NYJ", "status": "ACT"},
        {"gsis_id": "P2", "full_name": "Player Two", "position": "TE", "team": "BUF", "status": "ACT"},
        # Inactive player should be dropped.
        {"gsis_id": "P3", "full_name": "Player Three", "position": "WR", "team": "NYJ", "status": "RES"},
        # Non-WR/TE should be dropped.
        {"gsis_id": "P4", "full_name": "Player Four", "position": "QB", "team": "NYJ", "status": "ACT"},
    ])


def _raises_valueerror(seasons):
    raise ValueError("Season must be between 1999 and 2025")


def test_load_week_roster_falls_back_to_season_roster_on_valueerror():
    out = predict_wr.load_week_roster(
        2026, 1,
        load_rosters_weekly=_raises_valueerror,
        load_rosters=_season_roster_fake,
    )

    assert set(out["player_id"]) == {"P1", "P2"}
    assert set(out["position"]) == {"WR", "TE"}
    assert list(out.columns) == ["player_id", "player_display_name", "position", "team"]


@pytest.mark.parametrize("source_code, expected_code", [("AZ", "ARI"), ("LAR", "LA"), ("LVR", "LV"), ("NYJ", "NYJ")])
def test_load_week_roster_normalizes_evidenced_team_aliases_at_boundary(source_code, expected_code):
    def season_roster(_seasons):
        return pd.DataFrame([{
            "gsis_id": "P1", "full_name": "Player One", "position": "WR",
            "team": source_code, "status": "ACT",
        }])

    out = predict_wr.load_week_roster(
        2026, 1,
        load_rosters_weekly=_raises_valueerror,
        load_rosters=season_roster,
    )

    assert out.iloc[0]["team"] == expected_code


def _weekly_roster_fake(seasons):
    return pd.DataFrame([
        {"gsis_id": "P1", "full_name": "Player One", "position": "WR", "team": "A", "status": "ACT", "week": 1},
        {"gsis_id": "P1", "full_name": "Player One", "position": "WR", "team": "B", "status": "ACT", "week": 3},
        {"gsis_id": "P2", "full_name": "Player Two", "position": "TE", "team": "C", "status": "ACT", "week": 3},
    ])


def _unused_load_rosters(seasons):
    raise AssertionError("load_rosters should not be called when weekly data is available")


def test_load_week_roster_uses_weekly_data_scoped_to_week():
    out = predict_wr.load_week_roster(
        2026, 3,
        load_rosters_weekly=_weekly_roster_fake,
        load_rosters=_unused_load_rosters,
    )

    assert len(out) == 2
    p1 = out[out["player_id"] == "P1"].iloc[0]
    assert p1["team"] == "B"


# --- merge_name ---

def test_merge_name_reexports_odds_match():
    import odds_match
    assert predict_wr.merge_name is odds_match.merge_name


# --- join_odds ---

TEAM_MAP = {
    "New York Jets": "NYJ",
    "Buffalo Bills": "BUF",
    "Los Angeles Chargers": "LAC",
    "Denver Broncos": "DEN",
}


def _predictions_two_mike_williams():
    return pd.DataFrame([
        {"player_id": "P1", "player_display_name": "Mike Williams", "team": "NYJ",
         "predicted_touchdown_probability": 0.20},
        {"player_id": "P2", "player_display_name": "Mike Williams", "team": "LAC",
         "predicted_touchdown_probability": 0.15},
    ])


def _odds_two_mike_williams():
    return pd.DataFrame([
        {"description": "Mike Williams", "home_team": "New York Jets", "away_team": "Buffalo Bills",
         "price": 350, "bookmaker": "book1"},
        {"description": "Mike Williams", "home_team": "Los Angeles Chargers", "away_team": "Denver Broncos",
         "price": 280, "bookmaker": "book1"},
    ])


def test_join_odds_disambiguates_by_team_and_keeps_highest_price():
    out = predict_wr.join_odds(_predictions_two_mike_williams(), _odds_two_mike_williams(), TEAM_MAP)

    assert len(out) == 2
    nyj = out[out["team"] == "NYJ"].iloc[0]
    lac = out[out["team"] == "LAC"].iloc[0]
    assert nyj["price"] == 350
    assert lac["price"] == 280
    assert nyj["market_implied_prob"] == pytest.approx(100 / 450)


def test_join_odds_keeps_highest_price_across_bookmakers():
    odds = pd.concat([_odds_two_mike_williams(), pd.DataFrame([{
        "description": "Mike Williams", "home_team": "New York Jets", "away_team": "Buffalo Bills",
        "price": 375, "bookmaker": "book2",
    }])], ignore_index=True)

    out = predict_wr.join_odds(_predictions_two_mike_williams(), odds, TEAM_MAP)

    nyj = out[out["team"] == "NYJ"].iloc[0]
    assert nyj["price"] == 375
    assert nyj["bookmaker"] == "book2"


def test_join_odds_drops_predictions_without_a_match():
    predictions = pd.concat([_predictions_two_mike_williams(), pd.DataFrame([{
        "player_id": "P3", "player_display_name": "Nobody Here", "team": "SF",
        "predicted_touchdown_probability": 0.05,
    }])], ignore_index=True)

    out = predict_wr.join_odds(predictions, _odds_two_mike_williams(), TEAM_MAP)

    assert len(out) == 2
    assert "P3" not in out["player_id"].values


def test_join_odds_market_implied_prob_negative_price():
    predictions = pd.DataFrame([{
        "player_id": "P1", "player_display_name": "Mike Williams", "team": "NYJ",
        "predicted_touchdown_probability": 0.5,
    }])
    odds = pd.DataFrame([{
        "description": "Mike Williams", "home_team": "New York Jets", "away_team": "Buffalo Bills",
        "price": -150, "bookmaker": "book1",
    }])

    out = predict_wr.join_odds(predictions, odds, TEAM_MAP)

    assert out.iloc[0]["market_implied_prob"] == pytest.approx(150 / 250)


def test_join_odds_computes_model_edge():
    out = predict_wr.join_odds(_predictions_two_mike_williams(), _odds_two_mike_williams(), TEAM_MAP)
    nyj = out[out["team"] == "NYJ"].iloc[0]
    expected_edge = 0.20 - (100 / 450)
    assert nyj["model_edge"] == pytest.approx(expected_edge)


# --- predict_touchdown_scorers ---

class _FakeModel:
    def predict_proba(self, X):
        # Always predict 0.7 for the first row's feature ordering, 0.3 otherwise;
        # shape must match (n_rows, 2).
        n = len(X)
        return np.array([[0.7, 0.3]] * n)


def _tiny_feature_df():
    """Two players, 3 prior weeks (weeks 1-3), season 2026, target week 4."""
    defense_cols = [
        'passing_tds_allowed_to_WR', 'passing_tds_allowed_to_TE',
        'receiving_yards_allowed', 'receiving_epa_allowed',
        'receiving_air_yards_allowed', 'explosive_receiving_plays_allowed',
    ]
    rows = []
    for player_id, team, opponent_team in [("P1", "NYJ", "BUF"), ("P2", "LAC", "DEN")]:
        for week in [1, 2, 3]:
            row = {
                "season": 2026, "week": week, "player_id": player_id,
                "team": team, "opponent_team": opponent_team,
            }
            for stat in PLAYER_EWM_STATS:
                row[stat] = 1.0
            for col in defense_cols:
                row[col] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def _tiny_lines_df():
    return data_collection.schedule_to_team_lines(pd.DataFrame([
        {"season": 2026, "week": 4, "game_type": "REG", "home_team": "NYJ", "away_team": "BUF",
         "spread_line": -3.0, "total_line": 45.0},
        {"season": 2026, "week": 4, "game_type": "REG", "home_team": "LAC", "away_team": "DEN",
         "spread_line": 2.0, "total_line": 41.0},
    ]))


def _tiny_depth_df():
    # Only P1 has a depth chart entry; P2 should fall back to rank 4.
    return pd.DataFrame([
        {"player_id": "P1", "season": 2026, "week": 4, "depth_chart_rank": 1},
    ])


def _tiny_roster_df():
    return pd.DataFrame([
        {"player_id": "P1", "player_display_name": "Player One", "position": "WR", "team": "NYJ"},
        {"player_id": "P2", "player_display_name": "Player Two", "position": "TE", "team": "LAC"},
    ])


def test_predict_touchdown_scorers_smoke():
    out = predict_wr.predict_touchdown_scorers(
        _tiny_feature_df(), _FakeModel(), None, 2026, 4,
        _tiny_lines_df(), _tiny_depth_df(), _tiny_roster_df(),
    )

    for col in WR_TE_FEATURES:
        assert col in out.columns, f"missing feature column {col}"

    assert len(out) == 2
    p2 = out[out["player_id"] == "P2"].iloc[0]
    assert p2["depth_chart_rank"] == 4
    p1 = out[out["player_id"] == "P1"].iloc[0]
    assert p1["depth_chart_rank"] == 1

    assert "predicted_touchdown_probability" in out.columns
    assert (out["predicted_touchdown_probability"] == 0.3).all()


class _DiagnosticCalibrator:
    def predict_proba(self, X):
        p = np.asarray(X).ravel() * 0.5 + 0.25
        return np.column_stack([1.0 - p, p])


def test_direct_prediction_call_applies_optional_calibrator_for_diagnostics():
    out = predict_wr.predict_touchdown_scorers(
        _tiny_feature_df(), _FakeModel(), _DiagnosticCalibrator(), 2026, 4,
        _tiny_lines_df(), _tiny_depth_df(), _tiny_roster_df(),
    )

    assert np.allclose(out["predicted_touchdown_probability"], 0.4)


def test_production_main_loads_raw_forest_only_and_records_variant(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    vegas_dir = tmp_path / "vegas" / "2026"
    predictions_dir = tmp_path / "predictions"
    for path in (data_dir, models_dir, vegas_dir):
        path.mkdir(parents=True)
    pd.DataFrame({"team_name": ["New York Jets"], "team_id": ["NYJ"]}).to_csv(
        data_dir / "nfl_teams.csv", index=False
    )
    pd.DataFrame({"feature": [1]}).to_csv(data_dir / "raw_nfl_data.csv", index=False)
    pd.DataFrame({"description": ["Player One"]}).to_csv(
        vegas_dir / "week_1_td_odds_open.csv", index=False
    )

    monkeypatch.setattr(predict_wr.config, "DATA_DIR", data_dir)
    monkeypatch.setattr(predict_wr.config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(predict_wr.config, "VEGAS_DIR", tmp_path / "vegas")
    monkeypatch.setattr(predict_wr.config, "PREDICTIONS_DIR", predictions_dir)
    monkeypatch.setattr(predict_wr.config, "SEASON", 2026)
    monkeypatch.setattr(predict_wr.config, "WEEK", 1)

    loaded = []

    def fake_load(path):
        loaded.append(path)
        return object()

    captured = {}

    def fake_predict(feature_df, model, calibrator, season, week, lines, depth, roster):
        captured["calibrator"] = calibrator
        return pd.DataFrame([{
            "season": season, "week": week, "player_id": "P1",
            "player_display_name": "Player One", "team": "NYJ", "opponent_team": "BUF",
            "position": "WR", "predicted_touchdown_probability": 0.3,
        }])

    monkeypatch.setattr(predict_wr, "load_joblib_locally", fake_load)
    fetch_calls = []
    monkeypatch.setattr(
        predict_wr.fetch_live_odds,
        "fetch",
        lambda *args, **kwargs: fetch_calls.append((args, kwargs))
        or pd.DataFrame({"description": ["Player One"]}),
    )
    monkeypatch.setattr(predict_wr, "predict_touchdown_scorers", fake_predict)
    monkeypatch.setattr(predict_wr.data, "get_week_lines", lambda season, week: pd.DataFrame())
    monkeypatch.setattr(predict_wr.data, "get_depth_chart_for_week", lambda season, week: pd.DataFrame())
    monkeypatch.setattr(predict_wr, "load_week_roster", lambda season, week: pd.DataFrame())
    monkeypatch.setattr(predict_wr, "join_odds", lambda predictions, odds, team_map: predictions.assign(
        price=200, market_implied_prob=1 / 3, model_edge=-1 / 30, bookmaker="book"
    ))

    predict_wr.main()

    assert fetch_calls == [((2026, 1, "open", ["draftkings"]), {})]
    assert loaded == [models_dir / "wr_te_rf_final.pkl"]
    assert captured["calibrator"] is None
    output = pd.read_csv(predictions_dir / "2026" / "week_1.csv")
    assert output.loc[0, "probability_variant"] == "random_forest_current/raw"


def test_production_main_rejects_empty_odds_without_writing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_wr.config, "SEASON", 2026)
    monkeypatch.setattr(predict_wr.config, "WEEK", 1)
    predictions_dir = tmp_path / "predictions"
    monkeypatch.setattr(predict_wr.config, "PREDICTIONS_DIR", predictions_dir)
    monkeypatch.setattr(
        predict_wr.fetch_live_odds, "fetch", lambda *args, **kwargs: pd.DataFrame()
    )

    with pytest.raises(ValueError, match="contains no player odds rows"):
        predict_wr.main()
    assert not predictions_dir.exists()


def test_prediction_history_excludes_target_and_future_rows():
    feature_df = _tiny_feature_df()
    future_rows = feature_df[feature_df["week"] == 3].copy()
    future_rows["week"] = [4] * len(future_rows)
    future_rows["receptions"] = 999.0
    feature_df = pd.concat([feature_df, future_rows], ignore_index=True)

    history = predict_wr.prediction_history(feature_df, 2026, 4)

    assert set(history["week"]) == {1, 2, 3}
    assert history["receptions"].max() == 1.0


class _CaptureModel:
    def predict_proba(self, X):
        self.features = X.copy()
        return np.tile([[0.4, 0.6]], (len(X), 1))


def test_prediction_integration_uses_completed_prior_weeks_only():
    feature_df = _tiny_feature_df()
    future_rows = feature_df[feature_df["week"] == 3].copy()
    future_rows["week"] = [4] * len(future_rows)
    future_rows["receptions"] = 999.0
    future_rows["receiving_yards"] = 999.0
    feature_df = pd.concat([feature_df, future_rows], ignore_index=True)
    model = _CaptureModel()

    out = predict_wr.predict_touchdown_scorers(
        feature_df,
        model,
        None,
        2026,
        4,
        _tiny_lines_df(),
        _tiny_depth_df(),
        _tiny_roster_df(),
    )

    assert len(out) == 2
    assert (model.features["avg_receptions"] == 1.0).all()


def test_prediction_fails_when_schedule_line_context_is_missing():
    missing_lines = pd.DataFrame(
        columns=["team", "opponent", "implied_total", "spread_line"]
    )

    with pytest.raises(ValueError, match="schedule line"):
        predict_wr.predict_touchdown_scorers(
            _tiny_feature_df(),
            _FakeModel(),
            None,
            2026,
            4,
            missing_lines,
            _tiny_depth_df(),
            _tiny_roster_df(),
        )


def test_prediction_excludes_active_bye_team_before_line_validation():
    roster = pd.concat([
        _tiny_roster_df(),
        pd.DataFrame([{
            "player_id": "P3", "player_display_name": "Bye Player",
            "position": "WR", "team": "BYE",
        }]),
    ], ignore_index=True)

    out = predict_wr.predict_touchdown_scorers(
        _tiny_feature_df(), _FakeModel(), None, 2026, 4,
        _tiny_lines_df(), _tiny_depth_df(), roster,
    )

    assert set(out["team"]) == {"NYJ", "LAC"}
    assert "P3" not in set(out["player_id"])
