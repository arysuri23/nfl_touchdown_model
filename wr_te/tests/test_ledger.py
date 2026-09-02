import pandas as pd
import pytest

import ledger


TEAM_MAP = {
    "New York Jets": "NYJ",
    "Buffalo Bills": "BUF",
    "Los Angeles Chargers": "LAC",
    "Denver Broncos": "DEN",
}


# --------------------------------------------------------------------------
# implied_prob / decimal_odds
# --------------------------------------------------------------------------

def test_implied_prob_positive_price():
    assert ledger.implied_prob(350) == pytest.approx(100 / 450)


def test_implied_prob_negative_price():
    assert ledger.implied_prob(-150) == pytest.approx(0.6)


def test_decimal_odds_positive_price():
    assert ledger.decimal_odds(350) == 4.5


def test_decimal_odds_negative_price():
    assert ledger.decimal_odds(-150) == pytest.approx(5 / 3)


# --------------------------------------------------------------------------
# record_picks
# --------------------------------------------------------------------------

def _predictions_7rows():
    rows = []
    # 7 players, varying probability, edge, and price.
    data = [
        # player_id, prob,  price, edge
        ("P1", 0.40, 150, 0.10),
        ("P2", 0.35, 200, 0.30),
        ("P3", 0.30, 600, 0.50),  # highest edge but price > 400 -> excluded from top5_edge_le400
        ("P4", 0.25, 300, 0.05),
        ("P5", 0.20, 120, 0.02),
        ("P6", 0.15, 250, 0.20),
        ("P7", 0.10, 180, 0.01),
    ]
    for pid, prob, price, edge in data:
        rows.append({
            "season": 2026,
            "week": 1,
            "player_id": pid,
            "player_display_name": f"Player {pid}",
            "team": "NYJ",
            "opponent_team": "BUF",
            "position": "WR",
            "predicted_touchdown_probability": prob,
            "price": price,
            "market_implied_prob": ledger.implied_prob(price),
            "model_edge": edge,
            "bookmaker": "book1",
        })
    return pd.DataFrame(rows)


def test_record_picks_writes_top5_by_probability(tmp_path):
    preds = _predictions_7rows()
    ledger_path = tmp_path / "bets.csv"

    added = ledger.record_picks(preds, 2026, 1, "top5_prob", 1.0, ledger_path, now="2026-09-02T12:00:00")

    assert added == 5
    out = pd.read_csv(ledger_path)
    assert len(out) == 5
    expected_ids = set(preds.sort_values("predicted_touchdown_probability", ascending=False).head(5)["player_id"])
    assert set(out["player_id"]) == expected_ids

    # price_open / implied_open populated
    assert out["price_open"].notna().all()
    assert out["implied_open"].notna().all()

    # outcome/pnl/price_close/clv are empty
    assert out["outcome"].isna().all()
    assert out["pnl"].isna().all()
    assert out["price_close"].isna().all()
    assert out["clv"].isna().all()


def test_record_picks_is_idempotent(tmp_path):
    preds = _predictions_7rows()
    ledger_path = tmp_path / "bets.csv"

    ledger.record_picks(preds, 2026, 1, "top5_prob", 1.0, ledger_path, now="2026-09-02T12:00:00")
    added_again = ledger.record_picks(preds, 2026, 1, "top5_prob", 1.0, ledger_path, now="2026-09-02T12:05:00")

    assert added_again == 0
    out = pd.read_csv(ledger_path)
    assert len(out) == 5


def test_record_picks_top5_edge_le400_excludes_expensive_price(tmp_path):
    preds = _predictions_7rows()
    ledger_path = tmp_path / "bets.csv"

    ledger.record_picks(preds, 2026, 1, "top5_edge_le400", 1.0, ledger_path, now="2026-09-02T12:00:00")

    out = pd.read_csv(ledger_path)
    # P3 has the highest edge (0.50) but price 600 > 400, must be excluded.
    assert "P3" not in out["player_id"].values
    assert len(out) == 5


# --------------------------------------------------------------------------
# attach_closing
# --------------------------------------------------------------------------

def test_attach_closing_updates_matched_and_leaves_unmatched_empty(tmp_path):
    ledger_path = tmp_path / "bets.csv"
    preds = pd.DataFrame([
        {
            "season": 2026, "week": 1, "player_id": "P1",
            "player_display_name": "Mike Williams", "team": "NYJ",
            "opponent_team": "BUF", "position": "WR",
            "predicted_touchdown_probability": 0.4,
            "price": 300, "market_implied_prob": ledger.implied_prob(300),
            "model_edge": 0.1, "bookmaker": "book1",
        },
        {
            "season": 2026, "week": 1, "player_id": "P2",
            "player_display_name": "Nobody Matched", "team": "LAC",
            "opponent_team": "DEN", "position": "WR",
            "predicted_touchdown_probability": 0.3,
            "price": 280, "market_implied_prob": ledger.implied_prob(280),
            "model_edge": 0.05, "bookmaker": "book1",
        },
    ])
    ledger.record_picks(preds, 2026, 1, "top5_prob", 1.0, ledger_path, now="2026-09-02T12:00:00")

    close_odds = pd.DataFrame([
        {
            "description": "Mike Williams",
            "home_team": "New York Jets",
            "away_team": "Buffalo Bills",
            "price": 250,
            "bookmaker": "book2",
        },
        # No closing-line row for "Nobody Matched" -> stays empty.
    ])

    updated = ledger.attach_closing(ledger_path, close_odds, TEAM_MAP)

    assert updated == 1
    out = pd.read_csv(ledger_path).set_index("player_id")
    assert out.loc["P1", "price_close"] == 250
    expected_clv = ledger.implied_prob(250) - ledger.implied_prob(300)
    assert out.loc["P1", "clv"] == pytest.approx(expected_clv)
    assert expected_clv > 0

    assert pd.isna(out.loc["P2", "price_close"])
    assert pd.isna(out.loc["P2", "clv"])


# --------------------------------------------------------------------------
# settle
# --------------------------------------------------------------------------

def _settle_ledger(tmp_path):
    ledger_path = tmp_path / "bets.csv"
    preds = pd.DataFrame([
        {
            "season": 2026, "week": 1, "player_id": "A",
            "player_display_name": "Player A", "team": "NYJ",
            "opponent_team": "BUF", "position": "WR",
            "predicted_touchdown_probability": 0.5,
            "price": 300, "market_implied_prob": ledger.implied_prob(300),
            "model_edge": 0.1, "bookmaker": "book1",
        },
        {
            "season": 2026, "week": 1, "player_id": "B",
            "player_display_name": "Player B", "team": "NYJ",
            "opponent_team": "BUF", "position": "WR",
            "predicted_touchdown_probability": 0.4,
            "price": 200, "market_implied_prob": ledger.implied_prob(200),
            "model_edge": 0.08, "bookmaker": "book1",
        },
        {
            "season": 2026, "week": 1, "player_id": "C",
            "player_display_name": "Player C", "team": "SF",
            "opponent_team": "SEA", "position": "WR",
            "predicted_touchdown_probability": 0.3,
            "price": 400, "market_implied_prob": ledger.implied_prob(400),
            "model_edge": 0.06, "bookmaker": "book1",
        },
    ])
    ledger.record_picks(preds, 2026, 1, "top5_prob", 1.0, ledger_path, now="2026-09-02T12:00:00")
    return ledger_path


def _fake_pbp_pandas():
    return pd.DataFrame([
        {"season": 2026, "week": 1, "posteam": "NYJ", "defteam": "BUF", "touchdown": 1, "td_player_id": "A"},
        {"season": 2026, "week": 1, "posteam": "NYJ", "defteam": "BUF", "touchdown": 0, "td_player_id": None},
        {"season": 2026, "week": 1, "posteam": "BUF", "defteam": "NYJ", "touchdown": 1, "td_player_id": "Z"},
        # C's team (SF) has no pbp rows this week -> stays unsettled.
    ])


def test_settle_updates_scorers_and_non_scorers_leaves_no_pbp_unsettled(tmp_path):
    ledger_path = _settle_ledger(tmp_path)

    def fake_load_pbp(years):
        return _fake_pbp_pandas()

    settled = ledger.settle(ledger_path, 2026, 1, load_pbp=fake_load_pbp)

    assert settled == 2
    out = pd.read_csv(ledger_path).set_index("player_id")

    assert out.loc["A", "outcome"] == 1
    assert out.loc["A", "pnl"] == pytest.approx(3.0)

    assert out.loc["B", "outcome"] == 0
    assert out.loc["B", "pnl"] == pytest.approx(-1.0)

    assert pd.isna(out.loc["C", "outcome"])
    assert pd.isna(out.loc["C", "pnl"])


def test_settle_defensive_return_td_counts_as_score(tmp_path):
    ledger_path = tmp_path / "bets.csv"
    preds = pd.DataFrame([
        {
            "season": 2026, "week": 1, "player_id": "Z",
            "player_display_name": "Defender Z", "team": "BUF",
            "opponent_team": "NYJ", "position": "CB",
            "predicted_touchdown_probability": 0.05,
            "price": 900, "market_implied_prob": ledger.implied_prob(900),
            "model_edge": 0.01, "bookmaker": "book1",
        },
    ])
    ledger.record_picks(preds, 2026, 1, "top5_prob", 1.0, ledger_path, now="2026-09-02T12:00:00")

    def fake_load_pbp(years):
        return _fake_pbp_pandas()

    settled = ledger.settle(ledger_path, 2026, 1, load_pbp=fake_load_pbp)
    assert settled == 1
    out = pd.read_csv(ledger_path).set_index("player_id")
    assert out.loc["Z", "outcome"] == 1


def test_settle_accepts_polars_pbp(tmp_path):
    polars = pytest.importorskip("polars")
    ledger_path = _settle_ledger(tmp_path)

    def fake_load_pbp(years):
        return polars.from_pandas(_fake_pbp_pandas())

    settled = ledger.settle(ledger_path, 2026, 1, load_pbp=fake_load_pbp)
    assert settled == 2
    out = pd.read_csv(ledger_path).set_index("player_id")
    assert out.loc["A", "outcome"] == 1
    assert out.loc["B", "outcome"] == 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def test_report_cumulative_stats(tmp_path):
    ledger_path = _settle_ledger(tmp_path)

    def fake_load_pbp(years):
        return _fake_pbp_pandas()

    ledger.settle(ledger_path, 2026, 1, load_pbp=fake_load_pbp)

    rep = ledger.report(ledger_path)
    cumulative = rep[(rep["strategy"] == "top5_prob") & (rep["week"] == "ALL")].iloc[0]

    assert cumulative["n_settled"] == 2
    assert cumulative["hits"] == 1
    assert cumulative["pnl"] == pytest.approx(2.0)
    assert cumulative["roi"] == pytest.approx(1.0)
