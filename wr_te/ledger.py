"""Bet ledger: record picks at open odds, attach the closing line, settle
against play-by-play, and report hit rate / ROI / CLV.

CLI:
    python ledger.py record --strategy top5_prob [--stake 1.0] [--predictions PATH]
    python ledger.py close [--odds PATH]
    python ledger.py settle
    python ledger.py report
"""
import argparse
import csv
from pathlib import Path
from typing import Callable

import nflreadpy as nfl
import pandas as pd

import config
from odds_match import match_odds_to_players

LEDGER_COLUMNS = [
    "bet_id", "season", "week", "placed_at", "strategy", "player_id",
    "player_display_name", "team", "opponent_team", "position", "model_prob",
    "bookmaker", "price_open", "implied_open", "price_close", "implied_close",
    "stake", "outcome", "pnl", "clv",
]

# Any TD type (rush/rec/return) counts for the anytime-TD (ATD) market this
# ledger tracks.
_TD_TYPE_AGNOSTIC = True


def implied_prob(price) -> float:
    """American odds -> break-even (implied) probability."""
    p = float(price)
    if p > 0:
        return 100.0 / (p + 100.0)
    return abs(p) / (abs(p) + 100.0)


def decimal_odds(price) -> float:
    """American odds -> decimal odds."""
    p = float(price)
    if p > 0:
        return 1.0 + p / 100.0
    return 1.0 + 100.0 / abs(p)


def _top5_prob(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("predicted_touchdown_probability", ascending=False).head(5)


def _top5_edge_le400(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["price"] <= 400]
    return filtered.sort_values("model_edge", ascending=False).head(5)


STRATEGIES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "top5_prob": _top5_prob,
    "top5_edge_le400": _top5_edge_le400,
}


def _read_ledger(ledger_path: Path) -> pd.DataFrame:
    ledger_path = Path(ledger_path)
    if ledger_path.exists():
        return pd.read_csv(ledger_path)
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def _write_ledger(ledger_path: Path, df: pd.DataFrame) -> None:
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ledger_path, index=False, quoting=csv.QUOTE_MINIMAL)


def record_picks(
    predictions: pd.DataFrame,
    season: int,
    week: int,
    strategy: str,
    stake: float,
    ledger_path: Path,
    now: str,
) -> int:
    """Select picks per `strategy` and append new rows to the ledger CSV.

    Idempotent: rows whose `bet_id` already exists in the ledger are skipped.
    Returns the number of rows added.
    """
    selector = STRATEGIES[strategy]
    picks = selector(predictions)

    existing = _read_ledger(ledger_path)
    existing_ids = set(existing["bet_id"].astype(str)) if not existing.empty else set()

    new_rows = []
    for _, row in picks.iterrows():
        bet_id = f"{season}-{week}-{strategy}-{row['player_id']}"
        if bet_id in existing_ids:
            continue
        price_open = row["price"]
        new_rows.append({
            "bet_id": bet_id,
            "season": season,
            "week": week,
            "placed_at": now,
            "strategy": strategy,
            "player_id": row["player_id"],
            "player_display_name": row["player_display_name"],
            "team": row["team"],
            "opponent_team": row["opponent_team"],
            "position": row["position"],
            "model_prob": row["predicted_touchdown_probability"],
            "bookmaker": row.get("bookmaker", ""),
            "price_open": price_open,
            "implied_open": implied_prob(price_open),
            "price_close": float("nan"),
            "implied_close": float("nan"),
            "stake": stake,
            "outcome": float("nan"),
            "pnl": float("nan"),
            "clv": float("nan"),
        })

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows, columns=LEDGER_COLUMNS)
    combined = pd.concat([existing, new_df], ignore_index=True)
    _write_ledger(ledger_path, combined)
    return len(new_rows)


def attach_closing(ledger_path: Path, close_odds: pd.DataFrame, team_map: dict) -> int:
    """Fill price_close/implied_close/clv for unsettled ledger rows matched
    to `close_odds` (same name+team-in-game rule as `predict_wr.join_odds`).
    Returns the number of rows updated.
    """
    ledger_df = _read_ledger(ledger_path)
    if ledger_df.empty:
        return 0

    unsettled = ledger_df[ledger_df["outcome"].isna()]
    if unsettled.empty:
        return 0

    candidates = unsettled[["bet_id", "player_display_name", "team"]].copy()
    matched = match_odds_to_players(candidates, close_odds, team_map)
    if matched.empty:
        return 0

    matched = matched.set_index("bet_id")
    updated = 0
    for bet_id, row in matched.iterrows():
        idx = ledger_df.index[ledger_df["bet_id"] == bet_id]
        if len(idx) == 0:
            continue
        i = idx[0]
        price_close = row["price"]
        close_prob = implied_prob(price_close)
        ledger_df.at[i, "price_close"] = price_close
        ledger_df.at[i, "implied_close"] = close_prob
        ledger_df.at[i, "clv"] = close_prob - ledger_df.at[i, "implied_open"]
        updated += 1

    _write_ledger(ledger_path, ledger_df)
    return updated


def settle(ledger_path: Path, season: int, week: int, load_pbp=nfl.load_pbp) -> int:
    """Settle unsettled ledger rows for `season`/`week` whose team appears in
    that week's play-by-play. Scorers = any pbp row with `touchdown == 1` and
    a non-null `td_player_id` (any TD type, matching how the ATD market
    pays). Returns the number of rows settled.
    """
    ledger_df = _read_ledger(ledger_path)
    if ledger_df.empty:
        return 0

    pbp = load_pbp([season])
    if hasattr(pbp, "to_pandas"):
        pbp = pbp.to_pandas()

    pbp_week = pbp[(pbp["season"] == season) & (pbp["week"] == week)]

    scorers = set(
        pbp_week.loc[
            (pbp_week["touchdown"] == 1) & pbp_week["td_player_id"].notna(),
            "td_player_id",
        ].astype(str)
    )

    teams_with_pbp = set(pbp_week["posteam"].dropna()) | set(pbp_week["defteam"].dropna())

    mask = (
        (ledger_df["season"] == season)
        & (ledger_df["week"] == week)
        & (ledger_df["outcome"].isna())
        & (ledger_df["team"].isin(teams_with_pbp))
    )

    settled = 0
    for i in ledger_df.index[mask]:
        player_id = str(ledger_df.at[i, "player_id"])
        stake = ledger_df.at[i, "stake"]
        price_open = ledger_df.at[i, "price_open"]
        if player_id in scorers:
            outcome = 1
            pnl = stake * (decimal_odds(price_open) - 1)
        else:
            outcome = 0
            pnl = -stake
        ledger_df.at[i, "outcome"] = outcome
        ledger_df.at[i, "pnl"] = pnl
        settled += 1

    if settled:
        _write_ledger(ledger_path, ledger_df)
    return settled


def report(ledger_path: Path) -> pd.DataFrame:
    """Per-(strategy, week) and per-strategy-cumulative (week == 'ALL')
    summary: n_bets, n_settled, hits, hit_rate, staked, pnl, roi,
    n_with_close, mean_clv.
    """
    ledger_df = _read_ledger(ledger_path)

    def _agg(df: pd.DataFrame) -> pd.Series:
        n_bets = len(df)
        settled = df[df["outcome"].notna()]
        n_settled = len(settled)
        hits = int(settled["outcome"].sum()) if n_settled else 0
        hit_rate = hits / n_settled if n_settled else float("nan")
        staked = settled["stake"].sum() if n_settled else 0.0
        pnl = settled["pnl"].sum() if n_settled else 0.0
        roi = pnl / staked if staked else float("nan")
        with_close = df[df["implied_close"].notna()]
        n_with_close = len(with_close)
        mean_clv = with_close["clv"].mean() if n_with_close else float("nan")
        return pd.Series({
            "n_bets": n_bets,
            "n_settled": n_settled,
            "hits": hits,
            "hit_rate": hit_rate,
            "staked": staked,
            "pnl": pnl,
            "roi": roi,
            "n_with_close": n_with_close,
            "mean_clv": mean_clv,
        })

    if ledger_df.empty:
        cols = ["strategy", "week", "n_bets", "n_settled", "hits", "hit_rate",
                "staked", "pnl", "roi", "n_with_close", "mean_clv"]
        return pd.DataFrame(columns=cols)

    per_week = ledger_df.groupby(["strategy", "week"]).apply(_agg, include_groups=False).reset_index()

    cumulative = ledger_df.groupby("strategy").apply(_agg, include_groups=False).reset_index()
    cumulative.insert(1, "week", "ALL")

    result = pd.concat([per_week, cumulative], ignore_index=True)
    return result


def _cli_record(args: argparse.Namespace) -> None:
    predictions_path = args.predictions or config.predictions_path(args.season, args.week)
    predictions = pd.read_csv(predictions_path)
    ledger_path = config.LEDGER_DIR / "bets.csv"
    now = pd.Timestamp.utcnow().isoformat()
    added = record_picks(predictions, args.season, args.week, args.strategy, args.stake, ledger_path, now)
    print(f"record: added {added} row(s) to {ledger_path}")


def _cli_close(args: argparse.Namespace) -> None:
    odds_path = args.odds or config.odds_snapshot_path(args.season, args.week, "close")
    close_odds = pd.read_csv(odds_path)
    team_map = _load_team_map()
    ledger_path = config.LEDGER_DIR / "bets.csv"
    updated = attach_closing(ledger_path, close_odds, team_map)
    print(f"close: updated {updated} row(s) in {ledger_path}")


def _cli_settle(args: argparse.Namespace) -> None:
    ledger_path = config.LEDGER_DIR / "bets.csv"
    settled = settle(ledger_path, args.season, args.week)
    print(f"settle: settled {settled} row(s) in {ledger_path}")


def _cli_report(args: argparse.Namespace) -> None:
    ledger_path = config.LEDGER_DIR / "bets.csv"
    rep = report(ledger_path)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(rep.to_string(index=False))


def _load_team_map() -> dict:
    teams = pd.read_csv(config.DATA_DIR / "nfl_teams.csv")
    return dict(zip(teams["team_name"], teams["team_id"]))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bet ledger CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Record picks at open odds")
    record_parser.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    record_parser.add_argument("--stake", type=float, default=1.0)
    record_parser.add_argument("--predictions", type=Path, default=None)
    record_parser.add_argument("--season", type=int, default=config.SEASON)
    record_parser.add_argument("--week", type=int, default=config.WEEK)
    record_parser.set_defaults(func=_cli_record)

    close_parser = subparsers.add_parser("close", help="Attach closing odds")
    close_parser.add_argument("--odds", type=Path, default=None)
    close_parser.add_argument("--season", type=int, default=config.SEASON)
    close_parser.add_argument("--week", type=int, default=config.WEEK)
    close_parser.set_defaults(func=_cli_close)

    settle_parser = subparsers.add_parser("settle", help="Settle against play-by-play")
    settle_parser.add_argument("--season", type=int, default=config.SEASON)
    settle_parser.add_argument("--week", type=int, default=config.WEEK)
    settle_parser.set_defaults(func=_cli_settle)

    report_parser = subparsers.add_parser("report", help="Print hit rate / ROI / CLV report")
    report_parser.set_defaults(func=_cli_report)

    return parser


if __name__ == "__main__":
    arg_parser = _build_arg_parser()
    parsed_args = arg_parser.parse_args()
    parsed_args.func(parsed_args)
