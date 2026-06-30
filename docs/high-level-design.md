# High-Level Design — NFL Anytime Touchdown Model

## Purpose

Estimate, each week, the probability that an NFL skill-position player scores at least
one touchdown in an upcoming game, compare those probabilities against sportsbook
Anytime Touchdown (ATD) Scorer prices, and surface positive-expected-value (+EV)
betting opportunities as a ranked weekly bet sheet.

The system exists to find and exploit a **pricing edge**. Predicting touchdowns is the
means; the deliverable is a defensible list of bets where the model's probability
diverges from the market's implied probability by enough to be worth a stake.

## Goals

- Produce calibrated P(≥1 TD) for every active RB/WR/TE in a given week's slate.
- Translate model probabilities and book prices into edge, expected value, and a
  recommended (fractional-Kelly) stake.
- Validate the edge thesis honestly via a leakage-free historical backtest measuring
  calibration, closing-line value (CLV), and ROI.
- Deliver a reproducible weekly artifact from a single command.

## Non-Goals (v1)

- QB rushing touchdowns (noisier, distinct process — deferred to a later version).
- Other prop markets (receptions, rushing yards, first TD, multi-TD) — the modeling
  substrate may generalize, but only ATD is in scope.
- Live/in-game updating. The system operates pre-kickoff only.
- Fully automated scheduling/deployment. v1 is a manually triggered weekly script.

## Architecture Overview

A linear weekly pipeline of six components, each owning one leaf LLD and EARS prefix:

```
 [data] ──► [features] ──► [modeling] ──► [betting] ──► weekly bet sheet
   │             │              │             ▲
   │             │              │             │
   └──────► (historical) ──► [backtest] ──────┘   (validation loop)

 [pipeline] orchestrates the weekly forward path end-to-end.
```

| Segment      | EARS prefix | Owns |
|--------------|-------------|------|
| `data`       | `DATA-`     | Ingest nflverse play-by-play, rosters, schedules, snap counts; ingest live/forward ATD odds from The Odds API and the on-hand historical ATD odds dataset (2023–2025), both behind a pluggable odds adapter; produce clean point-in-time tables. |
| `features`   | `FEAT-`     | Build per-player-game features: rolling usage (carries, targets, red-zone & goal-line touches, snap share), role, opponent defense vs. position, game context (implied total, spread, home/away, pace). |
| `modeling`   | `MODEL-`    | Two-stage estimator: Stage 1 derives team expected TDs anchored to the Vegas implied total; Stage 2 learns each player's share of team TDs; combine to λ and P(≥1 TD) = 1 − e^(−λ); calibrate and persist. |
| `betting`    | `BET-`      | De-vig book prices to implied probabilities; compute edge and EV; size stakes with fractional Kelly under an edge threshold; emit the ranked bet sheet. |
| `backtest`   | `BACK-`     | Replay the forward path over historical weeks with strict temporal separation; report calibration, CLV, and ROI. |
| `pipeline`   | `PIPE-`     | Single-command weekly orchestration: run data → features → modeling → betting for the current slate; emit the edge-filtered bet sheet as the primary artifact plus a clearly-labeled informational top-probability view (diagnostic instrument, not a bet signal). |

Depth-2 structure (one HLD over flat leaves) is the default. `modeling` carries the
most internal depth (two stages + calibration); it stays a single leaf for v1 with
within-leaf facets (`MODEL-TEAM-`, `MODEL-SHARE-`, `MODEL-CAL-`) and is a candidate for
promotion to a sub-HLD if it outgrows one doc.

## Data Flow

1. `data` pulls open nflverse data and market odds, normalizing to point-in-time tables
   keyed by season/week/player/team.
2. `features` joins those into a per-player-game feature matrix using only information
   available before kickoff.
3. `modeling` produces P(≥1 TD) per player for the target week.
4. `betting` joins model probabilities with current ATD prices, de-vigs, computes
   edge/EV, applies the staking rule, and ranks.
5. `backtest` runs steps 2–4 over the 2023–2025 historical weeks against the on-hand
   historical ATD odds at bet time to measure whether the edge is real.
6. `pipeline` sequences the weekly forward path and emits the bet sheet, plus a
   labeled informational top-probability view for diagnostics.

## Tenets

Tie-breakers for decisions no spec will fully anticipate:

- **Anchor to the market; spend modeling effort where the edge is.** Treat the Vegas
  implied total as ground truth for the game environment; concentrate learning on the
  player-share component, where the book's priors are stickiest.
- **Calibration over discrimination.** A probability that is right on average at each
  level beats one that merely ranks players well. Model quality is judged by proper
  scoring rules (log loss, Brier) and reliability curves — never by classification
  accuracy, which is meaningless for a rare, low-base-rate event. When calibration and
  ranking trade off, optimize the reliability curve.
- **Probability is not a recommendation.** A high chance to score is not a good bet —
  the market already prices obvious scorers, and TD props carry a favorite-longshot bias
  that overprices chalk. Only *edge* drives the bet sheet. Any raw-probability view is an
  instrument, labeled non-actionable, and never feeds staking.
- **Closing-line value is the primary edge signal; realized ROI is secondary.** Beating
  the closing line is the leading indicator that the edge is real and survives variance;
  ROI is the lagging confirmation.
- **No leakage, ever.** Every feature and every odds value used in *prediction or*
  evaluation must have been available at the moment a bet would have been placed. A
  faster or richer feature that cannot be reconstructed point-in-time is not used. This
  is operationalized as the *Temporal Integrity* invariant below and enforced in every
  segment, not just the backtest.
- **Conservative by default.** Recommend bets only above a minimum edge threshold and
  stake at fractional Kelly. Aggressive sizing is opt-in, not the default.

## Temporal Integrity (No-Leakage Invariant)

Information from the target week (or later) must never influence a prediction for that
week. This is a hard invariant enforced in **every** segment — `data`, `features`,
`modeling`, `betting`, `backtest`, `pipeline` — not a backtest-only concern. It is the
single most important correctness property of the system: a leak makes a useless model
look profitable.

Operational rules:

- **As-of boundary.** A prediction for week _W_ of season _S_ is made as of the start of
  week _W_. Only events before that boundary are admissible.
- **Feature windows are backward-only.** Features for _(S, W)_ are computed from completed
  games in weeks _1..W-1_ of season _S_ and all prior seasons — never the current or a
  future week. No season-to-date figure may include week _W_.
- **The label is never an input.** Whether a player scored is attached only after the game
  completes and is used solely as a training/evaluation target.
- **Same-week pre-game signals excluded in v1.** Injury reports, inactives, and other
  week-_W_-specific pre-game info are deferred; v1 stays strictly prior-weeks-only so the
  boundary is unimpeachable. Vetted same-week signals may be added later.
- **Backtesting is walk-forward.** For each _(S, W)_ the model trains only on data strictly
  before the boundary; no random k-fold or future-fold leakage across weeks.
- **Odds provenance is respected.** Bet-time prices use the pre-kickoff snapshot; no price
  captured after kickoff enters the pipeline.
- **Enforcement is testable.** Each segment carries EARS specs and tests asserting its
  outputs depend only on pre-boundary data (a leakage guard), so the invariant is
  verified, not assumed. Leakage is a cross-cutting concern: each leaf owns a
  point-in-time facet of its EARS namespace.

## Key Decisions

| Decision | Choice | Rationale | Alternatives considered |
|----------|--------|-----------|-------------------------|
| Modeling approach | Two-stage hierarchical (Vegas-anchored team TDs × learned player share → Poisson P(≥1)) | Isolates the edge in the player-share component; anchors game environment to the market we can't beat; interpretable and debuggable. | Single Poisson expected-TD GBM (blends signal sources); direct GBM classifier (poorly calibrated on rare events, no market anchor). |
| Odds source | The Odds API for live/forward odds; on-hand historical ATD odds dataset (2023–2025) for backtest; both behind a pluggable odds adapter | Clean multi-book live odds plus a real historical record are required for an honest backtest; an adapter keeps both sources behind one interface and swappable. | Free scraping (brittle, no clean history); defer odds entirely (blocks edge validation). |
| Player/TD scope | RB/WR/TE, rushing + receiving | Covers the bulk of ATD market volume with clean signal. | All positions incl. QB rushing (noisier, distinct process); RBs only (too narrow). |
| Output / automation | Single-command weekly script → ranked bet sheet | Reproducible deliverable without premature infra. | Notebook-only (not reproducible); automated scheduled pipeline (over-engineered for v1). |

## Open Questions

- How many prior seasons of nflverse data to train on, and whether to weight recent
  seasons. (The backtest window is bounded to 2023–2025 by the available historical
  ATD odds; model training history may extend further back.)
- Exact schema/format of the on-hand 2023–2025 historical ATD odds dataset (books
  covered, granularity, fields) — to be pinned down in the `data` LLD.
- Minimum edge threshold and Kelly fraction defaults (to be set in `betting`).
```
