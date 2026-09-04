# Task 1 Report: Odds API guardrails

## Red/green verification

- Red: `wr_te/.venv/bin/pytest -q wr_te/tests/test_fetch_live_odds.py` — 25 passed, 8 failed after adding the guardrail regressions (failures were the expected missing behavior and stricter-contract fixture updates).
- Green: `wr_te/.venv/bin/pytest -q wr_te/tests/test_fetch_live_odds.py` — 36 passed.
- Full suite: `wr_te/.venv/bin/pytest -q` — 177 passed.
- Additional: `git diff --check` — clean.

## Files changed

- `wr_te/fetch_live_odds.py`: exact-week schedule/event preflight, one-to-one identity plan, strict normalized quote validation, cache attestations and coverage checks, atomic full-mode publishing, and `--canary-one-event`.
- `wr_te/tests/test_fetch_live_odds.py`: focused regression coverage for preflight, identity, malformed quotes, cache rejection, atomic failure behavior, and canary publishing behavior.
- `wr_te/README.md`: documents exact-week filtering, cache contract, and non-publishing canary use.

## Commit

Implementation commit: `9c8167d` (`Harden Odds API weekly collection guardrails`).

## Self-review

- Requested bookmaker keys are normalized and rejected when empty or duplicated before credentials, schedule loading, or HTTP access.
- Full mode validates the complete match plan before paid calls; paid responses are checked against the selected provider event and canonical IDs never fall back to provider IDs.
- Only intended `player_anytime_td`/`Yes` quotes are emitted, and emitted rows require nonblank identity fields, pregame timestamps, and valid American prices.
- Cache reuse requires repeated request attestations, canonical/provider identity coverage, bookmaker coverage, and row semantics; malformed or partial snapshots are rejected without HTTP access.
- Canary mode performs at most one paid event call and does not write the canonical snapshot.
- Unrelated untracked `wr_te/evaluation/` was not staged or modified.

## Concerns

- Existing pre-guardrail snapshots without the new identity/attestation columns are intentionally rejected and require an explicit refresh.
- Tests use local fakes only; no credentials or network/API calls were used.
