# NFL Anytime Touchdown Model

A weekly machine-learning pipeline that estimates each NFL player's probability of
scoring at least one touchdown in an upcoming game, compares those probabilities to
sportsbook Anytime Touchdown Scorer prices, and surfaces positive-expected-value
betting opportunities.

## LID

This project follows Linked-Intent Development. Consult the `linked-intent-dev` skill
for all code changes.

- Mode: Full
- Version: 1.3.0

Design intent lives under `docs/`:
- `docs/high-level-design.md` — root HLD (architecture, tenets).
- `docs/intent/<segment>/` — leaf LLDs and their EARS specs.

Changes walk the arrow: HLD → LLD → EARS → edge audit → tests → code, with stops
between phases. EARS specs and tests carry `@spec` IDs; code cites them at behavior
entry points.
