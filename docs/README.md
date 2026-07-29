# Internal Documentation Map

Use this folder for durable project specifications. For operational rules,
start with `../CLAUDE.md`.

## Current Files

- `implementation-spec.md` - exhaustive target spec for the local platform
  overhaul. It is normative for the future target, not proof of completion.
- `local-overhaul-design.md` - original local product and SQLite design
  baseline.
- `schema-evolution.md` - current-to-target schema analysis.
- `arena-future-architecture.md` - future arena/service boundary and research
  framing.

## Current Status

The live code partially implements the local platform:

- `rl/results_db.py` has schema v2 for teams, cards, decks, submissions,
  tournaments, matches, replay steps/options/events/snapshots, and card/deck
  Elo.
- `scripts/tournament.py` persists local tournament aggregates and normalized
  local match replay rows.
- `scripts/dashboard.py` provides a Streamlit dashboard for overview, cards,
  decks, deck builder, agents, arena, replays, and configuration.
- `scripts/rebuild_db.py` atomically rebuilds the remote-context database from
  canonical sources.

The full `TASK.md` target remains open for model revisions, experiments,
training runs, dashboard-editable tournament configs, submission lifecycle,
rating policies/epochs/events, and official visualizer reconstruction.
