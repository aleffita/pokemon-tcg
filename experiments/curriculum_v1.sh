#!/bin/bash
# Curriculum V1: 3-stage coarse->fine, all encadeados por resume, partindo de
# suite_5d_10ep_OFF/5d_10ep_OFF.pkl. Cada estágio salva checkpoint por epoch
# e roda tournament ao terminar.
#
# Stage 1: OFF, all days,     15ep, 80k rows/day,  lr=1.5e-4
# Stage 2: ON top-600 real,   5 days (últimos),  5ep, 300k rows/day, lr=8e-5
# Stage 3: ON top-100 real,   1 day  (último),  10ep, 300k rows/day, lr=3e-5
#
# All three stages: optimizer state reset, scheduler state reset (obrigatório),
# checkpoint_every_epochs=1. Elo Kaggle real é populado via tcg-elo-reconcile
# ANTES do stage 2 e 3 (não afeta stage 1, que é OFF).
#
# Tournament ao fim de cada estágio:
#   S1: 30 games, --no-sweep. Oponentes: 2 baselines + 4 starters + 3 strong.
#   S2: idem
#   S3: idem + tournament EXTRA de 30 games COM --sweep-source remote.
#
# Resume-aware: qualquer estágio cujo _final.pkl já existir é pulado (o
# tournament dele também é pulado se seu _tourn.json já existir).

set -e
cd /Users/alefita/workdir/pokemon-tcg

# ---------------- config ----------------
EXP_ROOT=experiments/curriculum_v1
REPORTS=$EXP_ROOT/reports
BASE_CHECKPOINT=model/checkpoint/suite_5d_10ep_OFF/5d_10ep_OFF.pkl
CONFIG=configs/train_config.json

mkdir -p "$REPORTS"

# Per-run tournament opponents (mesma composição usada na bc_curriculum_suite).
# 2 baselines (random+first) são adicionados implicitamente pela tournament.py.
declare -a OPPONENTS=(
  public_agents/starters/lb510_mega_abomasnow_ex
  public_agents/starters/lb526_iono
  public_agents/starters/lb600_dragapult_ex
  public_agents/starters/lb600_mega_lucario_ex
  public_agents/lb826_alakazam_seok
  public_agents/lb945_multiply_ivan
  public_agents/lb1009_mega_lucario_ex_islet
)

TOURNAMENT_GAMES=30

# ---------------- helpers ----------------
format_dur() {
  local s=$1
  local h=$(( s / 3600 ))
  local m=$(( (s % 3600) / 60 ))
  local sec=$(( s % 60 ))
  if [ $h -gt 0 ]; then printf "%dh%02dm%02ds" $h $m $sec
  elif [ $m -gt 0 ]; then printf "%dm%02ds" $m $sec
  else printf "%ds" $sec
  fi
}

overall_wr() {
  uv run python -c "
import json
print(json.load(open('$1'))['overall']['wr_pct'])
" 2>/dev/null
}

package_and_swap() {
  # $1 = pkl path, $2 = tag → refresh model/bc_model/bc_best_torch_fp16.pt
  # (this is what agent/main.py loads) and produce $EXP_ROOT/models/${TAG}.tar.gz
  local pkl="$1" tag="$2"
  local dst="$EXP_ROOT/models/${tag}.tar.gz"
  mkdir -p "$EXP_ROOT/models"
  cp "$pkl" model/checkpoint/bc_best_mlx.pkl
  uv run tcg-build --checkpoint model/checkpoint/bc_best_mlx.pkl --out "$dst" >/dev/null 2>&1
  if [ ! -f "$dst" ]; then
    echo "!! build_submission produced no tarball for $tag" >&2
    return 1
  fi
  echo "  packaged $tag -> $dst"
}

run_tournament() {
  # $1 = tag, $2 = json output path, $3 = extra flags (e.g. "--sweep-source remote")
  local tag="$1" json="$2"; shift 2
  local extra="$*"
  local log="${json%.json}.log"

  if [ -f "$json" ]; then
    echo ">>> tournament $tag already done at $json — skipping"
    return 0
  fi

  local OPP_ARGS=()
  for opp in "${OPPONENTS[@]}"; do
    OPP_ARGS+=(--opponent "$opp")
  done

  local NOTE="curriculum_v1 $tag${extra:+ ($extra)}"
  # shellcheck disable=SC2086
  uv run tcg-tournament \
    --games $TOURNAMENT_GAMES \
    $extra \
    "${OPP_ARGS[@]}" \
    --note "$NOTE" \
    --report-json "$json" \
    2>&1 | tee "$log" | tail -12

  local WR
  WR=$(overall_wr "$json")
  echo ">>> tournament $tag DONE: overall_wr=${WR:-?}%"
}

# ---------------- pre-suite enrichment (idempotent) ----------------
# 1) download missing Kaggle days (idempotent — skips existing zips)
# 2) build parquets for any zip not yet ingested
# 3) build_card_stats for new days (populates matches, agents, card_elo, deck_elo)
# 4) reconcile agent_elo_daily(source='remote') with LIVE Kaggle leaderboard
#
# Any of these that has nothing to do exits cleanly and cheaply.

echo ""
echo "======================================================================"
echo "  PRE-SUITE ENRICHMENT"
echo "======================================================================"

# Discover the newest known Kaggle-published day (yesterday, in UTC), and pull
# from the day after our latest local zip up to that. --range is idempotent
# and skips days already on disk / not yet published.
LATEST_LOCAL=$(ls data/bc_replay_zip/*.zip 2>/dev/null | sort | tail -1 | sed -E 's|.*/([0-9]{4}-[0-9]{2}-[0-9]{2})\.zip|\1|')
YESTERDAY=$(python3 -c "from datetime import datetime, timedelta, timezone; print((datetime.now(timezone.utc)-timedelta(days=1)).date().isoformat())")
if [ -n "$LATEST_LOCAL" ] && [ "$LATEST_LOCAL" \< "$YESTERDAY" ]; then
  NEXT=$(python3 -c "from datetime import datetime, timedelta; print((datetime.strptime('$LATEST_LOCAL','%Y-%m-%d')+timedelta(days=1)).date().isoformat())")
  echo "----- downloading Kaggle days $NEXT .. $YESTERDAY -----"
  uv run tcg-data --range "$NEXT" "$YESTERDAY" 2>&1 | tail -5
else
  echo "----- Kaggle days: nothing to download (latest local = $LATEST_LOCAL) -----"
fi

echo "----- refreshing days.competition_day (required before build_bc) -----"
# tcg-data inserts new days with competition_day=NULL. The encoder's
# meta_lookup then blows up on any episode with `day_id 25 not registered`,
# even for old days, because latest_day_id() returns the MAX id including
# NULL rows. Idempotent: reassigns 1-indexed competition_day by calendar
# order every call.
uv run python -c "
from rl.results_db import ResultsDB
db = ResultsDB('model/results.db'); db.refresh_competition_days(); db.close()
print('  refreshed competition_day for all days in calendar order')
"

echo "----- building Parquets for any new zips -----"
uv run tcg-build-bc --all --resume 2>&1 | tail -5

echo "----- building card/deck Elo stats for any new days -----"
# Discover days that have a parquet but no matches yet, then compute stats
# only for that range. Idempotent per-day (build_card_stats internally deletes
# the day's slice before recomputing).
NEW_DAYS=$(uv run python -c "
import sqlite3
from pathlib import Path
c = sqlite3.connect('model/results.db')
have_stats = {r[0] for r in c.execute(
  \"SELECT DISTINCT d.date FROM days d JOIN matches m ON m.day_id=d.id WHERE m.source='remote'\"
).fetchall()}
parquet_days = sorted(p.stem for p in Path('data/bc_data').glob('*.parquet'))
missing = [d for d in parquet_days if d not in have_stats]
if missing:
  print(missing[0], missing[-1])
")
if [ -n "$NEW_DAYS" ]; then
  START=$(echo "$NEW_DAYS" | awk '{print $1}')
  END=$(echo "$NEW_DAYS" | awk '{print $2}')
  echo "  new stats range: $START .. $END"
  uv run tcg-build-card-stats --range "$START" "$END" 2>&1 | tail -5
else
  echo "  all days already have matches populated — skipping"
fi

echo "----- reconciling agent_elo_daily(source='remote') with Kaggle LB -----"
uv run tcg-elo-reconcile 2>&1 | tail -6

# ---------------- STAGE 1: OFF, all days, 15ep, 80k/day, lr=1.5e-4 ----------------
S1_ROOT=$EXP_ROOT/stage1
S1_FINAL=$S1_ROOT/curriculum_v1_stage1.pkl
S1_LATEST=$S1_ROOT/curriculum_v1_stage1_latest.pkl
S1_TOURN=$REPORTS/stage1_tourn.json

mkdir -p "$S1_ROOT"

echo ""
echo "======================================================================"
echo "  STAGE 1: OFF, all_days, 15ep, 80k/day, lr=1.5e-4"
echo "  resume from base: $BASE_CHECKPOINT"
echo "======================================================================"

if [ -f "$S1_FINAL" ]; then
  echo ">>> stage 1 final checkpoint already exists at $S1_FINAL — skipping training"
else
  # Semantic-clean resume policy: _final.pkl marks stage completion. If it does
  # not exist, we start the stage fresh from BASE. Mid-stage crashes cost time
  # but never leave the run in an ambiguous partial state.
  rm -rf "$S1_ROOT"; mkdir -p "$S1_ROOT"

  S1_START=$(python3 -c "import time; print(time.time())")
  uv run tcg-train --config "$CONFIG" \
    --all-days --max-rows-per-day 80000 --epochs 15 \
    --lr 1.5e-4 \
    --resume "$BASE_CHECKPOINT" --optimizer-state reset --scheduler-state reset \
    --out "$S1_FINAL" --checkpoint-every-epochs 1 \
    2>&1 | tee -a "$S1_ROOT/_train.log"
  S1_END=$(python3 -c "import time; print(time.time())")
  echo ">>> stage 1 training DONE in $(format_dur $((S1_END - S1_START)))"

  rm -rf "$S1_ROOT/.cache_spill"
fi

package_and_swap "$S1_FINAL" "stage1"
run_tournament "stage1" "$S1_TOURN" "--no-sweep"

# ---------------- STAGE 2: ON top-600, last 5 days, 5ep, 300k/day, lr=8e-5 -----
# Elo reconcile already done in pre-suite; the state used by --top-elo 600 is
# the LIVE Kaggle leaderboard as of the pre-suite step.
S2_ROOT=$EXP_ROOT/stage2
S2_FINAL=$S2_ROOT/curriculum_v1_stage2.pkl
S2_LATEST=$S2_ROOT/curriculum_v1_stage2_latest.pkl
S2_TOURN=$REPORTS/stage2_tourn.json

mkdir -p "$S2_ROOT"

echo ""
echo "======================================================================"
echo "  STAGE 2: ON top-600 (bronze+), last-5-days, 5ep, 300k/day, lr=8e-5"
echo "  resume from stage 1 final: $S1_FINAL"
echo "======================================================================"

if [ -f "$S2_FINAL" ]; then
  echo ">>> stage 2 final checkpoint already exists at $S2_FINAL — skipping training"
else
  rm -rf "$S2_ROOT"; mkdir -p "$S2_ROOT"

  S2_START=$(python3 -c "import time; print(time.time())")
  uv run tcg-train --config "$CONFIG" \
    --last-n-days 5 --max-rows-per-day 300000 --epochs 5 \
    --top-elo 600 \
    --lr 8e-5 \
    --resume "$S1_FINAL" --optimizer-state reset --scheduler-state reset \
    --out "$S2_FINAL" --checkpoint-every-epochs 1 \
    2>&1 | tee -a "$S2_ROOT/_train.log"
  S2_END=$(python3 -c "import time; print(time.time())")
  echo ">>> stage 2 training DONE in $(format_dur $((S2_END - S2_START)))"

  rm -rf "$S2_ROOT/.cache_spill"
fi

package_and_swap "$S2_FINAL" "stage2"
run_tournament "stage2" "$S2_TOURN" "--no-sweep"

# ---------------- STAGE 3: ON top-100, last 1 day, 10ep, 300k/day, lr=3e-5 -----
S3_ROOT=$EXP_ROOT/stage3
S3_FINAL=$S3_ROOT/curriculum_v1_stage3.pkl
S3_LATEST=$S3_ROOT/curriculum_v1_stage3_latest.pkl
S3_TOURN_NOSWEEP=$REPORTS/stage3_tourn_nosweep.json
S3_TOURN_SWEEP=$REPORTS/stage3_tourn_sweep.json

mkdir -p "$S3_ROOT"

echo ""
echo "======================================================================"
echo "  STAGE 3: ON top-100 (elite), last-1-day, 10ep, 300k/day, lr=3e-5"
echo "  resume from stage 2 final: $S2_FINAL"
echo "======================================================================"

if [ -f "$S3_FINAL" ]; then
  echo ">>> stage 3 final checkpoint already exists at $S3_FINAL — skipping training"
else
  rm -rf "$S3_ROOT"; mkdir -p "$S3_ROOT"

  S3_START=$(python3 -c "import time; print(time.time())")
  uv run tcg-train --config "$CONFIG" \
    --last-n-days 1 --max-rows-per-day 300000 --epochs 10 \
    --top-elo 100 \
    --lr 3e-5 \
    --resume "$S2_FINAL" --optimizer-state reset --scheduler-state reset \
    --out "$S3_FINAL" --checkpoint-every-epochs 1 \
    2>&1 | tee -a "$S3_ROOT/_train.log"
  S3_END=$(python3 -c "import time; print(time.time())")
  echo ">>> stage 3 training DONE in $(format_dur $((S3_END - S3_START)))"

  rm -rf "$S3_ROOT/.cache_spill"
fi

package_and_swap "$S3_FINAL" "stage3"

# Two tournaments for stage 3: comparable to S1/S2 first, then sweep on top.
run_tournament "stage3_nosweep" "$S3_TOURN_NOSWEEP" "--no-sweep"
run_tournament "stage3_sweep"   "$S3_TOURN_SWEEP"   "--sweep-source remote"

echo ""
echo "======================================================================"
echo "  Curriculum V1 complete."
echo "  Reports:   $REPORTS"
echo "  Models:    $EXP_ROOT/models/{stage1,stage2,stage3}.tar.gz"
echo "  Checkpts:  $EXP_ROOT/{stage1,stage2,stage3}/*.pkl"
echo "======================================================================"
