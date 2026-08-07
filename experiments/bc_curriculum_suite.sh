#!/bin/bash
# BC curriculum ablation suite: 10 training runs varying (days, epochs, top-elo curriculum)
# plus a round-robin tournament between the resulting 10 checkpoints.
#
# Phase 1 (per-run):
#   for each of 10 configs:
#     train (--resume from latest.pkl if present)
#     build submission tarball into experiments/bc_curriculum_suite/models/${TAG}.tar.gz
#     tournament: sweep ON, --sweep-source remote, N games vs
#       baselines (random + first) + starters (4) + strong (3 high-Elo public agents)
#     emit structured JSON report
#
# Phase 2 (round-robin):
#   for each unique pair (i<j) of the 10 tags:
#     swap bc_best_mlx.pkl to model_i, tournament vs models/${TAG_j}.tar.gz
#     --no-sweep --skip-baselines, N games
#     emit structured JSON report
#
# Phase 3: aggregate reports into results.md.
#
# Resume-aware: any per-run whose _tourn.json already exists is skipped.
# Any round-robin pair whose JSON already exists is skipped.

set -e
cd /Users/alefita/workdir/pokemon-tcg

# ---------------- config ----------------
EXP_ROOT=experiments/bc_curriculum_suite
CKPT_ROOT=model/checkpoint
MODELS_DIR=$EXP_ROOT/models
REPORTS_PER_RUN=$EXP_ROOT/reports/per_run
REPORTS_RR=$EXP_ROOT/reports/round_robin
RESULTS_MD=$EXP_ROOT/results.md
SUITE_LOG=$EXP_ROOT/_suite_stdout.log

mkdir -p "$MODELS_DIR" "$REPORTS_PER_RUN" "$REPORTS_RR"

DAYS_1="--days 2026-08-01"
DAYS_3="--days 2026-07-30,2026-07-31,2026-08-01"
DAYS_5="--days 2026-07-28,2026-07-29,2026-07-30,2026-07-31,2026-08-01"

declare -a MATRIX=(
  "1d_10ep_OFF|$DAYS_1|10"
  "1d_10ep_ON|$DAYS_1|10|--top-elo 50"
  "3d_1ep_OFF|$DAYS_3|1"
  "3d_1ep_ON|$DAYS_3|1|--top-elo 50"
  "3d_10ep_OFF|$DAYS_3|10"
  "3d_10ep_ON|$DAYS_3|10|--top-elo 50"
  "5d_1ep_OFF|$DAYS_5|1"
  "5d_1ep_ON|$DAYS_5|1|--top-elo 50"
  "5d_10ep_OFF|$DAYS_5|10"
  "5d_10ep_ON|$DAYS_5|10|--top-elo 50"
)

# Per-run tournament opponent subset (option C):
#   2 baselines (random+first) added implicitly by tournament.py
#   4 starters + 3 strong high-Elo public agents passed explicitly.
declare -a PER_RUN_OPPONENTS=(
  public_agents/starters/lb510_mega_abomasnow_ex
  public_agents/starters/lb526_iono
  public_agents/starters/lb600_dragapult_ex
  public_agents/starters/lb600_mega_lucario_ex
  public_agents/lb826_alakazam_seok
  public_agents/lb945_multiply_ivan
  public_agents/lb1009_mega_lucario_ex_islet
)

PER_RUN_GAMES=20
ROUND_ROBIN_GAMES=30

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

# Read state["epoch"] (0-indexed loop var) from a checkpoint pickle.
# Returns completed = epoch + 1, or empty string on failure.
read_completed_epochs() {
  local pkl="$1"
  uv run python -c "
import pickle
with open('$pkl','rb') as f: s = pickle.load(f)
print(int(s.get('epoch', -1)) + 1)
" 2>/dev/null
}

# Extract per-opponent WR and overall WR from a tournament JSON report.
# Prints newline-separated: label=WR,... then a final line 'overall=WR'.
summarize_report() {
  local json="$1"
  uv run python -c "
import json, sys
with open('$json') as f: r = json.load(f)
# aggregate rows by opponent_label (sweep produces one row per (label, deck))
agg = {}
for row in r['rows']:
    lbl = row['opponent_label']
    if row.get('error'):
        agg[lbl] = None; continue
    a = agg.setdefault(lbl, {'w':0,'l':0,'d':0})
    if a is None: continue
    a['w'] += row['wins']; a['l'] += row['losses']; a['d'] += row['draws']
for lbl, a in agg.items():
    if a is None:
        print(f'{lbl}=ERR')
    else:
        n = max(a['w'] + a['l'], 1)
        print(f'{lbl}={a[\"w\"]*100/n:.1f}')
print(f'overall={r[\"overall\"][\"wr_pct\"]:.1f}')
"
}

# Extract only the overall WR from a report (used by round-robin table).
overall_wr() {
  uv run python -c "
import json
print(json.load(open('$1'))['overall']['wr_pct'])
" 2>/dev/null
}

package_model() {
  # $1 = checkpoint pkl path, $2 = tag → produces $MODELS_DIR/${2}.tar.gz
  # Also refreshes model/bc_model/bc_best_torch_fp16.pt so subsequent
  # tournament runs pick up this model as OUR agent.
  local pkl="$1" tag="$2"
  local dst="$MODELS_DIR/${tag}.tar.gz"
  cp "$pkl" "$CKPT_ROOT/bc_best_mlx.pkl"
  uv run tcg-build --checkpoint "$CKPT_ROOT/bc_best_mlx.pkl" --out "$dst" >/dev/null 2>&1
  if [ ! -f "$dst" ]; then
    echo "!! build_submission produced no tarball for $tag at $dst" >&2
    return 1
  fi
  echo "  packaged $tag -> $dst"
}

# ---------------- phase 1: train + per-run tournament ----------------
SUITE_START=$(python3 -c "import time; print(time.time())")

echo ""
echo "======================================================================"
echo "  PHASE 1: 10 training runs + per-run tournaments (sweep ON, remote)"
echo "======================================================================"
echo "  per-run opponents: baselines + $(echo "${#PER_RUN_OPPONENTS[@]}" | tr -d ' ') public agents"
echo "  per-run games: $PER_RUN_GAMES"

runs_executed=0
TOTAL=${#MATRIX[@]}

for i in "${!MATRIX[@]}"; do
  entry="${MATRIX[$i]}"
  IFS='|' read -ra parts <<< "$entry"
  TAG="${parts[0]}"
  DAYS_ARG="${parts[1]}"
  EPOCHS="${parts[2]}"
  EXTRA="${parts[3]:-}"
  OUT_DIR="$CKPT_ROOT/suite_${TAG}"
  LATEST_PKL="$OUT_DIR/${TAG}_latest.pkl"
  FINAL_PKL="$OUT_DIR/${TAG}.pkl"
  TOURN_JSON="$REPORTS_PER_RUN/${TAG}_tourn.json"

  RUN_NUM=$((i + 1))
  NOW=$(python3 -c "import time; print(time.time())")
  ELAPSED=$(python3 -c "print(int($NOW - $SUITE_START))")

  if [ $runs_executed -gt 0 ]; then
    AVG=$(python3 -c "print($ELAPSED / $runs_executed)")
    REMAINING_RUNS=$((TOTAL - i))
    ETA=$(python3 -c "print(int($AVG * $REMAINING_RUNS))")
    HDR=" — elapsed $(format_dur $ELAPSED) — ETA remaining $(format_dur $ETA)"
  else
    HDR=" — first executed run this session"
  fi
  echo ""
  echo "======================================================"
  echo "=== [$RUN_NUM/$TOTAL] $TAG${HDR} ==="
  echo "======================================================"

  # Skip if this run's per-run tournament already emitted its JSON.
  if [ -f "$TOURN_JSON" ]; then
    echo ">>> [$RUN_NUM/$TOTAL] $TAG per-run tournament already done — skipping"
    continue
  fi

  # Train (fresh or resume).
  RESUME_ARGS=""
  RUN_EPOCHS=$EPOCHS
  if [ -f "$LATEST_PKL" ]; then
    COMPLETED=$(read_completed_epochs "$LATEST_PKL")
    if [ -n "$COMPLETED" ] && [ "$COMPLETED" -gt 0 ] && [ "$COMPLETED" -lt "$EPOCHS" ]; then
      REMAINING=$((EPOCHS - COMPLETED))
      RESUME_ARGS="--resume $LATEST_PKL --optimizer-state resume --scheduler-state resume"
      RUN_EPOCHS=$REMAINING
      echo ">>> resuming from epoch $COMPLETED, running $REMAINING more"
    elif [ -n "$COMPLETED" ] && [ "$COMPLETED" -ge "$EPOCHS" ]; then
      RUN_EPOCHS=0
      echo ">>> training already at $COMPLETED/$EPOCHS — skipping train, going straight to tournament"
    else
      rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"
    fi
  else
    rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"
  fi

  RUN_START=$(python3 -c "import time; print(time.time())")

  if [ "$RUN_EPOCHS" -gt 0 ]; then
    uv run tcg-train $DAYS_ARG --max-rows-per-day 30000 --epochs $RUN_EPOCHS \
      --batch 1024 --tbptt-chunk 16 --val-batch-size 1024 \
      --lr 2.46e-4 --warmup-steps 20 --aux-return-weight 1.0 \
      $EXTRA $RESUME_ARGS \
      --out "$FINAL_PKL" --checkpoint-every-epochs 999 \
      2>&1 | tee -a "$OUT_DIR/_train.log"
  fi

  # KV cache SSD spill dir is training-scoped storage. Left in place it grows
  # unbounded across runs (10-25 GiB per big run) and fills the disk. Delete
  # as soon as training is done — checkpoints and _train.log are unaffected.
  rm -rf "$OUT_DIR/.cache_spill"

  # Package this model as a submission tarball for the round-robin phase.
  BEST_PKL="$FINAL_PKL"
  [ ! -f "$BEST_PKL" ] && BEST_PKL="$LATEST_PKL"
  package_model "$BEST_PKL" "$TAG"

  # Per-run tournament (sweep ON, remote-Elo top decks, option C opponents).
  OPP_ARGS=()
  for opp in "${PER_RUN_OPPONENTS[@]}"; do
    OPP_ARGS+=(--opponent "$opp")
  done

  uv run tcg-tournament \
    --games $PER_RUN_GAMES \
    --sweep-source remote \
    "${OPP_ARGS[@]}" \
    --note "suite $TAG per-run" \
    --report-json "$TOURN_JSON" \
    2>&1 | tee "$REPORTS_PER_RUN/${TAG}_tourn.log" | tail -20

  RUN_END=$(python3 -c "import time; print(time.time())")
  RUN_DUR=$(python3 -c "print(int($RUN_END - $RUN_START))")

  OVERALL=$(overall_wr "$TOURN_JSON")
  echo ">>> [$RUN_NUM/$TOTAL] $TAG DONE in $(format_dur $RUN_DUR): overall_wr=${OVERALL:-?}%"

  runs_executed=$((runs_executed + 1))
done

# ---------------- phase 2: round-robin ----------------
echo ""
echo "======================================================================"
echo "  PHASE 2: round-robin between the 10 trained models"
echo "======================================================================"
echo "  --no-sweep --skip-baselines, $ROUND_ROBIN_GAMES games/pair"

# Collect TAG list preserving MATRIX order.
declare -a TAGS=()
for entry in "${MATRIX[@]}"; do
  TAG=$(echo "$entry" | cut -d'|' -f1)
  TAGS+=("$TAG")
done
NTAGS=${#TAGS[@]}
TOTAL_PAIRS=$((NTAGS * (NTAGS - 1) / 2))
pair_num=0

RR_START=$(python3 -c "import time; print(time.time())")

for i in $(seq 0 $((NTAGS - 1))); do
  for j in $(seq $((i + 1)) $((NTAGS - 1))); do
    pair_num=$((pair_num + 1))
    TAG_I="${TAGS[$i]}"
    TAG_J="${TAGS[$j]}"
    PAIR_JSON="$REPORTS_RR/${TAG_I}_vs_${TAG_J}.json"

    if [ -f "$PAIR_JSON" ]; then
      echo "[$pair_num/$TOTAL_PAIRS] $TAG_I vs $TAG_J — cached, skipping"
      continue
    fi

    MODEL_I_PKL="$CKPT_ROOT/suite_${TAG_I}/${TAG_I}.pkl"
    [ ! -f "$MODEL_I_PKL" ] && MODEL_I_PKL="$CKPT_ROOT/suite_${TAG_I}/${TAG_I}_latest.pkl"
    MODEL_J_TAR="$MODELS_DIR/${TAG_J}.tar.gz"

    if [ ! -f "$MODEL_I_PKL" ] || [ ! -f "$MODEL_J_TAR" ]; then
      echo "!! [$pair_num/$TOTAL_PAIRS] missing artifacts for $TAG_I vs $TAG_J — skipping"
      continue
    fi

    # Swap OUR agent's checkpoint to model_i. tcg-build converts MLX -> the
    # PyTorch FP16 mirror at model/bc_model/bc_best_torch_fp16.pt, which is what
    # agent/main.py loads. We reuse the model_i tarball we already packaged in
    # phase 1 as the --out target (idempotent, no extra tarball on disk).
    cp "$MODEL_I_PKL" "$CKPT_ROOT/bc_best_mlx.pkl"
    uv run tcg-build --checkpoint "$CKPT_ROOT/bc_best_mlx.pkl" \
      --out "$MODELS_DIR/${TAG_I}.tar.gz" >/dev/null 2>&1

    PAIR_START=$(python3 -c "import time; print(time.time())")
    uv run tcg-tournament \
      --games $ROUND_ROBIN_GAMES \
      --no-sweep --skip-baselines \
      --opponent "$MODEL_J_TAR" \
      --note "round-robin ${TAG_I} vs ${TAG_J}" \
      --report-json "$PAIR_JSON" \
      2>&1 | tail -5
    PAIR_END=$(python3 -c "import time; print(time.time())")
    PAIR_DUR=$(python3 -c "print(int($PAIR_END - $PAIR_START))")

    WR=$(overall_wr "$PAIR_JSON")
    NOW=$(python3 -c "import time; print(time.time())")
    RR_ELAPSED=$(python3 -c "print(int($NOW - $RR_START))")
    if [ $pair_num -gt 0 ]; then
      AVG=$(python3 -c "print($RR_ELAPSED / $pair_num)")
      REMAIN=$((TOTAL_PAIRS - pair_num))
      ETA=$(python3 -c "print(int($AVG * $REMAIN))")
      echo "[$pair_num/$TOTAL_PAIRS] $TAG_I vs $TAG_J: wr(i)=${WR:-?}% (pair $(format_dur $PAIR_DUR), RR ETA $(format_dur $ETA))"
    fi
  done
done

# ---------------- phase 3: aggregate ----------------
echo ""
echo "======================================================================"
echo "  PHASE 3: aggregating reports into $RESULTS_MD"
echo "======================================================================"

uv run python - "$EXP_ROOT" "$RESULTS_MD" <<'PY'
import glob, json, os, sys
exp_root, results_md = sys.argv[1], sys.argv[2]
per_run_dir = os.path.join(exp_root, "reports", "per_run")
rr_dir = os.path.join(exp_root, "reports", "round_robin")

def load(p):
    with open(p) as f: return json.load(f)

per_run = {}
for p in sorted(glob.glob(os.path.join(per_run_dir, "*_tourn.json"))):
    tag = os.path.basename(p).replace("_tourn.json", "")
    per_run[tag] = load(p)

tags = list(per_run.keys())

lines = []
lines.append("# BC curriculum ablation suite\n\n")
lines.append("## Phase 1: per-run tournaments (sweep ON, remote-Elo top decks)\n\n")

# Header: overall + per-opponent columns using union of opponent labels
opp_labels: list[str] = []
seen = set()
for tag in tags:
    for row in per_run[tag]["rows"]:
        lbl = row["opponent_label"]
        if lbl not in seen:
            seen.add(lbl); opp_labels.append(lbl)

lines.append("| run | overall | " + " | ".join(opp_labels) + " |\n")
lines.append("|-----|---------|" + "|".join(["---"] * len(opp_labels)) + "|\n")
for tag in tags:
    rep = per_run[tag]
    ov = rep["overall"]["wr_pct"]
    # aggregate by opponent (sum across decks in sweep)
    agg: dict[str, dict[str, int]] = {}
    for row in rep["rows"]:
        lbl = row["opponent_label"]
        if row.get("error"): agg[lbl] = None; continue
        a = agg.setdefault(lbl, {"w":0,"l":0,"d":0})
        if a is None: continue
        a["w"] += row["wins"]; a["l"] += row["losses"]; a["d"] += row["draws"]
    cells = []
    for lbl in opp_labels:
        a = agg.get(lbl)
        if a is None or a == {}: cells.append("—")
        else:
            n = max(a["w"] + a["l"], 1)
            cells.append(f"{a['w']*100/n:.1f}%")
    lines.append(f"| {tag} | {ov:.1f}% | " + " | ".join(cells) + " |\n")

# Phase 2: round-robin
lines.append("\n## Phase 2: round-robin between trained models (no-sweep, no baselines)\n\n")
# matrix[i][j] = WR of tag_i vs tag_j (from the report where tag_i is our_agent)
matrix: dict[tuple[str,str], float] = {}
for p in glob.glob(os.path.join(rr_dir, "*.json")):
    name = os.path.basename(p).replace(".json", "")
    # split on "_vs_" (safe because our tag scheme uses underscores but no "_vs_")
    if "_vs_" not in name: continue
    ti, tj = name.split("_vs_", 1)
    r = load(p)
    matrix[(ti, tj)] = r["overall"]["wr_pct"]
    # mirror: tag_j's WR = 100 - tag_i's WR (draws ignored in this cell)
    matrix[(tj, ti)] = 100.0 - r["overall"]["wr_pct"]

if matrix:
    lines.append("| our \\ opp | " + " | ".join(tags) + " | avg |\n")
    lines.append("|-----------|" + "|".join(["---"] * (len(tags) + 1)) + "|\n")
    for ti in tags:
        cells = []
        wrs = []
        for tj in tags:
            if ti == tj: cells.append("—")
            elif (ti, tj) in matrix:
                v = matrix[(ti, tj)]
                cells.append(f"{v:.1f}")
                wrs.append(v)
            else: cells.append("?")
        avg = f"{sum(wrs)/len(wrs):.1f}" if wrs else "—"
        lines.append(f"| {ti} | " + " | ".join(cells) + f" | **{avg}** |\n")
else:
    lines.append("_no round-robin pairs found yet_\n")

with open(results_md, "w") as f: f.writelines(lines)
print(f"[aggregate] wrote {results_md}")
PY

TOTAL_DUR=$(python3 -c "import time; print(int(time.time() - $SUITE_START))")
echo ""
echo "======================================================================"
echo "SUITE COMPLETE in $(format_dur $TOTAL_DUR)."
echo "Results: $RESULTS_MD"
echo "Log:     $SUITE_LOG"
echo "======================================================================"
