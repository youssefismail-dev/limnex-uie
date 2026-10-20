#!/usr/bin/env bash
# Chain phase 2 onto phase 1 without needing anyone awake to press go.
#
# Phase 1 (already running) is the likelihood suite and four canonical
# run_eval jobs. Phase 2 is the calibrated harness: 4 models x 2 classes, each
# G=5 generations at n=1800, scored by D=3 discriminator seeds.
#
# OMP_NUM_THREADS=1 IS THE POINT. Phase 1 was launched without it, so one job
# grabbed ~4 torch threads while four others got one each -- 8 threads of demand
# on 6 cores, and everything ran 3-4x slower than measured. With tiny batches
# torch's intra-op threading buys almost nothing here, so pinning to 1 turns
# eight arms into eight clean single-core jobs.
set -u

R="$(cd "$(dirname "$0")/../.." && pwd)"   # the repository root
T="$R/data/scratch"                         # git-ignored working folder
LOG="$T/overnight.log"
CACHE="$T/harness_cache"
cd "$R" || exit 1

say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

say "orchestrator started"

# ---- wait for phase 1 -------------------------------------------------------
while :; do
    done1=0
    [ -f "$R/data/s4_likelihood.json" ] && done1=$((done1+1))
    for n in E5 w1 w2 w4; do
        grep -q "EXIT_$n=" "$T/eval/$n.log" 2>/dev/null && done1=$((done1+1))
    done
    [ "$done1" -ge 5 ] && break
    say "phase 1: $done1/5 complete, waiting"
    sleep 120
done
say "phase 1 COMPLETE"

# ---- honour the bit-exactness check ----------------------------------------
# If today's edit-and-revert perturbed the decoder, every cached E5 generation
# is from a different code path and reusing it would confound the model
# comparison with a code change. Drop them and let them regenerate.
if grep -q "bit-identical to cache: False" "$T/tasks/bfut97rvg.output" 2>/dev/null; then
    say "BIT-EXACT CHECK FAILED -- discarding cached E5 generations"
    rm -f "$CACHE"/best_e5_*_n1800_*.json
else
    say "bit-exact check passed or inconclusive -- cache retained"
fi

# ---- phase 2: eight arms, six at a time ------------------------------------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$T/arms"
running=0
for m in E5 w1 w2 w4; do
    for c in benign phishing; do
        (
          python -u research/s4/arms.py --model "$m" --label "$c" -n 1800 -D 3 \
              > "$T/arms/${m}_${c}.log" 2>&1
          echo "EXIT=$?" >> "$T/arms/${m}_${c}.log"
        ) &
        running=$((running+1))
        say "launched $m/$c"
        if [ "$running" -ge 6 ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
    done
done
wait
say "phase 2 COMPLETE"

for f in "$T/arms"/*.log; do say "$(basename "$f"): $(tail -1 "$f")"; done
say "orchestrator done"
