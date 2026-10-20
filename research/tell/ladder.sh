#!/usr/bin/env bash
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$R" || exit 1   # the repository root
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for c in benign phishing; do
  ( python -u research/tell/ablate.py --label "$c" -n 1800 -D 3 \
      > "$R/data/tell/ladder_$c.log" 2>&1; echo "EXIT=$?" >> "$R/data/tell/ladder_$c.log" ) &
done
wait
echo "LADDER DONE"
