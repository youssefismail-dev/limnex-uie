#!/usr/bin/env bash
# Finish the missing arms. Written as a FILE rather than a compound one-liner:
# the previous attempt lost $T inside backgrounded subshells and several arms
# never launched while the wrapper still reported success.
#
# Absolute paths throughout, no reliance on the caller's cwd, and each arm logs
# its own exit code so a silent miss is impossible to mistake for a completion.
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"   # the repository root
T="$R/data/scratch"                         # git-ignored working folder
D="$T/d0"
mkdir -p "$D"
cd "$R" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run () {   # model class delta tag
    local m="$1" c="$2" d="$3" tag="$4"
    if [ "$d" = "adopted" ]; then
        python -u research/s4/arms.py --model "$m" --label "$c" -n 1800 -D 3 \
            > "$D/$tag.log" 2>&1
    else
        python -u research/s4/arms.py --model "$m" --label "$c" -n 1800 -D 3 \
            --delta "$d" > "$D/$tag.log" 2>&1
    fi
    echo "EXIT=$?" >> "$D/$tag.log"
}

run E5 phishing adopted  fix_E5_phishing_adopted &
run w2 benign   0        fix_w2_benign_d0        &
run w2 phishing 0        fix_w2_phishing_d0      &
run w4 benign   0        fix_w4_benign_d0        &
run w4 phishing 0        fix_w4_phishing_d0      &
wait

echo "=== results ==="
for f in "$D"/fix_*.log; do echo "$(basename "$f" .log): $(tail -1 "$f")"; done
echo "=== files on disk ==="
ls -1 "$R/data/s4_arms/"
