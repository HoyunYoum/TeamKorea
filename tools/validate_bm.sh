#!/bin/bash
# Hidden BM validation runner.
#
# Usage:
#   tools/validate_bm.sh <bm_number> [timeout_override]
#
# Example:
#   tools/validate_bm.sh 6
#   tools/validate_bm.sh 24 300   # custom timeout for stress test
#
# Produces:
#   /tmp/bm<N>_pattern.json   — pattern_solver output
#   /tmp/bm<N>_baseline.json  — baseline_solver floor
#   stderr summary with status + anomaly flags

set -u
cd "$(dirname "$0")/.."

BM="${1:?usage: validate_bm.sh <bm_number>}"
PROB="benchmarks/mlsys-2026-${BM}.json"

if [ ! -f "$PROB" ]; then
  echo "FAIL: $PROB not found. Did the BM release?"
  exit 1
fi

# Competition timeouts (memory: feedback_no_10x_timeouts).
case "$BM" in
  1|2|3|4)     TO=2   ;;
  5|6|7|8)     TO=5   ;;
  9|10|11|12)  TO=15  ;;
  13|14|15|16) TO=30  ;;
  17|18|19|20) TO=60  ;;
  21|22|23|24) TO=120 ;;
  *) echo "FAIL: unknown BM number $BM"; exit 1 ;;
esac
TO="${2:-$TO}"

# Topology profile.
python3 -c "
import json
p = json.load(open('$PROB'))
n = len(p['op_types'])
n_mm = sum(1 for o in p['op_types'] if o == 'MatMul')
n_pw = n - n_mm
shapes = sorted(set(zip(p['widths'], p['heights'])))
max_t = max(w*h for w,h in shapes)
cap = p['fast_memory_capacity']
print(f'BM-$BM  N={n} ({n_mm}MM+{n_pw}PW)  cap={cap}  bw={p[\"slow_memory_bandwidth\"]}  max_tensor/cap={max_t/cap:.1f}x  shapes={len(shapes)}')
"

# Baseline floor.
./build/baseline_solver "$PROB" /tmp/bm${BM}_baseline.json 2>/dev/null
BASE=$(./build/eval_cli "$PROB" /tmp/bm${BM}_baseline.json 2>/dev/null | tail -1)
echo "  baseline: $BASE"

# Lower bound (analytical).
LB=$(./build/lower_bound "$PROB" 2>&1 | grep overall_lb | awk '{print $2}')
echo "  lower_bound: $LB"

# Pattern solver.
t0=$(date +%s.%N)
timeout $((TO + 15)) ./build/pattern_solver "$PROB" /tmp/bm${BM}_pattern.json "$TO" \
  > /tmp/bm${BM}_pattern.log 2>&1
status=$?
t1=$(date +%s.%N)
el=$(echo "$t1 - $t0" | bc)

if [ $status -eq 124 ]; then
  echo "  pattern_solver: TIMED OUT after ${el}s"
  exit 2
fi
if [ $status -ne 0 ]; then
  echo "  pattern_solver: CRASHED (exit $status)"
  cat /tmp/bm${BM}_pattern.log
  exit 2
fi

PS=$(./build/eval_cli "$PROB" /tmp/bm${BM}_pattern.json 2>/dev/null | tail -1)
if [ -z "$PS" ] || [[ "$PS" == Error* ]]; then
  echo "  pattern_solver: EVALUATE REJECTED — $PS"
  ./build/eval_cli "$PROB" /tmp/bm${BM}_pattern.json 2>&1
  exit 3
fi

echo "  pattern_solver: $PS  ($(printf '%.1f' $el)s / ${TO}s budget)"

# Anomaly flags.
FLAGS=""

# (1) Worse than baseline = catastrophic library gap.
awk -v ps="$PS" -v base="$BASE" 'BEGIN { if (ps > base * 0.99) exit 1 }'
if [ $? -eq 1 ]; then
  FLAGS="$FLAGS  [ANOMALY] pattern ≥ baseline — library or WS gap, check [enum] counts"
fi

# (2) Budget exhaustion.
awk -v el="$el" -v to="$TO" 'BEGIN { if (el > to * 0.9) exit 1 }'
if [ $? -eq 1 ]; then
  FLAGS="$FLAGS  [ANOMALY] used >90% timeout — consider S3 decomposition"
fi

# (3) MIP non-optimal (check [mip] line status specifically).
if grep -E '^\s*\[mip\]' /tmp/bm${BM}_pattern.log | grep -qv OPTIMAL; then
  FLAGS="$FLAGS  [ANOMALY] MIP did not reach OPTIMAL — large column count"
fi
if grep -q 'greedy fallback' /tmp/bm${BM}_pattern.log; then
  FLAGS="$FLAGS  [ANOMALY] MIP failed — greedy fallback engaged"
fi

# (4) MIP obj SHOULD-BE upper bound, but Evaluate exceeds by >2% → real problem
# (retention + gran re-opt can only LOWER actual cost from MIP estimate).
MIP_OBJ=$(grep -oE 'obj=[0-9]+' /tmp/bm${BM}_pattern.log | head -1 | cut -d= -f2)
if [ -n "$MIP_OBJ" ]; then
  awk -v mip="$MIP_OBJ" -v ps="$PS" 'BEGIN {
    if (ps > mip * 1.02) exit 1
  }'
  if [ $? -eq 1 ]; then
    FLAGS="$FLAGS  [ANOMALY] Evaluate > MIP obj by >2% — column cost underestimate"
  fi
fi

if [ -n "$FLAGS" ]; then
  echo "---"
  echo "$FLAGS"
  echo "---"
  echo "  log: /tmp/bm${BM}_pattern.log"
fi
