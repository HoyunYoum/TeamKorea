#!/bin/bash
# Run pattern_solver + legacy solvers on synthetic benchmarks.
# Produces a matrix showing where pattern_solver wins/ties/loses.
#
# Usage:
#   synthetic/compare.sh                    # all generated scenarios
#   synthetic/compare.sh mlp_small ffn_large  # specific scenarios
#
# Requires: CMake -DBUILD_LEGACY_SOLVERS=ON for legacy executables.

set -u
cd "$(dirname "$0")/.."

# Default timeout per op-count (calibrated vs competition scale).
choose_timeout() {
  local n=$1
  if   [ $n -le 10  ]; then echo 2
  elif [ $n -le 20  ]; then echo 5
  elif [ $n -le 40  ]; then echo 15
  elif [ $n -le 80  ]; then echo 30
  elif [ $n -le 150 ]; then echo 60
  else                      echo 120
  fi
}

if [ $# -eq 0 ]; then
  SCENARIOS=$(ls synthetic/generated/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//')
else
  SCENARIOS="$@"
fi

printf "%-28s %5s %8s %14s %14s %14s %14s\n" \
  "scenario" "N" "cap/max" "baseline" "io" "unified" "pattern"
echo "--------------------------------------------------------------------------------------------------------"

for sc in $SCENARIOS; do
  prob="synthetic/generated/${sc}.json"
  [ -f "$prob" ] || { echo "MISSING: $prob"; continue; }

  # Extract config.
  read N MAX_CAP_RATIO <<< $(python3 -c "
import json
p = json.load(open('$prob'))
n = len(p['op_types'])
max_t = max(w*h for w,h in zip(p['widths'], p['heights']))
ratio = max_t / p['fast_memory_capacity']
print(n, f'{ratio:.1f}x')
")

  TO=$(choose_timeout "$N")

  # Run each solver.
  ./build/baseline_solver "$prob" /tmp/sc_${sc}_baseline.json 2>/dev/null
  BASE=$(./build/eval_cli "$prob" /tmp/sc_${sc}_baseline.json 2>/dev/null | tail -1 2>/dev/null || echo "N/A")

  timeout $((TO + 10)) ./build/io_solver "$prob" /tmp/sc_${sc}_io.json \
    >/dev/null 2>/dev/null
  IO=$(./build/eval_cli "$prob" /tmp/sc_${sc}_io.json 2>/dev/null | tail -1 2>/dev/null || echo "FAIL")

  timeout $((TO + 10)) ./build/unified_solver "$prob" /tmp/sc_${sc}_unified.json "" "$TO" \
    >/dev/null 2>/dev/null
  UF=$(./build/eval_cli "$prob" /tmp/sc_${sc}_unified.json 2>/dev/null | tail -1 2>/dev/null || echo "FAIL")

  timeout $((TO + 10)) ./build/pattern_solver "$prob" /tmp/sc_${sc}_pattern.json "$TO" \
    >/dev/null 2>/dev/null
  PS=$(./build/eval_cli "$prob" /tmp/sc_${sc}_pattern.json 2>/dev/null | tail -1 2>/dev/null || echo "FAIL")

  # Format numbers.
  fmt() {
    local v="$1"
    if [[ "$v" =~ ^[0-9] ]] || [[ "$v" =~ ^-?[0-9] ]]; then
      printf "%g" "$v"
    else
      echo "$v"
    fi
  }

  # Find minimum across valid numeric results.
  BEST=$(for v in "$IO" "$UF" "$PS"; do
    [[ "$v" =~ ^[0-9] ]] && echo "$v"
  done | sort -g | head -1)

  # Delta vs best for pattern_solver.
  DELTA=""
  if [[ "$PS" =~ ^[0-9] ]] && [[ "$BEST" =~ ^[0-9] ]]; then
    DELTA=$(awk -v p="$PS" -v b="$BEST" 'BEGIN { printf "%+.1f%%", (p/b - 1) * 100 }')
  fi

  printf "%-28s %5s %8s %14s %14s %14s %14s %s\n" \
    "$sc" "$N" "$MAX_CAP_RATIO" "$(fmt "$BASE")" "$(fmt "$IO")" "$(fmt "$UF")" "$(fmt "$PS")" "$DELTA"
done
