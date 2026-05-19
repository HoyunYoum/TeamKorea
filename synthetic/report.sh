#!/bin/bash
# Generate a full synthetic benchmark report with LB + all solvers.
#
# Usage: synthetic/report.sh [scenario...]
set -u
cd "$(dirname "$0")/.."

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
  SCENARIOS=$(ls synthetic/generated/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//' | sort)
else
  SCENARIOS="$@"
fi

# Header
printf "%-28s %4s %6s %5s %10s %12s %12s %12s %12s %8s %8s\n" \
  "scenario" "N" "cap/T" "bw" "LB" "baseline" "io" "unified" "pattern" "vs_LB" "vs_best"
echo "---------------------------------------------------------------------------------------------------------------------------------"

for sc in $SCENARIOS; do
  prob="synthetic/generated/${sc}.json"
  [ -f "$prob" ] || continue

  read N RATIO BW <<< $(python3 -c "
import json
p = json.load(open('$prob'))
n = len(p['op_types'])
max_t = max(w*h for w,h in zip(p['widths'], p['heights']))
print(n, f'{max_t/p[\"fast_memory_capacity\"]:.1f}', p['slow_memory_bandwidth'])
")

  TO=$(choose_timeout "$N")

  # LB
  LB=$(./build/lower_bound "$prob" 2>&1 | grep overall_lb | awk '{print $2}')

  # Solvers — reuse cached outputs (from compare.sh run), compute costs.
  BASE=$(./build/eval_cli "$prob" /tmp/sc_${sc}_baseline.json 2>/dev/null | tail -1 2>/dev/null)
  IO=$(./build/eval_cli "$prob" /tmp/sc_${sc}_io.json 2>/dev/null | tail -1 2>/dev/null)
  UF=$(./build/eval_cli "$prob" /tmp/sc_${sc}_unified.json 2>/dev/null | tail -1 2>/dev/null)
  PS=$(./build/eval_cli "$prob" /tmp/sc_${sc}_pattern.json 2>/dev/null | tail -1 2>/dev/null)

  # Filter non-numerics.
  valid() { [[ "$1" =~ ^[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$ ]]; }
  valid "$BASE" || BASE=""
  valid "$IO"   || IO=""
  valid "$UF"   || UF=""
  valid "$PS"   || PS=""

  # Best across non-baseline solvers (io/unified/pattern).
  BEST=""
  for v in "$IO" "$UF" "$PS"; do
    valid "$v" || continue
    if [ -z "$BEST" ]; then BEST="$v";
    elif [ "$(awk -v a="$v" -v b="$BEST" 'BEGIN{print (a+0<b+0)?1:0}')" = "1" ]; then BEST="$v"; fi
  done

  # Deltas.
  vs_lb=""
  vs_best=""
  if valid "$PS" && valid "$LB"; then
    vs_lb=$(awk -v p="$PS" -v l="$LB" 'BEGIN{printf "%.2fx", p/l}')
  fi
  if valid "$PS" && valid "$BEST"; then
    vs_best=$(awk -v p="$PS" -v b="$BEST" 'BEGIN{printf "%+.1f%%", (p/b - 1)*100}')
  fi

  fmt() { [ -z "$1" ] && echo "-" || printf "%g" "$1"; }

  printf "%-28s %4s %6sx %5s %10s %12s %12s %12s %12s %8s %8s\n" \
    "$sc" "$N" "$RATIO" "$BW" "$(fmt "$LB")" "$(fmt "$BASE")" "$(fmt "$IO")" \
    "$(fmt "$UF")" "$(fmt "$PS")" "$vs_lb" "$vs_best"
done

echo ""
echo "Notes:"
echo "  cap/T  = max_tensor_size / fast_memory_capacity (higher = tighter WS pressure)"
echo "  vs_LB  = pattern_solver / lower_bound  (roofline gap; 1.0 = LB-tight)"
echo "  vs_best = pattern_solver vs min(io, unified, pattern)  (+% = pattern loses)"
