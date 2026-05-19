#!/bin/bash
# Run validate_bm.sh on every BM present in benchmarks/.
# Summary at the end: status per BM, elapsed time, any anomaly count.
#
# Usage: tools/validate_all.sh [bm_min] [bm_max]
#   tools/validate_all.sh       → all present BMs
#   tools/validate_all.sh 1 24  → BM-1 through BM-24 (skip missing)

set -u
cd "$(dirname "$0")/.."

MIN="${1:-1}"
MAX="${2:-24}"

PRESENT=()
for i in $(seq $MIN $MAX); do
  [ -f "benchmarks/mlsys-2026-${i}.json" ] && PRESENT+=("$i")
done
if [ ${#PRESENT[@]} -eq 0 ]; then
  echo "No benchmarks in range [$MIN, $MAX]."
  exit 0
fi

echo "Validating ${#PRESENT[@]} BMs: ${PRESENT[*]}"
echo ""

OK=0; ANOMALIES=0; FAILURES=0
RESULTS=()

for bm in "${PRESENT[@]}"; do
  OUT=$(tools/validate_bm.sh "$bm" 2>&1)
  if echo "$OUT" | grep -q '\[ANOMALY\]\|FAIL\|REJECTED\|CRASHED\|TIMED OUT'; then
    if echo "$OUT" | grep -q 'FAIL\|REJECTED\|CRASHED\|TIMED OUT'; then
      FAILURES=$((FAILURES+1))
    else
      ANOMALIES=$((ANOMALIES+1))
    fi
    RESULTS+=("BM-$bm: ISSUES")
  else
    OK=$((OK+1))
    RESULTS+=("BM-$bm: OK")
  fi
  echo "$OUT"
  echo ""
done

echo "=========================================="
echo "SUMMARY: $OK ok, $ANOMALIES anomalies, $FAILURES failures / ${#PRESENT[@]} BMs"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo ""
if [ $FAILURES -gt 0 ]; then
  echo "Failures need immediate investigation (solution rejected or crashed)."
  exit 1
fi
if [ $ANOMALIES -gt 0 ]; then
  echo "Anomalies are warnings — solution is valid but behavior unusual."
  exit 0
fi
