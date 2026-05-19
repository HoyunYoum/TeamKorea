#!/usr/bin/env bash
# test_submission_sigterm.sh — force SIGTERM path by giving mlsys a much
# larger internal timeout (10x) than the external container timeout.
#
# Normally the solver self-terminates ~0.5s before timeout (LocalSearch
# safety margin). To actually exercise the SIGTERM handler + atomic-write
# invariant, we trick mlsys into thinking it has 10x budget so it keeps
# running until the external `timeout -s TERM` fires at the real
# competition deadline.
#
# Expected: exit=143 (128+15=SIGTERM), wall≈external_to, output still
# valid (atomic write guarantees last-written solution survives).
#
# Usage: scripts/test_submission_sigterm.sh [zip_path]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ZIP="${1:-$ROOT/TeamKorea_TrackA_1.zip}"
if [ ! -f "$ZIP" ]; then
  echo "✗ zip not found: $ZIP" >&2
  exit 2
fi

EVAL_CLI="$ROOT/build/eval_cli"
if [ ! -x "$EVAL_CLI" ]; then
  echo "✗ host eval_cli missing — cmake --build build --target eval_cli" >&2
  exit 2
fi

declare -A TO=( [1]=2 [5]=5 [9]=15 [13]=30 [17]=60 )
BMS=(1 5 9 13 17)

STAGE="$(mktemp -d -t mlsys-sigterm-XXXXXX)"
trap "rm -rf '$STAGE'" EXIT

echo "▶ Extracting $(basename "$ZIP") → $STAGE"
unzip -q "$ZIP" -d "$STAGE"
chmod +x "$STAGE/mlsys"

IMG="ubuntu:22.04"
DRUN=(docker run --rm --platform=linux/amd64
      --cpus=8 --memory=32g --network=none
      -v "$STAGE:/work" -w /work
      "$IMG")

echo "▶ Pulling $IMG if needed..."
docker pull -q --platform=linux/amd64 "$IMG" >/dev/null

printf '\n%-6s %-9s %-9s %-7s %-4s %-12s %-12s %s\n' \
  BM ext_to int_to wall exit cost status notes
printf -- '------ --------- --------- ------- ---- ------------ ------------ ----------------------\n'

FAILED=0
for bm in "${BMS[@]}"; do
  ext_to=${TO[$bm]}
  int_to=$((ext_to * 10))   # solver thinks it has 10x
  prob="$ROOT/benchmarks/mlsys-2026-${bm}.json"
  out="$STAGE/out-${bm}.json"
  rm -f "$out"
  cp "$prob" "$STAGE/prob-${bm}.json"

  t0=$(date +%s.%N)
  set +e
  "${DRUN[@]}" timeout -s TERM "${ext_to}s" \
      /work/mlsys "/work/prob-${bm}.json" "/work/out-${bm}.json" "${int_to}" \
      > "$STAGE/run-${bm}.log" 2>&1
  rc=$?
  set -e
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")

  notes=""
  # GNU timeout semantics (verified): rc=124 is the normal "timed out"
  # return even when the child caught SIGTERM and exited cleanly (only
  # --preserve-status would propagate the child's 0). rc=137 (128+9)
  # would mean --kill-after had to escalate to SIGKILL — that's the
  # real "handler failed" signal. rc=0 = solver self-terminated despite
  # the 10× budget (unlikely unless the BM is tiny).
  case $rc in
    124) notes="SIGTERM caught, _Exit(0) path" ;;
    0)   notes="self-terminated before deadline" ;;
    137) notes="SIGKILL escalation — handler failed!"; FAILED=$((FAILED+1)) ;;
    *)   notes="unexpected exit=$rc" ;;
  esac

  if [ ! -s "$out" ]; then
    status="FAIL"
    notes="${notes}; no output file"
    printf '%-6s %-9s %-9s %-7s %-4s %-12s %-12s %s\n' \
      "$bm" "${ext_to}s" "${int_to}s" "${wall}s" "$rc" "-" "$status" "$notes"
    FAILED=$((FAILED+1))
    continue
  fi

  raw=$("$EVAL_CLI" "$prob" "$out" 2>/dev/null | tail -1)
  if [[ "$raw" =~ ^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
    cost=$(awk -v v="$raw" 'BEGIN{printf "%.0f", v}')
    status="ok"
  else
    cost="$raw"
    status="INVALID"
    notes="${notes}; eval_cli rejected"
    FAILED=$((FAILED+1))
  fi
  printf '%-6s %-9s %-9s %-7s %-4s %-12s %-12s %s\n' \
    "$bm" "${ext_to}s" "${int_to}s" "${wall}s" "$rc" "$cost" "$status" "$notes"
done

echo
if [ $FAILED -eq 0 ]; then
  echo "✓ atomic-write + SIGTERM handler survive on all ${#BMS[@]} BMs"
else
  echo "✗ $FAILED / ${#BMS[@]} BMs failed SIGTERM test"
  exit 1
fi
