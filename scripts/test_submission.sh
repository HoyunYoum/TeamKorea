#!/usr/bin/env bash
# test_submission.sh — smoke-test a packed Track A zip in a clean Docker
# container matching the organizer's grading host spec (issue #72/#83):
#   Ubuntu 22.04 LTS, x86-64-v3, 8 cores, 32 GB RAM, CPU-only, offline.
#
# For each public BM, runs the bundled `mlsys` under the real competition
# timeout (SIGTERM, matching the harness) and scores the output with the
# host-side eval_cli.
#
# Usage:
#   scripts/test_submission.sh [zip_path]
#   zip_path defaults to TeamKorea_TrackA_1.zip at repo root.

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
  echo "✗ host eval_cli missing at $EVAL_CLI — build it first:" >&2
  echo "    cmake --build build --target eval_cli" >&2
  exit 2
fi

if ! command -v docker >/dev/null; then
  echo "✗ docker not found in PATH" >&2
  exit 2
fi

# Competition timeouts (feedback_no_10x_timeouts memory: never use 10×).
declare -A TO=( [1]=2 [5]=5 [9]=15 [13]=30 [17]=60 )
BMS=(1 5 9 13 17)

STAGE="$(mktemp -d -t mlsys-test-XXXXXX)"
trap "rm -rf '$STAGE'" EXIT

echo "▶ Extracting $(basename "$ZIP") → $STAGE"
unzip -q "$ZIP" -d "$STAGE"

# Required contents per README §Track A.
for f in mlsys source writeup.pdf; do
  [ -e "$STAGE/$f" ] || { echo "✗ missing from zip: $f" >&2; exit 1; }
done
chmod +x "$STAGE/mlsys"

IMG="ubuntu:22.04"
DRUN=(docker run --rm --platform=linux/amd64
      --cpus=8 --memory=32g --network=none
      -v "$STAGE:/work" -w /work
      "$IMG")

echo "▶ Pulling $IMG (if needed)..."
docker pull -q --platform=linux/amd64 "$IMG" >/dev/null

echo "▶ ldd mlsys (inside container):"
"${DRUN[@]}" ldd /work/mlsys 2>&1 | sed 's/^/    /'
if "${DRUN[@]}" ldd /work/mlsys 2>&1 \
    | grep -qE 'libabsl|libhighs|libstdc\+\+|libgcc_s'; then
  echo "✗ non-libc/libm dependency — would fail on clean Ubuntu 22.04" >&2
  exit 1
fi
echo "  ✓ static except for libc/libm"

printf '\n%-6s %-8s %-7s %-12s %-12s %s\n' BM timeout wall cost status notes
printf -- '------ -------- ------- ------------ ------------ ----------------------\n'

FAILED=0
for bm in "${BMS[@]}"; do
  to=${TO[$bm]}
  prob="$ROOT/benchmarks/mlsys-2026-${bm}.json"
  out="$STAGE/out-${bm}.json"
  log="$STAGE/run-${bm}.log"
  rm -f "$out"
  cp "$prob" "$STAGE/prob-${bm}.json"

  # `timeout -s TERM` mirrors the grading harness SIGTERM (our mlsys has
  # an atexit-safe SIGTERM handler that _Exit(0)s so the last atomic
  # write survives). Give the container a 3s grace window on top of $to.
  t0=$(date +%s.%N)
  set +e
  "${DRUN[@]}" timeout -s TERM "${to}s" \
      /work/mlsys "/work/prob-${bm}.json" "/work/out-${bm}.json" \
      > "$log" 2>&1
  rc=$?
  set -e
  t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")

  notes=""
  # rc=124 = timeout fired but kill signal (not our path). rc=143 =
  # SIGTERM (128+15): harness-equivalent kill. Both leave the last
  # atomic write on disk.
  if [ $rc -ne 0 ] && [ $rc -ne 124 ] && [ $rc -ne 143 ]; then
    notes="exit=$rc (see $log)"
  fi
  if [ ! -s "$out" ]; then
    status="FAIL"
    notes="${notes:+$notes; }no output file"
    printf '%-6s %-8s %-7s %-12s %-12s %s\n' \
      "$bm" "${to}s" "${wall}s" "-" "$status" "$notes"
    FAILED=$((FAILED+1))
    continue
  fi

  raw=$("$EVAL_CLI" "$prob" "$out" 2>/dev/null | tail -1)
  # eval_cli prints scientific notation for large values (e.g. 4.9664e+06).
  # Costs are spec-integer — render as plain int via awk.
  if [[ "$raw" =~ ^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
    cost=$(awk -v v="$raw" 'BEGIN{printf "%.0f", v}')
    status="ok"
  else
    cost="$raw"
    status="INVALID"
    notes="${notes:+$notes; }eval_cli rejected: $raw"
    FAILED=$((FAILED+1))
  fi
  printf '%-6s %-8s %-7s %-12s %-12s %s\n' \
    "$bm" "${to}s" "${wall}s" "$cost" "$status" "$notes"
done

echo
if [ $FAILED -eq 0 ]; then
  echo "✓ all ${#BMS[@]} public BMs passed inside 8c/32g Ubuntu 22.04 container"
else
  echo "✗ $FAILED / ${#BMS[@]} BMs failed — see per-BM logs in $STAGE (will be wiped on exit)"
  echo "  (to preserve logs, copy $STAGE before this script returns)"
  exit 1
fi
