#!/usr/bin/env bash
# bench.sh — benchmark harness. Runs every solver on every benchmark and
# stores solutions under a named run directory so later reports can be
# generated without re-running the solvers.
#
# Usage:
#   scripts/bench.sh run [run-name] [--timeout N] [--solvers s1,s2,...] [--bms glob]
#   scripts/bench.sh report [run-name]   # default: latest
#   scripts/bench.sh list
#   scripts/bench.sh diff <run-a> <run-b>
#
# Layout:
#   results/
#     latest -> <run-name>/   (symlink, updated on every run)
#     <run-name>/
#       solutions/<solver>/<bm>.json
#       logs/<solver>/<bm>.log
#       meta.tsv          (bm, solver, status, elapsed_s)
#       lb.tsv            (bm, lb_value)
#       metadata.txt      (commit, date, build-dir, solver set)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/results"
BUILD="$ROOT/build"

ALL_SOLVERS="pattern io partition unified"
DEFAULT_TIMEOUT=60

# ── helpers ────────────────────────────────────────────────────────────

bm_files() {
  # Prints one path per line for every problem file we benchmark.
  # Public BMs are renamed public-N for display.
  for f in "$ROOT"/benchmarks/mlsys-2026-*.json; do
    [ -f "$f" ] || continue
    printf '%s\t%s\n' "$f" "public-$(basename "$f" .json | sed 's/mlsys-2026-//')"
  done
  for f in "$ROOT"/synthetic/generated/*.json; do
    [ -f "$f" ] || continue
    printf '%s\t%s\n' "$f" "$(basename "$f" .json)"
  done
}

# ── run subcommand ─────────────────────────────────────────────────────

cmd_run() {
  local run_name="${1:-}"
  shift || true
  local timeout_s=$DEFAULT_TIMEOUT
  local solvers=""
  local bm_filter=""

  while [ $# -gt 0 ]; do
    case "$1" in
      --timeout) timeout_s="$2"; shift 2 ;;
      --solvers) solvers="$2"; shift 2 ;;
      --bms)     bm_filter="$2"; shift 2 ;;
      *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
  done

  if [ -z "$run_name" ]; then
    local commit
    commit="$(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo nogit)"
    run_name="$(date +%Y-%m-%d_%H%M)_${commit}"
  fi
  [ -n "$solvers" ] || solvers="$ALL_SOLVERS"

  local run_dir="$RESULTS/$run_name"
  mkdir -p "$run_dir"
  mkdir -p "$run_dir/logs"
  for s in $(echo "$solvers" | tr ',' ' '); do
    mkdir -p "$run_dir/solutions/$s" "$run_dir/logs/$s"
  done

  {
    echo "run_name:    $run_name"
    echo "date:        $(date -Iseconds)"
    echo "commit:      $(cd "$ROOT" && git rev-parse HEAD 2>/dev/null || echo nogit)"
    echo "commit_msg:  $(cd "$ROOT" && git log -1 --oneline 2>/dev/null || echo nogit)"
    echo "build_dir:   $BUILD"
    echo "solvers:     $solvers"
    echo "timeout_s:   $timeout_s"
    echo "bm_filter:   ${bm_filter:-<none>}"
  } > "$run_dir/metadata.txt"

  echo -e "bm\tsolver\tstatus\telapsed_s" > "$run_dir/meta.tsv"
  echo -e "bm\tlb" > "$run_dir/lb.tsv"

  echo "▶ run: $run_name  (solvers: $solvers, timeout: ${timeout_s}s)"
  echo "  dir: $run_dir"

  # Collect LBs once (cheap).
  while IFS=$'\t' read -r path name; do
    [ -z "$bm_filter" ] || [[ "$name" == $bm_filter ]] || continue
    local lb
    lb="$("$BUILD/lower_bound" "$path" 2>/dev/null | awk '/overall_lb:/ {print $2}')"
    echo -e "${name}\t${lb}" >> "$run_dir/lb.tsv"
  done < <(bm_files)

  # Run solvers SERIALLY (one solver at a time), BMs serial within each
  # solver. Avoids CPU contention between parallel solver processes that
  # would otherwise inflate timeout counts and make runs non-reproducible.
  # Wall time scales with #solvers × #bms × avg-time; on our BM suite this
  # fits in ~20 minutes.
  for solver in $(echo "$solvers" | tr ',' ' '); do
    while IFS=$'\t' read -r path name; do
      [ -z "$bm_filter" ] || [[ "$name" == $bm_filter ]] || continue
      local out="$run_dir/solutions/$solver/$name.json"
      local log="$run_dir/logs/$solver/$name.log"
      local t0 t1 elapsed rc status
      t0=$(date +%s)
      # Disable pipefail briefly so set -e doesn't kill the whole run on a
      # solver crash (segfault, assert, timeout-exit-124). We already handle
      # non-zero rc below.
      set +e
      timeout "$timeout_s" "$BUILD/${solver}_solver" "$path" "$out" \
        >"$log" 2>&1
      rc=$?
      set -e
      t1=$(date +%s)
      elapsed=$((t1 - t0))
      if [ $rc -eq 124 ]; then
        status=TIMEOUT
        rm -f "$out"
      elif [ $rc -ne 0 ]; then
        status=ERROR
        rm -f "$out"
      else
        status=OK
      fi
      echo -e "${name}\t${solver}\t${status}\t${elapsed}" >> "$run_dir/meta.tsv"
      printf '  %-10s %-26s %-8s %3ds\n' "$solver" "$name" "$status" "$elapsed"
    done < <(bm_files)
  done

  ln -sfn "$run_name" "$RESULTS/latest"
  echo "✓ done. Results in $run_dir"
  echo "  Report: scripts/bench.sh report $run_name"
}

# ── report subcommand ──────────────────────────────────────────────────

cmd_report() {
  local run_name="${1:-latest}"
  local run_dir="$RESULTS/$run_name"
  if [ ! -d "$run_dir" ]; then
    echo "No such run: $run_name" >&2
    echo "Available: $(ls "$RESULTS" 2>/dev/null | grep -v '^latest$' | tr '\n' ' ')" >&2
    exit 1
  fi

  python3 "$ROOT/scripts/bench_report.py" "$run_dir" "$BUILD/eval_cli" "$ROOT"
}

# ── list subcommand ────────────────────────────────────────────────────

cmd_list() {
  if [ ! -d "$RESULTS" ]; then
    echo "(no runs yet — use: scripts/bench.sh run)"
    return
  fi
  local latest=""
  [ -L "$RESULTS/latest" ] && latest="$(readlink "$RESULTS/latest")"
  for d in "$RESULTS"/*/; do
    [ -d "$d" ] || continue
    local name
    name="$(basename "$d")"
    [ "$name" = "latest" ] && continue
    local commit=""
    [ -f "$d/metadata.txt" ] && commit="$(grep '^commit_msg:' "$d/metadata.txt" | cut -d: -f2-)"
    local marker=""
    [ "$name" = "$latest" ] && marker=" (latest)"
    printf '  %-40s %s%s\n' "$name" "$commit" "$marker"
  done
}

# ── diff subcommand ────────────────────────────────────────────────────

cmd_diff() {
  local a="${1:?usage: diff <run-a> <run-b>}"
  local b="${2:?usage: diff <run-a> <run-b>}"
  python3 "$ROOT/scripts/bench_report.py" \
    --diff "$RESULTS/$a" "$RESULTS/$b" "$BUILD/eval_cli" "$ROOT"
}

# ── dispatch ───────────────────────────────────────────────────────────

cmd="${1:-}"
shift || true
case "$cmd" in
  run)    cmd_run    "$@" ;;
  report) cmd_report "$@" ;;
  list)   cmd_list   "$@" ;;
  diff)   cmd_diff   "$@" ;;
  ""|-h|--help)
    cat <<EOF
Usage:
  scripts/bench.sh run [run-name] [--timeout N] [--solvers a,b] [--bms glob]
  scripts/bench.sh report [run-name]          # default: latest
  scripts/bench.sh list
  scripts/bench.sh diff <run-a> <run-b>

Examples:
  scripts/bench.sh run                        # new run, auto-named
  scripts/bench.sh run my-experiment          # named run
  scripts/bench.sh run quick --solvers pattern --bms 'public-*'
  scripts/bench.sh report                     # table from latest run
  scripts/bench.sh report 2026-04-20_1430_ffa3f27
  scripts/bench.sh diff baseline my-experiment
EOF
    ;;
  *) echo "Unknown subcommand: $cmd" >&2; exit 2 ;;
esac
