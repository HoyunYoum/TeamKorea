#!/usr/bin/env bash
# pack_submission.sh — assemble a Track A submission zip per README spec.
#
# Output layout (per README):
#   <TeamName>_TrackA_<N>.zip
#     mlsys                   — statically linked binary
#     source/                 — complete source tree, buildable standalone
#       CMakeLists.txt
#       evaluator/ core/ solvers/ tools/ tests/
#       benchmarks/           — the 5 public BMs (verifiers can reproduce)
#       README_BUILD.md       — how-to-rebuild instructions
#       (existing README.md, LICENSE, CONTRIBUTING.md, spec docs)
#     writeup.pdf             — must be placed by hand before running this
#                               script, or script warns and leaves a stub
#
# Usage:
#   scripts/pack_submission.sh <TeamName> [submission_number]
#
# Example:
#   scripts/pack_submission.sh TeamKorea 1
#   → TeamKorea_TrackA_1.zip

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <TeamName> [submission_number]" >&2
  exit 2
fi

TEAM="$1"
SUBN="${2:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ZIP_NAME="${TEAM}_TrackA_${SUBN}.zip"
STAGE="$(mktemp -d -t mlsys-submit-XXXXXX)"
trap "rm -rf '$STAGE'" EXIT

echo "▶ Staging in: $STAGE"
mkdir -p "$STAGE/source"

# ── 1. Build the binary from a clean build dir ─────────────────────────
echo "▶ Building mlsys (Release, static)..."
BUILD_DIR="$ROOT/build-submit"
rm -rf "$BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  > "$STAGE/build.log" 2>&1 || { cat "$STAGE/build.log"; exit 1; }
cmake --build "$BUILD_DIR" --target mlsys \
  >> "$STAGE/build.log" 2>&1 || { cat "$STAGE/build.log"; exit 1; }

if [ ! -x "$BUILD_DIR/mlsys" ]; then
  echo "✗ Build failed — no mlsys binary" >&2
  exit 1
fi

# Verify ldd shows only libc/libm (no absl/highs/stdc++)
LDD_OUT="$(ldd "$BUILD_DIR/mlsys" 2>&1)"
if echo "$LDD_OUT" | grep -qE "libabsl|libhighs|libstdc\+\+|libgcc_s"; then
  echo "✗ Binary has non-libc dynamic deps — not properly static" >&2
  echo "$LDD_OUT" >&2
  exit 1
fi

BINARY_SIZE=$(du -h "$BUILD_DIR/mlsys" | cut -f1)
echo "  ✓ mlsys built (${BINARY_SIZE}, static except for libc/libm)"

# Smoke-test: run on the smallest public BM and check output file exists.
echo "▶ Smoke test on mlsys-2026-1..."
TMP_OUT="$STAGE/_smoke.json"
"$BUILD_DIR/mlsys" "$ROOT/benchmarks/mlsys-2026-1.json" "$TMP_OUT" \
  > "$STAGE/smoke.log" 2>&1 || { cat "$STAGE/smoke.log"; exit 1; }
if [ ! -f "$TMP_OUT" ] || [ ! -s "$TMP_OUT" ]; then
  echo "✗ Smoke test: no output or empty file produced" >&2
  exit 1
fi
# Parse the stderr for the final-GT line — mlsys prints it.
GT=$(grep -oP 'final on-disk GT=\K[0-9]+' "$STAGE/smoke.log" | tail -1)
echo "  ✓ Smoke test passed (final GT=${GT:-?} on BM-1)"
rm -f "$TMP_OUT"

cp "$BUILD_DIR/mlsys" "$STAGE/mlsys"

# ── 2. Stage the source tree (include only what's needed) ──────────────
echo "▶ Staging source tree..."

# Apache 2.0 LICENSE from upstream — required by the license for
# redistribution of derivative works. Not a "doc" in the editorial
# sense; this is legal compliance.
cp "$ROOT/LICENSE" "$STAGE/source/LICENSE" 2>/dev/null || \
    echo "  ⚠ LICENSE missing at repo root — Apache 2.0 compliance at risk"

# Non-conditional directories (full copy).
for d in evaluator core benchmarks; do
  cp -r "$ROOT/$d" "$STAGE/source/"
done

# solvers/: pattern/ only (legacy/ is dev-only and not part of the submission).
mkdir -p "$STAGE/source/solvers"
cp -r "$ROOT/solvers/pattern" "$STAGE/source/solvers/"

# tools/: only eval_cli for reviewer verification.
# baseline_solver / cost_diag / lower_bound are dev diagnostics.
mkdir -p "$STAGE/source/tools"
cp "$ROOT/tools/eval_cli.cc" "$STAGE/source/tools/"

# tests/: eval_test + eval_corner_test + examples (reviewer validation).
mkdir -p "$STAGE/source/tests"
cp "$ROOT/tests/eval_test.cc" "$ROOT/tests/eval_corner_test.cc" \
   "$STAGE/source/tests/"
cp -r "$ROOT/tests/examples" "$STAGE/source/tests/"

# Generate a slim CMakeLists.txt for the submission — only targets that
# use the included sources (mlsys, eval_cli, eval_test, eval_corner_test).
# The repo's full CMakeLists (referencing legacy/dev tools) is
# omitted because those source files aren't bundled.
cat > "$STAGE/source/CMakeLists.txt" << 'CMAKE_EOF'
cmake_minimum_required(VERSION 3.16)
project(mlsys LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Grading host ISA baseline (organizer clarification, issue #72):
# Ubuntu 22.04 x86_64, x86-64-v3 (AVX2 / BMI2 / FMA3).
add_compile_options(-march=x86-64-v3)

# Static by default — `mlsys` has no runtime dep on absl/HiGHS/json.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so")

include(FetchContent)

# --- Abseil ---
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL    OFF CACHE BOOL "" FORCE)
FetchContent_Declare(absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG        20240722.0)
FetchContent_MakeAvailable(absl)

# --- nlohmann/json (header-only) ---
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# --- HiGHS (MIP solver) ---
set(HIGHS_BUILD_CXX ON CACHE BOOL "" FORCE)
FetchContent_Declare(highs
  GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
  GIT_TAG        v1.8.1)
FetchContent_MakeAvailable(highs)

# --- Evaluator library ---
add_library(mlsys_lib evaluator/mlsys.cc)
target_include_directories(mlsys_lib PUBLIC ${CMAKE_SOURCE_DIR}/evaluator)
target_link_libraries(mlsys_lib PUBLIC
  absl::status absl::statusor absl::strings
  nlohmann_json::nlohmann_json)

# --- Shared solver core ---
add_library(solver_common
  core/solver_common.cc core/dag.cc core/io_util.cc core/preprocess.cc
  core/tensor_roles.cc core/cost.cc core/granularity.cc
  core/partition_algo.cc core/solution_build.cc)
target_include_directories(solver_common PUBLIC ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(solver_common PUBLIC mlsys_lib)

# --- Pattern solver library ---
add_library(pattern_lib
  solvers/pattern/pattern_enum.cc
  solvers/pattern/mip_pack.cc
  solvers/pattern/retention_pass.cc)
target_include_directories(pattern_lib PUBLIC ${CMAKE_SOURCE_DIR}/solvers/pattern)
target_link_libraries(pattern_lib PUBLIC solver_common highs::highs)

# --- Submission binary (Track A deliverable) ---
add_executable(mlsys solvers/pattern/pattern_solver.cc)
target_link_libraries(mlsys PRIVATE pattern_lib)
target_link_options(mlsys PRIVATE
  -static-libgcc -static-libstdc++
  -Wl,--no-as-needed
  $<$<CONFIG:Release>:-s>)

# --- Reviewer tools ---
add_executable(eval_cli tools/eval_cli.cc)
target_link_libraries(eval_cli PRIVATE mlsys_lib)

add_executable(eval_test tests/eval_test.cc)
target_link_libraries(eval_test PRIVATE mlsys_lib)

add_executable(eval_corner_test tests/eval_corner_test.cc)
target_link_libraries(eval_corner_test PRIVATE mlsys_lib)
CMAKE_EOF

# A build/run README for the reviewer.
cat > "$STAGE/source/README_BUILD.md" << 'BUILD_EOF'
# Building `mlsys` from this source bundle

## Requirements (all standard Ubuntu 22.04 packages)
- `cmake` ≥ 3.16
- `ninja-build`
- `gcc` / `g++` with C++20 support (gcc 11+)
- `git` (CMake's FetchContent pulls absl, HiGHS, nlohmann/json)
- ~2 GB free disk space (dependencies build in-tree)

## Build
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mlsys
```

First build takes ~5 minutes (downloads + compiles abseil-cpp + HiGHS).
Subsequent builds finish in seconds.

## Run
```bash
./build/mlsys <problem.json> <solution.json>
```

The binary handles its own internal timeout via `CompetitionTimeout()` in
`core/io_util.cc`, derived from benchmark number (1–24) or op count.
An external SIGTERM is honored — the latest valid solution has already
been atomically written to the output path at that point.

## Verify (optional)
```bash
cmake --build build --target eval_test eval_corner_test
./build/eval_test           # 12/12 — evaluator invariants
./build/eval_corner_test    # 31/31 — corner cases (retention rules etc.)

./build/mlsys benchmarks/mlsys-2026-17.json /tmp/out17.json
./build/eval_cli benchmarks/mlsys-2026-17.json /tmp/out17.json
# Should print 4966400 (= LB-optimal on BM-17).
```

Approach details: see writeup.pdf.
BUILD_EOF

# ── 3. Pull in writeup.pdf if it sits at repo root; else warn ──────────
if [ -f "$ROOT/writeup.pdf" ]; then
  cp "$ROOT/writeup.pdf" "$STAGE/writeup.pdf"
  echo "  ✓ writeup.pdf bundled"
else
  echo "  ⚠ writeup.pdf NOT FOUND at $ROOT/writeup.pdf"
  echo "    Place it in the repo root, then re-run this script."
  echo "    Proceeding to assemble zip WITHOUT writeup (for dry-run purposes)."
fi

# ── 4. Zip it up ───────────────────────────────────────────────────────
OUT="$ROOT/$ZIP_NAME"
rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" mlsys source writeup.pdf 2>/dev/null || \
    zip -qr "$OUT" mlsys source)

SIZE=$(du -h "$OUT" | cut -f1)
echo
echo "✓ Submission zip: $OUT ($SIZE)"
echo
echo "Contents:"
unzip -l "$OUT" | tail -n +4 | head -n -2 | awk '{print "  " $NF}' | \
    grep -v '^  $' | head -40
TOTAL=$(unzip -l "$OUT" | tail -1 | awk '{print $2}')
echo "  (total files: $(unzip -l "$OUT" | tail -n +4 | head -n -2 | wc -l), $TOTAL bytes)"
echo
echo "Before submitting:"
echo "  1. Verify writeup.pdf is bundled:  unzip -l $ZIP_NAME | grep writeup"
echo "  2. Re-check ldd:                   unzip -p $ZIP_NAME mlsys > /tmp/m && chmod +x /tmp/m && ldd /tmp/m"
echo "  3. Smoke-test in a fresh dir:      cd /tmp && unzip $ZIP_NAME && ./mlsys ..."
