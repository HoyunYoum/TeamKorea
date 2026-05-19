cmake_minimum_required(VERSION 3.16)
project(mlsys LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Per organizer clarification (issue #72): grading host is Ubuntu 22.04 x86_64
# with the x86-64-v3 baseline (AVX2 / BMI2 / FMA3). Pin here so HiGHS and
# abseil (built via FetchContent below) also pick up the ISA.
add_compile_options(-march=x86-64-v3)

# ---------- Build mode ----------
# Static libs by default so the `mlsys` submission binary has no runtime
# dynamic dependency on absl/HiGHS/nlohmann_json. Competition machines
# are fresh Ubuntu 22.04 LTS with no guarantee of these libs installed.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Global: prefer .a over .so when both exist (belt-and-suspenders).
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so")

# ---------- Dependencies (all FetchContent for self-contained builds) ----------

include(FetchContent)

# Abseil — pulled in-tree so the source bundle doesn't depend on a
# system absl install. Compiles as STATIC via BUILD_SHARED_LIBS=OFF.
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG        20240722.0
)
FetchContent_MakeAvailable(absl)

# nlohmann/json — header-only, fetch a stable release tarball.
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# HiGHS — needed by pattern_solver's MIP set-cover.
# Force static + disable shared so we don't get libhighs.so in the binary's
# runtime dependency set.
set(BUILD_CXX_LIBRARY OFF CACHE BOOL "" FORCE)
set(HIGHS_BUILD_CXX ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  highs
  GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
  GIT_TAG        v1.8.1
)
FetchContent_MakeAvailable(highs)

# ---------- Evaluator (mlsys_lib) ----------
# Authoritative problem types + Evaluate function (mlsys::Evaluate).

add_library(mlsys_lib evaluator/mlsys.cc)
target_include_directories(mlsys_lib PUBLIC
  ${CMAKE_SOURCE_DIR}/evaluator
)
target_link_libraries(mlsys_lib PUBLIC
  absl::status
  absl::statusor
  absl::strings
  nlohmann_json::nlohmann_json
)

# ---------- Core solver library ----------

add_library(solver_common
  core/solver_common.cc
  core/dag.cc
  core/io_util.cc
  core/preprocess.cc
  core/tensor_roles.cc
  core/cost.cc
  core/granularity.cc
  core/partition_algo.cc
  core/solution_build.cc)
target_include_directories(solver_common PUBLIC ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(solver_common PUBLIC mlsys_lib)

# ---------- Primary solver ----------
# §8 pattern-library + HiGHS MIP set-cover. Dominant on 53/53 BMs.

add_library(pattern_lib
  solvers/pattern/pattern_enum.cc
  solvers/pattern/mip_pack.cc
  solvers/pattern/retention_pass.cc
  solvers/pattern/local_search.cc)
target_include_directories(pattern_lib PUBLIC
  ${CMAKE_SOURCE_DIR}/solvers/pattern
)
target_link_libraries(pattern_lib PUBLIC solver_common highs::highs)

# Submission binary (Track A requires a binary named `mlsys`).
# Same source as pattern_solver — kept as a separate target so internal
# scripts (bench.sh) can still reference `pattern_solver` without breakage.
add_executable(mlsys solvers/pattern/pattern_solver.cc)
target_link_libraries(mlsys PRIVATE pattern_lib)
# Static libgcc/libstdc++: binary runs on any modern Linux without
# caring about the host's libstdc++ version.
target_link_options(mlsys PRIVATE
  -static-libgcc
  -static-libstdc++
  # pthread needs a group so the static resolver doesn't drop symbols.
  -Wl,--no-as-needed
  # Strip debug + symbol table from the submission binary
  # (developer tooling still has pattern_solver target unstripped).
  $<$<CONFIG:Release>:-s>
)

# Same source, shared-name alias for scripts/bench.sh.
add_executable(pattern_solver solvers/pattern/pattern_solver.cc)
target_link_libraries(pattern_solver PRIVATE pattern_lib)

# ---------- Legacy solvers (behind a flag) ----------

option(BUILD_LEGACY_SOLVERS "Build legacy io/partition/unified solvers" OFF)
if(BUILD_LEGACY_SOLVERS)
  add_executable(io_solver solvers/legacy/io_solver.cc)
  target_link_libraries(io_solver PRIVATE solver_common)

  add_library(partition_lib
    solvers/legacy/partition_evaluator.cc
    solvers/legacy/partition_search.cc)
  target_include_directories(partition_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/solvers/legacy
  )
  target_link_libraries(partition_lib PUBLIC solver_common)

  add_executable(partition_solver solvers/legacy/partition_solver.cc)
  target_link_libraries(partition_solver PRIVATE partition_lib)

  add_executable(unified_solver solvers/legacy/unified_solver.cc)
  target_link_libraries(unified_solver PRIVATE solver_common highs::highs)
endif()

# ---------- Tools ----------

add_executable(eval_cli tools/eval_cli.cc)
target_link_libraries(eval_cli PRIVATE mlsys_lib)

add_executable(lower_bound tools/lower_bound.cc)
target_link_libraries(lower_bound PRIVATE mlsys_lib)

add_executable(baseline_solver tools/baseline_solver.cc)
target_link_libraries(baseline_solver PRIVATE mlsys_lib)

add_executable(cost_diag tools/cost_diag.cc)
target_link_libraries(cost_diag PRIVATE solver_common)

add_executable(enum_cost_diag tools/enum_cost_diag.cc)
target_link_libraries(enum_cost_diag PRIVATE pattern_lib)

# ---------- Tests ----------

add_executable(eval_test tests/eval_test.cc)
target_link_libraries(eval_test PRIVATE mlsys_lib)

add_executable(eval_corner_test tests/eval_corner_test.cc)
target_link_libraries(eval_corner_test PRIVATE mlsys_lib)
