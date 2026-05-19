// io_util.h — Pure I/O helpers + timing utilities.
//
// Minimal dependencies (mlsys.h only). Diagnostics helpers that need cost
// utilities (OutDims, etc.) stay in solver_common.

#ifndef IO_UTIL_H_
#define IO_UTIL_H_

#include <chrono>
#include <string>

#include "mlsys.h"

namespace solver {

// Write a Solution to JSON. Reported latencies are zeroed (the evaluator
// re-computes; spec treats 0 as "not provided"). Solver analytical cost is
// an approximation and should never be authoritative.
void WriteSolution(const mlsys::Solution& sol, const std::string& fname);

// Wall-clock elapsed seconds since `t0`.
double ElapsedSince(std::chrono::steady_clock::time_point t0);

// Competition timeout by op count (rough fallback when BM number unknown).
double CompetitionTimeout(int n_ops);

// Parse "mlsys-2026-<N>.json" filenames → N. Returns 0 on failure.
int ParseBenchmarkNumber(const std::string& filename);

// Competition timeout by explicit BM number (1-24). Falls back to op-count
// rule when bm_number is invalid (0).
double CompetitionTimeout(int n_ops, int bm_number);

}  // namespace solver

#endif  // IO_UTIL_H_
