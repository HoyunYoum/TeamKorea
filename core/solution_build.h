// solution_build.h — Build a Solution from a Partition + diagnostics printer.
//
// BuildSolution runs BestGranularity per subgraph + pairwise retention
// search. BuildSolutionExhaustive uses ExhaustiveGranularity. Both share
// post-partition structure.
//
// Extracted from solver_common.cc (Phase 3 refactor).

#ifndef SOLUTION_BUILD_H_
#define SOLUTION_BUILD_H_

#include "dag.h"
#include "mlsys.h"
#include "partition_algo.h"

namespace solver {

// Build solution with BestGranularity (AnalyticalCost) + pairwise retention.
// Legacy solvers' default path; scales O(|retention_candidates|^3) per sg
// (pattern_solver uses a size-tiered approach — minimal path for large N).
mlsys::Solution BuildSolution(const mlsys::Problem& p, const DAG& dag,
                               Partition& part);

// Build solution with ExhaustiveGranularity (ExactCost-based).
mlsys::Solution BuildSolutionExhaustive(const mlsys::Problem& p,
                                         const DAG& dag,
                                         Partition& part);

// Per-subgraph diagnostics to stderr (op count, MM/PW mix, granularity,
// tile count, snake mode, latency, retain count).
void PrintDiagnostics(const mlsys::Problem& p, const mlsys::Solution& sol);

}  // namespace solver

#endif  // SOLUTION_BUILD_H_
