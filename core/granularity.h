// granularity.h — Best/exhaustive granularity search for a given subgraph.
//
// BestGranularity: AnalyticalCost-based sweep over (w, h, k) × snake modes.
// ExhaustiveGranularity: top-K by ComputeTileWS, then ExactCost ranking.
//
// Extracted from solver_common.cc (Phase 3 refactor).

#ifndef GRANULARITY_H_
#define GRANULARITY_H_

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "cost.h"
#include "dag.h"
#include "mlsys.h"

namespace solver {

struct GranConfig {
  mlsys::Granularity gran;
  std::vector<int64_t> traversal;
  double cost;
};

// Standard granularity search (io_solver style). Iterates (w, h, k)
// candidates × snake modes; picks min AnalyticalCost. Includes fusion-
// validity guards (uniform-K, split-K feasibility) — rejects invalid grans.
//
// extra_wh_slots: additional w*h slots in WS (e.g. pass-through tensors).
GranConfig BestGranularity(const mlsys::Problem& p,
                           const std::vector<size_t>& ops,
                           const std::unordered_set<size_t>& retained_in,
                           int64_t retained_size,
                           const DAG* dag = nullptr,
                           bool assume_retain_out = false,
                           int64_t extra_wh_slots = 0);

struct BoundResult {
  mlsys::Granularity gran;
  int snake_mode;
  double exact_cost;
  int64_t ws;
  int64_t tiles_w, tiles_h, k_steps;
};

// Dense granularity search (optimal_solver style). Screens top-200 by
// ComputeTileWS, then ranks by ExactCost. More expensive than BestGranularity
// but catches AnalyticalCost ↔ Evaluate divergences.
BoundResult ExhaustiveGranularity(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const std::unordered_set<size_t>& retained_in,
    int64_t retained_size,
    const DAG* dag = nullptr,
    int64_t extra_wh_slots = 0,
    bool assume_retain_out = true);

}  // namespace solver

#endif  // GRANULARITY_H_
