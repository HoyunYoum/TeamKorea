// mip_pack.h — HiGHS-backed MIP set-cover pack for pattern_solver (S1).
//
// Given a list of feasible columns (each covering a subset of ops with an
// associated analytical cost), find the min-cost exact cover using HiGHS:
//
//   min Σ cost[j] · x[j]
//   s.t. Σ_{j : op i ∈ cols[j]} x[j] = 1,  ∀ op i
//        x[j] ∈ {0, 1}
//
// Falls back to greedy pack on solver failure or timeout.

#ifndef MIP_PACK_H_
#define MIP_PACK_H_

#include <cstddef>
#include <vector>

#include "pattern_enum.h"
#include "solver_common.h"

namespace pattern {

struct MipResult {
  solver::Partition partition;
  double objective = 0.0;   // total analytical cost of selected columns
  bool is_optimal = false;  // true iff HiGHS reported kOptimal
  int n_selected = 0;
  double solve_time_s = 0.0;
};

// Column cost variants that the MIP can minimize:
//   E = cost           (ExactCost, Evaluate-faithful, retain_out=false)
//   A = analytic_cost  (AnalyticalCost, O(1) formula matched to Evaluate)
//   R = cost_retain    (faithful cost − α-weighted expected retention saving)
// Different variants rank partitions differently; the pattern_solver driver
// runs all three and picks the min-GT solution after the full pipeline.
enum class CostVariant { E = 0, A = 1, R = 2 };

MipResult MipPack(const std::vector<Column>& columns,
                  size_t num_ops,
                  double time_limit_s,
                  CostVariant variant = CostVariant::E);

}  // namespace pattern

#endif  // MIP_PACK_H_
