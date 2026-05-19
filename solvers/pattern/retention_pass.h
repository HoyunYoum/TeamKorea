// retention_pass.h — Single-hop retention refinement.
//
// BuildSolution chooses retentions with approximate cost models and can
// miss single-hop (adjacent sg[i] → sg[i+1]) retentions whose benefit only
// shows up under the exact simulator. This pass rescans candidates and
// admits each one when the simulator confirms an improvement.
//
// Note: true multi-hop retention (carrying t from sg[i] to sg[i+2] via
// sg[i+1] passthrough) is forbidden by the spec — #34 limits retention
// lifetime to one step, and #52 requires every tensors_to_retain entry
// to be an op output of that subgraph. Carrying further legally requires
// recomputing the producing op in the intermediate subgraph, which this
// pass does NOT do; multi-hop candidates are filtered out before trial.
//
// Procedure:
//   1. Scan all non-ephemeral tensors for producer/consumer sg metadata.
//   2. Rank candidates by savings/footprint ratio.
//   3. Filter to span==1 (producer_sg + 1 == last_consumer_sg).
//   4. For each candidate, re-optimize gran in the affected sg and
//      re-Evaluate; keep iff ExactCost strictly improves.

#ifndef RETENTION_PASS_H_
#define RETENTION_PASS_H_

#include "mlsys.h"
#include "solver_common.h"

namespace pattern {

struct RetentionResult {
  mlsys::Solution solution;
  int n_retentions_added = 0;
  double cost_before = 0.0;  // 0 = unknown / skipped
  double cost_after = 0.0;
};

// deadline_s: wall-clock budget. Retention-pass exits early if exceeded
// (partial results kept iff Evaluate improves).
RetentionResult RetentionPass(const mlsys::Problem& p,
                               const solver::DAG& dag,
                               mlsys::Solution sol,
                               double deadline_s = 30.0);

// GranularityRefinement — ExhaustiveGranularity-based per-sg refinement.
// BestGranularity uses AnalyticalCost for gran selection, which can pick
// analytically-optimal but actually-worse grans (~0.2-0.3% gap on BM-17).
// This pass runs ExhaustiveGranularity (top-K screened with ExactCost) per
// sg, respecting retained_in carried from prior sg's tensors_to_retain, and
// swaps the gran if ExactCost improves. Evaluate-validated revert on hurt.
struct RefineResult {
  mlsys::Solution solution;
  int n_refined = 0;
  double cost_before = 0.0;
  double cost_after = 0.0;
};

// deadline_s: wall-clock budget for the refinement pass. Walks subgraphs and
// bails out when exceeded (partial refinement is kept iff Evaluate improves).
RefineResult GranularityRefinement(const mlsys::Problem& p,
                                    const solver::DAG& dag,
                                    mlsys::Solution sol,
                                    double deadline_s = 30.0);

}  // namespace pattern

#endif  // RETENTION_PASS_H_
