// partition_search.h — Layer 3 search strategies (Milestone 4a SA framework).
//
// Implements partition space exploration via simulated annealing, using
// EvaluatePartition (Layer 2) as the single-source-of-truth cost oracle.
//
// Design reference: PARTITION_SOLVER_PLAN.md Phase 1-B.
//
// Milestone 4a scope:
//   - SAPartition data structure (label + members + boundary)
//   - Move / Merge / Split neighborhood with constraint validation
//   - Full SA loop (acceptance, cooling, early termination)
//   - Multi-seed driver with budget reallocation
//   - Diagnostics (move acceptance, cost trajectory)
//
// Deferred to M4b:
//   - Delta evaluation with incremental XOR-hash caching

#ifndef PARTITION_SEARCH_H_
#define PARTITION_SEARCH_H_

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "mlsys.h"
#include "partition_evaluator.h"
#include "solver_common.h"

namespace partition {

// SA-friendly partition representation with incremental update support.
struct SAPartition {
  std::vector<int> label;                                  // label[op] = subgraph id
  std::vector<std::unordered_set<size_t>> members;         // members[sg] = ops
  // Boundary set: ops that have at least one DAG edge crossing their subgraph.
  // Boundary[sg] is a candidate pool for Move operations.
  std::vector<std::unordered_set<size_t>> boundary;
  // DAG is directed: only op_successors is provided by solver::DAG. Cache
  // reverse adjacency (predecessors) per op for incremental boundary updates.
  std::vector<std::vector<size_t>> op_predecessors;

  // Build from a solver::Partition. Re-indexes to remove empty subgraphs.
  static SAPartition FromPartition(const solver::Partition& part,
                                    const solver::DAG& dag);
  solver::Partition ToPartition() const;

  int num_subgraphs() const { return (int)members.size(); }
  int num_ops() const { return (int)label.size(); }

  // Basic moves (return false if move violates constraints — no mutation).
  // Move op v from its current subgraph to `target_sg`. Rebuilds only the
  // affected boundary sets.
  bool Move(size_t v, int target_sg, const solver::DAG& dag);
  // Merge subgraph b into a (b becomes empty, then removed).
  bool Merge(int a, int b, const solver::DAG& dag);
  // Split subgraph `s` into (side_a, side_b) where side_b becomes a new sg.
  // Caller provides the partition of members[s] as two op sets.
  bool Split(int s, const std::unordered_set<size_t>& side_a,
             const std::unordered_set<size_t>& side_b, const solver::DAG& dag);

 private:
  void RebuildBoundary(int sg, const solver::DAG& dag);
  void RebuildBoundaryFor(size_t op, const solver::DAG& dag);
};

// Configuration for a single SA run.
struct SAConfig {
  double time_budget_s = 5.0;             // wall-clock budget for this seed
  double initial_temp = 0.0;              // 0 → auto (stddev × 2)
  double final_temp_ratio = 0.01;         // T_final = T_initial × ratio
  int max_iterations = 50000;             // hard cap on iterations
  int no_improvement_patience = 1000;     // early stop after N iters without best
  uint32_t rng_seed = 42;

  // Move-type weights (sampled proportionally). Auto-adjusted by avg_sg_size.
  double p_move = 0.4;
  double p_merge = 0.3;
  double p_split = 0.3;

  // WS-guided split fraction (vs random split).
  double ws_split_frac = 0.7;
};

// Result of one SA run.
struct SAResult {
  solver::Partition best_partition;
  Evaluation best_eval;

  // Diagnostics.
  int iterations_done = 0;
  int move_attempts[3] = {0, 0, 0};   // [move, merge, split]
  int move_constraint_rejects[3] = {0, 0, 0};
  int move_cost_rejects[3] = {0, 0, 0};
  int move_accepts[3] = {0, 0, 0};
  double wall_time_s = 0.0;

  bool improved_over_seed = false;
  double seed_cost = 0.0;
};

// Run SA starting from the given seed partition. Returns the best partition
// found within the time budget (or earlier if patience exhausted).
SAResult RunSA(const mlsys::Problem& p,
               const solver::DAG& dag,
               const solver::Partition& seed,
               const SAConfig& cfg);

// Multi-seed driver: run SA sequentially from each seed, reallocating leftover
// budget from early-terminated seeds to later ones. Returns the global best.
struct MultiSeedResult {
  solver::Partition best_partition;
  Evaluation best_eval;
  std::vector<SAResult> per_seed;       // per seed in input order
  double total_wall_time_s = 0.0;
};

MultiSeedResult RunMultiSeedSA(const mlsys::Problem& p,
                                const solver::DAG& dag,
                                const std::vector<solver::Partition>& seeds,
                                double total_budget_s,
                                uint32_t base_seed = 42);

}  // namespace partition

#endif  // PARTITION_SEARCH_H_
