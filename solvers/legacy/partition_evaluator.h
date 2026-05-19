// partition_evaluator.h — Layer 2 primitives for partition-driven solver.
//
// Milestone 0: OptimalGranularity only. Future milestones add the full
// partition → cost pipeline (ordering, retention propagation, recompute).
//
// Design reference: PARTITION_SOLVER_PLAN.md (repo root).
//
// This header is standalone from the existing unified/two_phase/io solvers.
// It reuses primitives from solver_common (Classify, BuildRoles, ComputeTileWS,
// AnalyticalCost, OutDims, KCandidates) without modifying them.

#ifndef PARTITION_EVALUATOR_H_
#define PARTITION_EVALUATOR_H_

#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#include "mlsys.h"
#include "solver_common.h"

namespace partition {

// Result of OptimalGranularity: the best (w, h, k, snake) under the given
// retention assumptions, with its exact analytical cost and working set.
struct OptimalGranResult {
  mlsys::Granularity gran;                             // (width, height, depth=k)
  int snake_mode = 0;                                  // 0=row, 1=col, 2=raster
  double cost = std::numeric_limits<double>::infinity();  // exact cost via solver::AnalyticalCost
  int64_t ws = 0;                                      // working set size at the chosen config
  bool feasible = false;                               // true iff any (w,h,k) fit in fast_memory_capacity
};

// OptimalGranularity — exact granularity search via ceiling-step candidate
// enumeration.
//
// Covers all tile-count transition points ⌈out_dim / T⌉ for T=1..out_dim,
// plus divisors and powers of 2 as diversity. Within a bucket (fixed #tiles),
// cost is monotonic in (w, h), so the smallest (w, h) per bucket is optimal
// or tied-for-optimal. The enumeration therefore covers all possible optima
// with a sparser set than ExhaustiveGranularity (which densely iterates every
// integer 1..natW × 1..natH).
//
// Parameters:
//   p, ops       — subgraph to evaluate
//   retained_in  — set of tensors assumed pre-loaded from a previous subgraph
//                  (they skip first-load IO and occupy persistent WS)
//   retained_in_size — Σ(W_i × H_i) over retained_in, in elements
//   retained_out — set of output tensors assumed to be kept alive for a later
//                  subgraph (they skip eviction IO and occupy persistent WS
//                  instead of per-tile slots)
//   retained_out_size — Σ(W_i × H_i) over retained_out, in elements
//   dag          — optional DAG for tensor classification consistency
//
// Milestone 0 retained_out semantics (matches BestGranularity's bool
// assume_retain_out exactly — all-or-nothing for eviction cost):
//   * retained_out empty → assume_retain_out=false (every output evicts per
//     tile). WS unchanged vs retained_out=∅.
//   * retained_out non-empty → assume_retain_out=true (no eviction IO). WS
//     gains retained_out_size (persistent tensor cost).
// Future milestones may refine to per-tensor eviction control.
OptimalGranResult OptimalGranularity(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const std::unordered_set<size_t>& retained_in,
    int64_t retained_in_size,
    const std::unordered_set<size_t>& retained_out,
    int64_t retained_out_size,
    const solver::DAG* dag = nullptr);

// Diagnostics: return the ceiling-step candidate set for a dimension, for
// testing and profiling.  cap is the native-granularity cap.
std::vector<int64_t> CeilStepCandidates(int64_t out_dim, int64_t cap);

// ── OptimalGranularity cache (Milestone 4b) ──────────────────────────────────
//
// Per-run cache keyed on (SubgraphSig, retained_in_hash, retained_out_hash).
// Thread-unsafe. Caller typically constructs one per SA run (or pool
// evaluation batch) and reuses across many EvaluatePartition calls.
struct OptimalGranCache {
  struct Key {
    std::string sig;        // solver::SubgraphSig
    std::size_t ri_hash;    // XOR of std::hash<size_t>(tid) for retained_in
    std::size_t ro_hash;    // XOR of std::hash<size_t>(tid) for retained_out
    bool operator==(const Key& o) const {
      return ri_hash == o.ri_hash && ro_hash == o.ro_hash && sig == o.sig;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return std::hash<std::string>()(k.sig) ^ (k.ri_hash << 1)
             ^ (k.ro_hash << 3);
    }
  };
  std::unordered_map<Key, OptimalGranResult, KeyHash> table;
  int64_t hits = 0;
  int64_t misses = 0;

  void Clear() { table.clear(); hits = 0; misses = 0; }
  double hit_rate() const {
    int64_t total = hits + misses;
    return total > 0 ? (double)hits / total : 0.0;
  }
};

// Cached variant — returns cache hit or falls through to OptimalGranularity.
OptimalGranResult OptimalGranularityCached(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const std::unordered_set<size_t>& retained_in,
    int64_t retained_in_size,
    const std::unordered_set<size_t>& retained_out,
    int64_t retained_out_size,
    const solver::DAG* dag,
    OptimalGranCache* cache);

// ── Phase 2: Partition Evaluator ─────────────────────────────────────────────

// One step of the chosen execution ordering, with retention decisions.
// Populated by EvaluatePartition when feasible.
struct OrderedStep {
  int orig_index = -1;                       // index in input Partition.subgraphs
  std::unordered_set<size_t> retained_in;    // received from previous step
  std::unordered_set<size_t> retained_out;   // passed to next step
  OptimalGranResult gran;                    // with retained_in + retained_out
  double final_cost = std::numeric_limits<double>::infinity();
};

// Result of EvaluatePartition.
//
// Milestone 1: cost = Σ OptimalGranularity(sg, ∅, ∅).cost, ordering empty.
// Milestone 3+: ordering populated with greedy (or HK) retention-aware
//   sweep; cost reflects retention savings.
struct Evaluation {
  double cost = std::numeric_limits<double>::infinity();  // total cost = Σ OrderedStep.final_cost
  bool feasible = false;                                   // true iff every step admits a valid gran

  // Diagnostics — size = #subgraphs in input Partition (not execution order).
  std::vector<double> per_subgraph_cost;        // base cost per input subgraph
  std::vector<OptimalGranResult> per_subgraph_gran;  // base gran per input subgraph

  // Execution-order sequence with retention decisions (size = #subgraphs).
  std::vector<OrderedStep> ordering;

  // Partition-level validation flags.
  bool complete_cover = false;        // every op covered exactly once
  bool all_convex = false;            // every subgraph is DAG-convex
  int first_infeasible_sg = -1;       // index of first WS-infeasible subgraph
};

// Evaluate a partition: Milestone 1 semantics.
//   feasibility = (complete_cover) ∧ (all_convex) ∧ (all subgraphs feasible)
//   cost        = Σ_s OptimalGranularity(ops(s), retained_in=∅, retained_out=∅).cost
//
// Does NOT yet choose an ordering, propagate retention, or apply recomputation.
// Those are added in Milestone 3 / 5.
//
// The returned Evaluation always populates per_subgraph_cost / per_subgraph_gran
// up to the first infeasibility (and fills the rest with kInf). Callers who
// want a guaranteed number must check Evaluation.feasible.
Evaluation EvaluatePartition(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const solver::Partition& part);

// Cached / fast variant:
//   cache             — optional; pass nullptr for uncached behavior
//   compute_baseline  — if false, skip per-subgraph no-retention diagnostic
//                       (saves S × OptimalGranularity on large partitions).
//                       per_subgraph_cost / per_subgraph_gran arrays are
//                       left default-constructed when skipped.
Evaluation EvaluatePartitionCached(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const solver::Partition& part,
    OptimalGranCache* cache,
    bool compute_baseline = false);

// Convenience: return `input` permuted so subgraph[i] is the i-th step of
// the execution ordering chosen by EvaluatePartition. Required before
// passing the partition to solver::BuildSolution so retention decisions
// are respected (BuildSolution walks subgraphs in list order).
// If eval.ordering is empty (e.g., infeasible), returns input unchanged.
solver::Partition ApplyOrderingToPartition(
    const solver::Partition& input,
    const Evaluation& eval);

// ── Milestone 5: Recomputation post-pass ─────────────────────────────────────

// Try adding recomputable producers to subgraphs via solver::RecomputationPass,
// then re-evaluate. Returns (new_partition, new_eval) when the rewrite strictly
// improves cost; otherwise leaves the input unchanged and returns it.
struct RecompResult {
  solver::Partition partition;   // possibly-rewritten partition
  Evaluation evaluation;         // evaluation of `partition`
  bool applied = false;          // true iff rewrite was kept
};

RecompResult ApplyRecomputation(const mlsys::Problem& p,
                                 const solver::DAG& dag,
                                 const solver::Partition& part,
                                 const Evaluation& baseline,
                                 OptimalGranCache* cache);

// ── Phase 1-A: Initial Pool ──────────────────────────────────────────────────

// One evaluated candidate in the initial pool.
struct PoolEntry {
  std::string label;             // human-readable source description
  solver::Partition partition;
  Evaluation evaluation;
};

// Generate the Phase 1-A initial pool by calling each existing generator,
// then running EvaluatePartition on the result. Returns entries sorted by
// cost ascending; infeasible entries go last.
//
// Generators called (up to 17 candidates, depending on which produce
// feasible partitions):
//   - Singleton (trivial baseline)
//   - DPPartition × {BFS, DFS, MaxOutput, Random(seeds 1,2,3)} × {retain F, T}
//     (12 candidates)
//   - AgglomerativePartition (1)
//   - MaxFusionPartition(sink_first=true, false) (2)
//   - AntiChainPartition (1)
//
// ILP is deliberately excluded per the plan (empirical: ILP never strictly
// wins on the 5 released BMs; DP family dominates).
//
// rng_seeds optionally overrides the 3 seeds used for Random topo DAGs.
// time_budget_s caps total Phase 1-A wall time — if exceeded, remaining
// generators are skipped. 0 means no cap.
std::vector<PoolEntry> GenerateInitialPool(
    const mlsys::Problem& p,
    const solver::DAG& dag_default,
    const std::vector<uint32_t>& rng_seeds = {1, 2, 3},
    double time_budget_s = 0.0);

}  // namespace partition

#endif  // PARTITION_EVALUATOR_H_
