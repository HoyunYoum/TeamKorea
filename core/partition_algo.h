// partition_algo.h — Partition strategies: DP, agglomerative, top-down,
// random, anti-chain, recomputation, unfusion, retention, reordering.
//
// All produce a `Partition` (list of subgraphs) from an input DAG. Shared
// cost utilities come from cost.h; role classification from tensor_roles.h.
//
// Extracted from solver_common.cc (Phase 3 refactor).

#ifndef PARTITION_ALGO_H_
#define PARTITION_ALGO_H_

#include <cstdint>
#include <string>
#include <vector>

#include "dag.h"
#include "mlsys.h"

namespace solver {

// ── Partition ────────────────────────────────────────────────────────────────

struct Partition {
  std::vector<std::vector<size_t>> subgraphs;
};

// All-singleton partition in topological order. Starting point for DP etc.
Partition InitialPartition(const DAG& dag);

// Structural signature for cost caching (same ops + roles ⇒ same cost).
std::string SubgraphSig(const mlsys::Problem& p,
                        const std::vector<size_t>& ops, const DAG& dag);

// ── DP partition ─────────────────────────────────────────────────────────────

// DP-optimal contiguous partition: sweeps topo order, memoizes best cost
// for each (range → partition) cell via SubgraphSig.
Partition DPPartition(const mlsys::Problem& p, const DAG& dag,
                      Partition part,
                      bool assume_retain_out = false);

// ── Alternative Partition Strategies ─────────────────────────────────────────

// DAG validity check: can subgraphs sg_a and sg_b be merged without creating
// a cycle in the subgraph-level DAG?
bool CanMergeSubgraphs(const DAG& dag, const Partition& part,
                       int sg_a, int sg_b);

// Bottom-up: repeatedly merge the pair with highest cost savings.
Partition AgglomerativePartition(const mlsys::Problem& p, const DAG& dag,
                                  bool assume_retain_out = false);

// Top-down: greedily pack ops into subgraphs until WS limit.
// sink_first=true processes sinks first; false processes sources first.
Partition MaxFusionPartition(const mlsys::Problem& p, const DAG& dag,
                              bool sink_first = true,
                              bool assume_retain_out = false);

// Random partition with local search (time-budgeted).
Partition RandomPartition(const mlsys::Problem& p, const DAG& dag,
                           uint32_t seed = 42, double deadline_s = 5.0,
                           bool assume_retain_out = false);

// Horizontal grouping by DAG level (anti-chain packing).
Partition AntiChainPartition(const mlsys::Problem& p, const DAG& dag,
                              bool assume_retain_out = false);

// ── Recomputation ────────────────────────────────────────────────────────────

// Duplicate ops producing widely-reused tensors into downstream subgraphs
// to eliminate slow-memory spills.
Partition RecomputationPass(const mlsys::Problem& p, const DAG& dag,
                            Partition part,
                            bool assume_retain_out = false);

// Split non-convex subgraphs into singletons to prevent subgraph-level cycles.
Partition ConvexityRepair(const DAG& dag, Partition part);

// ── Unfusion ─────────────────────────────────────────────────────────────────

// Inverse of fusion: split subgraphs if the per-sg cost exceeds the sum of
// the split costs. Useful after aggressive DP/SA to remove over-fusion.
Partition UnfusionPass(const mlsys::Problem& p, const DAG& dag,
                       Partition part,
                       bool assume_retain_out = false);

// ── Retention ────────────────────────────────────────────────────────────────

// Tensors produced in cur_ops and consumed in next_ops — candidates for
// tensors_to_retain at the cur_ops → next_ops subgraph boundary.
std::vector<size_t> RetentionCandidates(const mlsys::Problem& p,
                                        const std::vector<size_t>& cur_ops,
                                        const std::vector<size_t>& next_ops,
                                        const DAG* dag = nullptr);

// ── Subgraph Reordering ─────────────────────────────────────────────────────

// Topological reorder maximizing shared-tensor adjacency between consecutive
// subgraphs. Legacy heuristic — scoring is greedy; can interleave attention
// branches suboptimally when ties aren't broken carefully.
Partition SubgraphReorder(const mlsys::Problem& p, const DAG& dag,
                          Partition part);

// Stable topological sort of subgraphs in partition-index order (min-heap
// of ready partition indices). Preserves the input partition's ordering on
// ties — matters when the input comes from a cost-ordered column generator
// (MIP, greedy) and the analytical-vs-retention tradeoff depends on which
// subgraph is scheduled first at equal-priority points.
//
// Behaviour: each subgraph picked at topo level = smallest partition index
// among ready candidates. Verified to recover 30% on BM-13-style attention
// where SubgraphReorder's shared-tensor tie-break interleaves head branches.
//
// Returns the original partition unchanged if a subgraph-level cycle is
// detected (ConvexityRepair is the caller's responsibility).
Partition StableTopoSort(const DAG& dag, Partition part);

}  // namespace solver

#endif  // PARTITION_ALGO_H_
