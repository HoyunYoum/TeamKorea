// dag.h — Directed-acyclic-graph utilities over a Problem.
//
// A `DAG` is a lightweight view over mlsys::Problem: topological order,
// tensor producer/consumer tables, and op-successor adjacency. Used by all
// solvers (legacy and pattern).
//
// Extracted from solver_common.cc. Previously inlined there but big enough
// to deserve its own module.

#ifndef DAG_H_
#define DAG_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mlsys.h"

namespace solver {

// ── DAG ─────────────────────────────────────────────────────────────────────

struct DAG {
  size_t num_ops, num_tensors;
  std::vector<int> tensor_producer;  // -1 = graph input
  std::vector<std::vector<size_t>> tensor_consumers;
  std::vector<std::vector<size_t>> op_successors;
  std::vector<size_t> topo_order;
};

// Kahn-style BFS topological sort. Deterministic given a problem.
DAG BuildDAG(const mlsys::Problem& p);

// Variants that produce the same adjacency but different topo orders:
DAG BuildDAG_DFS(const mlsys::Problem& p);                       // DFS post-order (chains contiguous)
DAG BuildDAG_Random(const mlsys::Problem& p, uint32_t seed);     // randomized
DAG BuildDAG_MaxOutput(const mlsys::Problem& p);                 // largest output first

// True iff `ops` is a convex subset: no op X ∉ ops lies on any path between
// two ops in `ops`. Non-convex subsets create unavoidable subgraph-level
// cycles when the intermediate op lives in a different subgraph.
bool IsDAGConvex(const DAG& dag, const std::vector<size_t>& ops);

// ── Graph Decomposition ─────────────────────────────────────────────────────

struct Component {
  std::vector<size_t> ops;          // op indices in this component
  std::vector<size_t> cut_tensors;  // boundary tensors (loaded from slow memory)
};

// Split the DAG into independent subproblems at narrow points (few live
// tensors). Each component can be solved independently; the splits reduce
// effective N for per-component max-size search.
std::vector<Component> DecomposeDAG(const mlsys::Problem& p, const DAG& dag);

}  // namespace solver

#endif  // DAG_H_
