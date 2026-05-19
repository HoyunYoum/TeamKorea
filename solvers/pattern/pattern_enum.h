// pattern_enum.h — §8-based column enumeration for pattern_solver.
//
// The set of valid fusion subgraphs is finite and classifiable under §8's
// reduction-axis accounting. Each column produced by this module is one
// candidate subgraph whose cost is computed via solver::BestGranularity
// (role-aware, RHS-chain symmetric).
//
// Column classes: C0 standalone MM, C1 standalone PW, C3 LHS-chain-2,
// C4 RHS-chain-2, C5 F2a single-side, C6 PW+MM at k=K, C7 3+ MM uniform-K.

#ifndef PATTERN_ENUM_H_
#define PATTERN_ENUM_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "mlsys.h"
#include "solver_common.h"

namespace pattern {

enum class PatternClass {
  StandaloneMM = 0,        // C0
  StandalonePW = 1,        // C1
  // C2 = PW chain (collapsed in preprocess; not a column class)
  LhsChain2 = 3,           // C3: (A @ B) @ C style — upstream's output feeds downstream LHS
  RhsChain2 = 4,           // C4: A @ (B @ C) style — mirror of C3
  F2aSingle = 5,           // C5: MM with 2 MM producers; fuse one side only
  PwMmKeqK = 6,            // C6: PW+MM fusion at nk=1 (full-K)
  Chain3plusUniformK = 7,  // C7: 3+ MM chain at w=k=K uniform (rare, usually OOM)
};

struct Column {
  std::vector<size_t> ops;           // op indices (topo-sorted on caller side)
  mlsys::Granularity gran{0, 0, 0};
  int snake_mode = -1;               // 0=row, 1=col, 2=raster; -1=auto (use traversal)
  std::vector<int64_t> traversal;    // optional explicit tile order
  double cost = std::numeric_limits<double>::infinity();         // ExactCost @ BestGranularity(assume_retain_out=false)
  double analytic_cost = std::numeric_limits<double>::infinity();// AnalyticalCost @ same gran (MIP legacy)
  double cost_retain = std::numeric_limits<double>::infinity();  // faithful cost minus α-weighted expected retention saving per non-ephemeral output (α ∈ [0, 1] based on external-consumer count; graph outputs / inputs get α=0). Biases MIP toward partitions whose single-hop retention RetentionPass can then confirm (multi-hop passthrough is spec-forbidden per #34/#52)
  int64_t ws = 0;
  PatternClass klass = PatternClass::StandaloneMM;
  bool feasible = false;             // true iff ws ≤ capacity && cost < inf
};

// Enumerate columns for the current stage (S0: C0 + C1 + C3).
// Returns only feasible columns (ws ≤ capacity, cost finite).
std::vector<Column> EnumerateColumns(const mlsys::Problem& p,
                                      const solver::DAG& dag);

// Greedy set-cover pack: sort columns by cost-per-op asc, take columns
// whose ops are all uncovered. Fill any uncovered ops with singleton
// fallbacks. Always produces a valid partition covering every op exactly once.
solver::Partition GreedyPack(const std::vector<Column>& columns,
                              size_t num_ops);

}  // namespace pattern

#endif  // PATTERN_ENUM_H_
