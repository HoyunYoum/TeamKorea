// preprocess.h — Problem preprocessing passes.
//
// Dead-op elimination + PW chain collapsing. Produces a reduced Problem that
// solvers run on; UnPreprocess expands the Solution back to original op
// indices for evaluation.

#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include <vector>

#include "dag.h"
#include "mlsys.h"

namespace solver {

// Pass 1: liveness + use-count analysis (read-only).
struct TensorAnalysis {
  std::vector<int> use_count;         // consumers per tensor
  std::vector<int> last_consumer_op;  // last op (in topo order); -1 if unused
  std::vector<bool> is_graph_output;  // true if no consumers (final output)
  std::vector<bool> is_single_use;    // use_count == 1
};

TensorAnalysis AnalyzeTensors(const mlsys::Problem& p, const DAG& dag);

// Pass 2: dead-op elimination. Returns a new Problem with dead ops/tensors
// removed and indices renumbered.
mlsys::Problem EliminateDeadOps(const mlsys::Problem& p,
                                 const TensorAnalysis& analysis);

// Pass 3: PW chain collapsing. Modifies Problem in-place. Returns number
// of ops removed (which merge into their chain head with summed base_cost).
int CollapsePWChains(mlsys::Problem& p, const TensorAnalysis& analysis);

// Pipeline result with inverse mapping for solution translation.
struct PreprocessResult {
  mlsys::Problem problem;          // preprocessed (reduced) problem
  mlsys::Problem orig_problem;     // original (before any reduction)
  // new_op_idx → list of original op indices (chain expansion).
  std::vector<std::vector<size_t>> op_expansion;
  // new_tensor_idx → original tensor index.
  std::vector<size_t> tensor_map;
  // Original op indices that were eliminated (dead ops).
  std::vector<size_t> dead_ops;
};

PreprocessResult Preprocess(mlsys::Problem p);

// Translate a solution from preprocessed indices back to original indices.
// Restores dead ops as singleton subgraphs and remaps tensor indices.
mlsys::Solution UnPreprocess(const mlsys::Solution& sol,
                              const PreprocessResult& pr);

}  // namespace solver

#endif  // PREPROCESS_H_
