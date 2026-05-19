// cost.h — Cost model + working-set computation.
//
// Authoritative tile-by-tile simulation (ExactCost) mirroring mlsys::Evaluate,
// plus the O(1) AnalyticalCost used by solvers' fast inner loops. Uses
// tensor_roles.h for per-tensor/per-op role info.
//
// Extracted from solver_common.cc (Phase 2 refactor).

#ifndef COST_H_
#define COST_H_

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dag.h"
#include "mlsys.h"
#include "tensor_roles.h"

namespace solver {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ── Working Set ──────────────────────────────────────────────────────────────

// Per-tile WS for a given (w, h, k) granularity. Mirrors evaluator exactly:
// each (op, input-pos) usage is a distinct buffer (#59). Chain-head flags
// in TensorRoleInfo override standard slice sizes per §8.
int64_t ComputeTileWS(
    const mlsys::Problem& p,
    const std::unordered_map<size_t, TensorRoleInfo>& roles,
    int64_t n_outputs,
    const std::unordered_set<size_t>& retained_in,
    int64_t w, int64_t h, int64_t k,
    int64_t extra_wh_slots = 0,
    int64_t retained_size = 0);

// Multi-role-aware working set (legacy; used by io_solver).
int64_t WorkingSet(const mlsys::Problem& p,
                   const std::vector<size_t>& ops,
                   const mlsys::Granularity& g,
                   int64_t extra = 0,
                   const DAG* dag = nullptr);

// ── Geometry ─────────────────────────────────────────────────────────────────

// Max W×H across non-ephemeral outputs (primary-output unified grid, #28).
std::pair<int64_t, int64_t> OutDims(const mlsys::Problem& p,
                                    const std::vector<size_t>& ops);

// ── Dominant Role (cost dispatch helper) ─────────────────────────────────────

// Returns (dom_type, max_slice) where dom_type: 0=LHS, 1=RHS, 2=PW.
// Used by ExactCostCore / AnalyticalCost to pick IO accounting rule.
std::pair<int, int64_t> DominantRole(const TensorRoleInfo& role,
                                      int64_t w, int64_t h, int64_t k);

// ── Traversal Orders ─────────────────────────────────────────────────────────

std::vector<int64_t> SnakeRow(int64_t ntw, int64_t nth);
std::vector<int64_t> SnakeCol(int64_t ntw, int64_t nth);

// ── Candidate Generation ─────────────────────────────────────────────────────

std::vector<int64_t> DimCandidates(int64_t out_dim, int64_t native);
std::vector<int64_t> KCandidates(int64_t maxK);

// ── Cost Evaluation ──────────────────────────────────────────────────────────

// Tile-by-tile simulation matching mlsys::Evaluate. Ground-truth cost model.
double ExactCost(const mlsys::Problem& p,
                 const std::vector<size_t>& ops,
                 const mlsys::Granularity& g,
                 const std::vector<size_t>& retain_out,
                 const std::unordered_set<size_t>& retained_in,
                 const std::vector<int64_t>* trav = nullptr,
                 const DAG* dag = nullptr);

// Cached variant: accepts pre-computed TensorSets + roles (avoids Classify
// + BuildRoles per call).
double ExactCostCached(const mlsys::Problem& p,
                       const std::vector<size_t>& ops,
                       const mlsys::Granularity& g,
                       const std::vector<size_t>& retain_out,
                       const std::unordered_set<size_t>& retained_in,
                       const TensorSets& ts,
                       const std::unordered_map<size_t, TensorRoleInfo>& roles,
                       const std::vector<int64_t>* trav = nullptr);

// O(1) analytical cost, algebraically equivalent to ExactCost for snake
// traversals on homogeneous subgraphs. Used for MIP column costs + SA inner
// loops. Known divergence from ExactCost in some chain scenarios.
double AnalyticalCost(const mlsys::Problem& p,
                 const std::vector<size_t>& ops,
                 const mlsys::Granularity& g,
                 int snake_mode,  // 0=row, 1=col, 2=raster
                 const std::unordered_set<size_t>& retained_in,
                 const TensorSets& ts,
                 const std::unordered_map<size_t, TensorRoleInfo>& tensor_roles,
                 int64_t maxK,
                 int64_t n_outputs,
                 bool assume_retain_out = false);

}  // namespace solver

#endif  // COST_H_
