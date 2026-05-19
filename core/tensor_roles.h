// tensor_roles.h — Tensor / op role classification for the unified cost model.
//
// Four kinds of role info:
//   TensorSets       — per-subgraph partition of tensors (produced, consumed,
//                      ephemeral, inputs, outputs) per spec §2.3–§2.5.
//   TensorRoleInfo   — per-tensor dispatch info (LHS/RHS/PW, chain-head K_up).
//                      Drives ComputeTileWS, DominantRole, AnalyticalCost.
//   MMRole           — per-MatMul-op chain membership (§8 LHS/RHS-chain
//                      symmetric). Two bits per direction: {lhs,rhs}_ephemeral
//                      (input side) and out_ephemeral{,_rhs} (output side).
//   MMRoleCounts     — aggregate head/tail/middle/standalone counts used by
//                      fusion-validity guards.
//
// Extracted from solver_common.cc (Phase 2 refactor). Depends only on dag.h.

#ifndef TENSOR_ROLES_H_
#define TENSOR_ROLES_H_

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dag.h"
#include "mlsys.h"
#include "mm_role.h"

namespace solver {

// ── Tensor Classification ────────────────────────────────────────────────────

struct TensorSets {
  std::unordered_set<size_t> produced, consumed, ephemeral, inputs, outputs;
};

// Partition a subgraph's tensors. If `dag` is provided, applies §5.1 F1 rule:
// a tensor is ephemeral only if ALL its graph-level consumers are inside ops.
TensorSets Classify(const mlsys::Problem& p, const std::vector<size_t>& ops,
                    const DAG* dag = nullptr);

// ── Per-tensor Role Info ─────────────────────────────────────────────────────

// Per-tensor dispatch for cost/WS. A tensor may play multiple roles (MM LHS,
// MM RHS, PW input); usage counts track distinct buffers per (op, input-pos).
//
// Chain-head (§62 + §8 symmetry) fields override standard streaming:
//   LHS-chain Head's LHS:  head_LHS_K > 0       → resident h × K_up
//   LHS-chain Head's RHS:  head_RHS_K > 0       → slice K_up × min(k, K_slicing)
//   RHS-chain Head's LHS:  rhs_head_LHS_K > 0   → slice min(k, K_slicing) × K_up
//   RHS-chain Head's RHS:  rhs_head_RHS_K > 0   → resident K_up × w
struct TensorRoleInfo {
  int64_t K_lhs = 0;     // max K across LHS roles (0 = unused as LHS)
  int64_t K_rhs = 0;     // max K across RHS roles (0 = unused as RHS)
  bool is_rhs = false;
  bool is_pw = false;

  int64_t head_LHS_K = 0;
  int64_t head_RHS_K = 0;
  int64_t rhs_head_LHS_K = 0;
  int64_t rhs_head_RHS_K = 0;

  // Per-role usage counts (#59: evaluator allocates one buffer per use).
  int lhs_uses = 0;
  int rhs_uses = 0;
  int pw_uses = 0;
};

std::unordered_map<size_t, TensorRoleInfo> BuildRoles(
    const mlsys::Problem& p, const std::vector<size_t>& ops,
    const TensorSets& ts);

// MMRole struct lives in mm_role.h (shared with mlsys.cc).

std::unordered_map<size_t, MMRole> ClassifyMMRoles(
    const mlsys::Problem& p, const std::vector<size_t>& ops,
    const TensorSets& ts);

// ── Aggregate Counts ────────────────────────────────────────────────────────

struct MMRoleCounts {
  int n_head = 0;
  int n_tail = 0;
  int n_middle = 0;
  int n_standalone = 0;
  int n_pw = 0;
  int total_mm() const { return n_head + n_tail + n_middle + n_standalone; }
  bool has_chain() const { return n_head + n_tail + n_middle > 0; }
  bool is_pure_chain_pair() const {
    return n_head == 1 && n_tail == 1 && n_middle == 0 && n_standalone == 0;
  }
  bool is_pure_standalone() const {
    return n_head == 0 && n_tail == 0 && n_middle == 0 && n_standalone > 0;
  }
};

MMRoleCounts CountMMRoles(
    const std::unordered_map<size_t, MMRole>& roles,
    const std::vector<size_t>& ops,
    const mlsys::Problem& p);

// Count ephemerals with ≥2 distinct internal consumers (would occupy a WS
// slot). Currently always returns 0 per PROBLEM_SPEC #64 (ephemerals cost
// zero capacity regardless of fan-out). Retained for API compatibility.
int64_t CountFanoutEphemerals(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const TensorSets& ts);

}  // namespace solver

#endif  // TENSOR_ROLES_H_
