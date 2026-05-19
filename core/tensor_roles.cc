// tensor_roles.cc — TensorSets, TensorRoleInfo, MMRole classification.

#include "tensor_roles.h"

#include <algorithm>

namespace solver {

TensorSets Classify(const mlsys::Problem& p, const std::vector<size_t>& ops,
                    const DAG* dag) {
  TensorSets ts;
  std::unordered_set<size_t> ops_set(ops.begin(), ops.end());
  for (size_t op : ops) {
    for (size_t t : p.ops[op].outputs) ts.produced.insert(t);
    for (size_t t : p.ops[op].inputs) ts.consumed.insert(t);
  }
  for (size_t t : ts.produced) {
    if (!ts.consumed.count(t)) continue;
    // Rule #51: ephemeral only if ALL graph-level consumers are inside ops.
    if (dag) {
      bool all_inside = true;
      for (size_t cons : dag->tensor_consumers[t]) {
        if (!ops_set.count(cons)) { all_inside = false; break; }
      }
      if (!all_inside) continue;
    }
    ts.ephemeral.insert(t);
  }
  for (size_t t : ts.consumed)
    if (!ts.produced.count(t)) ts.inputs.insert(t);
  for (size_t t : ts.produced)
    if (!ts.ephemeral.count(t)) ts.outputs.insert(t);
  return ts;
}

std::unordered_map<size_t, TensorRoleInfo> BuildRoles(
    const mlsys::Problem& p, const std::vector<size_t>& ops,
    const TensorSets& ts) {
  std::unordered_map<size_t, TensorRoleInfo> roles;
  for (size_t op : ops) {
    const auto& o = p.ops[op];
    for (int pos = 0; pos < (int)o.inputs.size(); ++pos) {
      size_t t = o.inputs[pos];
      if (ts.produced.count(t)) continue;  // ephemeral
      auto& role = roles[t];
      if (o.op_type == "MatMul") {
        if (pos == 0) {
          role.K_lhs = std::max(role.K_lhs, p.tensors[t].width);
          ++role.lhs_uses;
        } else {
          role.is_rhs = true;
          role.K_rhs = std::max(role.K_rhs, p.tensors[o.inputs[0]].width);
          ++role.rhs_uses;
        }
      } else {
        role.is_pw = true;
        ++role.pw_uses;
      }
    }
  }

  // Chain-upstream detection (#62 + §8 mirror): for each MatMul A whose
  // output is an ephemeral consumed by another MatMul B in `ops`, mark A's
  // operands with the appropriate streaming pattern based on which position
  // B consumes the ephemeral at.
  //
  //   Consumed as B's LHS (LHS-chain):
  //     A's LHS stays resident h × K_up (head_LHS_K)
  //     A's RHS slice = K_up × min(k, K_slicing) (head_RHS_K)
  //
  //   Consumed as B's RHS (RHS-chain):
  //     A's LHS slice = min(k, K_slicing) × K_up (rhs_head_LHS_K)
  //     A's RHS stays resident K_up × w (rhs_head_RHS_K)
  std::unordered_set<size_t> op_set(ops.begin(), ops.end());
  for (size_t up : ops) {
    const auto& upop = p.ops[up];
    if (upop.op_type != "MatMul") continue;
    if (upop.outputs.empty()) continue;
    size_t inter = upop.outputs[0];
    if (!ts.ephemeral.count(inter)) continue;
    bool feeds_mm_lhs = false;
    bool feeds_mm_rhs = false;
    for (size_t dn : ops) {
      if (dn == up) continue;
      const auto& dnop = p.ops[dn];
      if (dnop.op_type != "MatMul") continue;
      for (int pos = 0; pos < (int)dnop.inputs.size(); ++pos) {
        if (dnop.inputs[pos] != inter) continue;
        if (pos == 0) feeds_mm_lhs = true;
        else feeds_mm_rhs = true;
      }
    }
    if (!feeds_mm_lhs && !feeds_mm_rhs) continue;
    size_t lhs_t = upop.inputs[0];
    int64_t K_up = p.tensors[lhs_t].width;
    if (feeds_mm_lhs) {
      auto it = roles.find(lhs_t);
      if (it != roles.end()) {
        it->second.head_LHS_K = std::max(it->second.head_LHS_K, K_up);
      }
      for (int pos = 1; pos < (int)upop.inputs.size(); ++pos) {
        size_t rhs_t = upop.inputs[pos];
        auto rit = roles.find(rhs_t);
        if (rit != roles.end()) {
          rit->second.head_RHS_K = std::max(rit->second.head_RHS_K, K_up);
        }
      }
    }
    if (feeds_mm_rhs) {
      auto it = roles.find(lhs_t);
      if (it != roles.end()) {
        it->second.rhs_head_LHS_K = std::max(it->second.rhs_head_LHS_K, K_up);
      }
      for (int pos = 1; pos < (int)upop.inputs.size(); ++pos) {
        size_t rhs_t = upop.inputs[pos];
        auto rit = roles.find(rhs_t);
        if (rit != roles.end()) {
          rit->second.rhs_head_RHS_K =
              std::max(rit->second.rhs_head_RHS_K, K_up);
        }
      }
    }
  }
  return roles;
}

std::unordered_map<size_t, MMRole> ClassifyMMRoles(
    const mlsys::Problem& p, const std::vector<size_t>& ops,
    const TensorSets& ts) {
  std::unordered_set<size_t> op_set(ops.begin(), ops.end());
  std::unordered_map<size_t, size_t> producer_in_sg;
  for (size_t op : ops)
    for (size_t t : p.ops[op].outputs) producer_in_sg[t] = op;

  std::unordered_map<size_t, std::vector<size_t>> lhs_consumers_in_sg;
  std::unordered_map<size_t, std::vector<size_t>> rhs_consumers_in_sg;
  for (size_t op : ops) {
    if (p.ops[op].op_type != "MatMul") continue;
    const auto& opdef = p.ops[op];
    for (int pos = 0; pos < (int)opdef.inputs.size(); ++pos) {
      if (pos == 0) lhs_consumers_in_sg[opdef.inputs[pos]].push_back(op);
      else rhs_consumers_in_sg[opdef.inputs[pos]].push_back(op);
    }
  }

  std::unordered_map<size_t, MMRole> roles;
  for (size_t op : ops) {
    MMRole r;
    if (p.ops[op].op_type == "MatMul") {
      r.is_matmul = true;
      const auto& opdef = p.ops[op];
      for (int pos = 0; pos < (int)opdef.inputs.size(); ++pos) {
        size_t t = opdef.inputs[pos];
        if (!ts.ephemeral.count(t)) continue;
        auto it = producer_in_sg.find(t);
        if (it == producer_in_sg.end()) continue;
        if (p.ops[it->second].op_type != "MatMul") continue;
        if (pos == 0) r.lhs_ephemeral = true;
        else r.rhs_ephemeral = true;
      }
      if (!opdef.outputs.empty()) {
        size_t out_t = opdef.outputs[0];
        if (ts.ephemeral.count(out_t)) {
          auto lit = lhs_consumers_in_sg.find(out_t);
          if (lit != lhs_consumers_in_sg.end() && !lit->second.empty())
            r.out_ephemeral = true;
          auto rit = rhs_consumers_in_sg.find(out_t);
          if (rit != rhs_consumers_in_sg.end() && !rit->second.empty())
            r.out_ephemeral_rhs = true;
        }
      }
    }
    roles[op] = r;
  }
  return roles;
}

MMRoleCounts CountMMRoles(
    const std::unordered_map<size_t, MMRole>& roles,
    const std::vector<size_t>& ops,
    const mlsys::Problem& p) {
  MMRoleCounts c;
  for (size_t op : ops) {
    auto it = roles.find(op);
    if (it == roles.end()) continue;
    const MMRole& r = it->second;
    if (!r.is_matmul) {
      if (p.ops[op].op_type == "Pointwise") ++c.n_pw;
      continue;
    }
    if (r.is_head()) ++c.n_head;
    else if (r.is_tail()) ++c.n_tail;
    else if (r.is_middle()) ++c.n_middle;
    else if (r.is_standalone()) ++c.n_standalone;
  }
  return c;
}

int64_t CountFanoutEphemerals(const mlsys::Problem& p,
                               const std::vector<size_t>& ops,
                               const TensorSets& ts) {
  // Per PROBLEM_SPEC #64: ephemerals cost zero capacity regardless of fan-out.
  (void)p; (void)ops; (void)ts;
  return 0;
}

}  // namespace solver
