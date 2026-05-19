// mm_role.h — Per-MatMul topology role (LHS/RHS chain-symmetric).
//
// Minimal standalone header: no dependencies, used by BOTH the authoritative
// evaluator (mlsys.cc) and the solver infrastructure (tensor_roles.h etc.).
// Kept separate to avoid pulling solver internals into mlsys_lib.
//
// §8 semantics:
//   Four independent bits determine how an MM op behaves in cost/WS:
//     lhs_ephemeral     = LHS input is an ephemeral MM output.
//     out_ephemeral     = output is ephemeral consumed as LHS of another MM.
//     rhs_ephemeral     = RHS input is an ephemeral MM output.
//     out_ephemeral_rhs = output is ephemeral consumed as RHS of another MM.
//
// Chain-position (direction-agnostic, used by fusion guards):
//   Standalone = no chain bit set.
//   Head       = any ephemeral output (chain producer, either side).
//   Tail       = any ephemeral input  (chain consumer, either side).
//   Middle     = both producer and consumer (relay; only valid at nk=1).
//
// Direction-specific (shape dispatch):
//   is_lhs_head : upstream LHS-chain — LHS resident h×K_op, RHS strip K_op×ke.
//   is_lhs_tail : downstream LHS-chain — LHS ephemeral, RHS strip ke×w.
//   is_rhs_head : upstream RHS-chain — LHS strip ke×K_op, RHS resident K_op×w.
//   is_rhs_tail : downstream RHS-chain — LHS h×ke, RHS ephemeral.

#ifndef MM_ROLE_H_
#define MM_ROLE_H_

namespace solver {

struct MMRole {
  bool is_matmul = false;
  bool lhs_ephemeral = false;
  bool out_ephemeral = false;
  bool rhs_ephemeral = false;
  bool out_ephemeral_rhs = false;

  bool any_in_ephemeral() const { return lhs_ephemeral || rhs_ephemeral; }
  bool any_out_ephemeral() const { return out_ephemeral || out_ephemeral_rhs; }

  bool is_standalone() const {
    return is_matmul && !any_in_ephemeral() && !any_out_ephemeral();
  }
  bool is_head() const {
    return is_matmul && !any_in_ephemeral() && any_out_ephemeral();
  }
  bool is_tail() const {
    return is_matmul && any_in_ephemeral() && !any_out_ephemeral();
  }
  bool is_middle() const {
    return is_matmul && any_in_ephemeral() && any_out_ephemeral();
  }

  bool is_lhs_head() const { return is_matmul && !lhs_ephemeral && out_ephemeral; }
  bool is_lhs_tail() const { return is_matmul && lhs_ephemeral && !any_out_ephemeral(); }
  bool is_rhs_head() const { return is_matmul && !rhs_ephemeral && out_ephemeral_rhs; }
  bool is_rhs_tail() const { return is_matmul && rhs_ephemeral && !any_out_ephemeral(); }
};

}  // namespace solver

#endif  // MM_ROLE_H_
