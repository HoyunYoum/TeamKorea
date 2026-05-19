// granularity.cc — BestGranularity + ExhaustiveGranularity.
//
// Extracted from solver_common.cc (Phase 3 refactor).

#include "granularity.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "tensor_roles.h"

namespace solver {

namespace {
inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace

// ── BestGranularity (io_solver style) ────────────────────────────────────────

GranConfig BestGranularity(const mlsys::Problem& p,
                           const std::vector<size_t>& ops,
                           const std::unordered_set<size_t>& retained_in,
                           int64_t retained_size,
                           const DAG* dag,
                           bool assume_retain_out,
                           int64_t extra_wh_slots) {
  // Unified-grid invariant (#28): all non-ephemeral outputs in a subgraph
  // must share the same spatial shape, otherwise the evaluator rejects it.
  {
    auto ts_check = Classify(p, ops, dag);
    int64_t refW = 0, refH = 0;
    for (size_t op : ops) {
      for (size_t t : p.ops[op].outputs) {
        if (ts_check.ephemeral.count(t)) continue;
        int64_t tw = p.tensors[t].width, th = p.tensors[t].height;
        if (refW == 0 && refH == 0) { refW = tw; refH = th; continue; }
        if (tw != refW || th != refH) {
          return {{0, 0, 0}, {}, kInf};  // invalid subgraph
        }
      }
    }
  }

  auto [oW, oH] = OutDims(p, ops);
  int64_t natW = p.native_granularity.width;
  int64_t natH = p.native_granularity.height;
  int64_t C = p.fast_memory_capacity;

  int64_t maxK = 0;
  for (size_t op : ops)
    if (p.ops[op].op_type == "MatMul")
      maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);

  // Precompute per-tensor role info for multi-role-aware WS and IO.
  auto ts = Classify(p, ops, dag);
  auto tensor_roles = BuildRoles(p, ops, ts);
  int64_t n_outputs = (int64_t)ts.outputs.size();
  // Fan-out ephemerals occupy a w*h slot each (see mlsys::kFanoutEphemeralTakesSlot).
  int64_t n_fanout_eph = CountFanoutEphemerals(p, ops, ts);

  auto ComputeWS = [&](int64_t w, int64_t h, int64_t k) -> int64_t {
    return ComputeTileWS(p, tensor_roles, n_outputs, retained_in, w, h, k,
                         extra_wh_slots + n_fanout_eph, retained_size);
  };

  // ── Role-based fusion-validity guards (mirror evaluator exactly) ────
  auto mm_roles = ClassifyMMRoles(p, ops, ts);
  MMRoleCounts rc = CountMMRoles(mm_roles, ops, p);

  // Split-K (nk > 1) rules (evaluator refuses these):
  //   - Any Middle MM present (relay needs two k-partitions; impossible).
  //   - Anything other than a pure Head+Tail pair or a single Standalone
  //     (mix of chain + standalone violates unified-grid K_slicing = K_op).
  //   - MM+PW mix is NO LONGER a blanket ban (issue #71 supersedes #63).
  //     Granule-alignment (PW→MM(LHS): w≥K; PW→MM(RHS): h≥K) is enforced
  //     per-candidate in TryConfig below. Epilogue MM→PW is always OK.
  // Full-K (nk = 1) rules:
  //   - If any Middle present OR |MM| ≥ 3: require w = k = K_op uniform.
  const bool split_k_infeasible =
      rc.n_middle > 0 ||                          // Middle
      (rc.total_mm() >= 2 &&
       !rc.is_pure_chain_pair() && !rc.is_pure_standalone()) ||
      (rc.is_pure_standalone() && rc.total_mm() >= 2);

  // Precompute PW→MM alignment requirements (issue #71). For each PW op in
  // this subgraph, if its output feeds a downstream MM within the subgraph,
  // record the MM's K. Split-K (nk>1) configs must satisfy w ≥ pw_to_mm_lhs_K
  // and h ≥ pw_to_mm_rhs_K.
  int64_t pw_to_mm_lhs_K = 0;
  int64_t pw_to_mm_rhs_K = 0;
  {
    std::unordered_set<size_t> op_set(ops.begin(), ops.end());
    for (size_t pw_op : ops) {
      if (p.ops[pw_op].op_type != "Pointwise") continue;
      for (size_t t : p.ops[pw_op].outputs) {
        for (size_t mm_op : ops) {
          if (mm_op == pw_op) continue;
          if (p.ops[mm_op].op_type != "MatMul") continue;
          const auto& mins = p.ops[mm_op].inputs;
          int64_t mm_K = p.tensors[mins[0]].width;
          for (int pos = 0; pos < (int)mins.size(); ++pos) {
            if (mins[pos] != t) continue;
            if (pos == 0) pw_to_mm_lhs_K = std::max(pw_to_mm_lhs_K, mm_K);
            else pw_to_mm_rhs_K = std::max(pw_to_mm_rhs_K, mm_K);
          }
        }
      }
    }
  }

  const bool force_full_k = split_k_infeasible;
  auto k_cands = (force_full_k && maxK > 0)
                     ? std::vector<int64_t>{maxK}
                     : KCandidates(maxK);
  auto w_base = DimCandidates(oW, natW);
  auto h_base = DimCandidates(oH, natH);

  // Any chain of length ≥ 3 (Middle present) at nk=1 requires w = k = K_op
  // uniform.  Also fires for 3+ MM subgraphs forced to nk=1 above.
  //
  // Extended guard: subgraphs with 2+ MMs AND PW usually require uniform MM K.
  //
  // Exception: MM-PW-MM 3-op (FFN pattern) where the PW is a shape-preserving
  // relay between two Standalone MMs. Evaluator classifies both MMs as
  // Standalone (PW breaks the direct MM-MM chain detection), and at k=maxK
  // nk=1, each MM uses its OWN K_op as ke — no padding waste. Unified found
  // this on ffn_large (~22% improvement over uniform-K rejection). Gated on
  // ops.size()==3 to keep the relaxation conservative; multi-PW sandwiches
  // stay under the uniform-K constraint.
  const bool is_mm_pw_mm_3op = (ops.size() == 3 && rc.total_mm() == 2 &&
                                 rc.n_pw == 1 && rc.n_standalone == 2);
  const bool uniform_k_required =
      rc.n_middle > 0 ||
      rc.total_mm() >= 3 ||
      (rc.total_mm() >= 2 && rc.n_pw > 0 && !is_mm_pw_mm_3op);
  if (uniform_k_required) {
    bool uniform_K = true;
    for (size_t op : ops) {
      if (p.ops[op].op_type != "MatMul") continue;
      if (p.tensors[p.ops[op].inputs[0]].width != maxK) {
        uniform_K = false;
        break;
      }
    }
    if (!uniform_K || maxK > natW) {
      return {{0, 0, 0}, {}, kInf};  // infeasible
    }
    w_base = {maxK};  // force w = k = maxK
  }

  // AnalyticalCost is O(1) and exact — track single best directly.
  GranConfig best{{0, 0, 0}, {}, kInf};

  auto TryConfig = [&](int64_t w, int64_t h, int64_t k) {
    if (w <= 0 || h <= 0 || k <= 0) return;
    if (h > natH) h = natH;
    if (w > natW) w = natW;
    // Issue #71: PW→MM granule alignment on split-K (nk > 1).
    if (maxK > 0 && k < maxK) {
      if (pw_to_mm_lhs_K > 0 && w < pw_to_mm_lhs_K) return;
      if (pw_to_mm_rhs_K > 0 && h < pw_to_mm_rhs_K) return;
    }
    int64_t ws = ComputeWS(w, h, k);
    if (ws > C) return;

    for (int snake = 0; snake < 3; ++snake) {
      mlsys::Granularity g{w, h, k};
      double cost = AnalyticalCost(p, ops, g, snake, retained_in, ts,
                              tensor_roles, maxK, n_outputs, assume_retain_out);
      if (cost < best.cost) {
        int64_t ntw = CeilDiv(oW, w), nth = CeilDiv(oH, h);
        std::vector<int64_t> trav;
        if (snake == 0 && ntw * nth > 1)
          trav = SnakeRow(ntw, nth);
        else if (snake == 1 && ntw * nth > 1)
          trav = SnakeCol(ntw, nth);
        best = {g, std::move(trav), cost};
      }
    }
  };

  for (int64_t k : k_cands) {
    for (int64_t w : w_base) {
      // Try standard h candidates
      for (int64_t h : h_base)
        TryConfig(w, h, k);

      // Compute max feasible h analytically.  Chain-upstream LHS contributes
      // h × K_full (full reduction, resident) rather than h × k (#62).
      int64_t coeff_h = n_outputs * w;
      int64_t const_wk = retained_size;
      for (auto& [tid, role] : tensor_roles) {
        int64_t lhs_k_ws = role.head_LHS_K > 0 ? role.head_LHS_K : k;
        if (role.K_lhs > 0 && !role.is_rhs) {
          coeff_h += role.is_pw ? std::max(lhs_k_ws, w) : lhs_k_ws;
        } else if (role.K_lhs > 0 && role.is_rhs) {
          coeff_h += lhs_k_ws;
        } else if (role.is_rhs && !role.is_pw) {
          const_wk += k * w;
        } else if (role.is_rhs && role.is_pw) {
          coeff_h += w;
        } else if (role.is_pw) {
          coeff_h += w;
        }
      }
      if (coeff_h > 0) {
        int64_t h_max = (C - const_wk) / coeff_h;
        if (h_max > 0) {
          TryConfig(w, h_max, k);
          if (natH > 1) TryConfig(w, (h_max / natH) * natH, k);
        }
      }
    }

    // Symmetric: for each h, compute max feasible w
    for (int64_t h : h_base) {
      int64_t coeff_w = n_outputs * h;
      int64_t const_hk = retained_size;
      for (auto& [tid, role] : tensor_roles) {
        int64_t lhs_k_ws = role.head_LHS_K > 0 ? role.head_LHS_K : k;
        if (role.K_lhs > 0 && !role.is_rhs) {
          const_hk += h * lhs_k_ws;
        } else if (role.K_lhs > 0 && role.is_rhs) {
          const_hk += h * lhs_k_ws;
        } else if (role.is_rhs && !role.is_pw) {
          coeff_w += k;
        } else if (role.is_rhs && role.is_pw) {
          coeff_w += std::max(k, h);
        } else if (role.is_pw) {
          coeff_w += h;
        }
      }
      if (coeff_w > 0) {
        int64_t w_max = (C - const_hk) / coeff_w;
        if (w_max > 0) {
          TryConfig(w_max, h, k);
          if (natW > 1) TryConfig((w_max / natW) * natW, h, k);
        }
      }
    }
  }

  return best;
}

// ── ExhaustiveGranularity (optimal_solver style, top-200 screening) ──────────

BoundResult ExhaustiveGranularity(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const std::unordered_set<size_t>& retained_in,
    int64_t retained_size,
    const DAG* dag,
    int64_t extra_wh_slots,
    bool assume_retain_out) {
  auto [oW, oH] = OutDims(p, ops);
  int64_t natW = p.native_granularity.width;
  int64_t natH = p.native_granularity.height;
  int64_t C = p.fast_memory_capacity;

  int64_t maxK = 0;
  for (size_t op : ops)
    if (p.ops[op].op_type == "MatMul")
      maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);

  auto ts = Classify(p, ops, dag);
  auto tensor_roles = BuildRoles(p, ops, ts);
  int64_t n_outputs = (int64_t)ts.outputs.size();
  int64_t n_fanout_eph = CountFanoutEphemerals(p, ops, ts);

  auto ComputeWS = [&](int64_t w, int64_t h, int64_t k) -> int64_t {
    return ComputeTileWS(p, tensor_roles, n_outputs, retained_in, w, h, k,
                         extra_wh_slots + n_fanout_eph, retained_size);
  };

  // Dense w candidates (capped at native granularity)
  int64_t wCap = std::min(oW, natW);
  std::vector<int64_t> w_cands;
  for (int64_t w = 1; w <= wCap; ++w) w_cands.push_back(w);
  std::sort(w_cands.begin(), w_cands.end());
  w_cands.erase(std::unique(w_cands.begin(), w_cands.end()), w_cands.end());

  // Dense k candidates (divisors + powers of 2).
  // MM+PW split-K is no longer a blanket ban (issue #71 supersedes #63) —
  // granule alignment is enforced per-candidate in TryConfig below.
  std::vector<int64_t> k_cands;
  if (maxK == 0) {
    k_cands = {1};
  } else {
    std::set<int64_t> ks;
    for (int64_t d = 1; d * d <= maxK; ++d) {
      if (maxK % d == 0) { ks.insert(d); ks.insert(maxK / d); }
    }
    for (int64_t v = 1; v <= maxK; v *= 2) ks.insert(v);
    ks.insert(maxK);
    k_cands.assign(ks.begin(), ks.end());
  }

  // Precompute PW→MM granule alignment requirements (issue #71).
  int64_t pw_to_mm_lhs_K = 0;
  int64_t pw_to_mm_rhs_K = 0;
  for (size_t pw_op : ops) {
    if (p.ops[pw_op].op_type != "Pointwise") continue;
    for (size_t t : p.ops[pw_op].outputs) {
      for (size_t mm_op : ops) {
        if (mm_op == pw_op) continue;
        if (p.ops[mm_op].op_type != "MatMul") continue;
        const auto& mins = p.ops[mm_op].inputs;
        int64_t mm_K = p.tensors[mins[0]].width;
        for (int pos = 0; pos < (int)mins.size(); ++pos) {
          if (mins[pos] != t) continue;
          if (pos == 0) pw_to_mm_lhs_K = std::max(pw_to_mm_lhs_K, mm_K);
          else pw_to_mm_rhs_K = std::max(pw_to_mm_rhs_K, mm_K);
        }
      }
    }
  }

  // AnalyticalCost is O(1) and exact — track single best directly.
  BoundResult best{{0, 0, 0}, 0, kInf, 0, 0, 0, 0};

  auto TryConfig = [&](int64_t w, int64_t h, int64_t k) {
    if (w <= 0 || h <= 0 || k <= 0) return;
    if (h > natH) h = natH;
    if (w > natW) w = natW;
    // Issue #71: PW→MM granule alignment on split-K (nk > 1).
    if (maxK > 0 && k < maxK) {
      if (pw_to_mm_lhs_K > 0 && w < pw_to_mm_lhs_K) return;
      if (pw_to_mm_rhs_K > 0 && h < pw_to_mm_rhs_K) return;
    }
    int64_t ws = ComputeWS(w, h, k);
    if (ws > C) return;
    for (int snake = 0; snake < 3; ++snake) {
      mlsys::Granularity g{w, h, k};
      double cost = AnalyticalCost(p, ops, g, snake, retained_in, ts,
                              tensor_roles, maxK, n_outputs, assume_retain_out);
      if (cost < best.exact_cost) {
        int64_t ntw = CeilDiv(oW, w), nth = CeilDiv(oH, h);
        int64_t nk_val = (maxK > 0) ? CeilDiv(maxK, k) : 1;
        best = {g, snake, cost, ws, ntw, nth, nk_val};
      }
    }
  };

  int64_t hCap = std::min(oH, natH);

  for (int64_t k : k_cands) {
    for (int64_t w : w_cands) {
      if (ComputeWS(w, 1, k) > C) continue;
      int64_t lo = 1, hi = hCap;
      while (lo < hi) {
        int64_t mid = lo + (hi - lo + 1) / 2;
        if (ComputeWS(w, mid, k) <= C) lo = mid; else hi = mid - 1;
      }
      for (int64_t h = 1; h <= lo; ++h) {
        TryConfig(w, h, k);
      }
    }
  }
  return best;
}


}  // namespace solver
