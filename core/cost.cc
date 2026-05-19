// cost.cc — ExactCost simulation + AnalyticalCost formula + WS helpers.
//
// Extracted from solver_common.cc (Phase 2 refactor).

#include "cost.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>

#include "tensor_roles.h"

namespace solver {

namespace {
// CeilDiv — also declared in solver_common.h for external use. Local copy
// avoids cost.cc depending on solver_common.h (circular-include-prone).
inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace

int64_t ComputeTileWS(
    const mlsys::Problem& p,
    const std::unordered_map<size_t, TensorRoleInfo>& roles,
    int64_t n_outputs,
    const std::unordered_set<size_t>& retained_in,
    int64_t w, int64_t h, int64_t k,
    int64_t extra_wh_slots,
    int64_t retained_size) {
  // Mirror evaluator: each (op, input-pos) usage is a distinct buffer (#59).
  // Sum slice size times usage count per role.  §8 symmetric dispatch:
  //
  //   LHS slice:
  //     head_LHS_K > 0      : h × K_up           (LHS-chain Head LHS, resident)
  //     rhs_head_LHS_K > 0  : min(k, K_up) × K_up (RHS-chain Head LHS, strip)
  //     default             : h × k               (Standalone / chain Tail LHS)
  //
  //   RHS slice:
  //     head_RHS_K > 0      : K_up × min(k, K_rhs) (LHS-chain Head RHS, strip)
  //     rhs_head_RHS_K > 0  : K_up × w             (RHS-chain Head RHS, resident)
  //     default             : min(k, K_rhs) × w    (Standalone / chain Tail RHS)
  int64_t ws = 0;
  for (auto& [tid, role] : roles) {
    if (retained_in.count(tid)) continue;
    int64_t lhs = 0;
    if (role.K_lhs > 0) {
      if (role.head_LHS_K > 0) {
        lhs = h * role.head_LHS_K;
      } else if (role.rhs_head_LHS_K > 0) {
        lhs = std::min(k, role.K_lhs) * role.rhs_head_LHS_K;
      } else {
        lhs = h * k;
      }
    }
    int64_t rhs = 0;
    if (role.is_rhs) {
      if (role.head_RHS_K > 0) {
        rhs = role.head_RHS_K * std::min(k, role.K_rhs);
      } else if (role.rhs_head_RHS_K > 0) {
        rhs = role.rhs_head_RHS_K * w;
      } else {
        rhs = std::min(k, role.K_rhs > 0 ? role.K_rhs : k) * w;
      }
    }
    int64_t pw = role.is_pw ? w * h : 0;
    ws += lhs * role.lhs_uses + rhs * role.rhs_uses + pw * role.pw_uses;
  }
  for (size_t t : retained_in)
    ws += p.tensors[t].width * p.tensors[t].height;
  ws += (n_outputs + extra_wh_slots) * w * h;
  return ws + retained_size;
}

// Returns (dominant_type, max_slice) where type: 0=LHS, 1=RHS, 2=PW.
// Tie-break: LHS > RHS > PW (LHS has row-snake reuse, most beneficial).
// §8 symmetric — chain-head shapes override the standard slice sizes.
std::pair<int, int64_t> DominantRole(const TensorRoleInfo& role,
                                      int64_t w, int64_t h, int64_t k) {
  int64_t s_lhs = 0;
  if (role.K_lhs > 0) {
    if (role.head_LHS_K > 0) s_lhs = h * role.head_LHS_K;
    else if (role.rhs_head_LHS_K > 0) s_lhs = std::min(k, role.K_lhs) * role.rhs_head_LHS_K;
    else s_lhs = h * k;
  }
  int64_t s_rhs = 0;
  if (role.is_rhs) {
    if (role.head_RHS_K > 0) s_rhs = role.head_RHS_K * std::min(k, role.K_rhs);
    else if (role.rhs_head_RHS_K > 0) s_rhs = role.rhs_head_RHS_K * w;
    else s_rhs = std::min(k, role.K_rhs > 0 ? role.K_rhs : k) * w;
  }
  int64_t s_pw = role.is_pw ? w * h : 0;
  int64_t mx = std::max({s_lhs, s_rhs, s_pw});
  if (mx == 0) return {2, 0};  // degenerate
  if (s_lhs == mx) return {0, mx};
  if (s_rhs == mx) return {1, mx};
  return {2, mx};
}

// ── Output dims (max across all outputs) ─────────────────────────────────────

std::pair<int64_t, int64_t> OutDims(const mlsys::Problem& p,
                                    const std::vector<size_t>& ops) {
  // Non-ephemeral outputs must share shape under the unified-grid rule (#28).
  // Our evaluator derives the spatial tile grid from the primary (first)
  // non-ephemeral output, so we must mirror that here.  Ephemeral outputs
  // are excluded — their tile shape rides along with the grid.
  auto ts = Classify(p, ops);
  int64_t maxW = 0, maxH = 0;
  for (size_t op : ops) {
    for (size_t t : p.ops[op].outputs) {
      if (ts.ephemeral.count(t)) continue;
      maxW = std::max(maxW, p.tensors[t].width);
      maxH = std::max(maxH, p.tensors[t].height);
    }
  }
  if (maxW == 0 || maxH == 0) {
    // Fallback: all outputs ephemeral (degenerate case). Use max over all.
    for (size_t op : ops)
      for (size_t t : p.ops[op].outputs) {
        maxW = std::max(maxW, p.tensors[t].width);
        maxH = std::max(maxH, p.tensors[t].height);
      }
  }
  return {maxW, maxH};
}

// ── Working set (multi-role aware: max slice across all roles per tensor) ─────

int64_t WorkingSet(const mlsys::Problem& p,
                   const std::vector<size_t>& ops,
                   const mlsys::Granularity& g,
                   int64_t extra,
                   const DAG* dag) {
  auto ts = Classify(p, ops, dag);
  // Compute slice size per external input tensor across all roles.
  // Track LHS/RHS/PW separately to handle dual-role tensors (same tensor
  // as both LHS and RHS of MatMul → separate hardware buffers).
  struct SliceInfo { int64_t lhs = 0, rhs = 0, pw = 0; };
  std::unordered_map<size_t, SliceInfo> tensor_ws;
  for (size_t op : ops) {
    const auto& o = p.ops[op];
    for (int pos = 0; pos < (int)o.inputs.size(); ++pos) {
      size_t t = o.inputs[pos];
      if (ts.produced.count(t)) continue;  // ephemeral, skip
      auto& info = tensor_ws[t];
      if (o.op_type == "MatMul") {
        if (pos == 0)
          info.lhs = std::max(info.lhs, g.height * g.depth);
        else
          info.rhs = std::max(info.rhs, g.depth * g.width);
      } else {
        info.pw = std::max(info.pw, g.width * g.height);
      }
    }
  }
  int64_t ws = 0;
  for (auto& [tid, info] : tensor_ws) {
    if (info.lhs > 0 && info.rhs > 0)
      ws += info.lhs + info.rhs + info.pw;
    else
      ws += std::max({info.lhs, info.rhs, info.pw});
  }
  for (size_t t : ts.outputs) ws += g.width * g.height;
  return ws + extra;
}

// ── Snake traversals ─────────────────────────────────────────────────────────

std::vector<int64_t> SnakeRow(int64_t ntw, int64_t nth) {
  std::vector<int64_t> ord;
  ord.reserve(ntw * nth);
  for (int64_t r = 0; r < nth; ++r)
    if (r % 2 == 0)
      for (int64_t c = 0; c < ntw; ++c) ord.push_back(r * ntw + c);
    else
      for (int64_t c = ntw - 1; c >= 0; --c) ord.push_back(r * ntw + c);
  return ord;
}

std::vector<int64_t> SnakeCol(int64_t ntw, int64_t nth) {
  std::vector<int64_t> ord;
  ord.reserve(ntw * nth);
  for (int64_t c = 0; c < ntw; ++c)
    if (c % 2 == 0)
      for (int64_t r = 0; r < nth; ++r) ord.push_back(r * ntw + c);
    else
      for (int64_t r = nth - 1; r >= 0; --r) ord.push_back(r * ntw + c);
  return ord;
}

// ── ExactCost (tile-by-tile simulation matching corrected evaluator) ─────────

// Core implementation shared by both ExactCost variants.
static double ExactCostCore(const mlsys::Problem& p,
                            const std::vector<size_t>& ops,
                            const mlsys::Granularity& g,
                            const std::vector<size_t>& retain_out,
                            const std::unordered_set<size_t>& retained_in,
                            const TensorSets& ts,
                            const std::unordered_map<size_t, TensorRoleInfo>& tensor_roles,
                            const std::vector<int64_t>* trav) {
  std::unordered_set<size_t> retain_set(retain_out.begin(), retain_out.end());
  std::unordered_set<size_t> evict;
  for (size_t t : ts.outputs)
    if (!retain_set.count(t)) evict.insert(t);

  int64_t w = g.width, h = g.height, k = g.depth;
  // Guard: a 0 granularity (returned by BestGranularity when a subgraph is
  // infeasible — e.g. unified-grid violation) would cause FPE in CeilDiv
  // and the k-step accumulation. Return infinity instead.
  if (w <= 0 || h <= 0 || k <= 0) return kInf;
  auto [oW, oH] = OutDims(p, ops);
  int64_t ntw = CeilDiv(oW, w), nth = CeilDiv(oH, h);
  int64_t nspatial = ntw * nth;

  int64_t maxK = 0;
  for (size_t op : ops)
    if (p.ops[op].op_type == "MatMul")
      maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);
  int64_t nk = (maxK > 0) ? CeilDiv(maxK, k) : 1;

  // Per-op output dimensions for per-op tiling.
  std::unordered_map<size_t, std::pair<int64_t, int64_t>> op_od;
  for (size_t op : ops) {
    int64_t ow = 0, oh = 0;
    for (size_t t : p.ops[op].outputs) {
      ow = std::max(ow, p.tensors[t].width);
      oh = std::max(oh, p.tensors[t].height);
    }
    op_od[op] = {ow, oh};
  }

  // Tensor → consuming ops (for per-op IO activity check).
  std::unordered_map<size_t, std::vector<size_t>> tensor_consumers;
  for (size_t op : ops)
    for (size_t t : p.ops[op].inputs)
      if (!ts.produced.count(t))
        tensor_consumers[t].push_back(op);

  // Output tensor → producing op (for per-op eviction check).
  std::unordered_map<size_t, size_t> output_producer;
  for (size_t op : ops)
    for (size_t t : p.ops[op].outputs)
      output_producer[t] = op;

  // Pre-compute per-tensor IO entries. Dual-role tensors (both LHS and RHS
  // of MatMul) get separate entries since hardware allocates separate buffers.
  // §8 symmetric shapes — chain-head flags drive the size formula per tile.
  //
  // DomKind enumerates the 7 distinct size-vs-key dispatch cases that mirror
  // mlsys::Evaluate's SliceKey logic. The per-tile slice size uses h_eff /
  // w_eff / ke so boundary tiles (W_out % w != 0 or last-ks) don't over-count.
  //
  //   DK_PW             : h_eff * w_eff           at ks=0 only
  //   DK_LHS_STRIP      : h_eff * ke              per-ks (standalone/tail LHS)
  //   DK_RHS_STRIP      : ke * w_eff              per-ks (standalone/tail RHS)
  //   DK_LHS_HEAD_RES   : h_eff * head_LHS_K      at ks=0 only, resident
  //   DK_LHS_HEAD_STRIP : head_RHS_K * ke         per-ks (LHS-chain Head RHS)
  //   DK_RHS_HEAD_STRIP : ke * rhs_head_LHS_K     per-ks (RHS-chain Head LHS)
  //   DK_RHS_HEAD_RES   : rhs_head_RHS_K * w_eff  at ks=0 only, resident
  enum DomKind { DK_PW, DK_LHS_STRIP, DK_RHS_STRIP, DK_LHS_HEAD_RES,
                 DK_LHS_HEAD_STRIP, DK_RHS_HEAD_STRIP, DK_RHS_HEAD_RES };
  struct DomEntry {
    size_t tid;
    int dom;      // 0=LHS, 1=RHS, 2=PW (for any_consumer_active dispatch)
    DomKind kind;
    int64_t K_up;  // chain's K (0 for standalone)
  };
  std::vector<DomEntry> dom_entries;
  for (auto& [tid, role] : tensor_roles) {
    DomKind lhs_kind = DK_LHS_STRIP;
    int64_t lhs_Kup = 0;
    bool has_lhs = role.K_lhs > 0;
    if (has_lhs) {
      if (role.head_LHS_K > 0) { lhs_kind = DK_LHS_HEAD_RES; lhs_Kup = role.head_LHS_K; }
      else if (role.rhs_head_LHS_K > 0) { lhs_kind = DK_RHS_HEAD_STRIP; lhs_Kup = role.rhs_head_LHS_K; }
    }
    DomKind rhs_kind = DK_RHS_STRIP;
    int64_t rhs_Kup = 0;
    bool has_rhs = role.is_rhs;
    if (has_rhs) {
      if (role.head_RHS_K > 0) { rhs_kind = DK_LHS_HEAD_STRIP; rhs_Kup = role.head_RHS_K; }
      else if (role.rhs_head_RHS_K > 0) { rhs_kind = DK_RHS_HEAD_RES; rhs_Kup = role.rhs_head_RHS_K; }
    }
    bool has_pw = role.is_pw;
    if (has_lhs && has_rhs) {
      dom_entries.push_back({tid, 0, lhs_kind, lhs_Kup});
      dom_entries.push_back({tid, 1, rhs_kind, rhs_Kup});
      if (has_pw) dom_entries.push_back({tid, 2, DK_PW, 0});
    } else {
      auto [d, _s] = DominantRole(role, w, h, k);
      (void)_s;
      DomKind kind;
      int64_t Kup = 0;
      if (d == 0) { kind = lhs_kind; Kup = lhs_Kup; }
      else if (d == 1) { kind = rhs_kind; Kup = rhs_Kup; }
      else kind = DK_PW;
      dom_entries.push_back({tid, d, kind, Kup});
    }
  }

  // Helper: is any consuming op of tensor active at tile (r, c)?
  auto any_consumer_active = [&](size_t tid, int64_t r, int64_t c) -> bool {
    auto it = tensor_consumers.find(tid);
    if (it == tensor_consumers.end()) return false;
    for (size_t op : it->second) {
      auto [ow, oh] = op_od[op];
      if (c * w < ow && r * h < oh) return true;
    }
    return false;
  };

  // Single-slot per-tensor load cache (mirrors mlsys::Evaluate's `loaded`
  // map). Key is packed (ht, wt, ks) where -1 means "not part of this key".
  // Size-128 vector is fine for current subgraph scales.
  struct SliceKey { int ht, wt, ks; bool operator!=(const SliceKey& o) const {
    return ht != o.ht || wt != o.wt || ks != o.ks; } };
  std::unordered_map<size_t, SliceKey> loaded;

  const double B = p.slow_memory_bandwidth;
  double lat = 0;
  for (int64_t ti = 0; ti < nspatial; ++ti) {
    int64_t idx = trav ? (*trav)[ti] : ti;
    int64_t r = idx / ntw, c = idx % ntw;
    int64_t h_eff = std::min(h, oH - r * h);
    int64_t w_eff = std::min(w, oW - c * w);
    for (int64_t ks = 0; ks < nk; ++ks) {
      // Compute: every op runs at every (tile, ks), even when its output
      // doesn't cover this tile. Mirrors mlsys::Evaluate — the hardware-
      // model is "the sg's tile loop ticks every op per step," amortizing
      // small-output ops across the full grid.
      double cmp = 0;
      for (size_t op : ops) {
        const auto& o = p.ops[op];
        if (o.op_type == "MatMul") {
          int64_t op_K = p.tensors[o.inputs[0]].width;
          int64_t actual_k = std::min(k, op_K - k * ks);
          if (actual_k > 0)
            cmp += (double)o.base_cost * actual_k / op_K;
        } else if (ks == 0)
          cmp += (double)o.base_cost;
      }
      double mi = 0;
      int64_t ke = std::min(k, maxK > 0 ? maxK - k * ks : k);
      for (auto& de : dom_entries) {
        if (retained_in.count(de.tid)) continue;
        if (!any_consumer_active(de.tid, r, c)) continue;
        SliceKey key;
        int64_t sz = 0;
        bool should_load = true;
        switch (de.kind) {
          case DK_PW:
            key = {(int)r, (int)c, -1};
            sz = h_eff * w_eff;
            should_load = (ks == 0);
            break;
          case DK_LHS_STRIP:
            key = {(int)r, -1, (int)ks};
            sz = h_eff * ke;
            break;
          case DK_RHS_STRIP:
            key = {-1, (int)c, (int)ks};
            sz = ke * w_eff;
            break;
          case DK_LHS_HEAD_RES:
            key = {(int)r, -1, -1};
            sz = h_eff * de.K_up;
            should_load = (ks == 0);
            break;
          case DK_LHS_HEAD_STRIP:
            key = {-1, -1, (int)ks};
            sz = de.K_up * ke;
            break;
          case DK_RHS_HEAD_STRIP:
            key = {-1, -1, (int)ks};
            sz = ke * de.K_up;
            break;
          case DK_RHS_HEAD_RES:
            key = {-1, (int)c, -1};
            sz = de.K_up * w_eff;
            should_load = (ks == 0);
            break;
        }
        if (should_load) {
          auto it = loaded.find(de.tid);
          if (it == loaded.end() || it->second != key) {
            mi += (double)sz / B;
            loaded[de.tid] = key;
          }
        }
      }
      double mo = 0;
      if (ks == nk - 1)
        for (size_t t : evict) {
          auto pit = output_producer.find(t);
          if (pit != output_producer.end()) {
            auto [ow, oh] = op_od[pit->second];
            if (c * w >= ow || r * h >= oh) continue;
          }
          mo += (double)w_eff * h_eff / B;
        }
      lat += std::max(cmp, mi + mo);
    }
  }
  return lat;
}

double ExactCost(const mlsys::Problem& p,
                 const std::vector<size_t>& ops,
                 const mlsys::Granularity& g,
                 const std::vector<size_t>& retain_out,
                 const std::unordered_set<size_t>& retained_in,
                 const std::vector<int64_t>* trav,
                 const DAG* dag) {
  auto ts = Classify(p, ops, dag);
  auto tensor_roles = BuildRoles(p, ops, ts);
  return ExactCostCore(p, ops, g, retain_out, retained_in, ts, tensor_roles, trav);
}

double ExactCostCached(const mlsys::Problem& p,
                       const std::vector<size_t>& ops,
                       const mlsys::Granularity& g,
                       const std::vector<size_t>& retain_out,
                       const std::unordered_set<size_t>& retained_in,
                       const TensorSets& ts,
                       const std::unordered_map<size_t, TensorRoleInfo>& roles,
                       const std::vector<int64_t>* trav) {
  return ExactCostCore(p, ops, g, retain_out, retained_in, ts, roles, trav);
}

// ── Granularity candidates ───────────────────────────────────────────────────

std::vector<int64_t> DimCandidates(int64_t out_dim, int64_t native) {
  // Tile size cannot exceed native granularity (hardware constraint).
  int64_t cap = std::min(out_dim, native);
  std::set<int64_t> cands;
  // Powers of 2
  for (int64_t v = 1; v <= cap; v *= 2) cands.insert(v);
  // Multiples of native up to cap (only native itself, since cap <= native)
  if (native <= cap) cands.insert(native);
  // cap itself (= min(out_dim, native))
  cands.insert(cap);
  // Sub-native divisors of out_dim
  for (int64_t d = 1; d <= cap; ++d)
    if (out_dim % d == 0) cands.insert(d);
  return {cands.begin(), cands.end()};
}

std::vector<int64_t> KCandidates(int64_t maxK) {
  if (maxK == 0) return {1};
  std::set<int64_t> cands;
  // Powers of 2
  for (int64_t v = 1; v <= maxK; v *= 2) cands.insert(v);
  // Divisors of maxK
  for (int64_t d = 1; d * d <= maxK; ++d) {
    if (maxK % d == 0) {
      cands.insert(d);
      cands.insert(maxK / d);
    }
  }
  cands.insert(maxK);
  return {cands.begin(), cands.end()};
}

// ── AnalyticalCost (O(1) analytical cost estimate) ────────────────────────────────

double AnalyticalCost(const mlsys::Problem& p,
                 const std::vector<size_t>& ops,
                 const mlsys::Granularity& g,
                 int snake_mode,
                 const std::unordered_set<size_t>& retained_in,
                 const TensorSets& ts,
                 const std::unordered_map<size_t, TensorRoleInfo>& tensor_roles,
                 int64_t maxK,
                 int64_t n_outputs,
                 bool assume_retain_out) {
  int64_t w = g.width, h = g.height, k = g.depth;
  if (w <= 0 || h <= 0 || k <= 0) return kInf;  // infeasible-gran guard
  auto [oW, oH] = OutDims(p, ops);
  int64_t ntw = CeilDiv(oW, w), nth = CeilDiv(oH, h);
  int64_t nspatial = ntw * nth;
  double B = p.slow_memory_bandwidth;
  int64_t nk = (maxK > 0) ? CeilDiv(maxK, k) : 1;

  // Evaluate's canonical-grid rule (#68): every op runs at every canonical
  // tile once. Ops whose output extent doesn't cover a tile are MASKED
  // (zero compute, zero memory) — not scaled by a coverage ratio. Ops whose
  // output is LARGER than the grid still fire once per canonical tile
  // (producing a portion of their output each time). So the faithful weight
  // is always 1.0, and ExactCost uses exactly that. Kept as a named function
  // for clarity with the call sites that use it.
  auto OpWeight = [&](size_t /*op*/) -> double { return 1.0; };
  // Tried boundary-clamped average (w_eff = oW/ntw, h_eff = oH/nth).
  // Aggregate IO is exact by construction, but the snake-mode reuse
  // categorization (first / cat2 / cat3) assumes boundary tiles happen at
  // specific positions in the traversal — the average mutes the exact
  // per-category correction, and empirically BM-5 regressed +1.1%.
  // Left unaveraged (interior size) to match the snake-mode formulas.
  double w_eff = (double)w, h_eff = (double)h;

  // Chain-upstream detection by direction (§8 symmetric):
  //   chain_upstream_ops_lhs: output consumed as LHS of another MM (LHS-chain Head).
  //     RHS IO per k-step = K_op × ke (col-strip, full reduction).
  //   chain_upstream_ops_rhs: output consumed as RHS of another MM (RHS-chain Head).
  //     LHS IO per k-step = ke × K_op (row-strip, full reduction).
  std::unordered_set<size_t> chain_upstream_ops_lhs;
  std::unordered_set<size_t> chain_upstream_ops_rhs;
  {
    std::unordered_set<size_t> op_set(ops.begin(), ops.end());
    for (size_t up : ops) {
      const auto& upop = p.ops[up];
      if (upop.op_type != "MatMul") continue;
      if (upop.outputs.empty()) continue;
      size_t inter = upop.outputs[0];
      if (!ts.ephemeral.count(inter)) continue;
      for (size_t dn : ops) {
        if (dn == up) continue;
        const auto& dnop = p.ops[dn];
        if (dnop.op_type != "MatMul") continue;
        for (int pos = 0; pos < (int)dnop.inputs.size(); ++pos) {
          if (dnop.inputs[pos] != inter) continue;
          if (pos == 0) chain_upstream_ops_lhs.insert(up);
          else chain_upstream_ops_rhs.insert(up);
        }
      }
    }
  }

  // Compute per k-step (handles multi-K: ops with different K values finish at
  // different k-steps). Each MatMul op i is active for nk_i = ceil(op_K_i/k)
  // steps. At its last step (nk_i-1), it may have a partial k contribution.
  // Per-step costs are weighted by OpWeight for heterogeneous output dims.
  // MMInfo tracks per-k-step IO for each MM op.
  //   rhs_io: per-k-step RHS IO (per-op contribution to M_rhs in middle steps)
  //   lhs_io: per-k-step LHS IO (NEW, for RHS-chain upstream ops)
  struct MMInfo { int64_t nk_i; double per_step; double last_step;
                  double rhs_io; double lhs_io; };
  std::vector<MMInfo> mm_ops;
  double cmp_pw = 0;
  double cmp_mm_all = 0;  // sum of per_step for ALL MatMul ops (used for ks=0)
  double M_rhs_chain_up = 0;  // LHS-chain Head RHS per-step IO
  double M_lhs_chain_up = 0;  // RHS-chain Head LHS per-step IO (§8 mirror)
  for (size_t op : ops) {
    const auto& o = p.ops[op];
    double wt = OpWeight(op);
    if (o.op_type == "MatMul") {
      int64_t op_K = p.tensors[o.inputs[0]].width;
      int64_t nk_i = CeilDiv(op_K, k);
      int64_t eff_k = std::min(k, op_K);
      double ps = (double)o.base_cost * eff_k / op_K * wt;
      int64_t last_k = op_K - eff_k * (nk_i - 1);
      double ls = (last_k > 0) ? (double)o.base_cost * last_k / op_K * wt : 0;
      bool is_chain_up_lhs = chain_upstream_ops_lhs.count(op) > 0;
      bool is_chain_up_rhs = chain_upstream_ops_rhs.count(op) > 0;
      // Per-op RHS IO per k-step.
      //   LHS-chain Head : K_op × ke  (col strip, full reduction)
      //   RHS-chain Head : resident K_op × w (handled by M_rhs_chain, NOT here)
      //   standalone/tail: ke × w  (k-wide slice)
      double op_rhs = 0;
      if (!is_chain_up_rhs) {
        for (int pos = 1; pos < (int)o.inputs.size(); ++pos) {
          size_t tid = o.inputs[pos];
          if (ts.produced.count(tid)) continue;  // ephemeral
          if (retained_in.count(tid)) continue;
          double rhs_slice;
          if (is_chain_up_lhs) {
            rhs_slice = (double)op_K * eff_k;
          } else {
            int64_t rhs_k = std::min(k, p.tensors[tid].width);
            rhs_slice = (double)rhs_k * w_eff;
          }
          op_rhs += rhs_slice / B * OpWeight(op);
        }
      }
      // Per-op LHS IO per k-step — only for RHS-chain Head (strip ke × K_op).
      // LHS-chain Head LHS is resident (h × K_up) handled by M_lhs_chain.
      double op_lhs = 0;
      if (is_chain_up_rhs && !o.inputs.empty()) {
        size_t tid = o.inputs[0];
        if (!ts.produced.count(tid) && !retained_in.count(tid)) {
          int64_t lhs_slice = eff_k * op_K;
          op_lhs = (double)lhs_slice / B * OpWeight(op);
        }
      }
      if (is_chain_up_lhs) M_rhs_chain_up += op_rhs;
      if (is_chain_up_rhs) M_lhs_chain_up += op_lhs;
      mm_ops.push_back({nk_i, ps, ls, op_rhs, op_lhs});
      cmp_mm_all += ps;
    } else
      cmp_pw += (double)o.base_cost * wt;
  }
  // Sort by nk_i ascending for range-based processing
  std::sort(mm_ops.begin(), mm_ops.end(),
            [](const MMInfo& a, const MMInfo& b) { return a.nk_i < b.nk_i; });
  // cmp_mm_last: compute cost at the global last k-step (ks = nk-1)
  // Only ops with nk_i == nk contribute (others finished earlier)
  double cmp_mm = cmp_mm_all;  // full-rate MM compute (all ops at ks=0)
  double cmp_mm_last = 0;
  for (auto& m : mm_ops) {
    if (m.nk_i == nk)
      cmp_mm_last += m.last_step;
    else if (m.nk_i > nk)
      cmp_mm_last += m.per_step;  // shouldn't happen (nk = ceil(maxK/k))
  }

  // Per-tensor IO weight: fraction of tiles where the tensor's consumers are
  // active. For homogeneous subgraphs all weights are 1.0 (zero overhead).
  // Uses max output dims across all consuming ops within the subgraph.
  auto TensorIOWeight = [&](size_t tid) -> double {
    int64_t max_ow = 0, max_oh = 0;
    for (size_t op : ops) {
      for (size_t t : p.ops[op].inputs) {
        if (t != tid) continue;
        int64_t ow = 0, oh = 0;
        for (size_t out : p.ops[op].outputs) {
          ow = std::max(ow, p.tensors[out].width);
          oh = std::max(oh, p.tensors[out].height);
        }
        max_ow = std::max(max_ow, ow);
        max_oh = std::max(max_oh, oh);
      }
    }
    if (max_ow >= oW && max_oh >= oH) return 1.0;
    if (max_ow == 0 || max_oh == 0) return 1.0;
    int64_t consumer_tiles = CeilDiv(max_ow, w) * CeilDiv(max_oh, h);
    return (double)consumer_tiles / nspatial;
  };

  // IO components per load — classify by dominant role for this (w, h, k).
  // §8 symmetric treatment:
  //   M_lhs_chain: LHS-chain Head LHS (h × K_up, row-attached resident).
  //   M_rhs_chain: RHS-chain Head RHS (K_up × w, col-attached resident, mirror).
  //   M_rhs_chain_up: LHS-chain Head RHS per-step (K_up × ke, from per-op).
  //   M_lhs_chain_up: RHS-chain Head LHS per-step (ke × K_up, from per-op).
  //   Regular M_lhs / M_rhs / M_pw: standalone + chain-tail tensors.
  // Tensors are excluded from regular accounting when their per-step or
  // resident IO is tracked via the chain-specific variables above.
  double M_lhs = 0, M_rhs = 0, M_pw = 0, M_evict = 0;
  double M_lhs_chain = 0, M_rhs_chain = 0;
  for (auto& [tid, role] : tensor_roles) {
    if (retained_in.count(tid)) continue;  // fully in fast memory
    double tw = TensorIOWeight(tid);
    // LHS accounting.
    //   head_LHS_K > 0     → resident h × K_up, goes to M_lhs_chain.
    //   rhs_head_LHS_K > 0 → per-step strip (handled by per-op M_lhs_chain_up); skip regular.
    //   else               → standard h × k.
    double s_lhs = 0;
    double s_lhs_chain = 0;
    bool lhs_chain_lhs = role.head_LHS_K > 0;
    bool lhs_of_chain_up_rhs = role.rhs_head_LHS_K > 0;
    if (role.K_lhs > 0) {
      if (lhs_chain_lhs) s_lhs_chain = h_eff * role.head_LHS_K;
      else if (!lhs_of_chain_up_rhs) s_lhs = h_eff * k;
    }
    // RHS accounting (mirror):
    //   head_RHS_K > 0     → per-step strip (handled by per-op M_rhs_chain_up); skip regular.
    //   rhs_head_RHS_K > 0 → resident K_up × w, goes to M_rhs_chain.
    //   else               → standard ke × w.
    double s_rhs = 0;
    double s_rhs_chain = 0;
    bool rhs_of_chain_up_lhs = role.head_RHS_K > 0;
    bool rhs_chain_rhs = role.rhs_head_RHS_K > 0;
    if (role.is_rhs) {
      if (rhs_chain_rhs) s_rhs_chain = (double)role.rhs_head_RHS_K * w_eff;
      else if (!rhs_of_chain_up_lhs) s_rhs = (double)std::min(k, role.K_rhs > 0 ? role.K_rhs : k) * w_eff;
    }
    double s_pw = role.is_pw ? h_eff * w_eff : 0;
    // Attribute to the right bucket.
    if (s_lhs_chain > 0) M_lhs_chain += (double)s_lhs_chain / B * tw;
    if (s_rhs_chain > 0) M_rhs_chain += (double)s_rhs_chain / B * tw;
    if (s_lhs > 0 && s_rhs > 0) {
      M_lhs += (double)s_lhs / B * tw;
      M_rhs += (double)s_rhs / B * tw;
      if (s_pw > 0) M_pw += (double)s_pw / B * tw;
    } else if (s_lhs > 0) {
      M_lhs += (double)s_lhs / B * tw;
      if (s_pw > 0) M_pw += (double)s_pw / B * tw;
    } else if (s_rhs > 0) {
      M_rhs += (double)s_rhs / B * tw;
      if (s_pw > 0) M_pw += (double)s_pw / B * tw;
    } else if (s_pw > 0) {
      M_pw += (double)s_pw / B * tw;
    }
    // If only chain-head slots contributed (no regular), that's fine — nothing
    // more to add. If we fell through with all zero, tensor is fully chain-handled.
  }
  // Per-step chain IO: LHS-chain Head's RHS and RHS-chain Head's LHS —
  // fold into the respective per-tile per-step totals at ks=0.
  M_rhs += M_rhs_chain_up;
  M_lhs += M_lhs_chain_up;
  if (assume_retain_out) {
    M_evict = 0.0;
  } else {
    // Weight eviction by producing op's tile coverage for heterogeneous dims.
    M_evict = 0;
    for (size_t op : ops)
      for (size_t t : p.ops[op].outputs)
        if (ts.outputs.count(t))
          M_evict += h_eff * w_eff / B * OpWeight(op);
  }

  double cmp_ks0 = cmp_mm + cmp_pw;
  double cmp_mid = cmp_mm;
  double cmp_last = cmp_mm_last;  // last k-step compute (may differ when K%k != 0)

  // Tile categorization by snake mode (§8 symmetric).
  //   M_lhs_chain (LHS-chain Head LHS, ht-attached, h × K_up): reused where
  //     ht is stable (row-snake interior, col-snake zigzag col-start).
  //   M_rhs_chain (RHS-chain Head RHS, wt-attached, K_up × w): mirror —
  //     reused where wt is stable (row-snake zigzag row-start, col-snake
  //     interior).
  // Since both M_rhs and M_rhs_chain are wt-attached, they appear together
  // in the tile categorization; same for M_lhs and M_lhs_chain (ht-attached).
  double M_first, M_cat2, M_cat3;
  int64_t n_cat2, n_cat3;

  M_first = M_lhs + M_rhs + M_pw + M_lhs_chain + M_rhs_chain;

  if (snake_mode == 0) {  // Row-snake
    if (nk == 1) {
      M_cat2 = M_lhs + M_pw + M_lhs_chain;            // row-start: wt reused (RHS + M_rhs_chain)
      M_cat3 = M_rhs + M_pw + M_rhs_chain;            // interior: ht reused
    } else {
      M_cat2 = M_lhs + M_rhs + M_pw + M_lhs_chain + M_rhs_chain;
      M_cat3 = M_lhs + M_rhs + M_pw + M_rhs_chain;
    }
    n_cat2 = nth - 1;
    n_cat3 = nspatial - nth;
  } else if (snake_mode == 1) {  // Col-snake
    if (nk == 1) {
      M_cat2 = M_rhs + M_pw + M_rhs_chain;            // col-start: ht reused
      M_cat3 = M_lhs + M_pw + M_lhs_chain;            // interior: wt reused
    } else {
      M_cat2 = M_lhs + M_rhs + M_pw + M_lhs_chain + M_rhs_chain;
      M_cat3 = M_lhs + M_rhs + M_pw + M_lhs_chain;
    }
    n_cat2 = ntw - 1;
    n_cat3 = nspatial - ntw;
  } else {  // Raster
    if (ntw == 1 && nk == 1) {
      M_cat2 = M_lhs + M_pw + M_lhs_chain;            // ntw=1 → wt stable
    } else {
      M_cat2 = M_lhs + M_rhs + M_pw + M_lhs_chain + M_rhs_chain;
    }
    M_cat3 = (nk == 1) ? (M_rhs + M_pw + M_rhs_chain)
                       : (M_lhs + M_rhs + M_pw + M_rhs_chain);
    n_cat2 = nth - 1;
    n_cat3 = nspatial - nth;
  }

  double total = 0;
  if (nk == 1) {
    total += std::max(cmp_ks0, M_first + M_evict);
    total += n_cat2 * std::max(cmp_ks0, M_cat2 + M_evict);
    total += n_cat3 * std::max(cmp_ks0, M_cat3 + M_evict);
  } else {
    // ks=0
    total += std::max(cmp_ks0, M_first);
    total += n_cat2 * std::max(cmp_ks0, M_cat2);
    total += n_cat3 * std::max(cmp_ks0, M_cat3);
    // ks=1..nk-2: middle k-steps. Ops may finish at different nk_i boundaries.
    // Process ranges between nk_i boundaries where the active set is constant.
    // Track both active compute and active RHS IO (ops with smaller K finish
    // earlier and their RHS tensors stop being loaded).
    {
      double active_cmp = cmp_mm_all;  // starts with all MM ops active
      double active_rhs = 0;           // per-op RHS IO for active ops
      for (auto& m : mm_ops) active_rhs += m.rhs_io;
      int64_t prev_ks = 1;  // first middle k-step
      size_t mi = 0;  // index into sorted mm_ops

      while (mi < mm_ops.size() && mm_ops[mi].nk_i <= 1) {
        // Ops with nk_i <= 1 are only active at ks=0 (already counted)
        active_cmp -= mm_ops[mi].per_step;
        active_rhs -= mm_ops[mi].rhs_io;
        ++mi;
      }

      // Process each boundary where an op finishes
      while (mi < mm_ops.size()) {
        int64_t boundary_nk = mm_ops[mi].nk_i;
        if (boundary_nk >= nk) break;  // last k-step handled separately

        int64_t boundary_ks = boundary_nk - 1;  // ks where this op has its last step
        // Range [prev_ks, boundary_ks): all currently active ops at full rate
        if (boundary_ks > prev_ks && prev_ks <= nk - 2) {
          int64_t count = std::min(boundary_ks, nk - 1) - prev_ks;
          if (count > 0)
            total += (double)count * nspatial * std::max(active_cmp, M_lhs + active_rhs);
        }
        // At boundary_ks: ops finishing here use last_step instead of per_step
        if (boundary_ks >= 1 && boundary_ks <= nk - 2) {
          double cmp_at_boundary = active_cmp;
          // Collect all ops finishing at this boundary
          while (mi < mm_ops.size() && mm_ops[mi].nk_i == boundary_nk) {
            cmp_at_boundary -= mm_ops[mi].per_step;
            cmp_at_boundary += mm_ops[mi].last_step;
            active_cmp -= mm_ops[mi].per_step;
            active_rhs -= mm_ops[mi].rhs_io;
            ++mi;
          }
          total += nspatial * std::max(cmp_at_boundary, M_lhs + active_rhs);
        } else {
          // boundary is outside middle range, just remove ops
          while (mi < mm_ops.size() && mm_ops[mi].nk_i == boundary_nk) {
            active_cmp -= mm_ops[mi].per_step;
            active_rhs -= mm_ops[mi].rhs_io;
            ++mi;
          }
        }
        prev_ks = boundary_ks + 1;
      }
      // Remaining middle k-steps [prev_ks, nk-2] with surviving active ops
      if (prev_ks <= nk - 2) {
        int64_t count = (nk - 1) - prev_ks;
        if (count > 0)
          total += (double)count * nspatial * std::max(active_cmp, M_lhs + active_rhs);
      }
    }
    // ks=last: LHS + RHS + evict (only ops with nk_i == nk still active)
    double last_rhs = 0;
    for (auto& m : mm_ops)
      if (m.nk_i >= nk) last_rhs += m.rhs_io;
    total += nspatial * std::max(cmp_last, M_lhs + last_rhs + M_evict);
  }
  return total;
}


}  // namespace solver
