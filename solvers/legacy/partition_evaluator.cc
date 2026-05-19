// partition_evaluator.cc — OptimalGranularity implementation.
// See partition_evaluator.h for semantics and validation expectations.

#include "partition_evaluator.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace partition {

std::vector<int64_t> CeilStepCandidates(int64_t out_dim, int64_t cap) {
  std::set<int64_t> cands;

  // (1) Divisors of out_dim — padding-free tilings.
  for (int64_t d = 1; d * d <= out_dim; ++d) {
    if (out_dim % d == 0) {
      if (d <= cap) cands.insert(d);
      if (out_dim / d <= cap) cands.insert(out_dim / d);
    }
  }

  // (2) Powers of 2 — common hardware-friendly sizes.
  for (int64_t v = 1; v <= cap; v *= 2) cands.insert(v);

  // (3) Ceiling-step transition points — the value of w at which the tile
  // count along this dimension flips from T to T-1. Within a (T) bucket,
  // analytical cost is monotonic in w, so enumerating the smallest w per
  // bucket covers all possible optima. Smallest w with ⌈out_dim/w⌉ = T is
  // ⌈out_dim/T⌉.
  for (int64_t T = 1; T <= out_dim; ++T) {
    int64_t w = solver::CeilDiv(out_dim, T);
    if (w <= cap) cands.insert(w);
    if (w <= 1) break;  // further T only produce smaller w
  }

  // Safety bounds.
  cands.insert(1);
  cands.insert(std::min(out_dim, cap));

  return {cands.begin(), cands.end()};
}

OptimalGranResult OptimalGranularity(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const std::unordered_set<size_t>& retained_in,
    int64_t retained_in_size,
    const std::unordered_set<size_t>& retained_out,
    int64_t retained_out_size,
    const solver::DAG* dag) {

  OptimalGranResult best;  // feasible=false, cost=kInf

  if (ops.empty()) return best;

  auto [oW, oH] = solver::OutDims(p, ops);
  int64_t natW = p.native_granularity.width;
  int64_t natH = p.native_granularity.height;
  int64_t C = p.fast_memory_capacity;

  // maxK = largest K (= LHS width) across MatMul ops in this subgraph.
  int64_t maxK = 0;
  for (size_t op : ops)
    if (p.ops[op].op_type == "MatMul")
      maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);

  auto ts = solver::Classify(p, ops, dag);
  auto roles = solver::BuildRoles(p, ops, ts);
  int64_t n_outputs = (int64_t)ts.outputs.size();
  // Fan-out ephemerals occupy w*h slots (see mlsys::kFanoutEphemeralTakesSlot).
  int64_t n_fanout_eph = solver::CountFanoutEphemerals(p, ops, ts);

  // Milestone 0 retained_out semantics: all-or-nothing (matches
  // BestGranularity / ExhaustiveGranularity exactly).
  //
  // In the existing cost model, retain_out affects only eviction IO
  // (M_evict=0 in AnalyticalCost), NOT working set. The persistent storage
  // for a retained tensor is accounted for by the CONSUMER subgraph via
  // retained_in_size. The producer still uses per-tile output slots during
  // its own execution. Following this convention keeps Milestone 0 a strict
  // upgrade of BestGranularity's candidate generation with matching
  // numerics.
  //
  //   retained_out empty     → assume_retain_out=false, evict per tile.
  //   retained_out non-empty → assume_retain_out=true, no eviction IO.
  // (retained_out_size is accepted for API completeness; downstream
  // milestones may use it when introducing per-tensor retention control.)
  const bool assume_retain_out = !retained_out.empty();
  (void)retained_out_size;  // reserved for future per-tensor WS accounting

  auto ComputeWS = [&](int64_t w, int64_t h, int64_t k) -> int64_t {
    return solver::ComputeTileWS(p, roles, n_outputs, retained_in,
                                  w, h, k,
                                  /*extra_wh_slots=*/n_fanout_eph,
                                  retained_in_size);
  };

  // Candidate sets.
  const int64_t wCap = std::min(oW, natW);
  const int64_t hCap = std::min(oH, natH);
  auto w_cands = CeilStepCandidates(oW, wCap);
  const auto h_cands_base = CeilStepCandidates(oH, hCap);

  // Role-based fusion-validity guards (mirror evaluator exactly).
  auto mm_roles = solver::ClassifyMMRoles(p, ops, ts);
  solver::MMRoleCounts rc = solver::CountMMRoles(mm_roles, ops, p);

  // §8 reduction-axis accounting: N-chain has N reduction axes but unified
  // granularity slices exactly one (the `k` in [w,h,k]); one more is
  // absorbable via full-load on an input operand, so the split-K budget is
  // 2 ops. With k<K (nk>1), anything beyond a single Head+Tail pair forces
  // a second unsliced reduction into a non-ephemeral intermediate slab —
  // structurally impossible (§8.2 reduction-axis constraint).
  // is_head/is_tail/is_middle are direction-agnostic (LHS-chain ↔ RHS-chain
  // mirror), so this guard applies symmetrically.
  const bool split_k_infeasible =
      (rc.n_pw > 0 && rc.total_mm() > 0) ||
      rc.n_middle > 0 ||
      (rc.total_mm() >= 2 &&
       !rc.is_pure_chain_pair() && !rc.is_pure_standalone()) ||
      (rc.is_pure_standalone() && rc.total_mm() >= 2);

  const bool force_full_k = split_k_infeasible;
  std::vector<int64_t> k_cands;
  if (force_full_k && maxK > 0) {
    k_cands = {maxK};
  } else if (maxK == 0) {
    k_cands = {1};
  } else {
    k_cands = solver::KCandidates(maxK);
  }

  // §8.3: at k=K (nk=1) axis-counting collapses, but every intermediate
  // materializes at full size simultaneously. Require w=k=K uniform so
  // all reductions map to a single W×H grid — usually OOM in practice.
  // Extended (fix 1): PW+MM combos with 2+ MMs also require uniform K —
  // MM-PW-MM with non-uniform K forces k=max_K and wastes per-tile compute
  // on the smaller-K MM. Observed +68% on gqa_large.
  const bool uniform_k_required =
      rc.n_middle > 0 ||
      rc.total_mm() >= 3 ||
      (rc.total_mm() >= 2 && rc.n_pw > 0);
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
      return best;  // infeasible under evaluator rules
    }
    w_cands = {maxK};
  }

  auto TryConfig = [&](int64_t w, int64_t h, int64_t k) {
    if (w <= 0 || h <= 0 || k <= 0) return;
    // Cap at native grain (hard constraint; mirrors BestGranularity).
    if (w > natW) w = natW;
    if (h > natH) h = natH;

    const int64_t ws = ComputeWS(w, h, k);
    if (ws > C) return;

    const mlsys::Granularity g{w, h, k};
    for (int snake = 0; snake < 3; ++snake) {
      const double cost = solver::AnalyticalCost(
          p, ops, g, snake, retained_in, ts, roles, maxK,
          n_outputs, assume_retain_out);
      if (cost < best.cost) {
        best.gran = g;
        best.snake_mode = snake;
        best.cost = cost;
        best.ws = ws;
        best.feasible = true;
      }
    }
  };

  for (int64_t k : k_cands) {
    for (int64_t w : w_cands) {
      // Quick feasibility: if even h=1 overflows WS, skip this (w, k).
      if (ComputeWS(w, 1, k) > C) continue;

      // Binary search for max feasible h given (w, k).
      int64_t lo = 1, hi = hCap;
      while (lo < hi) {
        int64_t mid = lo + (hi - lo + 1) / 2;
        if (ComputeWS(w, mid, k) <= C) lo = mid; else hi = mid - 1;
      }
      const int64_t h_max = lo;

      // Try ceiling-step h candidates up to h_max.
      for (int64_t h : h_cands_base) {
        if (h <= h_max) TryConfig(w, h, k);
      }
      // Plus exact h_max boundary (may not be in ceiling-step set for some
      // pathological role mixes where h_max depends on non-monotonic terms).
      TryConfig(w, h_max, k);
    }
  }

  return best;
}

// ── OptimalGranularity cache (Milestone 4b) ──────────────────────────────────

OptimalGranResult OptimalGranularityCached(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const std::unordered_set<size_t>& retained_in,
    int64_t retained_in_size,
    const std::unordered_set<size_t>& retained_out,
    int64_t retained_out_size,
    const solver::DAG* dag,
    OptimalGranCache* cache) {
  if (!cache || !dag) {
    return OptimalGranularity(p, ops, retained_in, retained_in_size,
                               retained_out, retained_out_size, dag);
  }
  OptimalGranCache::Key key;
  key.sig = solver::SubgraphSig(p, ops, *dag);
  std::size_t rih = 0;
  for (size_t t : retained_in) rih ^= std::hash<std::size_t>()(t);
  std::size_t roh = 0;
  for (size_t t : retained_out) roh ^= std::hash<std::size_t>()(t);
  key.ri_hash = rih;
  key.ro_hash = roh;

  auto it = cache->table.find(key);
  if (it != cache->table.end()) { ++cache->hits; return it->second; }
  ++cache->misses;
  auto r = OptimalGranularity(p, ops, retained_in, retained_in_size,
                               retained_out, retained_out_size, dag);
  cache->table.emplace(std::move(key), r);
  return r;
}

// ── Phase 2: Partition Evaluator ─────────────────────────────────────────────

namespace {

// Compute subgraph-level predecessor sets: pred[s] = subgraphs whose output
// tensors feed into subgraph s's inputs.
std::vector<std::unordered_set<int>> BuildSubgraphPredecessors(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const solver::Partition& part) {
  const int S = (int)part.subgraphs.size();
  std::vector<int> op_to_sg(p.ops.size(), -1);
  for (int s = 0; s < S; ++s)
    for (size_t op : part.subgraphs[s]) op_to_sg[op] = s;

  std::vector<std::unordered_set<int>> preds(S);
  for (int s = 0; s < S; ++s) {
    for (size_t op : part.subgraphs[s]) {
      for (size_t t : p.ops[op].inputs) {
        int producer = dag.tensor_producer[t];
        if (producer < 0) continue;  // graph input
        int psg = op_to_sg[producer];
        if (psg >= 0 && psg != s) preds[s].insert(psg);
      }
    }
  }
  return preds;
}

// Σ(W × H) for a retention set, in elements.
int64_t TensorSetSize(const mlsys::Problem& p,
                      const std::unordered_set<size_t>& ts) {
  int64_t s = 0;
  for (size_t t : ts) s += p.tensors[t].width * p.tensors[t].height;
  return s;
}

// Enumerate retention subsets of R to try: {∅, R_full, each singleton}.
// Deduped when |R| ≤ 1 so we return at most max(1, |R|+1) distinct subsets.
std::vector<std::unordered_set<size_t>> RetentionSubsets(
    const std::vector<size_t>& R) {
  std::vector<std::unordered_set<size_t>> subsets;
  subsets.emplace_back();  // ∅
  if (R.empty()) return subsets;
  if (R.size() == 1) {
    subsets.push_back({R[0]});  // == R_full
    return subsets;
  }
  // R_full
  subsets.emplace_back(R.begin(), R.end());
  // singletons
  for (size_t t : R) subsets.push_back({t});
  return subsets;
}

struct PairCost {
  std::unordered_set<size_t> r;  // retention subset chosen
  double prev_cost = std::numeric_limits<double>::infinity();
  OptimalGranResult prev_gran;
  double next_cost = std::numeric_limits<double>::infinity();
  OptimalGranResult next_gran;
  double total = std::numeric_limits<double>::infinity();
};

// For a (prev, next) subgraph pair with fixed prev_ri (prev's retained_in),
// sweep retention subsets R = outputs(prev) ∩ inputs(next) and pick the
// retained_out = r that minimizes prev_cost_r + next_cost_r.
PairCost BestTransition(const mlsys::Problem& p,
                         const solver::DAG& dag,
                         const std::vector<size_t>& prev_ops,
                         const std::vector<size_t>& next_ops,
                         const std::unordered_set<size_t>& prev_ri,
                         int64_t prev_ri_size,
                         OptimalGranCache* cache) {
  PairCost best;
  const auto R_vec = solver::RetentionCandidates(p, prev_ops, next_ops, &dag);
  const auto subsets = RetentionSubsets(R_vec);
  std::unordered_set<size_t> empty_out;

  for (const auto& r : subsets) {
    int64_t r_size = TensorSetSize(p, r);

    auto prev_g = OptimalGranularityCached(p, prev_ops, prev_ri, prev_ri_size,
                                            r, r_size, &dag, cache);
    if (!prev_g.feasible) continue;

    auto next_g = OptimalGranularityCached(p, next_ops, r, r_size,
                                            empty_out, 0, &dag, cache);
    if (!next_g.feasible) continue;

    double total = prev_g.cost + next_g.cost;
    if (total < best.total) {
      best.r = r;
      best.prev_cost = prev_g.cost;
      best.prev_gran = prev_g;
      best.next_cost = next_g.cost;
      best.next_gran = next_g;
      best.total = total;
    }
  }
  return best;
}

// Max S for Held-Karp bitmask DP ordering. 2^15 × 15² ≈ 7M state×transition
// pairs — completes in ~tens of ms given our O(1) transition lookups.
constexpr int kHKMaxS = 15;

// Held-Karp bitmask DP over subgraph orderings (M3+ extension).
//
// Finds the ordering minimizing Σ base_cost[s] − Σ savings[ordering[i-1], ordering[i]]
// subject to the DAG constraint (each subgraph placed only after its
// predecessors). Uses a 2-stage decomposition:
//   1. precompute base_cost[s] (r=∅) and savings[a][b] (retention savings
//      with prev_ri=∅ — ignores long-range retention chains)
//   2. bitmask DP reconstructs optimal ordering
//
// After HK picks the ordering, the caller applies the greedy retention
// sweep along that fixed ordering to produce final per-step costs
// (which may differ slightly from the HK approximation cost, since
// greedy retention carries prev_ri forward).
//
// Returns true on success, false if S > kHKMaxS or the DAG is cyclic.
bool HKOrderSubgraphs(const mlsys::Problem& p,
                      const solver::DAG& dag,
                      const solver::Partition& part,
                      const std::vector<std::unordered_set<int>>& preds,
                      std::vector<int>* order_out,
                      OptimalGranCache* cache) {
  const int S = (int)part.subgraphs.size();
  if (S > kHKMaxS || S <= 1) return false;

  std::unordered_set<size_t> empty_set;

  // base_cost[s] = OptimalGranularity(s, ∅, ∅)
  std::vector<double> base_cost(S);
  std::vector<bool> base_ok(S, true);
  for (int s = 0; s < S; ++s) {
    auto r = OptimalGranularityCached(p, part.subgraphs[s], empty_set, 0,
                                       empty_set, 0, &dag, cache);
    if (!r.feasible) { base_ok[s] = false; base_cost[s] = solver::kInf; }
    else base_cost[s] = r.cost;
  }

  // savings[a][b] = max(0, base_cost[a] + base_cost[b] - BestTransition(a,b).total)
  std::vector<std::vector<double>> savings(S, std::vector<double>(S, 0.0));
  for (int a = 0; a < S; ++a) {
    if (!base_ok[a]) continue;
    for (int b = 0; b < S; ++b) {
      if (a == b || !base_ok[b]) continue;
      auto pc = BestTransition(p, dag, part.subgraphs[a], part.subgraphs[b],
                                empty_set, 0, cache);
      if (!std::isfinite(pc.total)) continue;
      double sv = base_cost[a] + base_cost[b] - pc.total;
      if (sv > 0) savings[a][b] = sv;
    }
  }

  // preds_mask[s] = bitmask of subgraphs that must be placed before s.
  std::vector<uint32_t> preds_mask(S, 0);
  for (int s = 0; s < S; ++s)
    for (int q : preds[s]) preds_mask[s] |= (1u << q);

  // DP. dp[mask][last] = min cost to reach state (mask, last).
  //     parent[mask][last] = best prev subgraph to reconstruct.
  // Infeasible states hold kInf.
  const size_t M = 1u << S;
  std::vector<std::vector<double>> dp(M, std::vector<double>(S, solver::kInf));
  std::vector<std::vector<int>> parent(M, std::vector<int>(S, -1));

  // Base: single-subgraph masks (s is a source: preds_mask[s] == 0).
  for (int s = 0; s < S; ++s) {
    if (preds_mask[s] == 0 && base_ok[s]) {
      uint32_t mask = (1u << s);
      dp[mask][s] = base_cost[s];
    }
  }

  // Ascending bit-count order: easier reasoning, and each mask evaluated
  // after all smaller masks.
  std::vector<std::vector<uint32_t>> by_bits(S + 1);
  for (uint32_t m = 0; m < M; ++m)
    by_bits[__builtin_popcount(m)].push_back(m);

  for (int bits = 1; bits < S; ++bits) {
    for (uint32_t mask : by_bits[bits]) {
      for (int last = 0; last < S; ++last) {
        if (!(mask & (1u << last))) continue;
        double cur = dp[mask][last];
        if (!std::isfinite(cur)) continue;

        // Extend to each valid next subgraph.
        for (int nxt = 0; nxt < S; ++nxt) {
          if (mask & (1u << nxt)) continue;
          if (!base_ok[nxt]) continue;
          if ((preds_mask[nxt] & mask) != preds_mask[nxt]) continue;  // preds not ready

          double nxt_cost = cur + base_cost[nxt] - savings[last][nxt];
          uint32_t nmask = mask | (1u << nxt);
          if (nxt_cost < dp[nmask][nxt]) {
            dp[nmask][nxt] = nxt_cost;
            parent[nmask][nxt] = last;
          }
        }
      }
    }
  }

  // Final mask: all subgraphs placed.
  uint32_t full = (1u << S) - 1;
  int best_last = -1;
  double best_cost = solver::kInf;
  for (int s = 0; s < S; ++s) {
    if (dp[full][s] < best_cost) { best_cost = dp[full][s]; best_last = s; }
  }
  if (best_last < 0 || !std::isfinite(best_cost)) return false;

  // Reconstruct order (reverse path).
  std::vector<int> rev_order;
  int cur = best_last;
  uint32_t cur_mask = full;
  while (cur >= 0) {
    rev_order.push_back(cur);
    int prev = parent[cur_mask][cur];
    cur_mask ^= (1u << cur);
    cur = prev;
  }
  std::reverse(rev_order.begin(), rev_order.end());
  if ((int)rev_order.size() != S) return false;
  *order_out = std::move(rev_order);
  return true;
}

// Apply a fixed ordering (from HK) to produce OrderedSteps with retention
// sweep. Mirrors the greedy retention logic but with the ordering
// predetermined (no choice of next subgraph).
bool ApplyOrderingWithRetention(const mlsys::Problem& p,
                                 const solver::DAG& dag,
                                 const solver::Partition& part,
                                 const std::vector<int>& order,
                                 Evaluation* eval,
                                 OptimalGranCache* cache) {
  const int S = (int)order.size();
  std::unordered_set<size_t> empty_set;
  eval->ordering.clear();
  eval->ordering.reserve(S);

  // First step: find best retained_out by looking ahead to order[1] (if any).
  {
    int c = order[0];
    OptimalGranResult first_gran;
    std::unordered_set<size_t> first_ro;
    double first_cost = solver::kInf;

    if (S == 1) {
      auto g = OptimalGranularityCached(p, part.subgraphs[c], empty_set, 0,
                                         empty_set, 0, &dag, cache);
      if (!g.feasible) return false;
      first_gran = g; first_cost = g.cost;
    } else {
      int c2 = order[1];
      auto pc = BestTransition(p, dag, part.subgraphs[c], part.subgraphs[c2],
                                empty_set, 0, cache);
      if (!std::isfinite(pc.total)) return false;
      first_gran = pc.prev_gran;
      first_cost = pc.prev_cost;
      first_ro = pc.r;
    }
    OrderedStep step;
    step.orig_index = c;
    step.retained_out = first_ro;
    step.gran = first_gran;
    step.final_cost = first_cost;
    eval->ordering.push_back(std::move(step));
  }

  // Subsequent steps: propagate retention forward along the fixed order.
  for (int i = 1; i < S; ++i) {
    int c = order[i];
    auto& prev_step = eval->ordering.back();
    int64_t prev_ri_size = 0;
    for (size_t t : prev_step.retained_in)
      prev_ri_size += p.tensors[t].width * p.tensors[t].height;

    // If this is the last step: prev's retained_out is already fixed by
    // the previous step's decision; just compute c's cost under it.
    const auto& c_ops = part.subgraphs[c];
    const auto& prev_ops = part.subgraphs[prev_step.orig_index];

    OptimalGranResult c_gran;
    double c_cost = solver::kInf;
    std::unordered_set<size_t> c_ro;

    if (i + 1 == S) {
      // No next subgraph: c_ro = ∅; just evaluate c with retained_in = prev.ro.
      int64_t r_size = 0;
      for (size_t t : prev_step.retained_out)
        r_size += p.tensors[t].width * p.tensors[t].height;
      auto g = OptimalGranularityCached(p, c_ops, prev_step.retained_out,
                                         r_size, empty_set, 0, &dag, cache);
      if (!g.feasible) return false;
      c_gran = g; c_cost = g.cost;
    } else {
      // Revise prev's retained_out based on next transition (c → order[i+1]).
      int next = order[i+1];
      // First: c's cost given prev's retained_out locked (can't change anymore).
      // Then: pick c's retained_out via BestTransition(c, next, prev_ro).
      auto pc = BestTransition(p, dag, c_ops, part.subgraphs[next],
                                prev_step.retained_out,
                                (int64_t)[&]() {
                                  int64_t sz = 0;
                                  for (size_t t : prev_step.retained_out)
                                    sz += p.tensors[t].width * p.tensors[t].height;
                                  return sz;
                                }(), cache);
      if (!std::isfinite(pc.total)) return false;
      c_gran = pc.prev_gran;
      c_cost = pc.prev_cost;
      c_ro = pc.r;
    }

    OrderedStep step;
    step.orig_index = c;
    step.retained_in = prev_step.retained_out;
    step.retained_out = c_ro;
    step.gran = c_gran;
    step.final_cost = c_cost;
    eval->ordering.push_back(std::move(step));
  }

  double total = 0;
  for (const auto& s : eval->ordering) total += s.final_cost;
  eval->cost = total;
  return true;
}

// Greedy ordering + retention. Fills eval.ordering and eval.cost.
// Returns true on success, false if a dead-end is reached (partition has
// no topologically valid ordering, or retention sweep finds no feasible
// gran for some step).
bool GreedyOrderWithRetention(const mlsys::Problem& p,
                               const solver::DAG& dag,
                               const solver::Partition& part,
                               Evaluation* eval,
                               OptimalGranCache* cache) {
  const int S = (int)part.subgraphs.size();
  if (S == 0) return false;

  auto preds = BuildSubgraphPredecessors(p, dag, part);

  std::vector<bool> placed(S, false);
  eval->ordering.clear();
  eval->ordering.reserve(S);

  auto IsReady = [&](int s) {
    if (placed[s]) return false;
    for (int q : preds[s]) if (!placed[q]) return false;
    return true;
  };

  std::unordered_set<size_t> empty_set;

  // ── Small-S fast path: Held-Karp bitmask DP ─────────────────────────
  if (S <= kHKMaxS && S >= 2) {
    std::vector<int> hk_order;
    if (HKOrderSubgraphs(p, dag, part, preds, &hk_order, cache)) {
      if (ApplyOrderingWithRetention(p, dag, part, hk_order, eval, cache))
        return true;
      // HK failed during retention sweep — fall through to greedy.
      eval->ordering.clear();
    }
  }

  // Degenerate case: S == 1 — just evaluate the single subgraph at base.
  if (S == 1) {
    const auto& ops = part.subgraphs[0];
    auto g = OptimalGranularityCached(p, ops, empty_set, 0, empty_set, 0,
                                       &dag, cache);
    if (!g.feasible) { eval->first_infeasible_sg = 0; return false; }
    OrderedStep step;
    step.orig_index = 0;
    step.gran = g;
    step.final_cost = g.cost;
    eval->ordering.push_back(std::move(step));
    eval->cost = g.cost;
    return true;
  }

  // ── Step 0 + 1 via 1-step lookahead ──────────────────────────────────
  // Pick (first, second, retained_out_from_first) that minimizes
  //   first_cost(ri=∅, ro=r) + second_cost(ri=r, ro=∅).
  //
  // For each first candidate c, compute the set of c2 candidates that
  // would be ready after c is placed (predecessors ⊆ {c}). Try each
  // (c, c2, r) and pick the minimum 2-step total.
  int best_first = -1, best_second = -1;
  std::unordered_set<size_t> best_r;
  double best_total = std::numeric_limits<double>::infinity();
  OptimalGranResult best_first_gran, best_second_gran;
  double best_first_cost = 0, best_second_cost = 0;

  // Candidates for first step: subgraphs with empty predecessor set.
  std::vector<int> sources;
  for (int s = 0; s < S; ++s) if (preds[s].empty()) sources.push_back(s);
  if (sources.empty()) return false;  // cycle

  for (int c : sources) {
    // Compute ready_after_c: subgraphs whose preds ⊆ {c}.
    std::vector<int> ready_after;
    for (int c2 = 0; c2 < S; ++c2) {
      if (c2 == c) continue;
      if (preds[c2].size() > 1) continue;
      if (preds[c2].size() == 1 && *preds[c2].begin() != c) continue;
      ready_after.push_back(c2);
    }

    if (ready_after.empty()) {
      // No successor available after c — degenerate (S ≥ 2 but this
      // source has no follower). Fall back to evaluating c alone.
      auto g = OptimalGranularityCached(p, part.subgraphs[c], empty_set, 0,
                                         empty_set, 0, &dag, cache);
      if (!g.feasible) continue;
      double total_only = g.cost;
      if (total_only < best_total) {
        best_first = c; best_second = -1;
        best_r.clear();
        best_first_gran = g;
        best_first_cost = g.cost;
        best_total = total_only;
      }
      continue;
    }

    for (int c2 : ready_after) {
      auto pc = BestTransition(p, dag, part.subgraphs[c], part.subgraphs[c2],
                                empty_set, 0, cache);
      if (!std::isfinite(pc.total)) continue;
      if (pc.total < best_total) {
        best_first = c;
        best_second = c2;
        best_r = pc.r;
        best_first_gran = pc.prev_gran;
        best_second_gran = pc.next_gran;
        best_first_cost = pc.prev_cost;
        best_second_cost = pc.next_cost;
        best_total = pc.total;
      }
    }
  }

  if (best_first < 0) return false;

  // Commit first step.
  {
    OrderedStep step;
    step.orig_index = best_first;
    step.retained_in.clear();
    step.retained_out = best_r;
    step.gran = best_first_gran;
    step.final_cost = best_first_cost;
    eval->ordering.push_back(std::move(step));
    placed[best_first] = true;
  }

  if (best_second >= 0) {
    OrderedStep step;
    step.orig_index = best_second;
    step.retained_in = best_r;
    step.retained_out.clear();
    step.gran = best_second_gran;
    step.final_cost = best_second_cost;  // provisional; may update below
    eval->ordering.push_back(std::move(step));
    placed[best_second] = true;
  }

  // ── Steps 2 .. S-1: greedy with retention sweep ──────────────────────
  while ((int)eval->ordering.size() < S) {
    OrderedStep& prev_step = eval->ordering.back();
    const auto& prev_ops = part.subgraphs[prev_step.orig_index];
    const auto& prev_ri = prev_step.retained_in;
    int64_t prev_ri_size = TensorSetSize(p, prev_ri);

    // Collect ready candidates.
    std::vector<int> ready;
    for (int s = 0; s < S; ++s) if (IsReady(s)) ready.push_back(s);
    if (ready.empty()) return false;  // dead-end

    int best_c = -1;
    PairCost best_pc;

    for (int c : ready) {
      auto pc = BestTransition(p, dag, prev_ops, part.subgraphs[c],
                                prev_ri, prev_ri_size, cache);
      if (!std::isfinite(pc.total)) continue;
      if (pc.total < best_pc.total) {
        best_c = c;
        best_pc = pc;
      }
    }

    if (best_c < 0) {
      // No feasible transition from prev to any ready candidate.
      eval->first_infeasible_sg = prev_step.orig_index;
      return false;
    }

    // Commit: revise prev's final_cost / retained_out / gran to the winning
    // values; append best_c with retained_in = r, provisional final_cost.
    prev_step.final_cost = best_pc.prev_cost;
    prev_step.retained_out = best_pc.r;
    prev_step.gran = best_pc.prev_gran;

    OrderedStep step;
    step.orig_index = best_c;
    step.retained_in = best_pc.r;
    step.retained_out.clear();
    step.gran = best_pc.next_gran;
    step.final_cost = best_pc.next_cost;
    eval->ordering.push_back(std::move(step));
    placed[best_c] = true;
  }

  // Total cost is Σ final_cost across ordered steps.
  double total = 0;
  for (const auto& step : eval->ordering) total += step.final_cost;
  eval->cost = total;
  return true;
}

}  // namespace

Evaluation EvaluatePartitionCached(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const solver::Partition& part,
    OptimalGranCache* cache,
    bool compute_baseline) {

  Evaluation eval;
  const size_t S = part.subgraphs.size();
  eval.per_subgraph_cost.assign(S, solver::kInf);
  eval.per_subgraph_gran.assign(S, OptimalGranResult{});

  // (1) Complete cover check: every op appears in exactly one subgraph.
  const size_t N = p.ops.size();
  std::vector<int> cover(N, 0);
  for (const auto& ops : part.subgraphs)
    for (size_t op : ops)
      if (op < N) ++cover[op];
  eval.complete_cover = true;
  for (size_t i = 0; i < N; ++i) {
    if (cover[i] != 1) { eval.complete_cover = false; break; }
  }

  // (2) DAG convexity check: every subgraph is topologically convex.
  eval.all_convex = true;
  for (const auto& ops : part.subgraphs) {
    if (ops.size() < 2) continue;
    if (!solver::IsDAGConvex(dag, ops)) { eval.all_convex = false; break; }
  }

  if (!eval.complete_cover || !eval.all_convex) return eval;

  // (2b) Heterogeneous ban (experimental): every op in a subgraph must share
  // the same output (W, H). Current evaluator's ephemeral classification
  // ignores producer-consumer extent compatibility, which silently allows
  // physically-infeasible schedules with heterogeneous fusion. Gated by
  // env var MLSYS_BAN_HETEROGENEOUS=1 so we can A/B cleanly.
  static const bool ban_hetero = [] {
    const char* e = std::getenv("MLSYS_BAN_HETEROGENEOUS");
    return e && std::string(e) == "1";
  }();
  if (ban_hetero) {
    for (size_t s = 0; s < S; ++s) {
      const auto& ops = part.subgraphs[s];
      if (ops.size() < 2) continue;
      int64_t w0 = 0, h0 = 0;
      bool first = true;
      for (size_t op : ops) {
        int64_t w = 0, h = 0;
        for (size_t t : p.ops[op].outputs) {
          w = std::max(w, p.tensors[t].width);
          h = std::max(h, p.tensors[t].height);
        }
        if (first) { w0 = w; h0 = h; first = false; }
        else if (w != w0 || h != h0) {
          eval.first_infeasible_sg = (int)s;
          return eval;  // infeasible under ban
        }
      }
    }
  }

  // (3) Empty-subgraph check (always).
  for (size_t s = 0; s < S; ++s) {
    if (part.subgraphs[s].empty()) {
      eval.first_infeasible_sg = (int)s;
      return eval;
    }
  }

  // (3b) Per-subgraph base costs (no retention) — diagnostic baseline.
  // Skipped in SA fast path (compute_baseline=false) to save S ×
  // OptimalGranularity calls on large partitions.
  std::unordered_set<size_t> empty_set;
  if (compute_baseline) {
    for (size_t s = 0; s < S; ++s) {
      auto r = OptimalGranularityCached(p, part.subgraphs[s], empty_set, 0,
                                         empty_set, 0, &dag, cache);
      eval.per_subgraph_gran[s] = r;
      eval.per_subgraph_cost[s] = r.cost;
      if (!r.feasible) {
        eval.first_infeasible_sg = (int)s;
        return eval;
      }
    }
  }

  // (4) Greedy ordering + retention sweep. Populates eval.ordering and
  // eval.cost with retention savings applied.
  if (!GreedyOrderWithRetention(p, dag, part, &eval, cache)) return eval;

  eval.feasible = true;
  return eval;
}

Evaluation EvaluatePartition(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const solver::Partition& part) {
  // Default: with diagnostic baseline, no cache.
  return EvaluatePartitionCached(p, dag, part, /*cache=*/nullptr,
                                  /*compute_baseline=*/true);
}

solver::Partition ApplyOrderingToPartition(
    const solver::Partition& input,
    const Evaluation& eval) {
  if (eval.ordering.empty()) return input;
  solver::Partition out;
  out.subgraphs.reserve(eval.ordering.size());
  for (const auto& step : eval.ordering) {
    if (step.orig_index < 0 ||
        (size_t)step.orig_index >= input.subgraphs.size())
      return input;  // malformed — fail safe
    out.subgraphs.push_back(input.subgraphs[step.orig_index]);
  }
  return out;
}

// ── Milestone 5: Recomputation post-pass ─────────────────────────────────────

RecompResult ApplyRecomputation(const mlsys::Problem& p,
                                 const solver::DAG& dag,
                                 const solver::Partition& part,
                                 const Evaluation& baseline,
                                 OptimalGranCache* cache) {
  RecompResult res;
  res.partition = part;
  res.evaluation = baseline;
  res.applied = false;

  // Delegate to solver::RecomputationPass (adds producer ops to subgraphs
  // when it locally improves BestGranularity cost). Pass a copy since it
  // mutates.
  solver::Partition rewritten = part;
  rewritten = solver::RecomputationPass(p, dag, std::move(rewritten),
                                         /*assume_retain_out=*/false);

  // If nothing changed structurally, skip re-evaluation.
  bool changed = rewritten.subgraphs.size() != part.subgraphs.size();
  if (!changed) {
    for (size_t s = 0; s < rewritten.subgraphs.size(); ++s) {
      auto a = rewritten.subgraphs[s];
      auto b = part.subgraphs[s];
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      if (a != b) { changed = true; break; }
    }
  }
  if (!changed) return res;

  // Re-evaluate rewritten partition under our cost model.
  auto eval_new = EvaluatePartitionCached(p, dag, rewritten, cache,
                                           /*compute_baseline=*/false);
  if (!eval_new.feasible) return res;

  if (eval_new.cost < baseline.cost - 1e-6) {
    res.partition = std::move(rewritten);
    res.evaluation = std::move(eval_new);
    res.applied = true;
  }
  return res;
}

// ── Phase 1-A: Initial Pool ──────────────────────────────────────────────────

namespace {

// Try one (generator, label) combination. Evaluate and push onto pool.
// Shares a cache across all pool candidates — dominant speedup on large
// partitions where per-subgraph OptimalGranularity dominates. Logs
// per-candidate eval time for Phase 1-A profiling.
void TryCandidate(const mlsys::Problem& p, const solver::DAG& dag,
                  solver::Partition part, const std::string& label,
                  std::vector<PoolEntry>* pool,
                  OptimalGranCache* cache) {
  if (part.subgraphs.empty()) return;
  auto t0 = std::chrono::high_resolution_clock::now();
  Evaluation eval = EvaluatePartitionCached(p, dag, part, cache,
                                             /*compute_baseline=*/false);
  double ms = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - t0).count();
  // Stash timing in label via stderr — the PoolEntry doesn't carry it.
  if (ms > 20.0) {
    std::fprintf(stderr, "    [pool] %s: %.0f ms (%zu sg)\n",
                 label.c_str(), ms, part.subgraphs.size());
  }
  pool->push_back({label, std::move(part), std::move(eval)});
}

}  // namespace

std::vector<PoolEntry> GenerateInitialPool(
    const mlsys::Problem& p,
    const solver::DAG& dag_default,
    const std::vector<uint32_t>& rng_seeds,
    double time_budget_s) {
  using clock = std::chrono::high_resolution_clock;
  auto pool_t0 = clock::now();
  auto budget_exceeded = [&]() {
    if (time_budget_s <= 0) return false;
    double el = std::chrono::duration<double>(clock::now() - pool_t0).count();
    return el >= time_budget_s;
  };
  std::vector<PoolEntry> pool;
  OptimalGranCache cache;  // shared across all candidates — huge speedup

  auto time_gen = [&](auto label_base, auto gen_fn, const solver::DAG& use_dag) {
    if (budget_exceeded()) return;
    auto t0 = clock::now();
    auto part = gen_fn();
    double gen_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    if (gen_ms > 20.0)
      std::fprintf(stderr, "    [pool-gen] %s: %.0f ms\n",
                   label_base.c_str(), gen_ms);
    TryCandidate(p, use_dag, std::move(part), label_base, &pool, &cache);
  };

  // Generators are ordered roughly fast-first so early budget exhaustion
  // loses the slow/marginal candidates rather than the fast/important ones.
  // Priorities inferred from 5 released BMs:
  //   Singleton (fast baseline) → Agglomerative (won BM-5) →
  //   MaxFusion → DP-BFS → DP-MaxOut (won BM-13) → AntiChain →
  //   DP-DFS (won BM-17 but slow) → DP-Rand (rarely dominant)

  // --- Singleton baseline -------------------------------------------------
  if (!budget_exceeded()) {
    solver::Partition part;
    for (size_t op : dag_default.topo_order)
      part.subgraphs.push_back({op});
    TryCandidate(p, dag_default, std::move(part), "Singleton", &pool, &cache);
  }

  // --- Agglomerative (bottom-up pair merging) — fast, often wins ----------
  for (bool retain : {false, true}) {
    std::string label = "Agglomerative";
    if (retain) label += "-R";
    time_gen(label, [&]() {
      return solver::AgglomerativePartition(p, dag_default, retain);
    }, dag_default);
  }

  // --- MaxFusion (top-down packing) — fast -------------------------------
  for (bool sink_first : {true, false}) {
    for (bool retain : {false, true}) {
      std::string label = std::string("MaxFusion-") +
                          (sink_first ? "Sink" : "Source");
      if (retain) label += "-R";
      time_gen(label, [&]() {
        return solver::MaxFusionPartition(p, dag_default, sink_first, retain);
      }, dag_default);
    }
  }

  // --- DP × {BFS, MaxOut} — often winning generators ----------------------
  // Split DP topos into "high priority" (BFS, MaxOut) first and the rest
  // after the main utility generators, so slow ones (DFS, Rand) lose to
  // time cap first.
  struct TopoSpec { std::string name; solver::DAG dag; };
  std::vector<TopoSpec> topos_hi;
  topos_hi.push_back({"BFS", solver::BuildDAG(p)});
  topos_hi.push_back({"MaxOut", solver::BuildDAG_MaxOutput(p)});
  for (const auto& t : topos_hi) {
    for (bool retain : {false, true}) {
      std::string label = "DP-" + t.name + (retain ? "-R" : "");
      time_gen(label, [&]() {
        auto init = solver::InitialPartition(t.dag);
        return solver::DPPartition(p, t.dag, init, retain);
      }, t.dag);
    }
  }

  // --- AntiChain (horizontal/level packing) -------------------------------
  for (bool retain : {false, true}) {
    std::string label = "AntiChain";
    if (retain) label += "-R";
    time_gen(label, [&]() {
      return solver::AntiChainPartition(p, dag_default, retain);
    }, dag_default);
  }

  // --- DP-DFS (slow on some BMs but won BM-17) ---------------------------
  {
    solver::DAG dag_dfs = solver::BuildDAG_DFS(p);
    for (bool retain : {false, true}) {
      std::string label = std::string("DP-DFS") + (retain ? "-R" : "");
      time_gen(label, [&]() {
        auto init = solver::InitialPartition(dag_dfs);
        return solver::DPPartition(p, dag_dfs, init, retain);
      }, dag_dfs);
    }
  }

  // --- DP-Rand (rarely dominant, last priority) --------------------------
  for (uint32_t seed : rng_seeds) {
    if (budget_exceeded()) break;
    solver::DAG dag_rand = solver::BuildDAG_Random(p, seed);
    for (bool retain : {false, true}) {
      std::string label = "DP-Rand" + std::to_string(seed) + (retain ? "-R" : "");
      time_gen(label, [&]() {
        auto init = solver::InitialPartition(dag_rand);
        return solver::DPPartition(p, dag_rand, init, retain);
      }, dag_rand);
    }
  }

  std::fprintf(stderr, "  [pool cache] hits=%lld misses=%lld hit_rate=%.1f%%\n",
               (long long)cache.hits, (long long)cache.misses,
               cache.hit_rate() * 100.0);

  // Sort: feasible first (by cost ascending), infeasible after (in order).
  std::sort(pool.begin(), pool.end(),
            [](const PoolEntry& a, const PoolEntry& b) {
              if (a.evaluation.feasible != b.evaluation.feasible)
                return a.evaluation.feasible;  // feasible < infeasible
              return a.evaluation.cost < b.evaluation.cost;
            });
  return pool;
}

}  // namespace partition
