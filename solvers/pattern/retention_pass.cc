// retention_pass.cc — Greedy multi-step retention.

#include "retention_pass.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pattern {

namespace {

// Compute per-subgraph retained_size from a Solution's tensors_to_retain.
// sg[i]'s retained_in set (as seen by ITS execution) is what sg[i-1]
// retained, carried to sg[i] and re-declared in sg[i]'s tensors_to_retain
// (if sg[i] itself keeps it alive for sg[i+1]).
//
// For purposes of budget tracking: sg[i]'s capacity available for WS =
// capacity - retained_in[i], where retained_in[i] = size of tensors_to_retain
// listed at sg[i-1]. Graph inputs cannot be retained per spec.
std::vector<int64_t> ComputeRetainedSizes(const mlsys::Problem& p,
                                           const mlsys::Solution& sol) {
  int S = static_cast<int>(sol.subgraphs.size());
  std::vector<int64_t> retained_into(S, 0);
  if (S == 0) return retained_into;
  for (int i = 0; i < S - 1; ++i) {
    for (size_t t : sol.subgraphs[i].tensors_to_retain) {
      retained_into[i + 1] += p.tensors[t].width * p.tensors[t].height;
    }
  }
  return retained_into;
}

}  // namespace

RetentionResult RetentionPass(const mlsys::Problem& p,
                               const solver::DAG& dag,
                               mlsys::Solution sol,
                               double deadline_s) {
  RetentionResult r;
  r.solution = sol;
  const int S = static_cast<int>(sol.subgraphs.size());
  if (S <= 1) return r;
  auto t_start = std::chrono::steady_clock::now();

  // Baseline cost (may fail if solution is invalid; guard with Evaluate).
  auto gt0 = mlsys::Evaluate(p, sol);
  if (gt0.ok()) r.cost_before = *gt0;

  // Map op → sg_idx.
  std::unordered_map<size_t, int> op_to_sg;
  for (int i = 0; i < S; ++i) {
    for (size_t op : sol.subgraphs[i].ops) op_to_sg[op] = i;
  }

  // Producer sg and consumer sgs per tensor.
  struct TensorInfo {
    int producer_sg = -1;  // -1 if graph input
    std::vector<int> consumer_sgs;  // distinct, ascending
    int64_t size = 0;
  };
  const size_t num_tensors = p.tensors.size();
  std::vector<TensorInfo> info(num_tensors);
  for (size_t t = 0; t < num_tensors; ++t) {
    info[t].size = p.tensors[t].width * p.tensors[t].height;
  }
  for (int i = 0; i < S; ++i) {
    for (size_t op : sol.subgraphs[i].ops) {
      for (size_t t : p.ops[op].outputs) {
        info[t].producer_sg = i;
      }
    }
  }
  for (int i = 0; i < S; ++i) {
    std::unordered_set<size_t> seen;
    for (size_t op : sol.subgraphs[i].ops) {
      for (size_t t : p.ops[op].inputs) {
        if (info[t].producer_sg == i) continue;  // internal, ephemeral
        if (seen.count(t)) continue;
        seen.insert(t);
        if (info[t].consumer_sgs.empty() ||
            info[t].consumer_sgs.back() != i) {
          info[t].consumer_sgs.push_back(i);
        }
      }
    }
  }

  const int64_t capacity = p.fast_memory_capacity;
  const double bw = p.slow_memory_bandwidth;
  auto retained_into = ComputeRetainedSizes(p, r.solution);

  // Currently-retained status per (tensor, sg_idx) — so we know which
  // tensors are already in tensors_to_retain at each boundary.
  auto is_retained_at = [&](size_t t, int sg_i) {
    if (sg_i < 0 || sg_i >= S) return false;
    for (size_t tt : r.solution.subgraphs[sg_i].tensors_to_retain)
      if (tt == t) return true;
    return false;
  };

  // Build candidates. For each tensor t with a producer sg P and at least
  // one consumer sg > P, compute the retention span [P .. last_consumer-1]
  // (boundaries where t must be re-declared in tensors_to_retain).
  struct Cand {
    size_t t;
    int producer_sg;
    int last_consumer_sg;
    int64_t size;
    int n_consumers_covered;  // external consumers reached by retention
    double ratio;             // savings / (size × span)
  };
  std::vector<Cand> cands;
  for (size_t t = 0; t < num_tensors; ++t) {
    const auto& ti = info[t];
    if (ti.producer_sg < 0) continue;  // graph input, cannot retain
    if (ti.consumer_sgs.empty()) continue;
    int last = ti.consumer_sgs.back();
    if (last <= ti.producer_sg) continue;
    // Saved reloads: every consumer in range reads from fast memory instead
    // of slow memory. Plus one saved eviction from the producer.
    int n_cons = 0;
    for (int sg_i : ti.consumer_sgs)
      if (sg_i > ti.producer_sg) ++n_cons;
    int span = last - ti.producer_sg;  // intermediate sg count + 1
    double savings = (n_cons + 1) * static_cast<double>(ti.size) / bw;
    double cost = static_cast<double>(ti.size) * span;
    double ratio = savings / std::max<double>(cost, 1.0);
    cands.push_back({t, ti.producer_sg, last, ti.size, n_cons, ratio});
  }

  std::sort(cands.begin(), cands.end(),
            [](const Cand& a, const Cand& b) { return a.ratio > b.ratio; });
  (void)capacity;  // used for future feasibility pruning; currently trusted to Evaluate

  // Per-candidate try-keep-or-revert loop. Each candidate:
  //   1. Save current state.
  //   2. Add retentions at required sg boundaries.
  //   3. Re-optimize gran for sgs whose carried-in retention set grew.
  //   4. Evaluate. If cost improved, keep; else revert this candidate only.
  //
  // O(|cands|) Evaluate calls. On N ≤ 200, Evaluate is ms-fast.
  double cur_cost = r.cost_before;
  int accepted = 0, rejected = 0;

  // Re-optimize gran only for sgs in [from_sg, to_sg] range (affected by a
  // retention addition). carries_in is the retained set at boundary sg[from_sg].
  auto reopt_range = [&](mlsys::Solution& s, int from_sg, int to_sg) -> bool {
    // Compute carried-in at from_sg from earlier tensors_to_retain state.
    std::unordered_set<size_t> carried;
    if (from_sg > 0) {
      for (size_t t : s.subgraphs[from_sg - 1].tensors_to_retain)
        carried.insert(t);
    }
    int end = std::min(to_sg + 1, S);
    for (int i = from_sg; i < end; ++i) {
      auto& sg = s.subgraphs[i];
      int64_t extra_sz = 0;
      auto ts_cur = solver::Classify(p, sg.ops, &dag);
      for (size_t t : carried) {
        if (!ts_cur.inputs.count(t))
          extra_sz += p.tensors[t].width * p.tensors[t].height;
      }
      auto cfg = solver::BestGranularity(p, sg.ops, carried, extra_sz, &dag,
                                          /*assume_retain_out=*/true);
      if (cfg.gran.width <= 0 || !std::isfinite(cfg.cost)) return false;
      sg.granularity = cfg.gran;
      if (!cfg.traversal.empty()) sg.traversal_order = cfg.traversal;
      else sg.traversal_order = std::nullopt;
      carried.clear();
      for (size_t t : sg.tensors_to_retain) carried.insert(t);
    }
    return true;
  };

  for (const auto& c : cands) {
    // Deadline check.
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    if (elapsed > deadline_s) break;

    // Per spec #34/#52, only the producing subgraph may re-retain a tensor;
    // multi-hop carry requires the intermediate subgraph to recompute the
    // producing op. Until recomputation is implemented here, limit this
    // pass to single-hop (adjacent) retentions — multi-hop candidates
    // would be constructed but then rejected by the evaluator's
    // produced_here check, wasting an Evaluate call per candidate.
    if (c.last_consumer_sg - c.producer_sg > 1) continue;
    std::vector<int> to_add;
    for (int i = c.producer_sg; i < c.last_consumer_sg; ++i) {
      if (!is_retained_at(c.t, i)) to_add.push_back(i);
    }
    if (to_add.empty()) continue;

    // Save; apply; re-opt only affected range; evaluate; commit or revert.
    mlsys::Solution trial = r.solution;
    for (int i : to_add)
      trial.subgraphs[i].tensors_to_retain.push_back(c.t);
    if (!reopt_range(trial, c.producer_sg, c.last_consumer_sg)) {
      ++rejected; continue;
    }
    for (auto& sg : trial.subgraphs) sg.subgraph_latency = 0.0;
    auto gt_trial = mlsys::Evaluate(p, trial);
    if (!gt_trial.ok()) { ++rejected; continue; }
    if (*gt_trial < cur_cost) {
      r.solution = std::move(trial);
      cur_cost = *gt_trial;
      r.n_retentions_added += static_cast<int>(to_add.size());
      ++accepted;
    } else {
      ++rejected;
    }
  }

  r.cost_after = cur_cost;
  (void)accepted; (void)rejected;
  return r;
}

RefineResult GranularityRefinement(const mlsys::Problem& p,
                                    const solver::DAG& dag,
                                    mlsys::Solution sol,
                                    double deadline_s) {
  RefineResult r;
  r.solution = sol;
  const int S = static_cast<int>(sol.subgraphs.size());
  if (S == 0) return r;

  // Skip for very large partitions to respect wall-clock budget.
  // ExhaustiveGranularity per sg scales O(|w_cands| × |h_cands| × |k_cands|)
  // so 400+ subgraphs can exceed 120s competition timeouts.
  auto t0 = std::chrono::steady_clock::now();
  if (S > 200 && deadline_s < 60.0) {
    r.cost_after = r.cost_before;
    return r;
  }

  auto gt0 = mlsys::Evaluate(p, sol);
  if (gt0.ok()) r.cost_before = *gt0;

  std::unordered_set<size_t> carried;
  for (int i = 0; i < S; ++i) {
    auto& sg = r.solution.subgraphs[i];
    // Respect deadline: bail out if we've spent the budget.
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed > deadline_s) break;

    int64_t extra_sz = 0;
    auto ts_cur = solver::Classify(p, sg.ops, &dag);
    for (size_t t : carried) {
      if (!ts_cur.inputs.count(t))
        extra_sz += p.tensors[t].width * p.tensors[t].height;
    }
    auto br = solver::ExhaustiveGranularity(p, sg.ops, carried, extra_sz,
                                             &dag, /*extra_wh_slots=*/0,
                                             /*assume_retain_out=*/true);
    if (br.gran.width > 0 && br.gran.height > 0 && br.gran.depth > 0) {
      if (br.gran != sg.granularity) {
        sg.granularity = br.gran;
        sg.traversal_order = std::nullopt;
        ++r.n_refined;
      }
    }
    carried.clear();
    for (size_t t : sg.tensors_to_retain) carried.insert(t);
  }

  if (r.n_refined == 0) {
    r.cost_after = r.cost_before;
    return r;
  }
  for (auto& sg : r.solution.subgraphs) sg.subgraph_latency = 0.0;
  auto gt1 = mlsys::Evaluate(p, r.solution);
  // Strict: only keep if ExactCost truly improves (not just tolerance-equal).
  if (gt1.ok() && r.cost_before > 0 && *gt1 < r.cost_before) {
    r.cost_after = *gt1;
  } else {
    r.solution = sol;
    r.cost_after = r.cost_before;
    r.n_refined = 0;
  }
  return r;
}

}  // namespace pattern
