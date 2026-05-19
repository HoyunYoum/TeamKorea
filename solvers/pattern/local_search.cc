// local_search.cc — See local_search.h.

#include "local_search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pattern {

namespace {

using TopoPos = std::unordered_map<size_t, size_t>;

TopoPos BuildTopoPos(const solver::DAG& dag) {
  TopoPos pos;
  pos.reserve(dag.topo_order.size());
  for (size_t i = 0; i < dag.topo_order.size(); ++i)
    pos[dag.topo_order[i]] = i;
  return pos;
}

void SortByTopo(std::vector<size_t>& ops, const TopoPos& pos) {
  std::sort(ops.begin(), ops.end(),
            [&](size_t a, size_t b) { return pos.at(a) < pos.at(b); });
}

// Re-run BestGranularity for sg[idx]. Carries retained_in from
// sg[idx-1].tensors_to_retain (unchanged neighbour). Mirrors
// retention_pass::reopt_range's single-sg case. Returns false on infeasible.
bool ReoptGran(const mlsys::Problem& p, const solver::DAG& dag,
                mlsys::Solution& s, int idx) {
  auto& sg = s.subgraphs[idx];
  std::unordered_set<size_t> carried;
  int64_t extra_sz = 0;
  if (idx > 0) {
    for (size_t t : s.subgraphs[idx - 1].tensors_to_retain) carried.insert(t);
  }
  auto ts_cur = solver::Classify(p, sg.ops, &dag);
  for (size_t t : carried) {
    if (!ts_cur.inputs.count(t))
      extra_sz += p.tensors[t].width * p.tensors[t].height;
  }
  auto cfg = solver::BestGranularity(p, sg.ops, carried, extra_sz, &dag,
                                       /*assume_retain_out=*/true);
  if (cfg.gran.width <= 0 || !std::isfinite(cfg.cost)) return false;
  sg.granularity = cfg.gran;
  if (!cfg.traversal.empty())
    sg.traversal_order = cfg.traversal;
  else
    sg.traversal_order = std::nullopt;
  sg.subgraph_latency = 0.0;
  return true;
}

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

LocalSearchResult LocalSearch(const mlsys::Problem& p,
                               const solver::DAG& dag,
                               mlsys::Solution sol,
                               double deadline_s,
                               ImproveCallback on_improve) {
  LocalSearchResult r;
  r.solution = std::move(sol);

  auto t_start = std::chrono::steady_clock::now();
  auto deadline_hit = [&] {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - t_start).count() > deadline_s;
  };

  for (auto& sg : r.solution.subgraphs) sg.subgraph_latency = 0.0;
  auto gt0 = mlsys::Evaluate(p, r.solution);
  if (!gt0.ok()) {
    r.cost_before = r.cost_after = kInf;
    return r;
  }
  r.cost_before = *gt0;
  double cur_cost = r.cost_before;

  auto topo_pos = BuildTopoPos(dag);

  // Move builders: each returns (trial_solution, trial_gt). gt=kInf on
  // infeasible (non-convex, BestGranularity failure, Evaluate error).
  // Semantics per move for tensors_to_retain:
  //   Merge(i,i+1):
  //     sg[i-1].tensors_to_retain         — unchanged (feeds merged sg).
  //     merged.tensors_to_retain          — inherit sg[i+1]'s (feeds sg[i+2]).
  //     sg[i].tensors_to_retain           — dropped (was internal to merged).
  //   Split(i, k)  [topo-sorted orig_ops, cut at k]:
  //     sg_a.tensors_to_retain            — empty (retention_pass can fill).
  //     sg_b.tensors_to_retain            — sg[i]'s original, filtered to the
  //                                         subset still produced in sg_b.
  //   Repartition(i, i+1, k):
  //     sg_a.tensors_to_retain            — empty.
  //     sg_b.tensors_to_retain            — sg[i+1]'s original, filtered to
  //                                         the subset still produced in sg_b.
  // Spec #52 requires every tensors_to_retain entry to be an op output of
  // that subgraph, so the filters above are necessary.

  auto TryMerge = [&](int i) -> std::pair<mlsys::Solution, double> {
    int S = static_cast<int>(r.solution.subgraphs.size());
    if (i < 0 || i + 1 >= S) return {{}, kInf};

    mlsys::Solution trial = r.solution;

    std::vector<size_t> combined;
    combined.reserve(trial.subgraphs[i].ops.size() +
                     trial.subgraphs[i + 1].ops.size());
    for (size_t op : trial.subgraphs[i].ops) combined.push_back(op);
    for (size_t op : trial.subgraphs[i + 1].ops) combined.push_back(op);
    SortByTopo(combined, topo_pos);

    if (!solver::IsDAGConvex(dag, combined)) return {{}, kInf};

    mlsys::Subgraph merged;
    merged.ops = std::move(combined);
    merged.tensors_to_retain = trial.subgraphs[i + 1].tensors_to_retain;
    merged.granularity = {};
    merged.traversal_order = std::nullopt;
    merged.subgraph_latency = 0.0;

    trial.subgraphs[i] = std::move(merged);
    trial.subgraphs.erase(trial.subgraphs.begin() + i + 1);

    if (!ReoptGran(p, dag, trial, i)) return {{}, kInf};

    auto gt = mlsys::Evaluate(p, trial);
    if (!gt.ok()) return {{}, kInf};
    return {std::move(trial), *gt};
  };

  auto TrySplit = [&](int i, int k) -> std::pair<mlsys::Solution, double> {
    int S = static_cast<int>(r.solution.subgraphs.size());
    if (i < 0 || i >= S) return {{}, kInf};

    std::vector<size_t> ops_sorted = r.solution.subgraphs[i].ops;
    SortByTopo(ops_sorted, topo_pos);
    int n = static_cast<int>(ops_sorted.size());
    if (k <= 0 || k >= n) return {{}, kInf};

    std::vector<size_t> ops_a(ops_sorted.begin(), ops_sorted.begin() + k);
    std::vector<size_t> ops_b(ops_sorted.begin() + k, ops_sorted.end());

    if (!solver::IsDAGConvex(dag, ops_a)) return {{}, kInf};
    if (!solver::IsDAGConvex(dag, ops_b)) return {{}, kInf};

    mlsys::Solution trial = r.solution;

    std::unordered_set<size_t> b_outputs;
    for (size_t op : ops_b)
      for (size_t t : p.ops[op].outputs) b_outputs.insert(t);
    std::vector<size_t> filtered_retain;
    for (size_t t : trial.subgraphs[i].tensors_to_retain)
      if (b_outputs.count(t)) filtered_retain.push_back(t);

    mlsys::Subgraph sg_a;
    sg_a.ops = std::move(ops_a);
    sg_a.granularity = {};
    sg_a.subgraph_latency = 0.0;
    mlsys::Subgraph sg_b;
    sg_b.ops = std::move(ops_b);
    sg_b.tensors_to_retain = std::move(filtered_retain);
    sg_b.granularity = {};
    sg_b.subgraph_latency = 0.0;

    trial.subgraphs[i] = std::move(sg_a);
    trial.subgraphs.insert(trial.subgraphs.begin() + i + 1, std::move(sg_b));

    if (!ReoptGran(p, dag, trial, i)) return {{}, kInf};
    if (!ReoptGran(p, dag, trial, i + 1)) return {{}, kInf};

    auto gt = mlsys::Evaluate(p, trial);
    if (!gt.ok()) return {{}, kInf};
    return {std::move(trial), *gt};
  };

  auto TryRepartition = [&](int i, int k)
      -> std::pair<mlsys::Solution, double> {
    int S = static_cast<int>(r.solution.subgraphs.size());
    if (i < 0 || i + 1 >= S) return {{}, kInf};

    int orig_a = static_cast<int>(r.solution.subgraphs[i].ops.size());

    std::vector<size_t> combined;
    combined.reserve(r.solution.subgraphs[i].ops.size() +
                     r.solution.subgraphs[i + 1].ops.size());
    for (size_t op : r.solution.subgraphs[i].ops) combined.push_back(op);
    for (size_t op : r.solution.subgraphs[i + 1].ops) combined.push_back(op);
    SortByTopo(combined, topo_pos);

    int n = static_cast<int>(combined.size());
    if (k <= 0 || k >= n) return {{}, kInf};
    if (k == orig_a) return {{}, kInf};  // identical to current split

    std::vector<size_t> ops_a(combined.begin(), combined.begin() + k);
    std::vector<size_t> ops_b(combined.begin() + k, combined.end());

    if (!solver::IsDAGConvex(dag, ops_a)) return {{}, kInf};
    if (!solver::IsDAGConvex(dag, ops_b)) return {{}, kInf};

    mlsys::Solution trial = r.solution;

    std::unordered_set<size_t> b_outputs;
    for (size_t op : ops_b)
      for (size_t t : p.ops[op].outputs) b_outputs.insert(t);
    std::vector<size_t> filtered_retain;
    for (size_t t : trial.subgraphs[i + 1].tensors_to_retain)
      if (b_outputs.count(t)) filtered_retain.push_back(t);

    mlsys::Subgraph sg_a;
    sg_a.ops = std::move(ops_a);
    sg_a.granularity = {};
    sg_a.subgraph_latency = 0.0;
    mlsys::Subgraph sg_b;
    sg_b.ops = std::move(ops_b);
    sg_b.tensors_to_retain = std::move(filtered_retain);
    sg_b.granularity = {};
    sg_b.subgraph_latency = 0.0;

    trial.subgraphs[i] = std::move(sg_a);
    trial.subgraphs[i + 1] = std::move(sg_b);

    if (!ReoptGran(p, dag, trial, i)) return {{}, kInf};
    if (!ReoptGran(p, dag, trial, i + 1)) return {{}, kInf};

    auto gt = mlsys::Evaluate(p, trial);
    if (!gt.ok()) return {{}, kInf};
    return {std::move(trial), *gt};
  };

  // Best-ever tracker. SA phase can drift to worse solutions; we only
  // fire on_improve for new all-time bests, and restore best-ever before
  // returning so a mid-walk deadline never regresses the caller's sol.
  double best_cost = cur_cost;
  mlsys::Solution best_sol = r.solution;

  auto hit_new_best = [&](double trial_cost) {
    if (trial_cost < best_cost - 1e-9) {
      best_cost = trial_cost;
      best_sol = r.solution;
      ++r.n_improvements;
      if (on_improve) on_improve(best_sol, best_cost);
    }
  };

  auto accept_hc = [&](mlsys::Solution&& trial, double trial_cost) {
    r.solution = std::move(trial);
    cur_cost = trial_cost;
    hit_new_best(trial_cost);
  };

  // ── Phase A: first-improvement hill-climb ────────────────────────────
  // Sweep Merge → Split → Repartition; on any accept, restart sweeps
  // (partition indices shifted). Exit when a full sweep of all three
  // move families yields no improvement.
  while (!deadline_hit()) {
    bool improved = false;

    {
      int S = static_cast<int>(r.solution.subgraphs.size());
      for (int i = 0; i + 1 < S && !deadline_hit(); ++i) {
        ++r.n_moves_tried; ++r.n_hc_moves;
        auto [trial, gt] = TryMerge(i);
        if (gt < cur_cost - 1e-9) {
          accept_hc(std::move(trial), gt);
          improved = true;
          break;
        }
      }
    }
    if (improved) continue;

    {
      int S = static_cast<int>(r.solution.subgraphs.size());
      for (int i = 0; i < S && !improved && !deadline_hit(); ++i) {
        int n = static_cast<int>(r.solution.subgraphs[i].ops.size());
        if (n < 2) continue;
        for (int k = 1; k < n && !deadline_hit(); ++k) {
          ++r.n_moves_tried; ++r.n_hc_moves;
          auto [trial, gt] = TrySplit(i, k);
          if (gt < cur_cost - 1e-9) {
            accept_hc(std::move(trial), gt);
            improved = true;
            break;
          }
        }
      }
    }
    if (improved) continue;

    {
      int S = static_cast<int>(r.solution.subgraphs.size());
      for (int i = 0; i + 1 < S && !improved && !deadline_hit(); ++i) {
        int n = static_cast<int>(r.solution.subgraphs[i].ops.size() +
                                  r.solution.subgraphs[i + 1].ops.size());
        for (int k = 1; k < n && !deadline_hit(); ++k) {
          ++r.n_moves_tried; ++r.n_hc_moves;
          auto [trial, gt] = TryRepartition(i, k);
          if (gt < cur_cost - 1e-9) {
            accept_hc(std::move(trial), gt);
            improved = true;
            break;
          }
        }
      }
    }

    if (!improved) break;  // stalled: no move in any sweep helps
  }

  // ── Phase B: simulated annealing ─────────────────────────────────────
  // Random-walk from the hill-climb local optimum; Metropolis acceptance.
  // SA schedule:
  //   T₀   = best·5e-3,  T_min = best·1e-6,  cool  = 0.9998
  // Move mix: 25% Merge / 25% Split / 50% Repartition (Repartition
  // subsumes boundary-shift via small k deltas from the current cut).
  // `r.solution`/`cur_cost` track the SA walker; `best_sol`/`best_cost`
  // track the all-time best. Only new all-time bests fire on_improve.
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  if (!deadline_hit() && std::isfinite(best_cost) && best_cost > 0.0 &&
      r.solution.subgraphs.size() >= 2) {
    double T = best_cost * 5e-3;
    const double T_min = best_cost * 1e-6;
    constexpr double kCool = 0.9998;

    while (!deadline_hit() && T > T_min) {
      int S = static_cast<int>(r.solution.subgraphs.size());
      if (S < 2) break;

      double roll = unif(rng);
      mlsys::Solution trial;
      double trial_cost = kInf;

      if (roll < 0.25) {
        // Merge
        int i = std::uniform_int_distribution<int>(0, S - 2)(rng);
        auto res = TryMerge(i);
        trial = std::move(res.first);
        trial_cost = res.second;
      } else if (roll < 0.50) {
        // Split
        int i = std::uniform_int_distribution<int>(0, S - 1)(rng);
        int n = static_cast<int>(r.solution.subgraphs[i].ops.size());
        if (n >= 2) {
          int k = std::uniform_int_distribution<int>(1, n - 1)(rng);
          auto res = TrySplit(i, k);
          trial = std::move(res.first);
          trial_cost = res.second;
        }
      } else {
        // Repartition
        int i = std::uniform_int_distribution<int>(0, S - 2)(rng);
        int n = static_cast<int>(r.solution.subgraphs[i].ops.size() +
                                  r.solution.subgraphs[i + 1].ops.size());
        if (n >= 2) {
          int k = std::uniform_int_distribution<int>(1, n - 1)(rng);
          auto res = TryRepartition(i, k);
          trial = std::move(res.first);
          trial_cost = res.second;
        }
      }

      ++r.n_moves_tried;
      ++r.n_sa_moves;
      T *= kCool;

      if (!std::isfinite(trial_cost)) continue;  // infeasible move

      double delta = trial_cost - cur_cost;
      bool accept = false;
      if (delta < 0) {
        accept = true;
      } else if (T > 0 && unif(rng) < std::exp(-delta / T)) {
        accept = true;
      }

      if (accept) {
        r.solution = std::move(trial);
        cur_cost = trial_cost;
        ++r.n_sa_accepts;
        hit_new_best(trial_cost);
      }
    }
  }

  // Restore all-time best (SA walker may be elsewhere).
  r.solution = std::move(best_sol);
  r.cost_after = best_cost;
  return r;
}

}  // namespace pattern
