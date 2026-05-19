// solution_build.cc — BuildSolution(Exhaustive) + PrintDiagnostics + TopoSortOps helper.
//
// Extracted from solver_common.cc (Phase 3 refactor).

#include "solution_build.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cost.h"
#include "granularity.h"
#include "tensor_roles.h"

namespace solver {

namespace {
inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace

// ── Topological sort of ops within a subgraph ───────────────────────────────

static void TopoSortOps(const mlsys::Problem& p, const DAG& dag,
                         std::vector<size_t>& ops) {
  if (ops.size() <= 1) return;
  std::unordered_set<size_t> op_set(ops.begin(), ops.end());
  std::unordered_map<size_t, int> in_deg;
  for (size_t op : ops) in_deg[op] = 0;
  for (size_t op : ops) {
    for (size_t succ : dag.op_successors[op]) {
      if (op_set.count(succ)) ++in_deg[succ];
    }
  }
  std::queue<size_t> q;
  for (size_t op : ops)
    if (in_deg[op] == 0) q.push(op);
  std::vector<size_t> sorted;
  sorted.reserve(ops.size());
  while (!q.empty()) {
    size_t u = q.front(); q.pop();
    sorted.push_back(u);
    for (size_t succ : dag.op_successors[u]) {
      if (op_set.count(succ) && --in_deg[succ] == 0)
        q.push(succ);
    }
  }
  if (sorted.size() == ops.size()) ops = std::move(sorted);
}

// ── BuildSolution (with BestGranularity + retention) ─────────────────────────

mlsys::Solution BuildSolution(const mlsys::Problem& p, const DAG& dag,
                               Partition& part) {
  mlsys::Solution sol;
  std::unordered_set<size_t> retained_in;
  int64_t retained_size = 0;

  for (size_t si = 0; si < part.subgraphs.size(); ++si) {
    auto& ops = part.subgraphs[si];
    TopoSortOps(p, dag, ops);
    auto cfg = BestGranularity(p, ops, retained_in, retained_size, &dag);

    mlsys::Subgraph sg;
    sg.ops = ops;
    sg.granularity = cfg.gran;
    sg.subgraph_latency = cfg.cost;
    if (!cfg.traversal.empty())
      sg.traversal_order = cfg.traversal;

    // Try retention for next subgraph
    if (si + 1 < part.subgraphs.size()) {
      auto cands = RetentionCandidates(p, ops, part.subgraphs[si + 1], &dag);
      auto ts = Classify(p, ops, &dag);
      auto ts_next = Classify(p, part.subgraphs[si + 1], &dag);

      // Also try retaining outputs that have NO consumers in subgraphs after
      // si+1. Retaining these saves eviction cost safely (no future subgraph
      // needs them from slow memory). But do NOT retain outputs consumed by
      // later subgraphs — retention expires after one step, making them lost.
      std::unordered_set<size_t> future_consumed;
      for (size_t sj = si + 2; sj < part.subgraphs.size(); ++sj) {
        for (size_t op : part.subgraphs[sj]) {
          for (size_t t : p.ops[op].inputs) future_consumed.insert(t);
        }
      }
      for (size_t t : ts.outputs) {
        if (std::find(cands.begin(), cands.end(), t) != cands.end()) continue;
        if (!future_consumed.count(t))
          cands.push_back(t);
      }

      auto RetainExtra = [&](const std::vector<size_t>& retain_set) -> int64_t {
        int64_t extra = 0;
        for (size_t t : retain_set) {
          if (!ts_next.inputs.count(t))
            extra += p.tensors[t].width * p.tensors[t].height;
        }
        return extra;
      };

      // Try subsets of candidates up to size 3 (multi-tensor retention)
      std::vector<std::vector<size_t>> subsets;
      for (size_t i = 0; i < cands.size(); ++i) {
        subsets.push_back({cands[i]});
        for (size_t j = i + 1; j < cands.size(); ++j) {
          subsets.push_back({cands[i], cands[j]});
          for (size_t k = j + 1; k < cands.size() && k < i + 5; ++k)
            subsets.push_back({cands[i], cands[j], cands[k]});
        }
      }

      double best_combined = cfg.cost;
      auto cfg_next_no = BestGranularity(p, part.subgraphs[si + 1],
                                          retained_in, retained_size, &dag);
      double baseline_total = cfg.cost + cfg_next_no.cost;
      double best_total_combined = baseline_total;
      std::vector<size_t> best_retain;

      for (auto& subset : subsets) {
        std::unordered_set<size_t> try_ri = retained_in;
        for (size_t t : subset) try_ri.insert(t);
        int64_t extra = RetainExtra(subset);

        const std::vector<int64_t>* tp =
            cfg.traversal.empty() ? nullptr : &cfg.traversal;
        double cur_cost =
            ExactCost(p, ops, cfg.gran, subset, retained_in, tp, &dag);

        auto cfg_next = BestGranularity(p, part.subgraphs[si + 1], try_ri,
                                        retained_size + extra, &dag);
        if (cfg_next.cost >= kInf) continue;

        double total_with = cur_cost + cfg_next.cost;
        if (total_with < best_total_combined) {
          best_total_combined = total_with;
          best_retain = subset;
          best_combined = cur_cost;
        }
      }

      sg.tensors_to_retain = best_retain;
      {
        const std::vector<int64_t>* tp =
            cfg.traversal.empty() ? nullptr : &cfg.traversal;
        sg.subgraph_latency =
            ExactCost(p, ops, cfg.gran, best_retain, retained_in, tp, &dag);
      }
    } else {
      // Last subgraph: retain non-graph-output tensors (saves eviction cost).
      // Graph outputs must end up in slow memory per evaluator rules.
      auto ts = Classify(p, ops, &dag);
      for (size_t t : ts.outputs) {
        if (!dag.tensor_consumers[t].empty())
          sg.tensors_to_retain.push_back(t);
      }
      const std::vector<int64_t>* tp =
          cfg.traversal.empty() ? nullptr : &cfg.traversal;
      sg.subgraph_latency =
          ExactCost(p, ops, cfg.gran, sg.tensors_to_retain, retained_in, tp,
                    &dag);
    }

    sol.subgraphs.push_back(std::move(sg));

    // Update retained state for next subgraph
    retained_in.clear();
    retained_size = 0;
    if (si + 1 < part.subgraphs.size()) {
      auto ts_next2 = Classify(p, part.subgraphs[si + 1], &dag);
      for (size_t t : sol.subgraphs.back().tensors_to_retain) {
        retained_in.insert(t);
        if (!ts_next2.inputs.count(t))
          retained_size += p.tensors[t].width * p.tensors[t].height;
      }
    } else {
      for (size_t t : sol.subgraphs.back().tensors_to_retain)
        retained_in.insert(t);
    }
  }
  return sol;
}

// ── BuildSolutionExhaustive (with ExhaustiveGranularity + retention) ─────────

mlsys::Solution BuildSolutionExhaustive(const mlsys::Problem& p,
                                         const DAG& dag,
                                         Partition& part) {
  mlsys::Solution sol;
  std::unordered_set<size_t> retained_in;
  int64_t retained_size = 0;

  for (size_t si = 0; si < part.subgraphs.size(); ++si) {
    auto& ops = part.subgraphs[si];
    TopoSortOps(p, dag, ops);
    auto bound = ExhaustiveGranularity(p, ops, retained_in, retained_size, &dag);

    mlsys::Subgraph sg;
    sg.ops = ops;
    sg.granularity = bound.gran;
    sg.subgraph_latency = bound.exact_cost;

    // Set traversal order
    auto [oW, oH] = OutDims(p, ops);
    int64_t ntw = CeilDiv(oW, bound.gran.width);
    int64_t nth = CeilDiv(oH, bound.gran.height);
    if (ntw * nth > 1) {
      if (bound.snake_mode == 0)
        sg.traversal_order = SnakeRow(ntw, nth);
      else if (bound.snake_mode == 1)
        sg.traversal_order = SnakeCol(ntw, nth);
    }

    // Try retention for next subgraph
    if (si + 1 < part.subgraphs.size()) {
      auto ts_cur = Classify(p, ops, &dag);
      auto ts_next = Classify(p, part.subgraphs[si + 1], &dag);

      // Retain candidates: outputs consumed by next subgraph + outputs with no
      // future consumers (safe to retain — saves eviction, no one needs them later).
      auto cands = RetentionCandidates(p, ops, part.subgraphs[si + 1], &dag);
      std::unordered_set<size_t> future_consumed;
      for (size_t sj = si + 2; sj < part.subgraphs.size(); ++sj) {
        for (size_t op : part.subgraphs[sj]) {
          for (size_t t : p.ops[op].inputs) future_consumed.insert(t);
        }
      }
      for (size_t t : ts_cur.outputs) {
        if (std::find(cands.begin(), cands.end(), t) != cands.end()) continue;
        if (!future_consumed.count(t))
          cands.push_back(t);
      }

      auto RetainExtra = [&](const std::vector<size_t>& retain_set) -> int64_t {
        int64_t extra = 0;
        for (size_t t : retain_set) {
          if (!ts_next.inputs.count(t))
            extra += p.tensors[t].width * p.tensors[t].height;
        }
        return extra;
      };

      // Try subsets up to size 4
      std::vector<std::vector<size_t>> subsets;
      for (size_t i = 0; i < cands.size(); ++i) {
        subsets.push_back({cands[i]});
        for (size_t j = i + 1; j < cands.size(); ++j) {
          subsets.push_back({cands[i], cands[j]});
          for (size_t k2 = j + 1; k2 < cands.size() && k2 < i + 6; ++k2) {
            subsets.push_back({cands[i], cands[j], cands[k2]});
            for (size_t l = k2 + 1; l < cands.size() && l < i + 7; ++l)
              subsets.push_back({cands[i], cands[j], cands[k2], cands[l]});
          }
        }
      }

      auto next_no_retain = ExhaustiveGranularity(p, part.subgraphs[si + 1],
                                                   retained_in, retained_size);
      double baseline = bound.exact_cost + next_no_retain.exact_cost;
      double best_combined = baseline;
      std::vector<size_t> best_retain;

      for (auto& subset : subsets) {
        std::unordered_set<size_t> try_ri = retained_in;
        for (size_t t : subset) try_ri.insert(t);
        int64_t extra = RetainExtra(subset);

        std::vector<int64_t> trav;
        if (ntw * nth > 1) {
          if (bound.snake_mode == 0) trav = SnakeRow(ntw, nth);
          else if (bound.snake_mode == 1) trav = SnakeCol(ntw, nth);
        }
        const std::vector<int64_t>* tp = trav.empty() ? nullptr : &trav;
        double cur_cost = ExactCost(p, ops, bound.gran, subset, retained_in, tp);

        auto next_with = ExhaustiveGranularity(p, part.subgraphs[si + 1],
                                               try_ri, retained_size + extra);
        if (next_with.exact_cost >= kInf) continue;

        double total_with = cur_cost + next_with.exact_cost;
        if (total_with < best_combined) {
          best_combined = total_with;
          best_retain = subset;
        }
      }

      sg.tensors_to_retain = best_retain;
      {
        std::vector<int64_t> trav;
        if (ntw * nth > 1) {
          if (bound.snake_mode == 0) trav = SnakeRow(ntw, nth);
          else if (bound.snake_mode == 1) trav = SnakeCol(ntw, nth);
        }
        const std::vector<int64_t>* tp = trav.empty() ? nullptr : &trav;
        sg.subgraph_latency =
            ExactCost(p, ops, bound.gran, best_retain, retained_in, tp);
      }
    } else {
      // Last subgraph: retain non-graph-output tensors.
      // Graph outputs must end up in slow memory per evaluator rules.
      auto ts = Classify(p, ops, &dag);
      for (size_t t : ts.outputs) {
        if (!dag.tensor_consumers[t].empty())
          sg.tensors_to_retain.push_back(t);
      }

      std::vector<int64_t> trav;
      if (ntw * nth > 1) {
        if (bound.snake_mode == 0) trav = SnakeRow(ntw, nth);
        else if (bound.snake_mode == 1) trav = SnakeCol(ntw, nth);
      }
      const std::vector<int64_t>* tp = trav.empty() ? nullptr : &trav;
      sg.subgraph_latency =
          ExactCost(p, ops, bound.gran, sg.tensors_to_retain, retained_in, tp);
    }

    sol.subgraphs.push_back(std::move(sg));

    // Update retained state for next subgraph
    retained_in.clear();
    retained_size = 0;
    if (si + 1 < part.subgraphs.size()) {
      auto ts_next2 = Classify(p, part.subgraphs[si + 1], &dag);
      for (size_t t : sol.subgraphs.back().tensors_to_retain) {
        retained_in.insert(t);
        if (!ts_next2.inputs.count(t))
          retained_size += p.tensors[t].width * p.tensors[t].height;
      }
    } else {
      for (size_t t : sol.subgraphs.back().tensors_to_retain)
        retained_in.insert(t);
    }
  }
  return sol;
}


// ── Diagnostics ──────────────────────────────────────────────────────────────

void PrintDiagnostics(const mlsys::Problem& p, const mlsys::Solution& sol) {
  std::cerr << "=== Per-subgraph diagnostics ===\n";
  double total = 0;
  for (size_t si = 0; si < sol.subgraphs.size(); ++si) {
    const auto& sg = sol.subgraphs[si];
    auto [oW, oH] = OutDims(p, sg.ops);
    int64_t w = sg.granularity.width, h = sg.granularity.height;
    int64_t k = sg.granularity.depth;
    int64_t ntw = CeilDiv(oW, w), nth = CeilDiv(oH, h);
    int64_t maxK = 0;
    int n_mm = 0, n_pw = 0;
    for (size_t op : sg.ops) {
      if (p.ops[op].op_type == "MatMul") {
        maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);
        ++n_mm;
      } else {
        ++n_pw;
      }
    }
    int64_t nk = (maxK > 0) ? CeilDiv(maxK, k) : 1;
    const char* snake = "raster";
    if (sg.traversal_order.has_value() && ntw * nth > 1) {
      auto& trav = *sg.traversal_order;
      if (trav.size() >= 2) {
        int64_t r0 = trav[0] / ntw, c0 = trav[0] % ntw;
        int64_t r1 = trav[1] / ntw, c1 = trav[1] % ntw;
        (void)c0; (void)r1;
        if (r0 == r1) snake = "row";
        else if (c0 == c1) snake = "col";
      }
    }
    total += sg.subgraph_latency;
    std::cerr << "  sg[" << si << "]"
              << " ops=" << sg.ops.size()
              << "(" << n_mm << "mm+" << n_pw << "pw)"
              << " gran=(" << w << "," << h << "," << k << ")"
              << " tiles=" << ntw << "x" << nth << "x" << nk
              << " snake=" << snake
              << " lat=" << (int64_t)sg.subgraph_latency
              << " retain=" << sg.tensors_to_retain.size()
              << "\n";
  }
  std::cerr << "  TOTAL: " << (int64_t)total << "\n";
}

}  // namespace solver
