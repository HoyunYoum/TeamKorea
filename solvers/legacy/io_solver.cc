// io_solver.cc — IO-first scheduler
//
// Design principle: All benchmarks are IO-bound. With the right snake direction,
// most IO is fixed at the IO lower bound. Only one set of weights (RHS for
// row-snake, LHS for col-snake) scales with tile count. The solver minimizes
// that scaling dimension's tile count by maximizing tile size within the
// working set constraint.
//
// Architecture:
//   1. Graph analysis (DAG, tensor classification)
//   2. Initial partition (one op per subgraph)
//   3. Greedy merge (fuse when intermediate IO saved > tiling IO increase)
//   4. Per-subgraph: optimal granularity + snake direction
//   5. Retention pass
//   6. Output solution

#include "solver_common.h"

namespace {

// ── Post-DP local search ─────────────────────────────────────────────────────
// Move ops between adjacent subgraphs to find boundary improvements.

solver::Partition LocalSearch(const mlsys::Problem& p, const solver::DAG& dag,
                      solver::Partition part,
                      std::chrono::steady_clock::time_point t0,
                      double time_limit) {
  std::unordered_set<size_t> empty_ri;
  bool improved = true;
  while (improved) {
    improved = false;
    int n = (int)part.subgraphs.size();
    for (int i = 0; i + 1 < n; ++i) {
      double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t0).count();
      if (elapsed > time_limit) return part;

      auto& sg_a = part.subgraphs[i];
      auto& sg_b = part.subgraphs[i + 1];
      double old_cost =
          solver::BestGranularity(p, sg_a, empty_ri, 0, &dag, true).cost +
          solver::BestGranularity(p, sg_b, empty_ri, 0, &dag, true).cost;

      // Try moving last op of sg_a to sg_b
      if (sg_a.size() > 1) {
        size_t op = sg_a.back();
        std::vector<size_t> new_a(sg_a.begin(), sg_a.end() - 1);
        std::vector<size_t> new_b;
        new_b.push_back(op);
        new_b.insert(new_b.end(), sg_b.begin(), sg_b.end());
        double new_cost =
            solver::BestGranularity(p, new_a, empty_ri, 0, &dag, true).cost +
            solver::BestGranularity(p, new_b, empty_ri, 0, &dag, true).cost;
        if (new_cost < old_cost * 0.999) {
          sg_a = std::move(new_a);
          sg_b = std::move(new_b);
          improved = true;
          continue;
        }
      }

      // Try moving first op of sg_b to sg_a
      if (sg_b.size() > 1) {
        size_t op = sg_b.front();
        std::vector<size_t> new_a = sg_a;
        new_a.push_back(op);
        std::vector<size_t> new_b(sg_b.begin() + 1, sg_b.end());
        double new_cost =
            solver::BestGranularity(p, new_a, empty_ri, 0, &dag, true).cost +
            solver::BestGranularity(p, new_b, empty_ri, 0, &dag, true).cost;
        if (new_cost < old_cost * 0.999) {
          sg_a = std::move(new_a);
          sg_b = std::move(new_b);
          improved = true;
          continue;
        }
      }

      // Try merging sg_a and sg_b entirely
      {
        std::vector<size_t> merged;
        merged.insert(merged.end(), sg_a.begin(), sg_a.end());
        merged.insert(merged.end(), sg_b.begin(), sg_b.end());
        double merge_cost = solver::BestGranularity(p, merged, empty_ri, 0, &dag, true).cost;
        if (merge_cost < old_cost * 0.999) {
          part.subgraphs[i] = std::move(merged);
          part.subgraphs.erase(part.subgraphs.begin() + i + 1);
          improved = true;
          break;  // restart — indices changed
        }
      }
    }

    // Try splitting subgraphs
    if (!improved) {
      n = (int)part.subgraphs.size();
      for (int i = 0; i < n; ++i) {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        if (elapsed > time_limit) return part;

        auto& sg = part.subgraphs[i];
        if (sg.size() <= 2) continue;
        double cur_cost = solver::BestGranularity(p, sg, empty_ri, 0, &dag, true).cost;
        for (size_t s = 1; s < sg.size(); ++s) {
          std::vector<size_t> a(sg.begin(), sg.begin() + (int)s);
          std::vector<size_t> b(sg.begin() + (int)s, sg.end());
          double split_cost =
              solver::BestGranularity(p, a, empty_ri, 0, &dag, true).cost +
              solver::BestGranularity(p, b, empty_ri, 0, &dag, true).cost;
          if (split_cost < cur_cost * 0.999) {
            part.subgraphs[i] = std::move(a);
            part.subgraphs.insert(part.subgraphs.begin() + i + 1,
                                  std::move(b));
            improved = true;
            break;
          }
        }
        if (improved) break;
      }
    }
  }
  return part;
}

}  // namespace

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <problem.json> <solution.json>\n";
    return 1;
  }

  auto t0 = std::chrono::steady_clock::now();

  auto problem = mlsys::ReadProblem(argv[1]);
  if (!problem.ok()) {
    std::cerr << problem.status() << "\n";
    return 1;
  }
  auto pr = solver::Preprocess(*problem);
  auto& p = pr.problem;

  // Time-budget based on benchmark number (competition timeouts)
  int n_ops = (int)problem->ops.size();
  int bm_number = solver::ParseBenchmarkNumber(argv[1]);
  double timeout_sec = solver::CompetitionTimeout(n_ops, bm_number);
  double hard_deadline = timeout_sec * 0.95;

  auto run_pipeline = [&](solver::DAG& dag) -> std::pair<mlsys::Solution, solver::Partition> {
    auto part = solver::InitialPartition(dag);
    part = solver::DPPartition(p, dag, part, /*assume_retain_out=*/true);
    part = LocalSearch(p, dag, part, t0, hard_deadline * 0.25);
    part = solver::RecomputationPass(p, dag, part, /*assume_retain_out=*/true);
    part = solver::UnfusionPass(p, dag, std::move(part), /*assume_retain_out=*/true);
    part = solver::SubgraphReorder(p, dag, part);
    auto sol = solver::BuildSolution(p, dag, part);
    return {std::move(sol), std::move(part)};
  };

  struct Candidate {
    mlsys::Solution sol;
    double cost;
    std::string label;
  };
  std::vector<Candidate> candidates;

  struct BestPartition {
    solver::Partition part;
    solver::DAG* dag;
    double cost;
  };
  BestPartition best_part{{}, nullptr, 1e18};

  auto try_variant = [&](const std::string& label, solver::DAG& dag) -> bool {
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed > hard_deadline * 0.55 && !candidates.empty()) {
      std::cerr << "  " << label << ": SKIPPED (time budget)\n";
      return false;
    }
    auto [sol, part] = run_pipeline(dag);
    auto eval = mlsys::Evaluate(p, sol);
    double c = eval.ok() ? *eval : 1e18;
    std::cerr << "  " << label << ": " << c << "\n";
    if (c < best_part.cost) {
      best_part = {std::move(part), &dag, c};
    }
    candidates.push_back({std::move(sol), c, label});
    return true;
  };

  std::cerr << "Running variants (budget=" << hard_deadline << "s, timeout=" << timeout_sec << "s):\n";

  // Deterministic orderings
  auto dag_bfs = solver::BuildDAG(p);
  try_variant("BFS+DP", dag_bfs);

  if (solver::ElapsedSince(t0) < hard_deadline * 0.35) {
    auto dag_dfs = solver::BuildDAG_DFS(p);
    try_variant("DFS+DP", dag_dfs);
  }

  // MaxOut only if time permits
  if (solver::ElapsedSince(t0) < hard_deadline * 0.25) {
    auto dag_maxout = solver::BuildDAG_MaxOutput(p);
    try_variant("MaxOut+DP", dag_maxout);
  }

  // Randomized orderings (try as many as time permits)
  for (uint32_t seed = 0; seed < 50; ++seed) {
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed > hard_deadline * 0.55) break;
    auto dag_rnd = solver::BuildDAG_Random(p, seed);
    try_variant("Rnd" + std::to_string(seed) + "+DP", dag_rnd);
  }

  // Try exhaustive granularity on best partition if time allows
  {
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double remaining = hard_deadline - elapsed;
    if (remaining > elapsed && best_part.dag) {
      auto sol_exh = solver::BuildSolutionExhaustive(p, *best_part.dag, best_part.part);
      auto eval_exh = mlsys::Evaluate(p, sol_exh);
      double c_exh = eval_exh.ok() ? *eval_exh : 1e18;
      std::cerr << "  Exhaustive: " << c_exh << "\n";
      candidates.push_back({std::move(sol_exh), c_exh, "Exhaustive"});
    }
  }

  // Pick best
  size_t best_idx = 0;
  for (size_t i = 1; i < candidates.size(); ++i)
    if (candidates[i].cost < candidates[best_idx].cost) best_idx = i;

  auto& sol = candidates[best_idx].sol;
  double best_cost = candidates[best_idx].cost;
  std::cerr << "\nEvaluated total latency: " << best_cost
            << " (" << candidates[best_idx].label << ")\n";

  solver::PrintDiagnostics(p, sol);
  auto final_sol = solver::UnPreprocess(sol, pr);
  solver::WriteSolution(final_sol, argv[2]);

  auto t1 = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration<double>(t1 - t0).count();
  std::cerr << "Elapsed: " << elapsed << "s\n";
  return 0;
}
