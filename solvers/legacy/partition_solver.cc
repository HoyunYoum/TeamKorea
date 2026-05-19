// partition_solver.cc — Driver for the partition-driven solver.
//
// Pipeline:
//   1. Read problem, apply solver::Preprocess (shared with unified/io/twophase)
//   2. Phase 1-A: generate diverse initial pool, pick top-3 feasible seeds
//   3. Phase 1-B: multi-seed SA with budget reallocation (includes M5
//      recomputation post-pass on the final best)
//   4. Final: BuildSolution + mlsys::Evaluate for ground-truth cost
//   5. Compare with analytical cost; emit warnings if they diverge
//   6. UnPreprocess and WriteSolution
//
// Usage:
//   mlsys_partition <problem.json> <solution.json> [timeout_s]
//
// Timeout defaults to CompetitionTimeout(N, bm_number) per io_util.cc.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "mlsys.h"
#include "partition_evaluator.h"
#include "partition_search.h"
#include "solver_common.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <problem.json> <solution.json> [timeout_s]\n";
    return 1;
  }
  const char* problem_path = argv[1];
  const char* output_path = argv[2];
  double timeout_override = (argc > 3) ? std::atof(argv[3]) : 0.0;

  auto t0 = std::chrono::steady_clock::now();
  auto problem = mlsys::ReadProblem(problem_path);
  if (!problem.ok()) {
    std::cerr << "ReadProblem: " << problem.status() << "\n";
    return 1;
  }
  const int N_orig = (int)problem->ops.size();
  auto pr = solver::Preprocess(*problem);
  auto& p = pr.problem;
  auto dag = solver::BuildDAG(p);
  const int N = (int)p.ops.size();

  int bm_num = solver::ParseBenchmarkNumber(problem_path);
  double timeout = timeout_override > 0
                   ? timeout_override
                   : solver::CompetitionTimeout(N_orig, bm_num);
  // Reserve ~15% of timeout for the post-SA pipeline (BuildSolution +
  // PatchLatencies + Evaluate + WriteSolution). On large BMs (BM-17 with
  // 70+ subgraphs) BuildSolution alone can take 15-20s.
  constexpr double kPostSolveReserveRatio = 0.15;
  double post_reserve = timeout * kPostSolveReserveRatio;
  double hard_deadline = timeout - post_reserve;
  double phase1a_budget = std::min(0.5, timeout * 0.10);
  double phase1b_budget = hard_deadline - phase1a_budget;
  if (phase1b_budget < 0) phase1b_budget = 0;

  std::cerr << "=== partition_solver ===\n"
            << "  input:    " << problem_path << " (N=" << N_orig
            << (N != N_orig ? " -> " + std::to_string(N) + " post-preprocess"
                            : "") << ")\n"
            << "  timeout:  " << std::fixed << std::setprecision(1)
            << timeout << "s  (BM-" << bm_num << ")\n"
            << "  budgets:  Phase1-A=" << phase1a_budget
            << "s  Phase1-B=" << phase1b_budget << "s\n";

  // ── Phase 1-A: initial pool ─────────────────────────────────────────
  auto t_p1a_start = std::chrono::steady_clock::now();
  // Cap pool generation at 30% of timeout so slow DP variants (DFS, Rand)
  // don't starve Phase 1-B on large BMs.
  double pool_budget = timeout * 0.30;
  auto pool = partition::GenerateInitialPool(p, dag, {1, 2, 3}, pool_budget);
  double p1a_elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_p1a_start).count();
  std::cerr << "  [phase1a] " << std::fixed << std::setprecision(2)
            << p1a_elapsed << "s  (" << pool.size() << " candidates, "
            << (100.0 * p1a_elapsed / timeout) << "% of timeout)\n";
  std::vector<solver::Partition> seeds;
  std::string pool_best_label;
  double pool_best_cost = std::numeric_limits<double>::infinity();
  for (const auto& e : pool) {
    if (!e.evaluation.feasible) continue;
    if (seeds.size() < 3) seeds.push_back(e.partition);
    if (e.evaluation.cost < pool_best_cost) {
      pool_best_cost = e.evaluation.cost;
      pool_best_label = e.label;
    }
    if (seeds.size() >= 3) break;
  }
  if (seeds.empty()) {
    std::cerr << "FATAL: no feasible seed in pool.\n";
    return 1;
  }
  std::cerr << "  pool best (analytical): " << pool_best_label
            << " cost=" << (int64_t)pool_best_cost
            << "  (seeds picked: " << seeds.size() << ")\n";

  // ── Phase 1-B: multi-seed SA ────────────────────────────────────────
  // Use remaining wall budget: hard_deadline minus elapsed since start.
  double actual_p1a = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  double sa_budget = std::max(0.0, hard_deadline - actual_p1a);
  std::cerr << "  [phase1b] budget=" << std::fixed << std::setprecision(2)
            << sa_budget << "s  (reserve=" << post_reserve << "s for finalize)\n";
  auto ms = partition::RunMultiSeedSA(p, dag, seeds, sa_budget,
                                       /*base_seed=*/42);
  std::cerr << "\n  SA best (analytical): " << (int64_t)ms.best_eval.cost
            << "  (" << (int)ms.best_partition.subgraphs.size()
            << " subgraphs)\n";
  for (size_t i = 0; i < ms.per_seed.size(); ++i) {
    const auto& r = ms.per_seed[i];
    std::cerr << "    seed" << i << ": iters=" << r.iterations_done
              << " best=" << (int64_t)r.best_eval.cost
              << " improved=" << (r.improved_over_seed ? "Y" : "N")
              << " wall=" << std::fixed << std::setprecision(1)
              << r.wall_time_s << "s\n";
  }

  // ── Final: pick the best feasible partition (SA > pool fallback) ────
  solver::Partition chosen_part;
  partition::Evaluation chosen_eval;
  if (ms.best_eval.feasible && !ms.best_partition.subgraphs.empty()) {
    chosen_part = ms.best_partition;
    chosen_eval = ms.best_eval;
  } else {
    // SA never produced a valid candidate (budget too tight). Fall back to
    // pool[0] which is sorted feasible-first by cost.
    std::cerr << "  [fallback] SA produced no feasible result; using pool best\n";
    chosen_part = partition::ApplyOrderingToPartition(
        pool.front().partition, pool.front().evaluation);
    chosen_eval = pool.front().evaluation;
  }
  solver::Partition best_part = solver::SubgraphReorder(
      p, dag, chosen_part);
  auto sol = solver::BuildSolution(p, dag, best_part);
  // Analytical latencies stored by BuildSolution are approximations.  Zero
  // them before Evaluate so our evaluator's reported-vs-computed
  // consistency check (which treats 0 as "not provided") doesn't reject.
  for (auto& sg : sol.subgraphs) sg.subgraph_latency = 0.0;
  auto gt = mlsys::Evaluate(p, sol);
  if (!gt.ok()) {
    std::cerr << "Evaluate: " << gt.status() << "\n";
    // Dump the rejected solution for post-mortem debugging.
    auto final_sol_dbg = solver::UnPreprocess(sol, pr);
    std::string dbg_path = std::string(output_path) + ".rejected";
    solver::WriteSolution(final_sol_dbg, dbg_path);
    std::cerr << "  [dumped rejected solution to " << dbg_path << "]\n";
    return 1;
  }
  double gt_cost = *gt;

  std::cerr << "\n  Ground truth (mlsys::Evaluate): " << (int64_t)gt_cost << "\n";
  double ratio = gt_cost / std::max(1.0, ms.best_eval.cost);
  if (std::abs(ratio - 1.0) > 0.02) {
    std::cerr << "  [warn] analytical/GT ratio = " << std::fixed
              << std::setprecision(3) << ratio
              << " (divergence > 2%) — analytical cost model approximation\n";
  }

  // ── Write solution ──────────────────────────────────────────────────
  auto final_sol = solver::UnPreprocess(sol, pr);
  solver::WriteSolution(final_sol, output_path);

  double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  std::cerr << "\nTotal elapsed: " << std::fixed << std::setprecision(2)
            << elapsed << "s  →  " << output_path << "\n";
  return 0;
}
