// pattern_solver.cc — Driver for the §8 pattern-coverage solver.
//
// Pipeline (S0 scope):
//   1. Read + preprocess
//   2. Build DAG
//   3. Enumerate columns via pattern_enum (C0, C1, C3 at S0)
//   4. Greedy set-cover pack → Partition
//   5. solver::BuildSolution (BestGranularity + retention + traversal)
//   6. mlsys::Evaluate for ground truth
//   7. UnPreprocess + WriteSolution
//
// Usage:
//   pattern_solver <problem.json> <solution.json> [timeout_s]

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "local_search.h"
#include "mip_pack.h"
#include "mlsys.h"
#include "pattern_enum.h"
#include "retention_pass.h"
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

  // SIGTERM handler: the competition harness kills the process when the
  // per-BM timeout expires. Every solution we emit is written via an
  // atomic tmp→rename, so the file on disk is always a complete valid
  // solution; _exit(0) preserves it. (SIGKILL can't be caught; our
  // atomic-write is the defense there.)
  std::signal(SIGTERM, +[](int) { std::_Exit(0); });

  auto t0 = std::chrono::steady_clock::now();

  auto problem = mlsys::ReadProblem(problem_path);
  if (!problem.ok()) {
    std::cerr << "ReadProblem: " << problem.status() << "\n";
    return 1;
  }

  const int N_orig = static_cast<int>(problem->ops.size());
  auto pr = solver::Preprocess(*problem);
  auto& p = pr.problem;
  auto dag = solver::BuildDAG(p);
  const int N = static_cast<int>(p.ops.size());

  int bm_num = solver::ParseBenchmarkNumber(problem_path);
  double timeout = (timeout_override > 0)
                       ? timeout_override
                       : solver::CompetitionTimeout(N_orig, bm_num);

  std::cerr << "=== pattern_solver ===\n"
            << "  input:    " << problem_path << " (N=" << N_orig
            << (N != N_orig ? " -> " + std::to_string(N) + " post-preprocess"
                            : "")
            << ")\n"
            << "  timeout:  " << std::fixed << std::setprecision(1) << timeout
            << "s  (BM-" << bm_num << ")\n";

  // ── Phase 1: column enumeration ─────────────────────────────────────
  auto t_enum = std::chrono::steady_clock::now();
  auto columns = pattern::EnumerateColumns(p, dag);
  double enum_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_enum).count();

  int n_by_class[8] = {0};
  for (const auto& c : columns) n_by_class[(int)c.klass]++;
  std::cerr << "  [enum] " << columns.size() << " feasible columns in "
            << std::fixed << std::setprecision(2) << enum_s << "s"
            << "  (C0=" << n_by_class[0]
            << " C1=" << n_by_class[1]
            << " C3=" << n_by_class[3]
            << " C4=" << n_by_class[4]
            << " C5=" << n_by_class[5]
            << " C6=" << n_by_class[6]
            << " C7=" << n_by_class[7] << ")\n";

  // ── Incremental best-so-far writer ──────────────────────────────────
  // Every variant's Evaluate result goes through try_write. If GT is a
  // new minimum, serialize the solution to <output_path>.tmp, then
  // rename() to output_path (atomic on a single filesystem). If the
  // harness kills us mid-execution, whatever was last renamed is on
  // disk. The double-checked lock keeps late-finishing variants from
  // overwriting a better-scoring earlier one.
  std::atomic<double> best_gt{std::numeric_limits<double>::infinity()};
  std::mutex write_mutex;
  auto try_write = [&](const mlsys::Solution& sol, double gt,
                        const char* tag) {
    if (!std::isfinite(gt) || gt >= best_gt.load()) return;
    std::lock_guard<std::mutex> g(write_mutex);
    if (gt >= best_gt.load()) return;
    auto final_sol = solver::UnPreprocess(sol, pr);
    std::string tmp = std::string(output_path) + ".tmp";
    solver::WriteSolution(final_sol, tmp);
    if (std::rename(tmp.c_str(), output_path) == 0) {
      best_gt.store(gt);
      std::cerr << "  [write " << tag << "] gt=" << (int64_t)gt << "\n";
    } else {
      std::remove(tmp.c_str());
    }
  };

  // ── Phase 1.5: greedy baseline write ────────────────────────────────
  // A cheap-but-valid solution on disk by t < 1s, so even an unexpectedly
  // early SIGTERM/SIGKILL leaves us with points instead of an empty file.
  // The triple-MIP below will overwrite this if any variant improves.
  {
    auto part0 = pattern::GreedyPack(columns, p.ops.size());
    part0 = solver::ConvexityRepair(dag, std::move(part0));
    part0 = solver::StableTopoSort(dag, std::move(part0));
    auto sol0 = solver::BuildSolution(p, dag, part0);
    for (auto& sg : sol0.subgraphs) sg.subgraph_latency = 0.0;
    auto g0 = mlsys::Evaluate(p, sol0);
    if (g0.ok()) try_write(sol0, *g0, "greedy");
  }

  // ── Phase 2–5: triple-MIP wrapper (E / A / R, run in parallel) ─────
  //
  // The MIP's column costs (per-sg, retained_in={}) ignore cross-sg
  // retention interactions. A partition with lower sum(column cost) can
  // have WORSE actual GT if its ops have fewer retention opportunities.
  //
  // Mitigation: three cost metrics in parallel:
  //   E = ExactCost                                 (Evaluate-faithful)
  //   A = AnalyticalCost                            (MIP-legacy, correlates with retention-friendly)
  //   R = AnalyticalCost with assume_retain_out=true(retention-optimistic)
  // Each variant runs the full downstream pipeline (BuildSolution +
  // RetentionPass + GranularityRefine) and evaluates GT; we keep min.
  //
  // Parallelism: the variants share only read-only data (p, dag, columns).
  // Each MipPack call creates its own Highs instance. Core passes
  // (BuildSolution, RetentionPass, GranularityRefinement, Evaluate) have
  // no mutable globals. Running on separate threads trades 1-thread wall
  // time for ≥3-core machines — on competition hardware (≥8 cores) this
  // brings wall time close to a single variant's cost.
  //
  // Per-thread stderr is buffered in a std::ostringstream and flushed
  // after the join, so per-variant output doesn't interleave.
  auto RunVariant = [&](pattern::CostVariant vari, const char* tag)
      -> std::tuple<mlsys::Solution, double, std::string> {
    std::ostringstream log;
    double remaining_v = timeout - std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double mip_budget_v = std::max(0.5, remaining_v * 0.5);
    auto mip_v = pattern::MipPack(columns, p.ops.size(), mip_budget_v, vari);
    solver::Partition part_v;
    if (mip_v.n_selected > 0) {
      part_v = std::move(mip_v.partition);
      log << "  [mip " << tag << "]  "
          << part_v.subgraphs.size() << " subgraphs, "
          << "obj=" << (int64_t)mip_v.objective
          << " " << (mip_v.is_optimal ? "OPT" : "feas")
          << " in " << std::fixed << std::setprecision(2)
          << mip_v.solve_time_s << "s\n";
    } else {
      part_v = pattern::GreedyPack(columns, p.ops.size());
      log << "  [greedy fallback " << tag << "]  "
          << part_v.subgraphs.size() << " subgraphs\n";
    }
    part_v = solver::ConvexityRepair(dag, std::move(part_v));
    part_v = solver::StableTopoSort(dag, std::move(part_v));

    // Build solution: BuildSolution for small N (includes pairwise retention
    // search); minimal BestGranularity path for large N.
    mlsys::Solution sol_v;
    if (part_v.subgraphs.size() <= 100) {
      sol_v = solver::BuildSolution(p, dag, part_v);
    } else {
      std::unordered_map<size_t, size_t> topo_pos;
      for (size_t i = 0; i < dag.topo_order.size(); ++i)
        topo_pos[dag.topo_order[i]] = i;
      std::unordered_set<size_t> retained_in_b;
      int64_t retained_size = 0;
      for (const auto& ops_vec : part_v.subgraphs) {
        auto ops_copy = ops_vec;
        std::sort(ops_copy.begin(), ops_copy.end(),
                  [&](size_t a, size_t b) { return topo_pos[a] < topo_pos[b]; });
        auto cfg = solver::BestGranularity(p, ops_copy, retained_in_b,
                                            retained_size, &dag);
        mlsys::Subgraph sg;
        sg.ops = std::move(ops_copy);
        sg.granularity = cfg.gran;
        if (!cfg.traversal.empty()) sg.traversal_order = cfg.traversal;
        sg.subgraph_latency = 0.0;
        sol_v.subgraphs.push_back(std::move(sg));
        retained_in_b.clear();
        retained_size = 0;
      }
    }
    for (auto& sg : sol_v.subgraphs) sg.subgraph_latency = 0.0;

    // Retention + gran refine — each variant gets its own budget (runs
    // concurrently). Upper caps are per-variant so 3× concurrency does
    // NOT multiply them.
    double remain_ret = timeout - std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double retain_budget = std::min(30.0, std::max(1.0, remain_ret * 0.4));
    auto ret = pattern::RetentionPass(p, dag, std::move(sol_v), retain_budget);
    sol_v = std::move(ret.solution);
    if (ret.n_retentions_added > 0) {
      log << "  [retain " << tag << "] +"
          << ret.n_retentions_added << " retentions  "
          << (int64_t)ret.cost_before << " -> "
          << (int64_t)ret.cost_after << "\n";
    }
    double remain2 = timeout - std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double refine_budget = std::min(15.0, std::max(1.0, remain2 * 0.3));
    auto refine = pattern::GranularityRefinement(p, dag, std::move(sol_v),
                                                  refine_budget);
    sol_v = std::move(refine.solution);
    if (refine.n_refined > 0) {
      log << "  [gran " << tag << "]   refined "
          << refine.n_refined << " sgs  " << (int64_t)refine.cost_before
          << " -> " << (int64_t)refine.cost_after << "\n";
    }
    for (auto& sg : sol_v.subgraphs) sg.subgraph_latency = 0.0;
    auto g = mlsys::Evaluate(p, sol_v);
    double gt = g.ok() ? *g : std::numeric_limits<double>::infinity();
    if (g.ok()) try_write(sol_v, gt, tag);
    return {std::move(sol_v), gt, log.str()};
  };

  // Launch three variants on separate threads. std::async with
  // std::launch::async forces thread creation (not deferred).
  auto t_ear = std::chrono::steady_clock::now();
  auto f_E = std::async(std::launch::async, RunVariant,
                         pattern::CostVariant::E, "E");
  auto f_A = std::async(std::launch::async, RunVariant,
                         pattern::CostVariant::A, "A");
  auto f_R = std::async(std::launch::async, RunVariant,
                         pattern::CostVariant::R, "R");
  auto [sol_E, gt_E, log_E] = f_E.get();
  auto [sol_A, gt_A, log_A] = f_A.get();
  auto [sol_R, gt_R, log_R] = f_R.get();
  double ear_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_ear).count();
  std::cerr << log_E << log_A << log_R
            << "  [variants] joined in " << std::fixed
            << std::setprecision(2) << ear_s << "s\n";
  mlsys::Solution sol;
  double gt_val;
  const char* picked;
  if (gt_E <= gt_A && gt_E <= gt_R) { sol = std::move(sol_E); gt_val = gt_E; picked = "E"; }
  else if (gt_A <= gt_R)            { sol = std::move(sol_A); gt_val = gt_A; picked = "A"; }
  else                              { sol = std::move(sol_R); gt_val = gt_R; picked = "R"; }
  // Report the ACTUAL on-disk GT (may be the greedy baseline if it beat every
  // variant). The per-variant trio is informational only.
  double on_disk_gt = best_gt.load();
  std::cerr << "\n  Ground truth (mlsys::Evaluate): " << (int64_t)on_disk_gt
            << "  [variants picked " << picked << ": E=" << (int64_t)gt_E
            << " A=" << (int64_t)gt_A << " R=" << (int64_t)gt_R << "]\n";
  // All three variants failed to Evaluate — very rare; the greedy
  // baseline write from Phase 1.5 should still be on disk if feasible.
  if (!std::isfinite(gt_val) && !std::isfinite(best_gt.load())) {
    std::cerr << "Evaluate: all variants failed and no baseline on disk\n";
    return 1;
  }
  // Solution already on disk via try_write (greedy baseline or best
  // variant, whichever scored lowest). No final WriteSolution needed.

  // ── Phase 6: local search ──────────────────────────────────────────
  // Hill-climb over Merge/Split/Repartition moves with the remaining
  // budget. Each accepted improvement is flushed via try_write, so the
  // on-disk solution keeps the atomic-write invariant the SIGTERM
  // handler relies on.
  double remain_ls = timeout - std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  // Keep a small safety margin so Evaluate + rename for the final
  // accepted improvement can finish before the competition harness
  // delivers SIGTERM.
  constexpr double kLsSafetyMargin = 0.5;
  double ls_budget = remain_ls - kLsSafetyMargin;
  if (std::isfinite(best_gt.load()) && ls_budget > 1.0) {
    auto ls_cb = [&](const mlsys::Solution& s, double gt) {
      try_write(s, gt, "ls");
    };
    auto t_ls = std::chrono::steady_clock::now();
    auto ls = pattern::LocalSearch(p, dag, sol, ls_budget, ls_cb);
    double ls_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_ls).count();
    if (ls.n_improvements > 0) {
      std::cerr << "  [ls]     +" << ls.n_improvements
                << " improvements  hc=" << ls.n_hc_moves
                << " sa=" << ls.n_sa_moves
                << " (sa_accept=" << ls.n_sa_accepts << ")  "
                << (int64_t)ls.cost_before << " -> "
                << (int64_t)ls.cost_after
                << "  (" << std::fixed << std::setprecision(2) << ls_s
                << "s)\n";
    } else {
      std::cerr << "  [ls]     no improvement  hc=" << ls.n_hc_moves
                << " sa=" << ls.n_sa_moves
                << " (sa_accept=" << ls.n_sa_accepts << ")  ("
                << std::fixed << std::setprecision(2) << ls_s << "s)\n";
    }
  }

  double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  std::cerr << "\nTotal elapsed: " << std::fixed << std::setprecision(2)
            << elapsed << "s  →  " << output_path
            << "  (final on-disk GT=" << (int64_t)best_gt.load() << ")\n";
  return 0;
}
