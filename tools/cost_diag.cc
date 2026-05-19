// cost_diag.cc — Per-subgraph diagnostic: isolate Evaluate's per-sg latency
// by abusing its subgraph_latency validation (any non-zero reported value is
// checked against Evaluate's own computation; mismatch error reports both).
//
// Usage:  cost_diag <problem.json> <solution.json>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "mlsys.h"
#include "solver_common.h"

namespace {

// Wedge a sentinel latency into sg[i] and read back Evaluate's true value
// from the validation error message ("reported X does not match computed Y").
double EvaluatePerSgViaValidation(const mlsys::Problem& p,
                                  mlsys::Solution sol, size_t sg_idx) {
  sol.subgraphs[sg_idx].subgraph_latency = 1.0;  // intentional mismatch
  auto res = mlsys::Evaluate(p, sol);
  if (res.ok()) return -1.0;  // sentinel somehow matched — sg is exactly 1.0
  std::string msg(res.status().message());
  // Parse "... computed Y"
  auto pos = msg.find("computed ");
  if (pos == std::string::npos) return -2.0;
  return std::atof(msg.c_str() + pos + 9);
}

void Report(const mlsys::Problem& p, const mlsys::Solution& sol) {
  auto dag = solver::BuildDAG(p);

  auto gt_or = mlsys::Evaluate(p, sol);
  if (!gt_or.ok()) {
    std::cerr << "Evaluate failed: " << gt_or.status() << "\n";
    return;
  }
  double gt = *gt_or;

  std::unordered_set<size_t> retained_in;
  double sum_exact = 0.0, sum_evaluate = 0.0;
  int diverge_count = 0;

  std::cout << std::fixed << std::setprecision(1);
  std::cout << "idx  nops  w   h   d   retain_in  retain_out  exact       evaluate    diff\n";
  std::cout << "---  ----  --- --- --- ---------  ----------  ----------  ----------  --------\n";

  for (size_t i = 0; i < sol.subgraphs.size(); ++i) {
    const auto& sg = sol.subgraphs[i];
    auto ts = solver::Classify(p, sg.ops, &dag);
    auto roles = solver::BuildRoles(p, sg.ops, ts);

    const std::vector<int64_t>* trav = nullptr;
    std::vector<int64_t> trav_buf;
    if (sg.traversal_order) {
      trav_buf.assign(sg.traversal_order->begin(), sg.traversal_order->end());
      trav = &trav_buf;
    }

    double exact = solver::ExactCostCached(p, sg.ops, sg.granularity,
                                           sg.tensors_to_retain, retained_in,
                                           ts, roles, trav);
    double evaluate = EvaluatePerSgViaValidation(p, sol, i);

    sum_exact += exact;
    sum_evaluate += evaluate;

    double diff = evaluate - exact;
    const char* flag = (std::abs(diff) > 0.5) ? "  <<" : "";
    if (std::abs(diff) > 0.5) ++diverge_count;
    std::cout << std::setw(3) << i << "  " << std::setw(4) << sg.ops.size()
              << "  " << std::setw(3) << sg.granularity.width
              << " " << std::setw(3) << sg.granularity.height
              << " " << std::setw(3) << sg.granularity.depth
              << "  " << std::setw(9) << retained_in.size()
              << "  " << std::setw(10) << sg.tensors_to_retain.size()
              << "  " << std::setw(10) << exact
              << "  " << std::setw(10) << evaluate
              << "  " << std::setw(8) << diff << flag << "\n";

    retained_in.clear();
    for (size_t t : sg.tensors_to_retain) retained_in.insert(t);
  }

  std::cout << "---\n";
  std::cout << "sum_exact     = " << sum_exact << "\n";
  std::cout << "sum_evaluate  = " << sum_evaluate << "\n";
  std::cout << "GT Evaluate   = " << gt << "\n";
  std::cout << "diverging sgs = " << diverge_count << "\n";
  std::cout << "exact - GT    = " << (sum_exact - gt) << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <problem.json> <solution.json>\n";
    return 1;
  }
  auto pr = mlsys::ReadProblem(argv[1]);
  if (!pr.ok()) { std::cerr << pr.status() << "\n"; return 1; }
  auto so = mlsys::ReadSolution(argv[2]);
  if (!so.ok()) { std::cerr << so.status() << "\n"; return 1; }
  Report(*pr, *so);
  return 0;
}
