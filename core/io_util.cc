// io_util.cc — Solution writer + timing helpers.

#include "io_util.h"

#include <cstdio>
#include <fstream>

#include "nlohmann/json.hpp"

namespace solver {

using json = nlohmann::json;

void WriteSolution(const mlsys::Solution& sol, const std::string& fname) {
  json j;
  json sgs = json::array(), grans = json::array();
  json rets = json::array(), travs = json::array(), lats = json::array();
  for (auto& sg : sol.subgraphs) {
    sgs.push_back(sg.ops);
    grans.push_back(
        {sg.granularity.width, sg.granularity.height, sg.granularity.depth});
    rets.push_back(sg.tensors_to_retain);
    travs.push_back(sg.traversal_order.has_value()
                         ? json(sg.traversal_order.value())
                         : json(nullptr));
    // Emit 0 — the evaluator recomputes per-subgraph latency, and the spec
    // treats 0 as "not provided" (skips the reported-vs-computed check).
    // Solver analytical cost is an approximation and should not be trusted
    // as authoritative.
    lats.push_back(0.0);
  }
  j["subgraphs"] = sgs;
  j["granularities"] = grans;
  j["tensors_to_retain"] = rets;
  j["traversal_orders"] = travs;
  j["subgraph_latencies"] = lats;
  std::ofstream f(fname);
  f << j.dump(2) << "\n";
}

double ElapsedSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
}

double CompetitionTimeout(int n_ops) {
  if (n_ops <= 8) return 2.0;
  if (n_ops <= 20) return 5.0;
  if (n_ops <= 40) return 15.0;
  if (n_ops <= 80) return 30.0;
  if (n_ops <= 110) return 60.0;
  return 120.0;
}

int ParseBenchmarkNumber(const std::string& filename) {
  auto pos = filename.rfind('/');
  const char* base = (pos != std::string::npos)
                         ? filename.c_str() + pos + 1
                         : filename.c_str();
  int n = 0;
  if (std::sscanf(base, "mlsys-2026-%d.json", &n) == 1 && n >= 1 && n <= 24)
    return n;
  return 0;
}

double CompetitionTimeout(int n_ops, int bm_number) {
  if (bm_number >= 1 && bm_number <= 4) return 2.0;
  if (bm_number >= 5 && bm_number <= 8) return 5.0;
  if (bm_number >= 9 && bm_number <= 12) return 15.0;
  if (bm_number >= 13 && bm_number <= 16) return 30.0;
  if (bm_number >= 17 && bm_number <= 20) return 60.0;
  if (bm_number >= 21 && bm_number <= 24) return 120.0;
  return CompetitionTimeout(n_ops);
}

}  // namespace solver
