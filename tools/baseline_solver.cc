// Baseline solver: each op in its own subgraph at native granularity.
// Picks the largest k that fits the working set for MatMul (may split K).
// Produces a valid solution on any benchmark; makes no fusion or reuse effort.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "mlsys.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace {

int64_t MaxKThatFits(int64_t h, int64_t w, int64_t K, int64_t capacity) {
  // MatMul WS: h*k + k*w + w*h  ≤  capacity.
  // ⇒ k ≤ (capacity - w*h) / (h + w).
  int64_t budget = capacity - w * h;
  if (budget <= 0) return 0;
  int64_t k_max = budget / (h + w);
  return std::min(K, std::max<int64_t>(1, k_max));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s <problem.json> <solution.json>\n", argv[0]);
    return 1;
  }

  auto p_or = mlsys::ReadProblem(argv[1]);
  if (!p_or.ok()) {
    std::fprintf(stderr, "read problem: %s\n",
                 std::string(p_or.status().message()).c_str());
    return 1;
  }
  const auto& p = *p_or;

  json out;
  out["subgraphs"] = json::array();
  out["granularities"] = json::array();
  out["tensors_to_retain"] = json::array();
  out["traversal_orders"] = json::array();
  out["subgraph_latencies"] = json::array();

  for (size_t op_idx = 0; op_idx < p.ops.size(); ++op_idx) {
    const auto& op = p.ops[op_idx];
    size_t out_t = op.outputs.front();
    int64_t W = p.tensors[out_t].width;
    int64_t H = p.tensors[out_t].height;
    int64_t w = std::min(p.native_granularity.width, W);
    int64_t h = std::min(p.native_granularity.height, H);
    int64_t k = 1;

    if (op.op_type == "MatMul") {
      size_t lhs_t = op.inputs[0];
      int64_t K = p.tensors[lhs_t].width;
      k = MaxKThatFits(h, w, K, p.fast_memory_capacity);
      if (k <= 0) {
        std::fprintf(stderr, "op %zu: cannot fit MatMul working set\n", op_idx);
        return 1;
      }
    } else {
      // Pointwise: WS = (N_inputs + 1) × w × h.  Shrink h (then w) until fits.
      int64_t slots = static_cast<int64_t>(op.inputs.size()) + 1;
      while (slots * w * h > p.fast_memory_capacity) {
        if (h > 1) {
          h = std::max<int64_t>(1, h / 2);
        } else if (w > 1) {
          w = std::max<int64_t>(1, w / 2);
        } else {
          std::fprintf(stderr, "op %zu: cannot fit Pointwise working set\n",
                       op_idx);
          return 1;
        }
      }
    }

    out["subgraphs"].push_back({op_idx});
    out["granularities"].push_back({w, h, k});
    out["tensors_to_retain"].push_back(json::array());
    out["traversal_orders"].push_back(nullptr);
    out["subgraph_latencies"].push_back(0);
  }

  std::ofstream f(argv[2]);
  f << out.dump(2) << "\n";
  return 0;
}
