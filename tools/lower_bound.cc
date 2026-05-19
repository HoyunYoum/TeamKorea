#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "mlsys.h"

namespace {

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

struct Bounds {
  double compute_lb;
  double io_lb;
  double overall_lb;
};

// Compute LB:  Σ base_cost_op × ceil(W_op/native_w) × ceil(H_op/native_h)
// IO LB:       (Σ graph_input_sizes + Σ graph_output_sizes) / bandwidth
Bounds ComputeLowerBounds(const mlsys::Problem& p) {
  const int64_t native_w = p.native_granularity.width;
  const int64_t native_h = p.native_granularity.height;
  const size_t num_tensors = p.tensors.size();

  std::vector<bool> is_graph_input(num_tensors, true);
  std::vector<bool> is_graph_output(num_tensors, true);
  for (const auto& op : p.ops) {
    for (size_t t : op.outputs) is_graph_input[t] = false;
    for (size_t t : op.inputs) is_graph_output[t] = false;
  }

  double compute_lb = 0.0;
  for (const auto& op : p.ops) {
    size_t out_t = op.outputs.front();
    int64_t W = p.tensors[out_t].width;
    int64_t H = p.tensors[out_t].height;
    int64_t tiles = CeilDiv(W, native_w) * CeilDiv(H, native_h);
    compute_lb += static_cast<double>(op.base_cost) * tiles;
  }

  int64_t io_bytes = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    if (is_graph_input[i] || is_graph_output[i]) {
      io_bytes += p.tensors[i].width * p.tensors[i].height;
    }
  }
  double io_lb = static_cast<double>(io_bytes) / p.slow_memory_bandwidth;

  return {compute_lb, io_lb, std::max(compute_lb, io_lb)};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s <problem.json>\n", argv[0]);
    return 1;
  }
  auto p_or = mlsys::ReadProblem(argv[1]);
  if (!p_or.ok()) {
    std::fprintf(stderr, "read problem: %s\n",
                 std::string(p_or.status().message()).c_str());
    return 1;
  }
  const auto& p = *p_or;

  Bounds b = ComputeLowerBounds(p);
  std::printf("problem:       %s\n", argv[1]);
  std::printf("ops:           %zu\n", p.ops.size());
  std::printf("tensors:       %zu\n", p.tensors.size());
  std::printf("fast_mem:      %ld\n",
              static_cast<long>(p.fast_memory_capacity));
  std::printf("bandwidth:     %ld\n",
              static_cast<long>(p.slow_memory_bandwidth));
  std::printf("native:        %ld x %ld\n",
              static_cast<long>(p.native_granularity.width),
              static_cast<long>(p.native_granularity.height));
  std::printf("compute_lb:    %.1f\n", b.compute_lb);
  std::printf("io_lb:         %.1f\n", b.io_lb);
  std::printf("overall_lb:    %.1f\n", b.overall_lb);
  return 0;
}
