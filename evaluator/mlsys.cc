#include "mlsys.h"

#include "mm_role.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "nlohmann/json.hpp"

namespace mlsys {

using json = nlohmann::json;

absl::StatusOr<Problem> ReadProblem(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return absl::NotFoundError("Could not open file: " + filename);
  }

  json j;
  try {
    file >> j;
  } catch (const json::parse_error& e) {
    return absl::InvalidArgumentError(std::string("JSON parse error: ") +
                                      e.what());
  }

  Problem problem;

  auto& widths = j["widths"];
  auto& heights = j["heights"];
  if (widths.size() != heights.size()) {
    return absl::InvalidArgumentError(
        "widths and heights arrays must have the same size");
  }
  for (size_t i = 0; i < widths.size(); ++i) {
    problem.tensors.push_back(
        {.width = widths[i].get<Width>(), .height = heights[i].get<Height>()});
  }

  auto& inputs = j["inputs"];
  auto& outputs = j["outputs"];
  auto& base_costs = j["base_costs"];
  auto& op_types = j["op_types"];
  size_t num_ops = inputs.size();
  if (outputs.size() != num_ops || base_costs.size() != num_ops ||
      op_types.size() != num_ops) {
    return absl::InvalidArgumentError("Op arrays must have the same size");
  }
  for (size_t i = 0; i < num_ops; ++i) {
    Op op;
    op.op_type = op_types[i].get<OpType>();
    op.base_cost = base_costs[i].get<BaseCost>();
    for (auto& idx : inputs[i]) {
      op.inputs.push_back(idx.get<size_t>());
    }
    for (auto& idx : outputs[i]) {
      op.outputs.push_back(idx.get<size_t>());
    }
    problem.ops.push_back(std::move(op));
  }

  problem.fast_memory_capacity =
      j["fast_memory_capacity"].get<FastMemoryCapacity>();
  problem.slow_memory_bandwidth =
      j["slow_memory_bandwidth"].get<SlowMemoryBandwidth>();

  auto& ng = j["native_granularity"];
  problem.native_granularity = {
      .width = ng[0].get<Width>(),
      .height = ng[1].get<Height>(),
      .depth = 1,
  };

  return problem;
}

absl::StatusOr<Solution> ReadSolution(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return absl::NotFoundError("Could not open file: " + filename);
  }

  json j;
  try {
    file >> j;
  } catch (const json::parse_error& e) {
    return absl::InvalidArgumentError(std::string("JSON parse error: ") +
                                      e.what());
  }

  Solution solution;

  auto& subgraphs = j["subgraphs"];
  auto& granularities = j["granularities"];
  auto& tensors_to_retain = j["tensors_to_retain"];
  auto& subgraph_latencies = j["subgraph_latencies"];

  size_t num_subgraphs = subgraphs.size();
  bool has_traversal = j.contains("traversal_orders");

  for (size_t i = 0; i < num_subgraphs; ++i) {
    Subgraph sg;

    for (auto& op_idx : subgraphs[i]) {
      sg.ops.push_back(op_idx.get<size_t>());
    }

    auto& g = granularities[i];
    sg.granularity = {
        .width = g[0].get<Width>(),
        .height = g[1].get<Height>(),
        .depth = g[2].get<Depth>(),
    };

    for (auto& t : tensors_to_retain[i]) {
      sg.tensors_to_retain.push_back(t.get<size_t>());
    }

    if (has_traversal && !j["traversal_orders"][i].is_null()) {
      TraversalOrder order;
      for (auto& idx : j["traversal_orders"][i]) {
        order.push_back(idx.get<int64_t>());
      }
      sg.traversal_order = std::move(order);
    }

    sg.subgraph_latency = subgraph_latencies[i].get<SubgraphLatency>();
    solution.subgraphs.push_back(std::move(sg));
  }

  return solution;
}

namespace {

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// MMRole lives in mm_role.h — shared with the solver for consistency.
using ::solver::MMRole;

struct SliceKey {
  int h_tile = -1;
  int w_tile = -1;
  int k_step = -1;
  bool operator==(const SliceKey& o) const = default;
  bool operator!=(const SliceKey& o) const = default;
};

}  // namespace

absl::StatusOr<TotalLatency> Evaluate(const Problem& problem,
                                      const Solution& solution) {
  const size_t num_tensors = problem.tensors.size();
  const size_t num_ops = problem.ops.size();
  const auto bw = static_cast<double>(problem.slow_memory_bandwidth);
  const int64_t native_w = problem.native_granularity.width;
  const int64_t native_h = problem.native_granularity.height;
  const int64_t capacity = problem.fast_memory_capacity;

  // Build graph metadata.
  std::vector<int> producer_of(num_tensors, -1);
  std::vector<std::vector<std::pair<size_t, int>>> consumers_of(num_tensors);
  std::vector<bool> is_graph_input(num_tensors, true);
  std::vector<bool> is_graph_output(num_tensors, true);

  for (size_t i = 0; i < num_ops; ++i) {
    for (size_t t : problem.ops[i].outputs) {
      producer_of[t] = static_cast<int>(i);
      is_graph_input[t] = false;
    }
    for (int pos = 0; pos < static_cast<int>(problem.ops[i].inputs.size());
         ++pos) {
      size_t t = problem.ops[i].inputs[pos];
      consumers_of[t].emplace_back(i, pos);
      is_graph_output[t] = false;
    }
  }

  for (size_t i = 0; i < num_ops; ++i) {
    if (problem.ops[i].op_type == "MatMul") {
      if (problem.ops[i].inputs.size() != 2) {
        return absl::InvalidArgumentError("MatMul must have exactly 2 inputs");
      }
      if (problem.ops[i].outputs.size() != 1) {
        return absl::InvalidArgumentError("MatMul must have exactly 1 output");
      }
      size_t lhs_t = problem.ops[i].inputs[0];
      size_t rhs_t = problem.ops[i].inputs[1];
      size_t out_t = problem.ops[i].outputs[0];
      if (problem.tensors[lhs_t].width != problem.tensors[rhs_t].height) {
        return absl::InvalidArgumentError(
            absl::StrCat("MatMul op ", i, ": LHS.width (",
                         problem.tensors[lhs_t].width, ") != RHS.height (",
                         problem.tensors[rhs_t].height, ")"));
      }
      if (problem.tensors[out_t].height != problem.tensors[lhs_t].height ||
          problem.tensors[out_t].width != problem.tensors[rhs_t].width) {
        return absl::InvalidArgumentError(absl::StrCat(
            "MatMul op ", i, ": output shape (w=",
            problem.tensors[out_t].width, ", h=",
            problem.tensors[out_t].height, ") does not match LHS.height (",
            problem.tensors[lhs_t].height, ") x RHS.width (",
            problem.tensors[rhs_t].width, ")"));
      }
    }
  }

  // Build op -> subgraphs map for global ephemeral check.
  // op_in_subgraphs[op] = set of subgraph indices containing that op.
  std::vector<std::set<size_t>> op_in_subgraphs(num_ops);
  for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
    for (size_t op : solution.subgraphs[sg_idx].ops) {
      op_in_subgraphs[op].insert(sg_idx);
    }
  }

  // Validate: every op must appear in at least one subgraph.
  for (size_t op = 0; op < num_ops; ++op) {
    if (op_in_subgraphs[op].empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Op ", op, " not scheduled in any subgraph"));
    }
  }

  std::set<size_t> retained_set;
  double total_latency = 0.0;

  for (size_t sg_idx = 0; sg_idx < solution.subgraphs.size(); ++sg_idx) {
    const Subgraph& sg = solution.subgraphs[sg_idx];
    const int64_t w = sg.granularity.width;
    const int64_t h = sg.granularity.height;
    const int64_t k = sg.granularity.depth;

    if (w > native_w || h > native_h) {
      return absl::InvalidArgumentError("Tile size exceeds native granularity");
    }
    if (w <= 0 || h <= 0 || k <= 0) {
      return absl::InvalidArgumentError("Granularity must be positive");
    }

    const std::set<size_t> sg_op_set(sg.ops.begin(), sg.ops.end());
    const std::set<size_t> retain_set(sg.tensors_to_retain.begin(),
                                      sg.tensors_to_retain.end());

    // Validate: no duplicate op indices within a subgraph.
    if (sg_op_set.size() != sg.ops.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Subgraph ", sg_idx, ": duplicate op indices in subgraph"));
    }

    // Validate: producer-before-consumer order for intra-subgraph tensor flow.
    // If op B consumes tensor T and T's producer A is also in this subgraph,
    // A must appear before B in sg.ops so ephemeral data flows correctly.
    {
      std::map<size_t, size_t> op_position;
      for (size_t i = 0; i < sg.ops.size(); ++i) op_position[sg.ops[i]] = i;
      for (size_t i = 0; i < sg.ops.size(); ++i) {
        size_t op_idx = sg.ops[i];
        for (size_t t : problem.ops[op_idx].inputs) {
          int prod = producer_of[t];
          if (prod < 0) continue;
          auto pit = op_position.find(static_cast<size_t>(prod));
          if (pit == op_position.end()) continue;
          if (pit->second >= i) {
            return absl::InvalidArgumentError(absl::StrCat(
                "Subgraph ", sg_idx, ": op ", op_idx,
                " consumes tensor ", t, " produced by op ", prod,
                " which appears at the same or later position"));
          }
        }
      }
    }

    for (size_t t : sg.tensors_to_retain) {
      if (t >= num_tensors) {
        return absl::InvalidArgumentError(
            absl::StrCat("Subgraph ", sg_idx, ": retained tensor index ", t,
                         " out of range"));
      }
      if (is_graph_input[t]) {
        return absl::InvalidArgumentError("Cannot retain graph input tensor");
      }
      // Spec #34 + #52: retention lifetime = exactly one step, and every
      // tensors_to_retain entry must be an op output of *this* subgraph.
      // Pass-through re-retention (just re-listing a carried-over tensor
      // without producing it here) is NOT legal — to carry a tensor past
      // one hop, the intermediate subgraph must recompute (re-execute the
      // producing op) and then retain.
      int prod = producer_of[t];
      bool produced_here =
          prod >= 0 && sg_op_set.count(static_cast<size_t>(prod)) > 0;
      if (!produced_here) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Subgraph ", sg_idx, ": cannot retain tensor ", t,
            " which is not produced by any op in this subgraph "
            "(spec #34/#52: retention lifetime is one step; to carry further, "
            "the intermediate subgraph must recompute the producer)"));
      }
      if (is_graph_output[t]) {
        // Graph outputs must be materialized to slow memory — that is
        // the purpose of a graph output. Retaining them would let a
        // solver bypass their eviction cost for a result that never
        // actually leaves fast memory, which is a scoring loophole.
        return absl::InvalidArgumentError(absl::StrCat(
            "Subgraph ", sg_idx, ": cannot retain tensor ", t,
            " which is a graph output (must be evicted to slow memory)"));
      }
    }

    // Validate: if a tensor produced in this subgraph is also consumed
    // internally (making it an ephemeral candidate), then every external
    // consumer must have the producer in its own subgraph (recompute).
    // Otherwise the tensor is ephemeral (never materialized) yet needed
    // externally — invalid per Issue #51.
    for (size_t op : sg.ops) {
      for (size_t t : problem.ops[op].outputs) {
        if (retain_set.count(t)) continue;
        if (is_graph_output[t]) continue;
        bool has_internal = false;
        bool has_uncovered_external = false;
        size_t uncovered_consumer = 0;
        for (auto& [consumer_op, unused_pos] : consumers_of[t]) {
          if (sg_op_set.count(consumer_op)) {
            has_internal = true;
          } else {
            bool covered = false;
            for (size_t other_sg : op_in_subgraphs[consumer_op]) {
              const auto& other_ops = solution.subgraphs[other_sg].ops;
              if (std::find(other_ops.begin(), other_ops.end(), op) !=
                  other_ops.end()) {
                covered = true;
                break;
              }
            }
            if (!covered) {
              has_uncovered_external = true;
              uncovered_consumer = consumer_op;
            }
          }
        }
        // Only error if tensor would be ephemeral (has internal consumer)
        // but also has an uncovered external consumer.
        if (has_internal && has_uncovered_external) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Subgraph ", sg_idx, ": tensor ", t,
              " would be ephemeral but has external consumer (op ",
              uncovered_consumer,
              ") whose subgraph does not recompute producer (op ", op, ")"));
        }
      }
    }

    // Ephemeral: produced in sg, consumed in sg, not graph output, not retained.
    // External consumer validity is already enforced by the validation above.
    auto is_ephemeral = [&](size_t t) -> bool {
      int prod = producer_of[t];
      if (prod < 0) return false;
      if (!sg_op_set.count(static_cast<size_t>(prod))) return false;
      bool consumed_in_sg = false;
      for (auto& [consumer_op, unused_pos] : consumers_of[t]) {
        if (sg_op_set.count(consumer_op)) {
          consumed_in_sg = true;
          break;
        }
      }
      if (!consumed_in_sg) return false;
      if (is_graph_output[t]) return false;
      if (retain_set.count(t)) return false;
      return true;
    };

    // Does this produced-in-sg tensor need eviction to slow memory?
    auto needs_eviction = [&](size_t t) -> bool {
      int prod = producer_of[t];
      if (prod < 0) return false;
      if (!sg_op_set.count(static_cast<size_t>(prod))) return false;
      if (is_ephemeral(t)) return false;
      if (retain_set.count(t)) return false;
      return true;
    };

    // Classify ops.
    std::vector<size_t> mm_ops, pw_ops;
    for (size_t op : sg.ops) {
      if (problem.ops[op].op_type == "MatMul")
        mm_ops.push_back(op);
      else
        pw_ops.push_back(op);
    }

    // ── Topology-based MMRole detection ─────────────────────────────────
    // For every MatMul op in the subgraph, determine per-direction chain
    // membership (§8 symmetry: LHS-chain and RHS-chain mirror):
    //   lhs_ephemeral     : LHS input produced by another MM in sg (ephemeral)
    //   out_ephemeral     : output ephemeral consumed as LHS by another MM
    //   rhs_ephemeral     : RHS input produced by another MM in sg (ephemeral)
    //   out_ephemeral_rhs : output ephemeral consumed as RHS by another MM
    // This classifies each MM as Standalone / Head / Tail / Middle in either
    // direction, without being tied to chain length.
    auto get_K_op = [&](size_t op) -> int64_t {
      return problem.tensors[problem.ops[op].inputs[0]].width;
    };

    std::map<size_t, MMRole> roles;
    for (size_t op : sg.ops) {
      MMRole r;
      if (problem.ops[op].op_type == "MatMul") r.is_matmul = true;
      roles[op] = r;
    }
    for (size_t op : mm_ops) {
      const auto& opdef = problem.ops[op];
      // Inputs: LHS (pos 0) and RHS (pos > 0) — ephemeral from another MM?
      for (int pos = 0; pos < static_cast<int>(opdef.inputs.size()); ++pos) {
        size_t t = opdef.inputs[pos];
        if (!is_ephemeral(t)) continue;
        int prod = producer_of[t];
        if (prod < 0 || !sg_op_set.count(static_cast<size_t>(prod))) continue;
        if (problem.ops[prod].op_type != "MatMul") continue;
        if (pos == 0) roles[op].lhs_ephemeral = true;
        else roles[op].rhs_ephemeral = true;
      }
      // Output: consumed by another MM in sg as LHS (pos 0) or RHS (pos > 0)?
      if (!opdef.outputs.empty()) {
        size_t out_t = opdef.outputs[0];
        if (is_ephemeral(out_t)) {
          for (auto& [consumer_op, pos] : consumers_of[out_t]) {
            if (!sg_op_set.count(consumer_op)) continue;
            if (problem.ops[consumer_op].op_type != "MatMul") continue;
            if (pos == 0) roles[op].out_ephemeral = true;
            else roles[op].out_ephemeral_rhs = true;
          }
        }
      }
    }

    // K_slicing: the dimension granularity k slices.  For any chain (either
    // direction) it is the downstream Tail's K_op:
    //   LHS-chain: Tail's K_op == width of intermediate  (Ex5B)
    //   RHS-chain: Tail's K_op == height of intermediate (§8 mirror)
    // K_up stores the Head's own reduction dim, used for the direction-
    // appropriate "full-load" operand shape (LHS-chain Head: LHS resident
    // h × K_up; RHS-chain Head: RHS resident K_up × w).  For Standalone-
    // only subgraphs, use max K across MMs.
    int64_t K_slicing = 0;
    int64_t K_up = 0;
    size_t chain_head_op = SIZE_MAX;
    for (size_t op : mm_ops) {
      if (roles[op].is_tail()) {
        K_slicing = get_K_op(op);
      }
      if (roles[op].is_head()) {
        chain_head_op = op;
        K_up = get_K_op(op);
      }
    }
    if (K_slicing == 0) {
      int64_t max_K = 0;
      for (size_t op : mm_ops) max_K = std::max(max_K, get_K_op(op));
      K_slicing = max_K;
    }
    int64_t nk = (K_slicing > 0) ? CeilDiv(K_slicing, k) : 1;

    // ── Fusion constraint validation (role-based, §8 symmetric) ────────
    // Cross-direction double-consume (output consumed as BOTH LHS of one MM
    // and RHS of another): structurally invalid under any granularity, since
    // LHS-chain requires per-step shape h×ke while RHS-chain requires ke×w —
    // a single output cannot satisfy both simultaneously.
    for (size_t op : mm_ops) {
      const MMRole& r = roles[op];
      if (r.out_ephemeral && r.out_ephemeral_rhs) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Op ", op, ": ephemeral output consumed as both LHS and RHS of "
            "downstream MMs — inconsistent per-step shape"));
      }
    }
    // Split-K (nk > 1):
    //   - Pointwise→MatMul edges require granule alignment (issue #71,
    //     superseding #63's blanket ban): one PW tile must cover the full
    //     K-axis of the operand it produces.
    //       PW → MM(LHS): w ≥ MM.K
    //       PW → MM(RHS): h ≥ MM.K
    //       MM → PW epilogue: always OK (PW runs on fully-formed output tile).
    //   - No Middle MMs (§8: 3+ reduction axes force a non-ephemeral slab).
    //   - Parallel MMs (Standalone) with their own K_op ≠ K_slicing: forbid
    //     mixing chains with unrelated standalones (simplifies cost model
    //     and mirrors the organiser's conservative split-K rule).
    // Full-K (nk = 1):
    //   - Chains of any length are allowed, but every Middle MM (or 3+ MM
    //     chain) requires w = k = K_op uniform so all intermediates collapse
    //     onto the same tile grid.
    if (nk > 1) {
      for (size_t pw_op : pw_ops) {
        for (size_t t : problem.ops[pw_op].outputs) {
          for (auto& [consumer_op, pos] : consumers_of[t]) {
            if (!sg_op_set.count(consumer_op)) continue;
            if (problem.ops[consumer_op].op_type != "MatMul") continue;
            int64_t mm_K = get_K_op(consumer_op);
            if (pos == 0 && w < mm_K) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Pointwise (op ", pw_op, ") → MatMul (op ", consumer_op,
                  ") LHS split-K granule alignment: w=", w, " < MM.K=", mm_K,
                  " (issue #71)"));
            }
            if (pos > 0 && h < mm_K) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Pointwise (op ", pw_op, ") → MatMul (op ", consumer_op,
                  ") RHS split-K granule alignment: h=", h, " < MM.K=", mm_K,
                  " (issue #71)"));
            }
          }
        }
      }
      for (size_t op : mm_ops) {
        if (roles[op].is_middle()) {
          return absl::InvalidArgumentError(
              "Middle MatMul (chain relay) in 3+ MM chain not allowed with "
              "split-K");
        }
      }
      // A Standalone MM inside a subgraph that also has a chain is blocked:
      // under unified grid every op shares one k, but Standalone's own K_op
      // may differ from K_slicing.  Only allow "all Standalone" or "exactly
      // one Head + exactly one Tail".
      int n_head = 0, n_tail = 0, n_standalone = 0;
      for (size_t op : mm_ops) {
        if (roles[op].is_head()) ++n_head;
        else if (roles[op].is_tail()) ++n_tail;
        else if (roles[op].is_standalone()) ++n_standalone;
      }
      bool is_pure_chain = (n_head == 1 && n_tail == 1 && n_standalone == 0 &&
                             mm_ops.size() == 2);
      bool is_pure_standalone = (n_head == 0 && n_tail == 0 &&
                                  n_standalone == static_cast<int>(mm_ops.size()));
      if (!is_pure_chain && !is_pure_standalone) {
        return absl::InvalidArgumentError(
            "Split-K mixes chain and non-chain MMs — not supported");
      }
      if (is_pure_standalone && mm_ops.size() >= 2) {
        return absl::InvalidArgumentError(
            "Two MMs without chain not allowed with split-K");
      }
    } else {
      // nk = 1.  For any chain position (Head / Tail / Middle) present,
      // require w = k = K_op uniform across all MMs in the subgraph so the
      // chain collapses cleanly onto the shared tile grid.
      bool any_chain = false;
      for (size_t op : mm_ops) {
        if (!roles[op].is_standalone()) { any_chain = true; break; }
      }
      // The uniform-K rule triggers when there's any Middle, or more than 2
      // MMs, or (any chain at all AND the unified grid needs to cover more
      // than one intermediate).  For simplicity mirror the previous rule:
      // 3+ MMs OR any Middle MM in the subgraph.
      bool has_middle = false;
      for (size_t op : mm_ops) {
        if (roles[op].is_middle()) { has_middle = true; break; }
      }
      if (mm_ops.size() >= 3 || has_middle) {
        if (w != k) {
          return absl::InvalidArgumentError(absl::StrCat(
              "3+ MM chain with k=K requires w == k (got w=", w, ", k=", k,
              ")"));
        }
        for (size_t op : mm_ops) {
          int64_t K_op = get_K_op(op);
          if (K_op != k) {
            return absl::InvalidArgumentError(absl::StrCat(
                "3+ MM chain with k=K requires all reduction dims == k, but op ",
                op, " has K=", K_op, " != k=", k));
          }
        }
      }
      (void)any_chain;  // reserved for future relaxation
    }

    // Find primary output tensor (for spatial tiling grid) and enforce
    // unified execution grid: all non-ephemeral outputs must share W×H.
    size_t primary_out_t = SIZE_MAX;
    for (size_t op : sg.ops) {
      for (size_t t : problem.ops[op].outputs) {
        if (is_ephemeral(t)) continue;
        if (primary_out_t == SIZE_MAX) {
          primary_out_t = t;
        } else if (problem.tensors[t].width !=
                       problem.tensors[primary_out_t].width ||
                   problem.tensors[t].height !=
                       problem.tensors[primary_out_t].height) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Subgraph ", sg_idx,
              ": non-ephemeral outputs have inconsistent shapes (tensor ",
              primary_out_t, ": ", problem.tensors[primary_out_t].width, "x",
              problem.tensors[primary_out_t].height, ", tensor ", t, ": ",
              problem.tensors[t].width, "x", problem.tensors[t].height, ")"));
        }
      }
    }
    if (primary_out_t == SIZE_MAX) {
      return absl::InvalidArgumentError("No non-ephemeral output in subgraph");
    }

    const int64_t W_out = problem.tensors[primary_out_t].width;
    const int64_t H_out = problem.tensors[primary_out_t].height;
    const int64_t nw = CeilDiv(W_out, w);
    const int64_t nh = CeilDiv(H_out, h);
    const int64_t num_tiles = nw * nh;

    // Per-op k_eff at a given k_step.  A chain-participating MM (Head or
    // Tail, either direction) slices against the chain's K_slicing (the
    // downstream's K), not its own K_op.  Standalone MMs slice their own K.
    auto get_k_eff = [&](size_t op, int64_t ks) -> int64_t {
      const MMRole& r = roles[op];
      if (!r.is_matmul) return 0;
      if (!r.is_standalone()) {
        return std::min(k, K_slicing - ks * k);
      }
      int64_t K_op = get_K_op(op);
      if (nk > 1) return std::min(k, K_op - ks * k);
      return K_op;
    };

    // Denominator for compute fraction (chain MMs use K_slicing).
    auto get_K_denom = [&](size_t op) -> int64_t {
      const MMRole& r = roles[op];
      if (!r.is_standalone() && r.is_matmul) return K_slicing;
      return get_K_op(op);
    };

    // Working set check (peak = non-last k-step with full k).
    {
      int64_t ws = 0;
      for (size_t t : retained_set) {
        ws += problem.tensors[t].width * problem.tensors[t].height;
      }

      // Working-set accounting per (op, input-pos) use.  Dispatch rules
      // (§8 symmetric LHS-chain / RHS-chain):
      //   Pointwise:    each input / output = w × h.
      //   Standalone MM: LHS = h × k_eff, RHS = k_eff × w.
      //   LHS-chain Head  : LHS resident h × K_up; RHS strip K_up × min(k, K_slicing).
      //   LHS-chain Tail  : LHS ephemeral (skipped); RHS strip min(k, K_slicing) × w.
      //   RHS-chain Head  : LHS strip min(k, K_slicing) × K_up; RHS resident K_up × w.
      //   RHS-chain Tail  : LHS strip h × min(k, K_slicing); RHS ephemeral (skipped).
      //   Middle MM (both ephemeral): valid only at nk=1 / w=k=K uniform;
      //     collapses to the Standalone formula with ephemerals filtered.
      for (size_t op : sg.ops) {
        const MMRole& role = roles[op];
        const Op& opdef = problem.ops[op];

        for (int pos = 0; pos < static_cast<int>(opdef.inputs.size()); ++pos) {
          size_t t = opdef.inputs[pos];
          if (retained_set.count(t)) continue;
          if (is_ephemeral(t)) continue;

          if (!role.is_matmul) {
            ws += h * w;  // Pointwise
          } else if (role.is_lhs_head()) {
            ws += (pos == 0) ? h * K_up : K_up * std::min(k, K_slicing);
          } else if (role.is_lhs_tail()) {
            if (pos == 0) continue;  // LHS ephemeral
            ws += std::min(k, K_slicing) * w;
          } else if (role.is_rhs_head()) {
            ws += (pos == 0) ? std::min(k, K_slicing) * K_up : K_up * w;
          } else if (role.is_rhs_tail()) {
            if (pos > 0) continue;  // RHS ephemeral
            ws += h * std::min(k, K_slicing);
          } else {  // Standalone (Middle at nk=1 falls here after is_ephemeral filter)
            int64_t k_ws = get_k_eff(op, 0);
            ws += (pos == 0) ? h * k_ws : k_ws * w;
          }
        }

        for (size_t t : opdef.outputs) {
          if (is_ephemeral(t)) continue;
          ws += h * w;
        }
      }

      if (ws > capacity) {
        return absl::ResourceExhaustedError(
            absl::StrCat("Subgraph ", sg_idx, ": working set ", ws,
                         " exceeds capacity ", capacity));
      }
    }

    // Traversal order: must be a permutation of [0, num_tiles).
    std::vector<int64_t> tile_order;
    if (sg.traversal_order.has_value()) {
      tile_order = sg.traversal_order.value();
      if (static_cast<int64_t>(tile_order.size()) != num_tiles) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Subgraph ", sg_idx, ": traversal_order size ", tile_order.size(),
            " does not match num_tiles ", num_tiles));
      }
      std::vector<bool> seen(num_tiles, false);
      for (int64_t idx : tile_order) {
        if (idx < 0 || idx >= num_tiles) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Subgraph ", sg_idx,
              ": traversal_order contains out-of-range index ", idx));
        }
        if (seen[idx]) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Subgraph ", sg_idx,
              ": traversal_order contains duplicate index ", idx));
        }
        seen[idx] = true;
      }
    } else {
      tile_order.resize(num_tiles);
      std::iota(tile_order.begin(), tile_order.end(), 0);
    }

    // Per-cell loop.
    double sg_latency = 0.0;
    std::map<size_t, SliceKey> loaded;

    for (int64_t seq = 0; seq < num_tiles; ++seq) {
      int64_t tile_idx = tile_order[seq];
      int64_t ht = tile_idx / nw;
      int64_t wt = tile_idx % nw;
      int64_t h_eff = std::min(h, H_out - ht * h);
      int64_t w_eff = std::min(w, W_out - wt * w);

      for (int64_t ks = 0; ks < nk; ++ks) {
        // Compute.
        double compute = 0;
        for (size_t op : sg.ops) {
          const MMRole& r = roles[op];
          if (!r.is_matmul) {
            if (ks == 0) compute += problem.ops[op].base_cost;
          } else {
            int64_t ke = get_k_eff(op, ks);
            int64_t Kd = get_K_denom(op);
            compute +=
                static_cast<double>(problem.ops[op].base_cost) * ke / Kd;
          }
        }

        // IO.
        double io_in = 0, io_out = 0;

        for (size_t op : sg.ops) {
          const MMRole& role = roles[op];
          const Op& opdef = problem.ops[op];
          int64_t K_op = role.is_matmul ? get_K_op(op) : 0;
          int64_t ke = get_k_eff(op, ks);

          for (int pos = 0; pos < static_cast<int>(opdef.inputs.size());
               ++pos) {
            size_t t = opdef.inputs[pos];

            if (retained_set.count(t)) continue;
            if (is_ephemeral(t)) continue;

            SliceKey key;
            double sz = 0;
            bool should_load = true;

            // Per-input SliceKey + size.  §8 symmetric dispatch:
            //   LHS-chain Head LHS : {ht, -1, -1}  h × K_up   (resident per tile, ks=0)
            //   LHS-chain Head RHS : {-1, -1, ks}  K_up × ke  (per k-step)
            //   LHS-chain Tail RHS : {-1, wt, ks}  ke × w     (per k-step; LHS skipped)
            //   RHS-chain Head LHS : {-1, -1, ks}  ke × K_up  (per k-step)
            //   RHS-chain Head RHS : {-1, wt, -1}  K_up × w   (resident per tile, ks=0)
            //   RHS-chain Tail LHS : {ht, -1, ks}  h × ke     (per k-step; RHS skipped)
            //   Stand LHS          : {ht, -1, ks}  h × ke     (per k-step)
            //   Stand RHS          : {-1, wt, ks}  ke × w     (per k-step)
            //   Pointwise          : {ht, wt, -1}  w × h      (at ks=0)
            if (!role.is_matmul) {
              key = {static_cast<int>(ht), static_cast<int>(wt), -1};
              sz = static_cast<double>(h_eff) * w_eff;
              should_load = (ks == 0);
            } else if (role.is_lhs_head()) {
              if (pos == 0) {
                key = {static_cast<int>(ht), -1, -1};
                sz = static_cast<double>(h_eff) * K_op;
                should_load = (ks == 0);
              } else {
                key = {-1, -1, static_cast<int>(ks)};
                sz = static_cast<double>(K_op) * ke;
              }
            } else if (role.is_lhs_tail()) {
              if (pos == 0) continue;  // LHS ephemeral, skip
              key = {-1, static_cast<int>(wt), static_cast<int>(ks)};
              sz = static_cast<double>(ke) * w_eff;
            } else if (role.is_rhs_head()) {
              if (pos == 0) {
                key = {-1, -1, static_cast<int>(ks)};
                sz = static_cast<double>(ke) * K_op;
              } else {
                key = {-1, static_cast<int>(wt), -1};
                sz = static_cast<double>(K_op) * w_eff;
                should_load = (ks == 0);
              }
            } else if (role.is_rhs_tail()) {
              if (pos > 0) continue;  // RHS ephemeral, skip
              key = {static_cast<int>(ht), -1, static_cast<int>(ks)};
              sz = static_cast<double>(h_eff) * ke;
            } else {  // Standalone (or Middle collapsed at nk=1)
              if (pos == 0) {
                key = {static_cast<int>(ht), -1, static_cast<int>(ks)};
                sz = static_cast<double>(h_eff) * ke;
              } else {
                key = {-1, static_cast<int>(wt), static_cast<int>(ks)};
                sz = static_cast<double>(ke) * w_eff;
              }
            }

            if (should_load) {
              auto it = loaded.find(t);
              if (it == loaded.end() || it->second != key) {
                io_in += sz / bw;
                loaded[t] = key;
              }
            }
          }
        }

        // Output eviction at each tile's last k-step.
        if (ks == nk - 1) {
          for (size_t op : sg.ops) {
            for (size_t t : problem.ops[op].outputs) {
              if (!needs_eviction(t)) continue;
              io_out += static_cast<double>(h_eff * w_eff) / bw;
            }
          }
        }

        sg_latency += std::max(compute, io_in + io_out);
      }
    }

    // Validate: when a solution reports a non-zero subgraph_latency, it must
    // match our computed value. Zero is treated as "not provided" (placeholder
    // convention) to allow tests and partial solutions.
    if (sg.subgraph_latency != 0.0) {
      double reported = sg.subgraph_latency;
      double denom = sg_latency > 0.0 ? sg_latency : 1.0;
      double rel_err = std::abs(reported - sg_latency) / denom;
      if (rel_err > 1e-6) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Subgraph ", sg_idx, ": reported latency ", reported,
            " does not match computed ", sg_latency));
      }
    }

    total_latency += sg_latency;

    retained_set.clear();
    for (size_t t : sg.tensors_to_retain) {
      retained_set.insert(t);
    }
  }

  return total_latency;
}

}  // namespace mlsys
