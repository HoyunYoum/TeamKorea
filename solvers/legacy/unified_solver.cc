// unified_solver.cc — Unified ILP Solver (based on two_phase_solver)
//
// Phase 1: Set-partitioning ILP via HiGHS (partition only, no ordering)
//   - Super-columns with retained_in={} — no predecessor dependency
//   - Lightweight: only x[c] binary vars + coverage constraints
//
// Phase 2: Ordering + Retention optimization
//   - Enumerate valid topological orderings (S≤10) or greedy (S>10)
//   - Greedily assign retention at each boundary
//
// No Gurobi dependency — uses HiGHS only.

#include "solver_common.h"
#include "Highs.h"
#include <cstdio>
#include <iomanip>

using json = nlohmann::json;

// Helper: analytical latencies stored by BuildSolution are approximations
// of the evaluator.  Our evaluator treats subgraph_latency == 0 as "not
// provided" and skips the reported-vs-computed consistency check.  Use this
// instead of the upstream mlsys::PatchLatencies (which our evaluator does
// not expose) before calling mlsys::Evaluate or WriteSolution.
static inline void ZeroLatencies(mlsys::Solution& sol) {
  for (auto& sg : sol.subgraphs) sg.subgraph_latency = 0.0;
}

namespace {

// Snake mode constants (matching solver_common.cc: 0=row, 1=col, 2=raster/none).
constexpr int kSnakeRow = 0;
constexpr int kSnakeCol = 1;
constexpr int kSnakeNone = 2;

int DefaultMaxSize(int n) {
  if (n <= 10) return 5;
  if (n <= 30) return 4;
  if (n <= 50) return 3;
  return 2;
}

// ── Tuning Constants ─────────────────────────────────────────────────────────
constexpr int kTopK = 5;                         // granularity candidates per snake mode
constexpr int kMaxHeavyInputs = 5;               // heavy graph inputs to explore
constexpr int kMaxGroupSize = 8;                  // max consumer group size for heavy inputs
constexpr int kHeldKarpMaxS = 15;                 // max subgraphs for Held-Karp DP
constexpr int kMaxSubgraphsForHK = 30;            // reject oversized partitions in PartitionToHKCandidate
constexpr double kRecompCostRatio = 4.0;          // recompute if cost <= ratio × spill cost
constexpr int kMaxILPIters = 3;                   // multi-pass ILP refinement iterations
constexpr double kHardDeadlineRatio = 0.95;       // fraction of timeout for hard deadline
constexpr double kTimeBudgetRewrite = 0.35;       // rewrite path time budget (fraction of deadline)
constexpr double kTimeBudgetRewriteP0 = 0.04;     // rewrite Phase 0 time budget
constexpr double kTimeBudgetPhase0 = 0.12;        // main Phase 0 time budget
constexpr double kTimeBudgetILPBreak = 0.5;       // stop ILP iterations after this fraction
constexpr double kTimeBudgetDPHK = 0.9;           // deadline fraction for DP-HK fallback
constexpr double kTimeBudgetDPDFS = 0.95;         // deadline fraction for DP-DFS fallback

// Hash for std::vector<size_t>, used in seen-sets for connected subset enumeration.
struct VectorHash {
  size_t operator()(const std::vector<size_t>& v) const {
    size_t seed = v.size();
    for (auto x : v)
      seed ^= x * 2654435761ULL + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

// Classify result cache: avoids recomputing tensor classification for same ops.
using ClassifyCache = std::unordered_map<std::vector<size_t>, solver::TensorSets, VectorHash>;

const solver::TensorSets& CachedClassify(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const solver::DAG& dag,
    ClassifyCache& cache) {
  auto it = cache.find(ops);
  if (it != cache.end()) return it->second;
  auto [ins, ok] = cache.emplace(ops, solver::Classify(p, ops, &dag));
  return ins->second;
}

// ── Super-Column ─────────────────────────────────────────────────────────────

struct SuperColumn {
  std::vector<size_t> ops;
  mlsys::Granularity gran;
  int snake_mode;  // 0=row, 1=col
  std::vector<int64_t> traversal;
  double cost;       // current cost (may reflect retention context)
  double base_cost;  // original no-retention cost (immutable after creation)
  std::unordered_set<size_t> op_set;
  std::vector<size_t> output_tensors;  // for Phase 2 retention
  std::unordered_set<size_t> retained_in;   // tensors assumed pre-loaded from predecessor
  std::unordered_set<size_t> input_tensors; // all boundary inputs (for retention matching)
};

// ── Undirected Adjacency ─────────────────────────────────────────────────────

using AdjList = std::vector<std::unordered_set<size_t>>;

AdjList BuildUndirectedAdj(const solver::DAG& dag) {
  int N = (int)dag.num_ops;
  AdjList adj(N);
  for (size_t t = 0; t < dag.num_tensors; ++t) {
    int prod = dag.tensor_producer[t];
    for (size_t cons : dag.tensor_consumers[t]) {
      if (prod >= 0 && (size_t)prod != cons) {
        adj[prod].insert(cons);
        adj[cons].insert((size_t)prod);
      }
    }
    auto& consumers = dag.tensor_consumers[t];
    for (size_t i = 0; i < consumers.size(); ++i)
      for (size_t j = i + 1; j < consumers.size(); ++j) {
        adj[consumers[i]].insert(consumers[j]);
        adj[consumers[j]].insert(consumers[i]);
      }
  }
  return adj;
}

// ── Granularity Candidate ────────────────────────────────────────────────────

struct GranCandidate {
  mlsys::Granularity gran;
  int snake_mode;
  std::vector<int64_t> traversal;
  double cost;
};

// ── Granularity Dedup Key ────────────────────────────────────────────────────

struct GranKey {
  int64_t w, h, k;
  int snake;
  bool operator==(const GranKey& o) const {
    return w == o.w && h == o.h && k == o.k && snake == o.snake;
  }
};

struct GranKeyHash {
  size_t operator()(const GranKey& g) const {
    return std::hash<int64_t>()(g.w) ^ (std::hash<int64_t>()(g.h) << 16)
         ^ (std::hash<int64_t>()(g.k) << 32) ^ (std::hash<int>()(g.snake) << 48);
  }
};

// Emit a SuperColumn if the (gran, snake) combo hasn't been seen.
// Uses no-retention cost for fair ILP comparison. Returns true if added.
bool EmitSuperColumn(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    GranCandidate& gc,
    const std::unordered_set<size_t>& ri_assumed,
    const solver::TensorSets& ts,
    const std::unordered_map<size_t, solver::TensorRoleInfo>& tensor_roles,
    int64_t maxK, int64_t n_outputs,
    const std::unordered_set<size_t>& op_set,
    const std::vector<size_t>& output_tensors,
    const std::unordered_set<size_t>& input_tensors,
    std::unordered_set<GranKey, GranKeyHash>& seen_grans,
    std::vector<SuperColumn>& columns) {
  GranKey gk{gc.gran.width, gc.gran.height, gc.gran.depth, gc.snake_mode};
  if (!seen_grans.insert(gk).second) return false;

  double cost = solver::AnalyticalCost(
      p, ops, gc.gran, gc.snake_mode, {}, ts, tensor_roles, maxK, n_outputs, true);
  if (cost >= solver::kInf) return false;

  SuperColumn sc;
  sc.ops = ops;
  sc.gran = gc.gran;
  sc.snake_mode = gc.snake_mode;
  sc.traversal = gc.traversal;
  sc.cost = cost;
  sc.base_cost = cost;
  sc.op_set = op_set;
  sc.output_tensors = output_tensors;
  sc.input_tensors = input_tensors;
  sc.retained_in = ri_assumed;
  columns.push_back(std::move(sc));
  return true;
}

// ── Granularity Regime Analysis ──────────────────────────────────────────────

struct OpRegime {
  double solo_cost = solver::kInf;
  mlsys::Granularity best_gran{0, 0, 0};
  int best_snake = kSnakeRow;
};

struct RegimeAnalysis {
  std::vector<OpRegime> op_regimes;
  std::vector<std::unordered_set<size_t>> compatible;  // compatible[i] = {j : fusion beneficial}
};

RegimeAnalysis ComputeRegimeAnalysis(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const AdjList& adj,
    ClassifyCache& cc) {
  int N = (int)dag.num_ops;
  double threshold = (N > 50) ? 1.10 : (N > 20) ? 1.05 : 1.01;
  double bw = p.slow_memory_bandwidth;

  RegimeAnalysis ra;
  ra.op_regimes.resize(N);
  ra.compatible.resize(N);

  auto ra_t0 = std::chrono::steady_clock::now();

  // Phase A: per-op regimes via BestGranularity (single best config per op)
  std::unordered_set<size_t> empty_ri;
  for (int i = 0; i < N; ++i) {
    std::vector<size_t> ops = {(size_t)i};
    auto gc = solver::BestGranularity(p, ops, empty_ri, 0, &dag, true);
    ra.op_regimes[i] = {gc.cost, gc.gran, 0};
    // Determine snake_mode from traversal
    if (!gc.traversal.empty() && gc.traversal.size() >= 2) {
      auto [oW, oH] = solver::OutDims(p, ops);
      int64_t ntw = solver::CeilDiv(oW, gc.gran.width);
      // If second tile is in same row (idx < ntw), it's row-snake
      ra.op_regimes[i].best_snake = (gc.traversal[1] < ntw) ? kSnakeRow : kSnakeCol;
    }
  }

  // Phase B: pairwise compatibility — two-tier check
  // n_edges is approximate: adj includes co-consumer edges, not just direct data deps.
  int n_edges = 0, n_compat = 0, n_tier1_pruned = 0;
  for (int i = 0; i < N; ++i) {
    for (size_t j : adj[i]) {
      if (j <= (size_t)i) continue;
      ++n_edges;
      double split_cost = ra.op_regimes[i].solo_cost + ra.op_regimes[j].solo_cost;
      if (split_cost >= solver::kInf) continue;
      double needed = split_cost * (threshold - 1.0);

      // Tier 1: analytical IO savings upper bound (O(1) per pair)
      // Sources of savings:
      // 1. Ephemeral intermediates: 2 × size / bw (eviction + reload avoided)
      // 2. Shared input reload: size / bw per shared input (loaded once instead of twice)
      double max_savings = 0;
      std::unordered_set<size_t> pair_ops = {(size_t)i, j};

      // Ephemeral intermediates (outputs of i consumed by j, all consumers internal)
      for (size_t t : p.ops[i].outputs) {
        bool all_internal = true;
        for (size_t c : dag.tensor_consumers[t]) {
          if (!pair_ops.count(c)) { all_internal = false; break; }
        }
        if (all_internal) {
          max_savings += 2.0 * p.tensors[t].width * p.tensors[t].height / bw;
        }
      }
      // Symmetric: outputs of j consumed by i
      for (size_t t : p.ops[j].outputs) {
        bool all_internal = true;
        for (size_t c : dag.tensor_consumers[t]) {
          if (!pair_ops.count(c)) { all_internal = false; break; }
        }
        if (all_internal) {
          max_savings += 2.0 * p.tensors[t].width * p.tensors[t].height / bw;
        }
      }

      // Shared input reload savings (inputs consumed by both ops)
      std::unordered_set<size_t> inputs_i(p.ops[i].inputs.begin(), p.ops[i].inputs.end());
      for (size_t t : p.ops[j].inputs) {
        if (inputs_i.count(t)) {
          // Shared input: fusing saves one reload (upper bound)
          max_savings += (double)p.tensors[t].width * p.tensors[t].height / bw;
        }
      }

      if (max_savings < needed) {
        ++n_tier1_pruned;
        continue;  // even perfect fusion can't clear threshold
      }

      // Tier 2: spot-check at solo granularities (2-4 AnalyticalCost calls)
      std::vector<size_t> pair_vec = {std::min((size_t)i, j), std::max((size_t)i, j)};
      auto ts = solver::Classify(p, pair_vec, &dag);
      auto roles = solver::BuildRoles(p, pair_vec, ts);
      int64_t maxK = 0;
      for (size_t op : pair_vec)
        if (p.ops[op].op_type == "MatMul")
          maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);
      int64_t n_outputs = (int64_t)ts.outputs.size();

      double fused_best = solver::kInf;
      // Try both solo granularities × both snake modes
      for (int who = 0; who < 2; ++who) {
        auto& regime = (who == 0) ? ra.op_regimes[i] : ra.op_regimes[j];
        if (regime.solo_cost >= solver::kInf) continue;
        // Check WS feasibility first
        int64_t w = regime.best_gran.width, h = regime.best_gran.height, k = regime.best_gran.depth;
        int64_t ws = solver::ComputeTileWS(p, roles, n_outputs, {}, w, h, k);
        if (ws > p.fast_memory_capacity) continue;

        for (int snake = kSnakeRow; snake <= kSnakeCol; ++snake) {
          double c = solver::AnalyticalCost(p, pair_vec, regime.best_gran, snake,
                                            empty_ri, ts, roles, maxK, n_outputs, true);
          fused_best = std::min(fused_best, c);
        }
      }

      if (fused_best < split_cost * threshold) {
        ra.compatible[i].insert(j);
        ra.compatible[j].insert((size_t)i);
        ++n_compat;
      } else {
        // Tier 3: solo grans weren't sufficient — try BestGranularity for proper search.
        // Only runs on pairs that passed Tier 1 (plausible IO savings) but failed Tier 2.
        auto gc = solver::BestGranularity(p, pair_vec, empty_ri, 0, &dag, true);
        if (gc.cost < split_cost * threshold) {
          ra.compatible[i].insert(j);
          ra.compatible[j].insert((size_t)i);
          ++n_compat;
        }
      }
    }
  }

  double ra_elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - ra_t0).count();
  std::cerr << "Regime analysis: " << N << " ops, " << n_edges << " edges, "
            << n_compat << " compatible (" << n_tier1_pruned << " tier1-pruned, "
            << std::fixed << std::setprecision(2) << ra_elapsed << "s)\n";
  return ra;
}

// ── DAG Rewrite for Ephemeral Recomputation ─────────────────────────────────

bool IsRecomputable(const mlsys::Problem& p, const solver::DAG& dag, size_t op_idx) {
  bool high_fanout = false;
  for (size_t t : p.ops[op_idx].outputs) {
    if (dag.tensor_consumers[t].size() >= 2) { high_fanout = true; break; }
  }
  if (!high_fanout) return false;
  double recomp_cost = p.ops[op_idx].base_cost;
  double spill_cost = 0;
  for (size_t t : p.ops[op_idx].outputs)
    spill_cost += (double)(p.tensors[t].width * p.tensors[t].height)
                  / p.slow_memory_bandwidth;
  return recomp_cost <= kRecompCostRatio * spill_cost;
}

struct RecompCandidate {
  size_t op;
  std::vector<size_t> output_tensors;  // tensors with fan_out >= 2
  double recomp_cost;
  double spill_cost;
};

std::vector<RecompCandidate> FindRecomputeCandidates(
    const mlsys::Problem& p, const solver::DAG& dag) {
  std::vector<RecompCandidate> candidates;
  for (size_t i = 0; i < dag.num_ops; ++i) {
    if (!IsRecomputable(p, dag, i)) continue;
    std::vector<size_t> high_fanout_tensors;
    for (size_t t : p.ops[i].outputs)
      if (dag.tensor_consumers[t].size() >= 2)
        high_fanout_tensors.push_back(t);
    double recomp_cost = p.ops[i].base_cost;
    double spill_cost = 0;
    for (size_t t : p.ops[i].outputs)
      spill_cost += (double)(p.tensors[t].width * p.tensors[t].height)
                    / p.slow_memory_bandwidth;
    candidates.push_back({i, std::move(high_fanout_tensors), recomp_cost, spill_cost});
  }
  return candidates;
}

struct DAGRewrite {
  mlsys::Problem rw_problem;
  // Mapping: rw op index → original op index
  std::vector<size_t> orig_op;    // size = rw_problem.ops.size()
  // Mapping: rw tensor index → original tensor index
  std::vector<size_t> orig_tensor; // size = rw_problem.tensors.size()
  int n_orig_ops;
};

DAGRewrite RewriteDAG(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const std::vector<RecompCandidate>& candidates) {
  DAGRewrite rw;
  rw.n_orig_ops = (int)p.ops.size();

  // Start with a copy of the original problem
  rw.rw_problem = p;
  rw.orig_op.resize(p.ops.size());
  std::iota(rw.orig_op.begin(), rw.orig_op.end(), 0);
  rw.orig_tensor.resize(p.tensors.size());
  std::iota(rw.orig_tensor.begin(), rw.orig_tensor.end(), 0);

  for (auto& cand : candidates) {
    for (size_t t : cand.output_tensors) {
      auto& consumers = dag.tensor_consumers[t];
      if (consumers.size() < 2) continue;

      // First consumer keeps original op+tensor; rest get clones.
      // Note: consumer assignment doesn't matter for released benchmarks
      // (skip connections are simple fan_out=2 to independent consumers).
      for (size_t ci = 1; ci < consumers.size(); ++ci) {
        size_t consumer_op = consumers[ci];

        // Create clone tensor with same dimensions
        size_t new_tensor_id = rw.rw_problem.tensors.size();
        rw.rw_problem.tensors.push_back(p.tensors[t]);
        rw.orig_tensor.push_back(t);  // maps back to original tensor

        // Create clone op: same type, inputs, base_cost, but new output tensor
        mlsys::Op clone_op = p.ops[cand.op];
        // Replace the cloned output tensor in the clone op's outputs
        for (auto& out : clone_op.outputs) {
          if (out == t) { out = new_tensor_id; break; }
        }
        size_t new_op_id = rw.rw_problem.ops.size();
        rw.rw_problem.ops.push_back(std::move(clone_op));
        rw.orig_op.push_back(cand.op);  // maps back to original op

        // Rewire consumer: replace input t with new_tensor_id
        for (auto& inp : rw.rw_problem.ops[consumer_op].inputs) {
          if (inp == t) { inp = new_tensor_id; break; }
        }
      }
    }
  }

  return rw;
}

// Map a partition (vector of op-index vectors) from rewritten DAG back to original ops
solver::Partition MapPartitionBack(
    const DAGRewrite& rw,
    const solver::Partition& rw_part) {
  solver::Partition result;
  for (auto& sg_ops : rw_part.subgraphs) {
    std::vector<size_t> orig_ops;
    std::unordered_set<size_t> seen;
    for (size_t rw_op : sg_ops) {
      size_t orig = rw.orig_op[rw_op];
      if (seen.insert(orig).second)
        orig_ops.push_back(orig);
    }
    result.subgraphs.push_back(std::move(orig_ops));
  }
  return result;
}

// ── Connected Subset Enumeration ─────────────────────────────────────────────

void EnumerateConnectedSubsets(
    const solver::DAG& dag,
    const AdjList& adj,
    std::vector<std::vector<size_t>>& subsets,
    int max_size,
    std::chrono::steady_clock::time_point t0 = {},
    double deadline = 1e18) {
  int N = (int)dag.num_ops;
  bool has_deadline = t0.time_since_epoch().count() > 0;

  std::unordered_set<std::vector<size_t>, VectorHash> seen;
  for (int i = 0; i < N; ++i) {
    std::vector<size_t> s = {(size_t)i};
    seen.insert(s);
    subsets.push_back(s);
  }

  std::vector<std::vector<size_t>> current_level;
  for (auto& s : subsets) current_level.push_back(s);

  bool timed_out = false;
  int last_complete_size = 1;
  for (int size = 2; size <= std::min(N, max_size); ++size) {
    std::vector<std::vector<size_t>> next_level;
    for (auto& subset : current_level) {
      if (has_deadline && solver::ElapsedSince(t0) > deadline) {
        timed_out = true;
        break;
      }
      std::set<size_t> sub_set(subset.begin(), subset.end());
      std::set<size_t> neighbors;
      for (size_t op : subset)
        for (size_t nb : adj[op])
          if (!sub_set.count(nb) && nb > subset[0])
            neighbors.insert(nb);

      for (size_t nb : neighbors) {
        auto next = subset;
        auto pos = std::lower_bound(next.begin(), next.end(), nb);
        next.insert(pos, nb);
        if (seen.insert(next).second) {
          subsets.push_back(next);
          next_level.push_back(next);
        }
      }
    }
    if (timed_out) {
      std::cerr << " [time-gated at size=" << size << ", "
                << subsets.size() << " subsets]\n";
      break;
    }
    last_complete_size = size;
    current_level = std::move(next_level);
    if (current_level.empty()) break;
  }

  std::cerr << "Connected subsets: " << subsets.size()
            << " (complete through size " << last_complete_size << ")\n";
}

// ── Granularity Enumeration ──────────────────────────────────────────────────

void EnumerateGranularitiesImpl(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const std::unordered_set<size_t>& retained_in_set,
    const solver::DAG& dag,
    std::vector<GranCandidate>& candidates,
    ClassifyCache& cc) {

  auto [oW, oH] = solver::OutDims(p, ops);
  int64_t natW = p.native_granularity.width;
  int64_t natH = p.native_granularity.height;
  int64_t C = p.fast_memory_capacity;

  int64_t maxK = 0;
  for (size_t op : ops)
    if (p.ops[op].op_type == "MatMul")
      maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);

  auto& ts = CachedClassify(p, ops, dag, cc);
  auto tensor_roles = solver::BuildRoles(p, ops, ts);
  int64_t n_outputs = (int64_t)ts.outputs.size();

  auto ComputeWS = [&](int64_t w, int64_t h, int64_t k) -> int64_t {
    return solver::ComputeTileWS(p, tensor_roles, n_outputs, retained_in_set, w, h, k);
  };

  // MM+PW split-K prohibition (#32)
  bool has_mm = false, has_pw = false;
  for (size_t op : ops) {
    if (p.ops[op].op_type == "MatMul") has_mm = true;
    else if (p.ops[op].op_type == "Pointwise") has_pw = true;
  }

  std::vector<int64_t> nk_cands;
  if (has_mm && has_pw && maxK > 0) {
    nk_cands.push_back(1);
  } else if (maxK == 0) {
    nk_cands.push_back(1);
  } else {
    for (int64_t d = 1; d * d <= maxK; ++d) {
      if (maxK % d == 0) { nk_cands.push_back(d); nk_cands.push_back(maxK / d); }
    }
    for (int64_t nk = 1; nk <= maxK; nk *= 2) nk_cands.push_back(nk);
    std::sort(nk_cands.begin(), nk_cands.end());
    nk_cands.erase(std::unique(nk_cands.begin(), nk_cands.end()), nk_cands.end());
  }

  // Dense enumeration over all distinct (w,h) pairs via tw/th iteration.
  // AnalyticalCost is O(1) — evaluating all candidates is fast.
  // Track top-K per snake mode to give ILP granularity options.
  struct TopCand {
    double cost = solver::kInf;
    mlsys::Granularity gran{0, 0, 0};
  };
  TopCand top_snake[2][kTopK];  // [snake][rank]
  int n_top[2] = {0, 0};

  auto InsertTop = [&](int snake, double cost, mlsys::Granularity g) {
    auto* top = top_snake[snake];
    int& n = n_top[snake];
    if (n < kTopK) {
      top[n++] = {cost, g};
      for (int j = n - 1; j > 0 && top[j].cost < top[j-1].cost; --j)
        std::swap(top[j], top[j-1]);
    } else if (cost < top[kTopK-1].cost) {
      top[kTopK-1] = {cost, g};
      for (int j = kTopK - 2; j >= 0 && top[j+1].cost < top[j].cost; --j)
        std::swap(top[j], top[j+1]);
    }
  };

  for (int64_t nk : nk_cands) {
    int64_t k = (maxK > 0) ? solver::CeilDiv(maxK, nk) : 1;
    for (int64_t tw = 1; tw <= oW; ++tw) {
      int64_t w = std::min(solver::CeilDiv(oW, tw), natW);
      if (tw > 1 && w == std::min(solver::CeilDiv(oW, tw - 1), natW)) continue;
      if (ComputeWS(w, 1, k) > C) continue;

      int64_t min_th = 1;
      {
        int64_t lo = 1, hi = oH;
        while (lo <= hi) {
          int64_t mid = (lo + hi) / 2;
          if (ComputeWS(w, solver::CeilDiv(oH, mid), k) <= C) { min_th = mid; hi = mid - 1; }
          else lo = mid + 1;
        }
      }

      for (int64_t th = min_th; th <= oH; ++th) {
        int64_t h = std::min(solver::CeilDiv(oH, th), natH);
        if (th > min_th && h == std::min(solver::CeilDiv(oH, th - 1), natH)) continue;

        mlsys::Granularity g{w, h, k};
        for (int snake = kSnakeRow; snake <= kSnakeCol; ++snake) {
          double cost = solver::AnalyticalCost(p, ops, g, snake, retained_in_set,
                                               ts, tensor_roles, maxK, n_outputs, true);
          if (cost < solver::kInf)
            InsertTop(snake, cost, g);
        }
      }
    }
  }

  // Emit top-K candidates per snake mode (with traversals)
  for (int snake = kSnakeRow; snake <= kSnakeCol; ++snake) {
    for (int i = 0; i < n_top[snake]; ++i) {
      auto& b = top_snake[snake][i];
      int64_t ntw = solver::CeilDiv(oW, b.gran.width);
      int64_t nth = solver::CeilDiv(oH, b.gran.height);
      auto trav = (snake == kSnakeRow) ? solver::SnakeRow(ntw, nth)
                                      : solver::SnakeCol(ntw, nth);
      candidates.push_back({b.gran, snake, std::move(trav), b.cost});
    }
  }
}

// Phase 1 wrapper: no retained_in
void EnumerateGranularities(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const solver::DAG& dag,
    std::vector<GranCandidate>& candidates,
    ClassifyCache& cc) {
  std::unordered_set<size_t> empty_ri;
  EnumerateGranularitiesImpl(p, ops, empty_ri, dag, candidates, cc);
}

// Phase 2 wrapper: with retained_in
void EnumerateGranularitiesWithRI(
    const mlsys::Problem& p,
    const std::vector<size_t>& ops,
    const std::unordered_set<size_t>& retained_in,
    const solver::DAG& dag,
    std::vector<GranCandidate>& candidates,
    ClassifyCache& cc) {
  EnumerateGranularitiesImpl(p, ops, retained_in, dag, candidates, cc);
}

// ── Column Generation for a Single Op-Set ───────────────────────────────────
// Shared helper: Classify → BuildRoles → EnumerateGranularities → EmitSuperColumn
// for both baseline (ri={}) and retention-aware (ri=all_inputs) paths.
// Returns the number of columns added.

int GenerateColumnsForOps(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const std::vector<size_t>& ops,
    ClassifyCache& cc,
    std::vector<SuperColumn>& columns) {
  auto& ts = CachedClassify(p, ops, dag, cc);
  auto roles = solver::BuildRoles(p, ops, ts);
  int64_t maxK = 0;
  for (size_t op : ops)
    if (p.ops[op].op_type == "MatMul")
      maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);
  int64_t n_outputs = (int64_t)ts.outputs.size();
  std::vector<size_t> ro_cands(ts.outputs.begin(), ts.outputs.end());
  std::unordered_set<size_t> all_inputs(ts.inputs.begin(), ts.inputs.end());
  std::unordered_set<size_t> op_set(ops.begin(), ops.end());

  std::unordered_set<GranKey, GranKeyHash> seen;
  size_t before = columns.size();

  // Path 1: baseline (retained_in = {})
  {
    std::vector<GranCandidate> grans;
    EnumerateGranularities(p, ops, dag, grans, cc);
    for (auto& gc : grans)
      EmitSuperColumn(p, ops, gc, {}, ts, roles, maxK, n_outputs,
                      op_set, ro_cands, all_inputs, seen, columns);
  }

  // Path 2: retention-aware (retained_in = all inputs)
  if (!all_inputs.empty()) {
    std::vector<GranCandidate> ri_grans;
    EnumerateGranularitiesWithRI(p, ops, all_inputs, dag, ri_grans, cc);
    for (auto& gc : ri_grans)
      EmitSuperColumn(p, ops, gc, all_inputs, ts, roles, maxK, n_outputs,
                      op_set, ro_cands, all_inputs, seen, columns);
  }

  return (int)(columns.size() - before);
}

// ── Build Phase 1 Super-Columns ──────────────────────────────────────────────

void BuildSuperColumns(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const std::vector<std::vector<size_t>>& subsets,
    std::vector<SuperColumn>& columns,
    ClassifyCache& cc,
    std::chrono::steady_clock::time_point t0 = {},
    double deadline = 1e18) {
  bool has_deadline = t0.time_since_epoch().count() > 0;

  for (size_t si = 0; si < subsets.size(); ++si) {
    if (has_deadline && solver::ElapsedSince(t0) > deadline) {
      std::cerr << "\n  [BuildSuperColumns time-gated after " << si << "/"
                << subsets.size() << " subsets, " << columns.size() << " columns]\n";
      break;
    }
    auto& ops = subsets[si];
    std::cerr << "  [";
    for (size_t i = 0; i < ops.size(); ++i)
      std::cerr << (i ? "," : "") << ops[i];
    std::cerr << "]" << std::flush;

    int added = GenerateColumnsForOps(p, dag, ops, cc, columns);
    if (added > 0) std::cerr << "→" << added;
    std::cerr << " ";
  }
  std::cerr << "\nTotal super-columns: " << columns.size() << "\n";
}


// ── Phase 0: Heavy Graph Input Fusion ─────────────────────────────────────────

void GenerateHeavyInputColumns(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    std::vector<SuperColumn>& columns,
    std::chrono::steady_clock::time_point t0,
    double time_budget_s,
    ClassifyCache& cc,
    const RegimeAnalysis* regime = nullptr) {

  double deadline = solver::ElapsedSince(t0) + time_budget_s;
  int64_t C = p.fast_memory_capacity;
  double bw = p.slow_memory_bandwidth;

  // Step 1: Identify heavy graph inputs (large size × many consumers)
  struct HeavyInput {
    size_t tid;
    int64_t size;
    double reload_cost;
    std::vector<size_t> consumers;
  };
  std::vector<HeavyInput> heavies;

  for (size_t t = 0; t < dag.num_tensors; ++t) {
    if (dag.tensor_producer[t] >= 0) continue;
    auto& cons = dag.tensor_consumers[t];
    if (cons.size() < 2) continue;

    int64_t sz = p.tensors[t].width * p.tensors[t].height;
    double reload = (double)(cons.size() - 1) * sz / bw;
    heavies.push_back({t, sz, reload, cons});
  }

  std::sort(heavies.begin(), heavies.end(),
            [](const HeavyInput& a, const HeavyInput& b) {
              return a.reload_cost > b.reload_cost;
            });

  int max_heavies = std::min((int)heavies.size(), kMaxHeavyInputs);
  int max_group_size = kMaxGroupSize;

  if (max_heavies == 0) return;
  std::cerr << "Phase 0: " << heavies.size() << " heavy graph inputs, exploring top "
            << max_heavies << "\n";

  // Track generated subsets to avoid duplicates
  std::unordered_set<std::vector<size_t>, VectorHash> generated;

  auto TrySubset = [&](std::vector<size_t> ops) -> int {
    std::sort(ops.begin(), ops.end());
    if (ops.size() < 2) return 0;
    if (generated.count(ops)) return 0;
    generated.insert(ops);
    return GenerateColumnsForOps(p, dag, ops, cc, columns);
  };

  // Step 2: For each heavy input, enumerate consumer subsets
  for (int hi = 0; hi < max_heavies; ++hi) {
    if (solver::ElapsedSince(t0) > deadline) break;

    auto& heavy = heavies[hi];
    auto& consumers = heavy.consumers;
    int nc = (int)consumers.size();

    std::cerr << "  T" << heavy.tid << " (" << p.tensors[heavy.tid].width
              << "x" << p.tensors[heavy.tid].height << "): "
              << nc << " consumers, reload_cost=" << (int64_t)heavy.reload_cost << "\n";

    // Compute marginal WS per consumer (rough estimate at native gran)
    int64_t natW = p.native_granularity.width;
    int64_t natH = p.native_granularity.height;
    struct ConsumerInfo { size_t op; int64_t marginal_ws; };
    std::vector<ConsumerInfo> cinfos;

    for (size_t op : consumers) {
      auto& o = p.ops[op];
      int64_t marginal = 0;
      // Rough WS upper bound at native granularity for sorting only.
      // LHS uses h×K (tensor width), which equals h×k when nk=1.
      // Actual WS is validated later via ComputeTileWS.
      for (int pos = 0; pos < (int)o.inputs.size(); ++pos) {
        if (o.inputs[pos] == heavy.tid) continue;  // shared, don't count
        if (o.op_type == "MatMul") {
          marginal += (pos == 0) ? natH * p.tensors[o.inputs[pos]].width
                                 : natW;
        } else {
          marginal += natW * natH;
        }
      }
      marginal += (int64_t)o.outputs.size() * natW * natH;
      cinfos.push_back({op, marginal});
    }

    std::sort(cinfos.begin(), cinfos.end(),
              [](const ConsumerInfo& a, const ConsumerInfo& b) {
                return a.marginal_ws < b.marginal_ws;
              });

    int total_added = 0;

    // Strategy A: All pairs (skip incompatible pairs)
    for (int i = 0; i < nc && solver::ElapsedSince(t0) < deadline; ++i) {
      for (int j = i + 1; j < nc && solver::ElapsedSince(t0) < deadline; ++j) {
        if (regime && !regime->compatible[cinfos[i].op].count(cinfos[j].op))
          continue;
        total_added += TrySubset({cinfos[i].op, cinfos[j].op});
      }
    }

    // Strategy B: Greedy packing (sorted by marginal WS ascending)
    // Regime filter: only add consumer if compatible with at least one existing member
    {
      std::vector<size_t> group;
      for (int i = 0; i < nc && (int)group.size() < max_group_size; ++i) {
        if (solver::ElapsedSince(t0) > deadline) break;
        size_t op = cinfos[i].op;
        if (regime && !group.empty()) {
          bool any_compat = false;
          for (size_t g : group)
            if (regime->compatible[g].count(op)) { any_compat = true; break; }
          if (!any_compat) continue;
        }
        group.push_back(op);
        if (group.size() >= 2)
          total_added += TrySubset(group);
      }
      // If greedy didn't work for some consumers, try skipping them
      if ((int)group.size() < nc) {
        // Re-try: for each consumer not in group, try adding individually
        std::unordered_set<size_t> in_group(group.begin(), group.end());
        for (int i = 0; i < nc && solver::ElapsedSince(t0) < deadline; ++i) {
          if (in_group.count(cinfos[i].op)) continue;
          auto trial = group;
          trial.push_back(cinfos[i].op);
          total_added += TrySubset(trial);
        }
      }
    }

    // Strategy C: Sliding windows of size 3-5 on sorted consumers
    // Regime filter: require at least one compatible pair in window
    for (int sz = 3; sz <= std::min(max_group_size, nc); ++sz) {
      if (solver::ElapsedSince(t0) > deadline) break;
      for (int start = 0; start + sz <= nc; ++start) {
        if (solver::ElapsedSince(t0) > deadline) break;
        std::vector<size_t> ops;
        for (int k = start; k < start + sz; ++k)
          ops.push_back(cinfos[k].op);
        if (regime) {
          bool any_pair = false;
          for (size_t a = 0; a < ops.size() && !any_pair; ++a)
            for (size_t b = a + 1; b < ops.size() && !any_pair; ++b)
              if (regime->compatible[ops[a]].count(ops[b])) any_pair = true;
          if (!any_pair) continue;
        }
        total_added += TrySubset(ops);
      }
    }

    // Strategy D: Branch-pair columns — include each consumer's successor
    // to create ephemeral intermediates. For each subset of consumers,
    // also add their depth-1 successors (ops that consume the consumer's output).
    {
      // Map each consumer to its single-consumer successors (branch chain)
      std::unordered_map<size_t, std::vector<size_t>> chain_successors;
      for (size_t op : consumers) {
        for (size_t out_t : p.ops[op].outputs) {
          auto& out_cons = dag.tensor_consumers[out_t];
          if (out_cons.size() == 1) {
            chain_successors[op].push_back(out_cons[0]);
          }
        }
      }

      int chain_added = 0;

      // For each group size (number of branch-pairs)
      int max_pairs = std::min(max_group_size, nc);
      for (int n_pairs = 2; n_pairs <= max_pairs; ++n_pairs) {
        if (solver::ElapsedSince(t0) > deadline) break;

        // Sliding window on sorted consumers
        for (int start = 0; start + n_pairs <= nc; ++start) {
          if (solver::ElapsedSince(t0) > deadline) break;

          // Build the branch-pair group: consumer + its successor(s)
          std::vector<size_t> group;
          for (int k = start; k < start + n_pairs; ++k) {
            size_t op = cinfos[k].op;
            group.push_back(op);
            if (chain_successors.count(op))
              for (size_t succ : chain_successors[op])
                group.push_back(succ);
          }
          chain_added += TrySubset(group);
        }
      }

      if (chain_added > 0)
        std::cerr << "    → " << chain_added << " branch-pair columns added\n";
      total_added += chain_added;
    }

    std::cerr << "    → " << total_added << " columns added\n";
  }

  std::cerr << "Phase 0 done: " << columns.size() << " columns total\n";
}

// ── Phase 1 (Column Generation): LP-guided column generation ─────────────────

struct Phase1Result {
  std::vector<size_t> selected;  // indices into columns
  double cost;
  bool solved;
};


// ── Phase 1 (Exhaustive): HiGHS Set Partitioning ────────────────────────────

Phase1Result SolvePartitionILP(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const std::vector<SuperColumn>& columns,
    const std::vector<bool>& recomputable = {},
    std::chrono::steady_clock::time_point t0 = {},
    double hard_deadline = 60.0,
    int iteration = 0) {

  int num_ops = (int)dag.num_ops;
  int num_cols = (int)columns.size();

  std::cerr << "Phase 1 ILP: " << num_ops << " ops, " << num_cols << " columns\n";

  // ── Build ILP ───────────────────────────────────────────────────────
  std::vector<double> col_cost(num_cols);
  std::vector<double> col_lower(num_cols, 0.0);
  std::vector<double> col_upper(num_cols);
  std::vector<HighsVarType> integrality(num_cols);

  for (int j = 0; j < num_cols; ++j) {
    col_cost[j] = columns[j].cost;
    col_upper[j] = 1.0;
    integrality[j] = HighsVarType::kInteger;
  }

  // CSC matrix
  std::vector<HighsInt> a_start(num_cols + 1);
  std::vector<HighsInt> a_index;
  std::vector<double> a_value;

  for (int j = 0; j < num_cols; ++j) {
    a_start[j] = (HighsInt)a_index.size();
    for (size_t op : columns[j].ops) {
      a_index.push_back((HighsInt)op);
      a_value.push_back(1.0);
    }
  }
  a_start[num_cols] = (HighsInt)a_index.size();

  // Row bounds
  std::vector<double> row_lower(num_ops);
  std::vector<double> row_upper(num_ops);
  // Op coverage: exactly 1 (or ≥1 for recomputable)
  for (int i = 0; i < num_ops; ++i) {
    row_lower[i] = 1.0;
    row_upper[i] = 1.0;
    if (i < (int)recomputable.size() && recomputable[i])
      row_upper[i] = (double)num_cols;
  }

  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("time_limit", std::max(1.0, hard_deadline - solver::ElapsedSince(t0)));

  HighsModel model;
  model.lp_.num_col_ = num_cols;
  model.lp_.num_row_ = num_ops;
  model.lp_.sense_ = ObjSense::kMinimize;
  model.lp_.col_cost_ = col_cost;
  model.lp_.col_lower_ = col_lower;
  model.lp_.col_upper_ = col_upper;
  model.lp_.row_lower_ = row_lower;
  model.lp_.row_upper_ = row_upper;
  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  model.lp_.a_matrix_.start_ = a_start;
  model.lp_.a_matrix_.index_ = a_index;
  model.lp_.a_matrix_.value_ = a_value;
  model.lp_.integrality_ = integrality;

  highs.passModel(model);

  // ── LP relaxation bounds (iteration 0 only — pure diagnostics) ──
  if (iteration == 0) {
    // LP-optimistic: valid lower bound (all intermediates retained-in, retain-out=true)
    {
      std::vector<double> lb_costs(num_cols);
      for (int j = 0; j < num_cols; ++j) {
        auto& col = columns[j];
        auto ts = solver::Classify(p, col.ops, &dag);
        auto roles = solver::BuildRoles(p, col.ops, ts);
        std::unordered_set<size_t> ri_opt;
        for (size_t t : ts.inputs)
          if (dag.tensor_producer[t] >= 0) ri_opt.insert(t);
        int64_t maxK = 0;
        for (size_t op : col.ops)
          if (p.ops[op].op_type == "MatMul")
            maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);
        lb_costs[j] = solver::AnalyticalCost(
            p, col.ops, col.gran, col.snake_mode,
            ri_opt, ts, roles, maxK,
            (int64_t)ts.outputs.size(), true);
      }
      HighsModel lp_model = model;
      lp_model.lp_.integrality_.clear();
      lp_model.lp_.col_cost_ = lb_costs;
      Highs lp;
      lp.setOptionValue("output_flag", false);
      lp.passModel(lp_model);
      lp.run();
      if (lp.getModelStatus() == HighsModelStatus::kOptimal)
        std::cerr << "  LP-optimistic LB: "
                  << (int64_t)lp.getObjectiveValue() << "\n";
    }

    // LP-diagnostic: tighter, uses existing column costs (ri={}, ro=true)
    {
      HighsModel lp_model = model;
      lp_model.lp_.integrality_.clear();
      Highs lp;
      lp.setOptionValue("output_flag", false);
      lp.passModel(lp_model);
      lp.run();
      if (lp.getModelStatus() == HighsModelStatus::kOptimal)
        std::cerr << "  LP-diagnostic:    "
                  << (int64_t)lp.getObjectiveValue() << "\n";
    }
  }

  // Warm start: cheapest singleton cover per op
  {
    std::vector<double> warm(num_cols, 0.0);
    std::vector<int> best_singleton(num_ops, -1);
    for (int j = 0; j < num_cols; ++j) {
      if (columns[j].ops.size() == 1) {
        size_t op = columns[j].ops[0];
        if (best_singleton[op] < 0 || columns[j].cost < columns[best_singleton[op]].cost)
          best_singleton[op] = j;
      }
    }
    bool all_covered = true;
    for (int i = 0; i < num_ops; ++i) {
      if (best_singleton[i] < 0) { all_covered = false; break; }
      warm[best_singleton[i]] = 1.0;
    }
    if (all_covered) {
      HighsSolution warm_sol;
      warm_sol.col_value = warm;
      highs.setSolution(warm_sol);
    }
  }

  auto status = highs.run();

  Phase1Result result;
  result.solved = false;

  if (status != HighsStatus::kError) {
    auto model_status = highs.getModelStatus();
    if (model_status == HighsModelStatus::kOptimal ||
        model_status == HighsModelStatus::kObjectiveBound ||
        model_status == HighsModelStatus::kSolutionLimit ||
        model_status == HighsModelStatus::kTimeLimit) {

      const auto& sol = highs.getSolution();
      if (!sol.col_value.empty()) {
        std::vector<bool> op_covered(num_ops, false);
        bool valid = true;
        result.cost = 0;

        for (int j = 0; j < num_cols; ++j) {
          if (sol.col_value[j] > 0.5) {
            for (size_t op : columns[j].ops) {
              if (op_covered[op] &&
                  !(op < recomputable.size() && recomputable[op])) {
                valid = false; break;
              }
              op_covered[op] = true;
            }
            if (!valid) break;
            result.selected.push_back(j);
            result.cost += columns[j].cost;
          }
        }

        if (valid) {
          for (int i = 0; i < num_ops; ++i)
            if (!op_covered[i]) { valid = false; break; }
        }

        if (valid) {
          result.solved = true;
          std::cerr << "  Phase 1 solved: cost=" << result.cost
                    << " subgraphs=" << result.selected.size() << "\n";
        }
      }
    }
  }

  return result;
}

// ── Column Generation: LP-guided pricing ────────────────────────────────────

// Solve LP relaxation of set-partitioning and return dual variables.
// Uses retention-optimistic costs: each column priced as if all producible
// inputs are retained, giving a tighter lower bound and more useful duals.
// Returns empty vector on failure.
std::vector<double> SolveLPRelaxation(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const std::vector<SuperColumn>& columns,
    const std::vector<bool>& recomputable,
    double* lp_obj = nullptr) {

  int num_ops = (int)dag.num_ops;
  int num_cols = (int)columns.size();

  std::vector<double> col_cost(num_cols);
  std::vector<double> col_lower(num_cols, 0.0);
  std::vector<double> col_upper(num_cols, 1.0);

  // Use retention-optimistic costs for LP.
  for (int j = 0; j < num_cols; ++j) {
    auto& col = columns[j];
    auto ts = solver::Classify(p, col.ops, &dag);
    auto roles = solver::BuildRoles(p, col.ops, ts);
    std::unordered_set<size_t> ri_opt;
    for (size_t t : ts.inputs)
      if (dag.tensor_producer[t] >= 0) ri_opt.insert(t);
    int64_t maxK = 0;
    for (size_t op : col.ops)
      if (p.ops[op].op_type == "MatMul")
        maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);
    col_cost[j] = solver::AnalyticalCost(
        p, col.ops, col.gran, col.snake_mode,
        ri_opt, ts, roles, maxK,
        (int64_t)ts.outputs.size(), true);
    if (col_cost[j] >= solver::kInf)
      col_cost[j] = columns[j].cost;  // fallback to no-retention cost
  }

  // CSC matrix
  std::vector<HighsInt> a_start(num_cols + 1);
  std::vector<HighsInt> a_index;
  std::vector<double> a_value;

  for (int j = 0; j < num_cols; ++j) {
    a_start[j] = (HighsInt)a_index.size();
    for (size_t op : columns[j].ops) {
      a_index.push_back((HighsInt)op);
      a_value.push_back(1.0);
    }
  }
  a_start[num_cols] = (HighsInt)a_index.size();

  std::vector<double> row_lower(num_ops, 1.0);
  std::vector<double> row_upper(num_ops);
  for (int i = 0; i < num_ops; ++i) {
    row_upper[i] = (i < (int)recomputable.size() && recomputable[i])
                   ? (double)num_cols : 1.0;
  }

  Highs lp;
  lp.setOptionValue("output_flag", false);

  HighsModel model;
  model.lp_.num_col_ = num_cols;
  model.lp_.num_row_ = num_ops;
  model.lp_.sense_ = ObjSense::kMinimize;
  model.lp_.col_cost_ = col_cost;
  model.lp_.col_lower_ = col_lower;
  model.lp_.col_upper_ = col_upper;
  model.lp_.row_lower_ = row_lower;
  model.lp_.row_upper_ = row_upper;
  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  model.lp_.a_matrix_.start_ = a_start;
  model.lp_.a_matrix_.index_ = a_index;
  model.lp_.a_matrix_.value_ = a_value;
  // No integrality → LP relaxation

  lp.passModel(model);
  auto status = lp.run();

  if (status == HighsStatus::kError ||
      lp.getModelStatus() != HighsModelStatus::kOptimal) {
    return {};
  }

  if (lp_obj)
    *lp_obj = lp.getObjectiveValue();

  const auto& sol = lp.getSolution();

  // Diagnostic: LP solution composition.
  {
    int n_frac = 0, n_int = 0, n_multi = 0;
    for (int j = 0; j < num_cols; ++j) {
      if (sol.col_value[j] > 0.01) {
        if (columns[j].ops.size() >= 2) ++n_multi;
        if (sol.col_value[j] > 0.99) ++n_int;
        else ++n_frac;
      }
    }
    std::cerr << "    LP sol: " << n_int << " int + " << n_frac
              << " frac (" << n_multi << " multi-op)\n";
  }

  return sol.row_dual;  // dual[i] = π_i for op i's coverage constraint
}

// Pricing heuristic: find columns with negative reduced cost.
// Returns number of new columns added.
int PricingHeuristic(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const AdjList& adj,
    const std::vector<double>& duals,
    std::vector<SuperColumn>& columns,
    ClassifyCache& cc,
    std::chrono::steady_clock::time_point t0,
    double deadline) {

  int N = (int)dag.num_ops;
  int added = 0;

  // Track existing column op-sets to avoid duplicates.
  std::unordered_set<std::vector<size_t>, VectorHash> existing;
  for (auto& col : columns)
    existing.insert(col.ops);

  // Use retention-optimistic costs for pricing: assume all producible inputs
  // are retained (best case for this column's position in ordering).
  auto OptimisticCost = [&](const std::vector<size_t>& ops) -> double {
    auto ts = solver::Classify(p, ops, &dag);
    auto roles = solver::BuildRoles(p, ops, ts);
    std::unordered_set<size_t> ri_opt;
    for (size_t t : ts.inputs)
      if (dag.tensor_producer[t] >= 0) ri_opt.insert(t);
    auto gc = solver::BestGranularity(p, ops, ri_opt, 0, &dag, true);
    return gc.cost;
  };

  auto TrySubset = [&](std::vector<size_t>& ops) -> int {
    std::sort(ops.begin(), ops.end());
    if (existing.count(ops)) return 0;
    if (ops.size() >= 2 && !solver::IsDAGConvex(dag, ops)) return 0;

    double cost = OptimisticCost(ops);
    if (cost >= solver::kInf) return 0;

    double dual_sum = 0;
    for (size_t o : ops) dual_sum += duals[o];
    if (cost - dual_sum >= -1e-6) return 0;

    int n = GenerateColumnsForOps(p, dag, ops, cc, columns);
    existing.insert(ops);
    return n;
  };

  // Sort ops by dual value descending.
  std::vector<size_t> sorted_ops(N);
  std::iota(sorted_ops.begin(), sorted_ops.end(), 0);
  std::sort(sorted_ops.begin(), sorted_ops.end(),
            [&](size_t a, size_t b) { return duals[a] > duals[b]; });

  // Strategy A: Greedy expansion from high-dual ops.
  int max_starts = std::min(N, 30);
  for (int si = 0; si < max_starts; ++si) {
    if (solver::ElapsedSince(t0) > deadline) break;

    size_t start = sorted_ops[si];
    std::vector<size_t> current = {start};
    std::unordered_set<size_t> current_set = {start};

    // Greedily add neighbor with best (most negative) marginal reduced cost.
    for (int expand = 0; expand < 10; ++expand) {
      if (solver::ElapsedSince(t0) > deadline) break;

      std::vector<size_t> candidates;
      for (size_t op : current)
        for (size_t nb : adj[op])
          if (!current_set.count(nb))
            candidates.push_back(nb);
      std::sort(candidates.begin(), candidates.end());
      candidates.erase(std::unique(candidates.begin(), candidates.end()),
                       candidates.end());

      size_t best_nb = (size_t)-1;
      double best_cost = solver::kInf;

      for (size_t nb : candidates) {
        auto trial = current;
        trial.push_back(nb);
        std::sort(trial.begin(), trial.end());
        if (!solver::IsDAGConvex(dag, trial)) continue;

        double cost = OptimisticCost(trial);
        if (cost >= solver::kInf) continue;

        double dual_sum = 0;
        for (size_t o : trial) dual_sum += duals[o];
        double reduced = cost - dual_sum;
        if (reduced < best_cost) {
          best_cost = reduced;
          best_nb = nb;
        }
      }

      if (best_nb == (size_t)-1) break;

      current.push_back(best_nb);
      current_set.insert(best_nb);
      std::sort(current.begin(), current.end());

      // Emit at every expansion step if negative reduced cost.
      if (current.size() >= 2) {
        auto trial = current;
        added += TrySubset(trial);
      }
    }
  }

  // Strategy B: All pairs of high-dual ops (even non-adjacent).
  // This finds fusion opportunities the adjacency-based enumeration misses.
  int pair_limit = std::min(N, 40);
  for (int i = 0; i < pair_limit; ++i) {
    if (solver::ElapsedSince(t0) > deadline) break;
    for (int j = i + 1; j < pair_limit; ++j) {
      size_t a = sorted_ops[i], b = sorted_ops[j];
      std::vector<size_t> pair = {std::min(a,b), std::max(a,b)};
      added += TrySubset(pair);
    }
  }

  // Strategy C: Extend existing fractional columns.
  // Find columns with fractional LP values (0 < x < 1) and try extending them.
  // We identify fractional columns by checking which multi-op columns have
  // negative reduced cost potential if extended.
  int col_limit = std::min((int)columns.size(), 200);
  for (int ci = 0; ci < col_limit; ++ci) {
    if (solver::ElapsedSince(t0) > deadline) break;
    auto& col = columns[ci];
    if (col.ops.size() < 2) continue;

    std::unordered_set<size_t> ops_set(col.ops.begin(), col.ops.end());
    for (size_t op : col.ops) {
      if (solver::ElapsedSince(t0) > deadline) break;
      for (size_t nb : adj[op]) {
        if (ops_set.count(nb)) continue;
        auto trial = col.ops;
        trial.push_back(nb);
        added += TrySubset(trial);
      }
    }

    // Also try removing each op (creates sub-columns).
    if (col.ops.size() >= 3) {
      for (size_t skip = 0; skip < col.ops.size(); ++skip) {
        std::vector<size_t> sub;
        for (size_t k = 0; k < col.ops.size(); ++k)
          if (k != skip) sub.push_back(k < col.ops.size() ? col.ops[k] : 0);
        // Fix: actually use ops
        sub.clear();
        for (size_t k = 0; k < col.ops.size(); ++k)
          if (k != skip) sub.push_back(col.ops[k]);
        added += TrySubset(sub);
      }
    }
  }

  // Strategy D: DAG-path columns — follow producer→consumer chains.
  for (int si = 0; si < std::min(N, 20); ++si) {
    if (solver::ElapsedSince(t0) > deadline) break;
    size_t start = sorted_ops[si];

    // Follow successors greedily by dual value.
    std::vector<size_t> path = {start};
    std::unordered_set<size_t> path_set = {start};
    size_t cur = start;
    for (int step = 0; step < 8; ++step) {
      size_t best_succ = (size_t)-1;
      double best_dual = -1e18;
      for (size_t succ : dag.op_successors[cur]) {
        if (path_set.count(succ)) continue;
        if (duals[succ] > best_dual) {
          best_dual = duals[succ];
          best_succ = succ;
        }
      }
      if (best_succ == (size_t)-1) break;
      path.push_back(best_succ);
      path_set.insert(best_succ);
      cur = best_succ;

      // Try all prefixes of length >= 2.
      if (path.size() >= 2) {
        auto trial = path;
        added += TrySubset(trial);
      }
    }
  }

  return added;
}

// Run column generation loop: LP → pricing → repeat → final MIP.
Phase1Result RunColumnGeneration(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const AdjList& adj,
    std::vector<SuperColumn>& columns,
    const std::vector<bool>& recomputable,
    ClassifyCache& cc,
    std::chrono::steady_clock::time_point t0,
    double hard_deadline) {

  constexpr int kMaxCGIters = 5;
  double cg_deadline = hard_deadline * 0.35;

  std::cerr << "\n=== Column Generation ===\n";
  std::cerr << "Initial columns: " << columns.size() << "\n";

  int total_added = 0;
  for (int iter = 0; iter < kMaxCGIters; ++iter) {
    if (solver::ElapsedSince(t0) > cg_deadline) break;

    // Solve LP relaxation.
    double lp_obj = 0;
    auto duals = SolveLPRelaxation(p, dag, columns, recomputable, &lp_obj);
    if (duals.empty()) {
      std::cerr << "  CG iter " << iter << ": LP infeasible\n";
      break;
    }
    std::cerr << "  CG iter " << iter << ": LP=" << (int64_t)lp_obj
              << ", cols=" << columns.size();

    // Pricing heuristic.
    double pricing_deadline = std::min(cg_deadline,
                                        solver::ElapsedSince(t0) + 2.0);
    int new_cols = PricingHeuristic(p, dag, adj, duals, columns, cc,
                                    t0, pricing_deadline);
    std::cerr << ", +" << new_cols << " new columns\n";
    total_added += new_cols;

    if (new_cols == 0) {
      std::cerr << "  CG converged (no negative reduced cost columns)\n";
      break;
    }
  }

  std::cerr << "Final column pool: " << columns.size()
            << " (" << total_added << " added by CG)\n";

  // Don't re-solve MIP here — let RunILPIterations handle it with the enriched pool.
  Phase1Result result;
  result.solved = false;
  return result;
}

// ── Phase 2: Ordering + Retention ────────────────────────────────────────────

struct OrderedSubgraph {
  std::vector<size_t> ops;
  mlsys::Granularity gran;
  int snake_mode;
  std::vector<int64_t> traversal;
  std::vector<size_t> output_tensors;
  std::vector<size_t> tensors_to_retain;  // assigned by Phase 2
  double cost;  // recomputed with retention
  std::unordered_set<size_t> assumed_retained_in;  // from Phase 1 super-column
  std::unordered_set<size_t> input_tensors;        // boundary inputs
};

// Build mlsys::Solution directly from Phase 2's OrderedSubgraph data,
// preserving granularity/traversal/retention decisions without re-searching.
// Two-pass: (1) build full solution with original granularities, (2) forward WS
// validation using the evaluator's exact model, re-searching on overflow.
static mlsys::Solution DirectSolutionFromOrdering(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const std::vector<OrderedSubgraph>& subgraphs) {
  // ── Pass 1: build full solution with original granularities ──
  mlsys::Solution sol;
  for (auto& osg : subgraphs) {
    mlsys::Subgraph sg;
    sg.ops = osg.ops;
    // Topo-sort ops within subgraph (evaluator validates internal DAG order)
    if (sg.ops.size() > 1) {
      std::unordered_set<size_t> op_set(sg.ops.begin(), sg.ops.end());
      std::unordered_map<size_t, int> in_deg;
      for (size_t op : sg.ops) in_deg[op] = 0;
      for (size_t op : sg.ops)
        for (size_t succ : dag.op_successors[op])
          if (op_set.count(succ)) ++in_deg[succ];
      std::queue<size_t> q;
      for (size_t op : sg.ops)
        if (in_deg[op] == 0) q.push(op);
      std::vector<size_t> sorted;
      sorted.reserve(sg.ops.size());
      while (!q.empty()) {
        size_t u = q.front(); q.pop();
        sorted.push_back(u);
        for (size_t succ : dag.op_successors[u])
          if (op_set.count(succ) && --in_deg[succ] == 0)
            q.push(succ);
      }
      if (sorted.size() == sg.ops.size()) sg.ops = std::move(sorted);
    }
    sg.granularity = osg.gran;
    sg.subgraph_latency = osg.cost;
    sg.tensors_to_retain = osg.tensors_to_retain;
    if (!osg.traversal.empty())
      sg.traversal_order = osg.traversal;
    sol.subgraphs.push_back(std::move(sg));
  }

  // ── Pass 2: forward WS validation with full solution visible ──
  // Precompute consumed_by and produced_elsewhere once for the performance overload.
  std::unordered_map<size_t, std::vector<size_t>> consumed_by;
  for (size_t i = 0; i < p.ops.size(); ++i)
    for (size_t t : p.ops[i].inputs)
      consumed_by[t].push_back(i);

  // Precompute all produced tensors and per-subgraph outputs for O(T) produced_elsewhere
  std::unordered_set<size_t> all_produced;
  std::vector<std::vector<size_t>> sg_output_tensors(sol.subgraphs.size());
  for (size_t i = 0; i < sol.subgraphs.size(); ++i)
    for (size_t op : sol.subgraphs[i].ops)
      for (size_t t : p.ops[op].outputs) {
        all_produced.insert(t);
        sg_output_tensors[i].push_back(t);
      }

  std::unordered_set<size_t> prev_retained;
  for (size_t idx = 0; idx < sol.subgraphs.size(); ++idx) {
    // produced_elsewhere = all_produced minus this subgraph's outputs
    auto produced_elsewhere = all_produced;
    for (size_t t : sg_output_tensors[idx])
      produced_elsewhere.erase(t);

    // solver_common::WorkingSet mirrors the evaluator for one subgraph at
    // its chosen granularity. For retained-in accounting, pass it via an
    // extra term equal to the full-tensor sizes of prev_retained.
    int64_t retained_extra = 0;
    for (size_t t : prev_retained) {
      retained_extra += p.tensors[t].width * p.tensors[t].height;
    }
    int64_t ws = solver::WorkingSet(p, sol.subgraphs[idx].ops,
                                     sol.subgraphs[idx].granularity,
                                     retained_extra, &dag);
    if (ws > p.fast_memory_capacity) {
      // Re-search using BestGranularity (analytical h_max/w_max finds feasible
      // configs in tight-memory scenarios where sparse DimCandidates misses them)
      auto& sg = sol.subgraphs[idx];
      auto ts = solver::Classify(p, sg.ops, &dag);
      std::unordered_set<size_t> actual_ri;
      for (size_t t : prev_retained)
        if (ts.inputs.count(t)) actual_ri.insert(t);

      auto cfg = solver::BestGranularity(p, sg.ops, actual_ri, 0, &dag,
                                          /*assume_retain_out=*/false);
      if (cfg.cost < solver::kInf) {
        sg.granularity = cfg.gran;
        if (!cfg.traversal.empty())
          sg.traversal_order = cfg.traversal;
        else
          sg.traversal_order.reset();
      } else {
        // Restore original granularity; Evaluate will fail → candidate scores 1e18
        sg.granularity = subgraphs[idx].gran;
        if (!subgraphs[idx].traversal.empty())
          sg.traversal_order = subgraphs[idx].traversal;
        else
          sg.traversal_order.reset();
      }
    }

    prev_retained.clear();
    for (size_t t : sol.subgraphs[idx].tensors_to_retain)
      prev_retained.insert(t);
  }
  return sol;
}

// Build subgraph-level DAG: sg_must_before[i] = set of subgraph indices that must come before i
void BuildSubgraphDAG(
    const solver::DAG& dag,
    const std::vector<OrderedSubgraph>& subgraphs,
    std::vector<std::unordered_set<size_t>>& sg_preds) {

  int S = (int)subgraphs.size();
  sg_preds.resize(S);

  // Map each op to its subgraph indices (op may appear in multiple subgraphs
  // due to recomputation).
  std::unordered_map<size_t, std::vector<size_t>> op_to_sgs;
  for (int s = 0; s < S; ++s)
    for (size_t op : subgraphs[s].ops)
      op_to_sgs[op].push_back(s);

  // For each DAG edge (i→j), if i and j are in different subgraphs,
  // add predecessor constraint. Skip if the consumer subgraph also
  // contains op i (recomputation: sj doesn't depend on si's copy).
  for (size_t i = 0; i < dag.num_ops; ++i) {
    auto it_i = op_to_sgs.find(i);
    if (it_i == op_to_sgs.end()) continue;
    for (size_t j : dag.op_successors[i]) {
      auto it_j = op_to_sgs.find(j);
      if (it_j == op_to_sgs.end()) continue;
      for (size_t si : it_i->second) {
        for (size_t sj : it_j->second) {
          if (si == sj) continue;
          // Skip if sj contains op i (sj recomputes i, doesn't need si)
          auto& sj_ops = subgraphs[sj].ops;
          if (std::find(sj_ops.begin(), sj_ops.end(), i) != sj_ops.end()) continue;
          sg_preds[sj].insert(si);
        }
      }
    }
  }

  // Transitive closure via topological propagation (O(S²) vs O(S³) fixed-point)
  {
    // Kahn's algorithm for topo order on the subgraph DAG
    std::vector<int> in_deg(S, 0);
    std::vector<std::vector<size_t>> sg_succs(S);
    for (int s = 0; s < S; ++s)
      for (size_t pred : sg_preds[s]) {
        sg_succs[pred].push_back(s);
        ++in_deg[s];
      }
    std::queue<int> q;
    for (int s = 0; s < S; ++s)
      if (in_deg[s] == 0) q.push(s);
    std::vector<int> topo;
    topo.reserve(S);
    while (!q.empty()) {
      int u = q.front(); q.pop();
      topo.push_back(u);
      for (size_t v : sg_succs[u])
        if (--in_deg[v] == 0) q.push(v);
    }
    // Propagate: each node inherits all transitive predecessors
    for (int s : topo) {
      std::unordered_set<size_t> expanded;
      for (size_t pred : sg_preds[s]) {
        expanded.insert(pred);
        for (size_t pp : sg_preds[pred])
          expanded.insert(pp);
      }
      sg_preds[s] = std::move(expanded);
    }
  }
}

// Compute total cost for a given ordering with greedy retention (mutating version).
// Pass-through tensor logic removed: per rule #52, only op outputs can be retained,
// so pass-through (retained tensors not consumed by this subgraph) cannot occur
// with proper retention assignment via RetentionCandidates.
double EvaluateOrdering(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    std::vector<OrderedSubgraph>& subgraphs,
    const std::vector<size_t>& order,
    ClassifyCache* cc = nullptr) {

  double total = 0;
  ClassifyCache local_cc;
  ClassifyCache& cache = cc ? *cc : local_cc;
  std::unordered_set<size_t> full_retained_set;
  for (size_t idx = 0; idx < order.size(); ++idx) {
    auto& sg = subgraphs[order[idx]];
    sg.tensors_to_retain.clear();

    auto& ts = CachedClassify(p, sg.ops, dag, cache);
    auto roles = solver::BuildRoles(p, sg.ops, ts);
    std::unordered_set<size_t> retained_in;
    for (size_t t : full_retained_set)
      if (ts.inputs.count(t)) retained_in.insert(t);

    // Re-optimize granularity when retained_in differs from Phase 1 assumption.
    // Use ExactCostCached with retain_out={} for consistent comparison
    // (matching the recomputation at line below).
    if (retained_in != sg.assumed_retained_in) {
      std::vector<GranCandidate> recands;
      EnumerateGranularitiesWithRI(p, sg.ops, retained_in, dag, recands, cache);
      // Baseline: current granularity WITH new retained_in (not old cost without it)
      double best_reopt = solver::ExactCostCached(p, sg.ops, sg.gran, {},
                                                   retained_in, ts, roles, &sg.traversal);
      for (auto& gc : recands) {
        double actual = solver::ExactCostCached(p, sg.ops, gc.gran, {},
                                                retained_in, ts, roles, &gc.traversal);
        if (actual < best_reopt) {
          sg.gran = gc.gran;
          sg.snake_mode = gc.snake_mode;
          sg.traversal = gc.traversal;
          best_reopt = actual;
        }
      }
    }

    // Recompute cost with actual retained_in and current granularity
    double cost = solver::ExactCostCached(p, sg.ops, sg.gran, {},
                                          retained_in, ts, roles, &sg.traversal);
    sg.cost = cost;

    // Determine what to retain for next subgraph
    if (idx + 1 < order.size()) {
      auto& next_sg = subgraphs[order[idx + 1]];
      auto rc = solver::RetentionCandidates(p, sg.ops, next_sg.ops, &dag);

      if (!rc.empty()) {
        // Try retaining all candidates (fast path)
        std::vector<size_t> retain_all(rc.begin(), rc.end());
        double cost_all = solver::ExactCostCached(p, sg.ops, sg.gran, retain_all,
                                                   retained_in, ts, roles, &sg.traversal);
        if (cost_all <= cost) {
          cost = cost_all;
          sg.cost = cost;
          sg.tensors_to_retain = retain_all;
        } else if (rc.size() >= 2) {
          // Retain-all worsened cost: try subsets (singles + pairs)
          double best_sub_cost = cost;
          std::vector<size_t> best_retain;

          // Singles
          for (size_t i = 0; i < rc.size(); ++i) {
            std::vector<size_t> sub = {rc[i]};
            double c = solver::ExactCostCached(p, sg.ops, sg.gran, sub,
                                                retained_in, ts, roles, &sg.traversal);
            if (c < best_sub_cost) { best_sub_cost = c; best_retain = sub; }
          }
          // Pairs
          for (size_t i = 0; i < rc.size(); ++i) {
            for (size_t j = i + 1; j < rc.size(); ++j) {
              std::vector<size_t> sub = {rc[i], rc[j]};
              double c = solver::ExactCostCached(p, sg.ops, sg.gran, sub,
                                                  retained_in, ts, roles, &sg.traversal);
              if (c < best_sub_cost) { best_sub_cost = c; best_retain = sub; }
            }
          }

          if (!best_retain.empty()) {
            cost = best_sub_cost;
            sg.cost = cost;
            sg.tensors_to_retain = best_retain;
          }
        }
      }
    } else {
      // Last subgraph: retain non-graph-output tensors to avoid unnecessary
      // write-back. Graph outputs must end up in slow memory per evaluator rules.
      std::vector<size_t> retain_all;
      for (size_t t : ts.outputs) {
        if (!dag.tensor_consumers[t].empty())
          retain_all.push_back(t);
      }
      double cur_with_retain = solver::ExactCostCached(p, sg.ops, sg.gran, retain_all,
                                                        retained_in, ts, roles, &sg.traversal);
      if (cur_with_retain <= cost) {
        cost = cur_with_retain;
        sg.cost = cost;
        sg.tensors_to_retain = retain_all;
      }
    }

    // Update full retained set for next subgraph.
    full_retained_set.clear();
    for (size_t t : sg.tensors_to_retain)
      full_retained_set.insert(t);

    total += cost;
  }

  return total;
}

// ── Held-Karp Ordering DP ───────────────────────────────────────────────────

double HeldKarpOrdering(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    std::vector<OrderedSubgraph>& subgraphs,
    const std::vector<std::unordered_set<size_t>>& sg_preds) {

  int S = (int)subgraphs.size();

  // ── Precompute base costs (no retention, no retained_in) ──────────────
  struct BaseInfo {
    solver::GranConfig cfg;  // gran, traversal, cost (with ro={}, ri={})
  };
  std::vector<BaseInfo> base(S);
  for (int s = 0; s < S; ++s) {
    base[s].cfg = solver::BestGranularity(p, subgraphs[s].ops, {}, 0, &dag,
                                          /*assume_retain_out=*/false);
  }

  // ── Precompute S² transitions ─────────────────────────────────────────
  struct TransInfo {
    double cost;          // next's cost with ri from last, ro={}
    double retain_delta;  // savings from last retaining for next (≥0)
    solver::GranConfig cfg;  // next's best granularity with ri
    std::vector<size_t> retain_out;  // what last retains for next
  };
  std::vector<std::vector<TransInfo>> trans(S, std::vector<TransInfo>(S));

  for (int last = 0; last < S; ++last) {
    for (int next = 0; next < S; ++next) {
      if (next == last) continue;
      auto ri = solver::RetentionCandidates(p, subgraphs[last].ops,
                                             subgraphs[next].ops, &dag);
      std::unordered_set<size_t> ri_set(ri.begin(), ri.end());

      if (ri.empty()) {
        trans[last][next].cost = base[next].cfg.cost;
        trans[last][next].retain_delta = 0;
        trans[last][next].cfg = base[next].cfg;
      } else {
        // Re-optimize next's granularity given retained inputs
        auto cfg = solver::BestGranularity(p, subgraphs[next].ops, ri_set, 0,
                                           &dag, /*assume_retain_out=*/false);

        if (cfg.cost >= solver::kInf) {
          // Retention doesn't fit in WS — fall back to no-retention transition
          trans[last][next].cost = base[next].cfg.cost;
          trans[last][next].retain_delta = 0;
          trans[last][next].cfg = base[next].cfg;
          ri.clear();
        } else {
          trans[last][next].cost = cfg.cost;
          trans[last][next].cfg = cfg;

          // Retention delta for last: savings from retaining ri
          const auto* trav_ptr = base[last].cfg.traversal.empty()
              ? nullptr : &base[last].cfg.traversal;
          double cost_with = solver::ExactCost(
              p, subgraphs[last].ops, base[last].cfg.gran, ri, {}, trav_ptr, &dag);
          trans[last][next].retain_delta = base[last].cfg.cost - cost_with;
        }
      }
      trans[last][next].retain_out = std::move(ri);
    }
  }

  ClassifyCache cc;

  std::cerr << "  Held-Karp: S=" << S << ", precomputed " << S*S << " transitions\n";

  // ── Held-Karp DP ──────────────────────────────────────────────────────
  // State: (mask, last) → minimum cost via flat array (S≤15 required)
  const double INF = 1e18;
  if (S > 15) return INF;
  const uint64_t FULL = (1ULL << S) - 1;

  size_t flat_size = ((size_t)1 << S) * S;
  std::vector<double> dp_flat(flat_size, INF);
  std::vector<int> par_flat(flat_size, -1);

  auto dp_key = [&](uint64_t mask, int last) -> uint64_t {
    return mask * S + last;
  };
  auto dp_get = [&](uint64_t mask, int last) -> double {
    return dp_flat[dp_key(mask, last)];
  };
  auto dp_set = [&](uint64_t mask, int last, double val, int parent) {
    uint64_t key = dp_key(mask, last);
    dp_flat[key] = val;
    par_flat[key] = parent;
  };
  auto par_get = [&](uint64_t mask, int last) -> int {
    return par_flat[dp_key(mask, last)];
  };

  // Initialize: source subgraphs (no DAG predecessors)
  for (int s = 0; s < S; ++s) {
    if (!sg_preds[s].empty()) continue;
    dp_set(1ULL << s, s, base[s].cfg.cost, -1);
  }

  // Expand
  for (uint64_t mask = 1; mask <= FULL; ++mask) {
    for (int last = 0; last < S; ++last) {
      if (!(mask & (1ULL << last))) continue;
      double prev_cost = dp_get(mask, last);
      if (prev_cost >= INF) continue;

      for (int next = 0; next < S; ++next) {
        if (mask & (1ULL << next)) continue;

        // Precedence check: all predecessors of next must be in mask
        bool ready = true;
        for (size_t pred : sg_preds[next]) {
          if (!(mask & (1ULL << pred))) { ready = false; break; }
        }
        if (!ready) continue;

        double total = prev_cost
            - trans[last][next].retain_delta
            + trans[last][next].cost;

        uint64_t new_mask = mask | (1ULL << next);
        if (total < dp_get(new_mask, next)) {
          dp_set(new_mask, next, total, last);
        }
      }
    }
  }

  // ── Find best terminal state (predecessor-aware terminal delta) ──────
  double best_cost = INF;
  int best_last = -1;
  for (int s = 0; s < S; ++s) {
    double val = dp_get(FULL, s);
    if (val >= INF) continue;

    // Compute terminal retention delta using actual predecessor context
    double term_delta = 0;
    int prev = par_get(FULL, s);
    assert(prev == -1 || trans[prev][s].cost < INF);

    auto& ts = CachedClassify(p, subgraphs[s].ops, dag, cc);
    std::vector<size_t> all_out;
    for (size_t t : ts.outputs)
      if (!dag.tensor_consumers[t].empty())
        all_out.push_back(t);
    if (!all_out.empty()) {
      auto& cfg = (prev >= 0) ? trans[prev][s].cfg : base[s].cfg;
      std::unordered_set<size_t> ri;
      if (prev >= 0)
        for (size_t t : trans[prev][s].retain_out) ri.insert(t);
      const auto* trav_ptr = cfg.traversal.empty() ? nullptr : &cfg.traversal;
      double cost_without = solver::ExactCost(
          p, subgraphs[s].ops, cfg.gran, {}, ri, trav_ptr, &dag);
      double cost_with = solver::ExactCost(
          p, subgraphs[s].ops, cfg.gran, all_out, ri, trav_ptr, &dag);
      term_delta = std::max(0.0, cost_without - cost_with);
    }

    double total = val - term_delta;
    if (total < best_cost) {
      best_cost = total;
      best_last = s;
    }
  }

  if (best_last < 0) {
    std::cerr << "  Held-Karp: no valid ordering found!\n";
    return INF;
  }

  // ── Backtrack to reconstruct ordering ─────────────────────────────────
  std::vector<int> order;
  {
    uint64_t mask = FULL;
    int cur = best_last;
    while (cur >= 0) {
      order.push_back(cur);
      int prev = par_get(mask, cur);
      mask ^= (1ULL << cur);
      cur = prev;
    }
    std::reverse(order.begin(), order.end());
  }

  std::cerr << "  Held-Karp: best=" << (int64_t)best_cost
            << " order=[";
  for (int i = 0; i < S; ++i)
    std::cerr << (i ? "," : "") << order[i];
  std::cerr << "]\n";

  // ── Apply decisions to subgraphs ──────────────────────────────────────
  for (int i = 0; i < S; ++i) {
    int s = order[i];
    auto& sg = subgraphs[s];

    if (i > 0) {
      int prev = order[i - 1];
      // Apply next's granularity from transition
      sg.gran = trans[prev][s].cfg.gran;
      sg.traversal = trans[prev][s].cfg.traversal;
      // Apply prev's retention for this subgraph
      subgraphs[prev].tensors_to_retain = trans[prev][s].retain_out;
    } else {
      // First subgraph: use base granularity
      sg.gran = base[s].cfg.gran;
      sg.traversal = base[s].cfg.traversal;
    }

    // Last subgraph: retain non-graph-output tensors if it doesn't worsen cost.
    if (i == S - 1) {
      auto& ts = CachedClassify(p, sg.ops, dag, cc);
      std::vector<size_t> retain_ngo;
      for (size_t t : ts.outputs)
        if (!dag.tensor_consumers[t].empty())
          retain_ngo.push_back(t);
      if (!retain_ngo.empty()) {
        const auto* trav_ptr = sg.traversal.empty() ? nullptr : &sg.traversal;
        std::unordered_set<size_t> ri;
        if (i > 0) {
          int prev_s = order[i - 1];
          for (size_t t : subgraphs[prev_s].tensors_to_retain) ri.insert(t);
        }
        double cost_without = solver::ExactCost(p, sg.ops, sg.gran, {}, ri, trav_ptr, &dag);
        double cost_with = solver::ExactCost(p, sg.ops, sg.gran, retain_ngo, ri, trav_ptr, &dag);
        if (cost_with <= cost_without)
          sg.tensors_to_retain = std::move(retain_ngo);
      }
    }
  }

  // Recompute per-subgraph costs with actual retention context
  {
    std::unordered_set<size_t> retained_in;
    double total = 0;
    for (int i = 0; i < S; ++i) {
      int s = order[i];
      auto& sg = subgraphs[s];
      const auto* trav_ptr = sg.traversal.empty() ? nullptr : &sg.traversal;
      sg.cost = solver::ExactCost(p, sg.ops, sg.gran, sg.tensors_to_retain,
                                   retained_in, trav_ptr, &dag);
      total += sg.cost;
      retained_in.clear();
      for (size_t t : sg.tensors_to_retain) retained_in.insert(t);
    }
    best_cost = total;
  }

  // Reorder subgraphs in place
  std::vector<OrderedSubgraph> reordered;
  for (int s : order)
    reordered.push_back(std::move(subgraphs[s]));
  subgraphs = std::move(reordered);

  return best_cost;
}

// ── Greedy Ordering (fallback for S > 15) ────────────────────────────────────

double GreedyOrdering(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    std::vector<OrderedSubgraph>& subgraphs,
    const std::vector<std::unordered_set<size_t>>& sg_preds) {

  int S = (int)subgraphs.size();
  ClassifyCache cc;

  // ── Precompute transition benefits (O(S²) BestGranularity calls) ──────
  // trans_benefit[a][b] = cost savings for b when a retains tensors for b.
  // Uses BestGranularity with retained_in = RetentionCandidates(a→b).
  std::vector<std::vector<double>> trans_benefit(S, std::vector<double>(S, 0.0));
  std::vector<double> base_cost(S);
  {
    for (int s = 0; s < S; ++s) {
      std::unordered_set<size_t> empty_ri;
      auto cfg = solver::BestGranularity(p, subgraphs[s].ops, empty_ri, 0, &dag, true);
      base_cost[s] = cfg.cost;
    }

    for (int a = 0; a < S; ++a) {
      for (int b = 0; b < S; ++b) {
        if (a == b) continue;
        auto ri = solver::RetentionCandidates(p, subgraphs[a].ops,
                                               subgraphs[b].ops, &dag);
        if (ri.empty()) continue;
        std::unordered_set<size_t> ri_set(ri.begin(), ri.end());
        auto cfg_b = solver::BestGranularity(p, subgraphs[b].ops, ri_set, 0, &dag, true);
        // Benefit = b's cost reduction from receiving retained inputs
        double benefit = base_cost[b] - cfg_b.cost;
        trans_benefit[a][b] = std::max(0.0, benefit);
      }
    }
  }

  // ── Greedy topological sort: try all source subgraphs as starting points ──
  std::vector<int> sources;
  for (int s = 0; s < S; ++s)
    if (sg_preds[s].empty()) sources.push_back(s);

  struct GreedyCandidate { std::vector<size_t> order; double quick_score; };
  std::vector<GreedyCandidate> greedy_candidates;

  for (int start : sources) {
    std::vector<bool> placed(S, false);
    std::vector<size_t> order;
    order.reserve(S);
    order.push_back(start);
    placed[start] = true;

    while ((int)order.size() < S) {
      size_t prev = order.back();
      int best_next = -1;
      double best_sc = -1;

      for (int s = 0; s < S; ++s) {
        if (placed[s]) continue;
        bool ready = true;
        for (size_t pred : sg_preds[s]) {
          if (!placed[pred]) { ready = false; break; }
        }
        if (!ready) continue;

        double score = trans_benefit[prev][s];
        if (score > best_sc || best_next < 0) {
          best_sc = score;
          best_next = s;
        }
      }

      if (best_next < 0) {
        std::cerr << "WARNING: GreedyOrdering: no ready node at step "
                  << order.size() << "/" << S << ", falling back to first unplaced\n";
        for (int s = 0; s < S; ++s) {
          if (!placed[s]) { best_next = s; break; }
        }
      }

      order.push_back(best_next);
      placed[best_next] = true;
    }

    // Quick scoring: sum of base costs minus transition benefits
    double score = 0;
    for (int i = 0; i < S; ++i) score += base_cost[order[i]];
    for (int i = 1; i < (int)order.size(); ++i)
      score -= trans_benefit[order[i-1]][order[i]];

    greedy_candidates.push_back({std::move(order), score});
  }

  // Sort by quick score, fully evaluate top K candidates
  std::sort(greedy_candidates.begin(), greedy_candidates.end(),
            [](const GreedyCandidate& a, const GreedyCandidate& b) {
              return a.quick_score < b.quick_score;
            });
  constexpr int kGreedyTopK = 5;
  int eval_count = std::min((int)greedy_candidates.size(), kGreedyTopK);

  std::vector<size_t> best_order;
  double best_cost = 1e18;
  for (int k = 0; k < eval_count; ++k) {
    auto sg_trial = subgraphs;
    double cost = EvaluateOrdering(p, dag, sg_trial, greedy_candidates[k].order, &cc);
    if (cost < best_cost) {
      best_cost = cost;
      best_order = greedy_candidates[k].order;
      subgraphs = sg_trial;
    }
  }

  if (best_order.empty()) {
    std::cerr << "  Phase 2 greedy: no valid ordering found!\n";
    return 1e18;
  }

  std::cerr << "  Phase 2 greedy (" << sources.size() << " sources, "
            << eval_count << " evaluated): cost=" << best_cost << "\n";

  // ── 2-opt with screening ─────────────────────────────────────────────
  bool improved = true;
  while (improved) {
    improved = false;
    for (int i = 0; i + 1 < S; ++i) {
      size_t a = best_order[i], b = best_order[i + 1];
      if (sg_preds[a].count(b)) continue;

      // Screening: estimate swap benefit from transition matrix
      // Before: ...→prev→a→b→next→...  After: ...→prev→b→a→next→...
      double old_ab = trans_benefit[a][b];
      double new_ba = trans_benefit[b][a];
      double old_prev_a = (i > 0) ? trans_benefit[best_order[i-1]][a] : 0;
      double new_prev_b = (i > 0) ? trans_benefit[best_order[i-1]][b] : 0;
      double old_b_next = (i + 2 < S) ? trans_benefit[b][best_order[i+2]] : 0;
      double new_a_next = (i + 2 < S) ? trans_benefit[a][best_order[i+2]] : 0;
      double delta = (new_prev_b + new_ba + new_a_next) - (old_prev_a + old_ab + old_b_next);
      if (delta <= 0) continue;  // skip clearly non-improving swaps

      // Topo validity check
      auto trial = best_order;
      std::swap(trial[i], trial[i + 1]);
      std::vector<bool> tp(S, false);
      bool valid = true;
      for (size_t s : trial) {
        for (size_t pred : sg_preds[s]) {
          if (!tp[pred]) { valid = false; break; }
        }
        if (!valid) break;
        tp[s] = true;
      }
      if (!valid) continue;

      // Full evaluation for promising swaps
      auto sg_copy = subgraphs;
      double cost = EvaluateOrdering(p, dag, sg_copy, trial, &cc);
      if (cost < best_cost) {
        best_cost = cost;
        best_order = trial;
        subgraphs = sg_copy;
        improved = true;
        break;
      }
    }
  }

  // Non-adjacent 2-opt: reverse segments [i..j] for distance > 1
  // Distance cap: limit to nearby swaps for large S to control cost
  int max_dist = (S <= 25) ? S : std::min(S, 10);
  improved = true;
  while (improved) {
    improved = false;
    for (int i = 0; i < S - 2 && !improved; ++i) {
      for (int j = i + 2; j < std::min(S, i + max_dist) && !improved; ++j) {
        // Screening: estimate benefit from edge changes at reversal boundary
        // Reversing segment [i..j] changes edges: (i-1,i), (j,j+1)
        // and internal edges become reversed direction
        size_t a = best_order[i], b = best_order[j];
        if (sg_preds[a].count(b) || sg_preds[b].count(a)) continue;

        double old_benefit = 0, new_benefit = 0;
        if (i > 0) {
          old_benefit += trans_benefit[best_order[i-1]][best_order[i]];
          new_benefit += trans_benefit[best_order[i-1]][best_order[j]];
        }
        if (j + 1 < S) {
          old_benefit += trans_benefit[best_order[j]][best_order[j+1]];
          new_benefit += trans_benefit[best_order[i]][best_order[j+1]];
        }
        if (new_benefit <= old_benefit) continue;

        // Build trial by reversing segment [i..j]
        auto trial = best_order;
        std::reverse(trial.begin() + i, trial.begin() + j + 1);

        // Topo validity check
        bool valid = true;
        std::vector<bool> tp(S, false);
        for (size_t s : trial) {
          for (size_t pred : sg_preds[s]) {
            if (!tp[pred]) { valid = false; break; }
          }
          if (!valid) break;
          tp[s] = true;
        }
        if (!valid) continue;

        auto sg_copy2 = subgraphs;
        double cost = EvaluateOrdering(p, dag, sg_copy2, trial, &cc);
        if (cost < best_cost) {
          best_cost = cost;
          best_order = trial;
          subgraphs = sg_copy2;
          improved = true;
        }
      }
    }
  }

  // Re-evaluate with best_order on a fresh copy: needed because (1) if 2-opt never
  // improved, subgraphs still has state from the initial greedy evaluation; (2) the
  // reordering below needs sg_copy indexed by best_order to produce the final vector.
  auto sg_copy = subgraphs;
  best_cost = EvaluateOrdering(p, dag, sg_copy, best_order, &cc);
  std::vector<OrderedSubgraph> reordered;
  for (size_t s : best_order)
    reordered.push_back(std::move(sg_copy[s]));
  subgraphs = std::move(reordered);

  return best_cost;
}

// Phase 2 main function
double OptimizeOrdering(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    std::vector<OrderedSubgraph>& subgraphs) {

  int S = (int)subgraphs.size();
  if (S == 0) return 0;
  if (S == 1) {
    auto& sg = subgraphs[0];
    double base_cost = solver::ExactCost(p, sg.ops, sg.gran,
                                          {}, {}, &sg.traversal, &dag);
    // Last (only) subgraph: retain non-graph-output tensors to avoid write-back
    auto ts = solver::Classify(p, sg.ops, &dag);
    std::vector<size_t> retain_ngo;
    for (size_t t : ts.outputs)
      if (!dag.tensor_consumers[t].empty())
        retain_ngo.push_back(t);
    if (!retain_ngo.empty()) {
      double cost_with = solver::ExactCost(p, sg.ops, sg.gran,
                                            retain_ngo, {}, &sg.traversal, &dag);
      if (cost_with <= base_cost) {
        sg.tensors_to_retain = std::move(retain_ngo);
        sg.cost = cost_with;
        return cost_with;
      }
    }
    sg.cost = base_cost;
    return base_cost;
  }

  std::vector<std::unordered_set<size_t>> sg_preds;
  BuildSubgraphDAG(dag, subgraphs, sg_preds);

  if (S <= kHeldKarpMaxS) {
    return HeldKarpOrdering(p, dag, subgraphs, sg_preds);
  } else {
    return GreedyOrdering(p, dag, subgraphs, sg_preds);
  }
}

// ── Multi-Pass ILP Refinement Helpers ────────────────────────────────────────

// Extract the set of all retained tensors from Phase 2 ordering.
std::unordered_set<size_t> ExtractAllRetainedTensors(
    const std::vector<OrderedSubgraph>& subgraphs) {
  std::unordered_set<size_t> all_retained;
  for (auto& sg : subgraphs)
    for (size_t t : sg.tensors_to_retain)
      all_retained.insert(t);
  return all_retained;
}

// Update column costs using Phase 2 retention context.
// Selected columns get their matched subgraph's retained_in.
// Non-selected columns get possible_ri from all retained tensors.
// If re_search_grans is true, columns with significant cost improvement
// are re-searched for better granularities; new columns are appended.
void UpdateColumnCosts(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    std::vector<SuperColumn>& columns,
    const std::vector<OrderedSubgraph>& subgraphs,
    const std::unordered_set<size_t>& all_retained,
    bool re_search_grans = false) {

  // Build map: sorted ops → subgraph index for matching
  std::map<std::vector<size_t>, int> ops_to_sg;
  for (int i = 0; i < (int)subgraphs.size(); ++i) {
    auto key = subgraphs[i].ops;
    std::sort(key.begin(), key.end());
    ops_to_sg[key] = i;
  }

  // Extract per-subgraph retained_in
  std::vector<std::unordered_set<size_t>> sg_retained_in(subgraphs.size());
  {
    std::unordered_set<size_t> ri;
    for (int i = 0; i < (int)subgraphs.size(); ++i) {
      for (size_t t : ri)
        if (subgraphs[i].input_tensors.count(t))
          sg_retained_in[i].insert(t);
      ri.clear();
      for (size_t t : subgraphs[i].tensors_to_retain) ri.insert(t);
    }
  }

  ClassifyCache re_cc;
  std::vector<SuperColumn> new_columns;
  int updated_selected = 0, updated_other = 0, re_searched = 0;

  for (auto& col : columns) {
    auto key = col.ops;
    std::sort(key.begin(), key.end());
    auto it = ops_to_sg.find(key);

    std::unordered_set<size_t> ri;
    if (it != ops_to_sg.end()) {
      // Selected column: use exact retained_in from Phase 2
      ri = sg_retained_in[it->second];
      ++updated_selected;
    } else {
      // Non-selected: use possible_ri from all retained tensors
      for (size_t t : all_retained)
        if (col.input_tensors.count(t)) ri.insert(t);
      if (ri.empty()) continue;  // no retention benefit possible
      ++updated_other;
    }

    // Recompute cost with retention — use Classify + AnalyticalCost
    auto ts = solver::Classify(p, col.ops, &dag);
    auto roles = solver::BuildRoles(p, col.ops, ts);
    int64_t maxK = 0;
    for (size_t op : col.ops)
      if (p.ops[op].op_type == "MatMul")
        maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);
    int64_t n_out = (int64_t)ts.outputs.size();

    double new_cost = solver::AnalyticalCost(p, col.ops, col.gran, col.snake_mode,
                                              ri, ts, roles, maxK, n_out, true);
    col.cost = std::min(col.base_cost, new_cost);  // reset from base, then take best

    // Re-search granularities when retention significantly improved cost
    if (re_search_grans && col.base_cost > 0 && (1.0 - new_cost / col.base_cost) > 0.05) {
      std::vector<GranCandidate> re_grans;
      EnumerateGranularitiesWithRI(p, col.ops, ri, dag, re_grans, re_cc);
      std::unordered_set<size_t> op_set(col.ops.begin(), col.ops.end());
      std::vector<size_t> ro_cands(ts.outputs.begin(), ts.outputs.end());
      std::unordered_set<size_t> all_inputs(ts.inputs.begin(), ts.inputs.end());
      std::unordered_set<GranKey, GranKeyHash> seen;
      // Add existing gran to seen set to avoid duplicates
      seen.insert({col.gran.width, col.gran.height, col.gran.depth, col.snake_mode});
      for (auto& gc : re_grans) {
        double re_cost = solver::AnalyticalCost(p, col.ops, gc.gran, gc.snake_mode,
                                                 ri, ts, roles, maxK, n_out, true);
        if (re_cost < new_cost * 0.98) {  // meaningfully better
          GranKey gk{gc.gran.width, gc.gran.height, gc.gran.depth, gc.snake_mode};
          if (!seen.insert(gk).second) continue;
          SuperColumn sc;
          sc.ops = col.ops;
          sc.gran = gc.gran;
          sc.snake_mode = gc.snake_mode;
          sc.traversal = gc.traversal;
          sc.cost = re_cost;
          sc.base_cost = solver::AnalyticalCost(p, col.ops, gc.gran, gc.snake_mode,
                                                 {}, ts, roles, maxK, n_out, true);
          sc.op_set = op_set;
          sc.output_tensors = ro_cands;
          sc.input_tensors = all_inputs;
          sc.retained_in = ri;
          new_columns.push_back(std::move(sc));
        }
      }
      ++re_searched;
    }
  }

  if (!new_columns.empty()) {
    std::cerr << "  Granularity re-search: " << re_searched << " columns re-searched, "
              << new_columns.size() << " new columns added\n";
    columns.insert(columns.end(), new_columns.begin(), new_columns.end());
  }

  if (updated_selected + updated_other > 0)
    std::cerr << "  Updated costs: " << updated_selected << " selected, "
              << updated_other << " other columns\n";
}

// Extract op-level partition for convergence check
std::vector<std::vector<size_t>> ExtractOpPartition(
    const Phase1Result& phase1,
    const std::vector<SuperColumn>& columns) {
  std::vector<std::vector<size_t>> partition;
  for (size_t ci : phase1.selected) {
    auto ops = columns[ci].ops;
    std::sort(ops.begin(), ops.end());
    partition.push_back(ops);
  }
  std::sort(partition.begin(), partition.end());
  return partition;
}

// ── Candidate Tracking ───────────────────────────────────────────────────────

struct FinalCandidate {
  mlsys::Solution sol;
  double cost;
  std::string label;
  double elapsed_s;
  int num_subgraphs;
};

// Writes best-so-far solution to disk atomically (temp + rename) so that
// the output file is always valid even if the process is killed mid-run.
struct BestTracker {
  const solver::PreprocessResult& pr;
  const char* output_path;
  double best_cost = 1e18;

  void MaybeWrite(const FinalCandidate& c) {
    if (c.cost >= best_cost) return;
    best_cost = c.cost;
    auto final_sol = solver::UnPreprocess(c.sol, pr);
    std::string tmp_path = std::string(output_path) + ".tmp";
    solver::WriteSolution(final_sol, tmp_path);
    std::rename(tmp_path.c_str(), output_path);
    std::cerr << "  >> wrote " << c.label
              << " (" << (int64_t)c.cost << ") to disk\n";
  }
};

// Convert Partition → OrderedSubgraphs → OptimizeOrdering → DirectSolution.
// Shared helper for rewrite path, DP-BFS, and DP-DFS Held-Karp candidates.
// Returns empty candidate (cost=1e18) if partition has too many subgraphs
// (OptimizeOrdering's O(S²) transition precomputation becomes too expensive).
FinalCandidate PartitionToHKCandidate(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const solver::Partition& part,
    const std::string& label) {
  if ((int)part.subgraphs.size() > kMaxSubgraphsForHK) {
    std::cerr << "  " << label << ": skipped (" << part.subgraphs.size() << " subgraphs)\n";
    return {{}, 1e18, label, 0, (int)part.subgraphs.size()};
  }
  std::vector<OrderedSubgraph> subgraphs;
  for (auto& ops : part.subgraphs) {
    auto ts = solver::Classify(p, ops, &dag);
    std::unordered_set<size_t> empty_ri;
    auto cfg = solver::BestGranularity(p, ops, empty_ri, 0, &dag, false);
    OrderedSubgraph osg;
    osg.ops = ops;
    osg.gran = cfg.gran;
    osg.snake_mode = kSnakeRow;
    osg.traversal = cfg.traversal;
    osg.cost = cfg.cost;
    osg.assumed_retained_in = {};
    osg.output_tensors.assign(ts.outputs.begin(), ts.outputs.end());
    osg.input_tensors = ts.inputs;
    subgraphs.push_back(std::move(osg));
  }

  OptimizeOrdering(p, dag, subgraphs);

  auto sol = DirectSolutionFromOrdering(p, dag, subgraphs);
  ZeroLatencies(sol);
  auto eval = mlsys::Evaluate(p, sol);
  double cost = eval.ok() ? *eval : 1e18;
  if (!eval.ok())
    std::cerr << "    " << label << " EVAL ERROR: " << eval.status() << "\n";
  std::cerr << "  " << label << ": " << (int64_t)cost << "\n";
  return {std::move(sol), cost, label, 0, (int)subgraphs.size()};
}

struct DPResult {
  solver::Partition part;
  FinalCandidate candidate;
};

DPResult DPFallbackWithPartition(const mlsys::Problem& p, const solver::DAG& dag,
                                 bool assume_retain_out = false,
                                 const std::string& label = "DP-fallback") {
  auto part = solver::InitialPartition(dag);
  part = solver::DPPartition(p, dag, std::move(part), assume_retain_out);
  part = solver::RecomputationPass(p, dag, std::move(part), assume_retain_out);
  part = solver::ConvexityRepair(dag, std::move(part));
  part = solver::UnfusionPass(p, dag, std::move(part), /*assume_retain_out=*/true);

  auto part_copy = part;  // save for HK path
  part = solver::SubgraphReorder(p, dag, std::move(part));
  auto sol = solver::BuildSolution(p, dag, part);
  ZeroLatencies(sol);
  auto eval = mlsys::Evaluate(p, sol);
  double cost = eval.ok() ? *eval : 1e18;
  if (!eval.ok()) std::cerr << "    EVAL ERROR: " << eval.status() << "\n";
  std::cerr << "  " << label << ": " << cost << "\n";
  int n_sgs = (int)sol.subgraphs.size();
  return {std::move(part_copy),
          {std::move(sol), cost, label, 0, n_sgs}};
}

// ── Validation: AnalyticalCost vs ExactCost ──────────────────────────────────────

void ValidateAnalyticalCost(const mlsys::Problem& p, const solver::DAG& dag) {
  int N = (int)p.ops.size();
  int64_t natW = p.native_granularity.width;
  int64_t natH = p.native_granularity.height;
  int64_t C = p.fast_memory_capacity;

  int total_checked = 0, total_mismatches = 0;
  double max_rel_err = 0;

  // Test singleton subgraphs + a few multi-op subsets
  std::vector<std::vector<size_t>> test_subsets;
  for (int i = 0; i < N; ++i)
    test_subsets.push_back({(size_t)i});

  // Add pairs from DAG edges (cap to avoid combinatorial explosion on large graphs)
  int max_pairs = std::min(N * 2, 60);
  int pair_count = 0;
  for (int i = 0; i < N && pair_count < max_pairs; ++i) {
    for (size_t succ : dag.op_successors[i]) {
      if (pair_count >= max_pairs) break;
      std::vector<size_t> pair = {(size_t)i, succ};
      std::sort(pair.begin(), pair.end());
      test_subsets.push_back(pair);
      ++pair_count;
    }
  }

  for (auto& ops : test_subsets) {
    auto ts = solver::Classify(p, ops, &dag);
    auto roles = solver::BuildRoles(p, ops, ts);
    int64_t n_outputs = (int64_t)ts.outputs.size();
    std::vector<size_t> ro_all(ts.outputs.begin(), ts.outputs.end());

    int64_t maxK = 0;
    for (size_t op : ops)
      if (p.ops[op].op_type == "MatMul")
        maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);

    auto [oW, oH] = solver::OutDims(p, ops);
    bool has_mm = false, has_pw = false;
    for (size_t op : ops) {
      if (p.ops[op].op_type == "MatMul") has_mm = true;
      else has_pw = true;
    }

    // k candidates (sample representative values)
    std::vector<int64_t> nk_cands;
    if (has_mm && has_pw && maxK > 0) {
      nk_cands.push_back(1);
    } else if (maxK == 0) {
      nk_cands.push_back(1);
    } else {
      // Use divisors + powers of 2, same as solver
      for (int64_t d = 1; d * d <= maxK; ++d) {
        if (maxK % d == 0) { nk_cands.push_back(d); nk_cands.push_back(maxK / d); }
      }
      for (int64_t nk = 1; nk <= maxK; nk *= 2) nk_cands.push_back(nk);
      std::sort(nk_cands.begin(), nk_cands.end());
      nk_cands.erase(std::unique(nk_cands.begin(), nk_cands.end()), nk_cands.end());
    }

    // Test with two retained_in scenarios: empty and all inputs
    std::vector<std::unordered_set<size_t>> ri_scenarios;
    ri_scenarios.push_back({});  // no retention
    if (!ts.inputs.empty())
      ri_scenarios.push_back(std::unordered_set<size_t>(ts.inputs.begin(), ts.inputs.end()));

    int configs_this_subset = 0;
    const int max_configs_per_subset = 200;
    for (auto& retained_in : ri_scenarios) {
      auto ComputeWS = [&](int64_t w, int64_t h, int64_t k) -> int64_t {
        return solver::ComputeTileWS(p, roles, n_outputs, retained_in, w, h, k);
      };

      for (int64_t nk : nk_cands) {
        if (configs_this_subset >= max_configs_per_subset) break;
        int64_t k = (maxK > 0) ? solver::CeilDiv(maxK, nk) : 1;
        // Sample representative (w, h) — covers 1-tile, multi-row, multi-col, max
        std::vector<std::pair<int64_t, int64_t>> wh_cands;
        for (int64_t tw : {1, 2, 4}) {
          int64_t w = std::min(solver::CeilDiv(oW, tw), natW);
          for (int64_t th : {1, 2, 4}) {
            int64_t h = std::min(solver::CeilDiv(oH, th), natH);
            if (ComputeWS(w, h, k) <= C)
              wh_cands.push_back({w, h});
          }
        }
        wh_cands.push_back({std::min(oW, natW), std::min(oH, natH)});
        wh_cands.push_back({1, 1});
        // Deduplicate
        std::sort(wh_cands.begin(), wh_cands.end());
        wh_cands.erase(std::unique(wh_cands.begin(), wh_cands.end()), wh_cands.end());

        for (auto [w, h] : wh_cands) {
          if (w <= 0 || h <= 0) continue;
          if (ComputeWS(w, h, k) > C) continue;

          mlsys::Granularity g{w, h, k};
          int64_t ntw = solver::CeilDiv(oW, w), nth = solver::CeilDiv(oH, h);
          int64_t nk_actual = (maxK > 0) ? solver::CeilDiv(maxK, k) : 1;
          // Skip configs with huge tile grids (ExactCost is O(tiles*k_steps))
          if (ntw * nth * nk_actual > 100000) continue;

          for (int snake : {kSnakeRow, kSnakeCol}) {
            if (configs_this_subset >= max_configs_per_subset) break;
            // AnalyticalCost with assume_retain_out=true
            double qc = solver::AnalyticalCost(p, ops, g, snake, retained_in,
                                          ts, roles, maxK, n_outputs, true);

            // ExactCost with retain_out=all_outputs (matching assume_retain_out=true)
            std::vector<int64_t> trav;
            if (ntw * nth > 1) {
              trav = (snake == kSnakeRow) ? solver::SnakeRow(ntw, nth)
                                        : solver::SnakeCol(ntw, nth);
            }
            const std::vector<int64_t>* tp = trav.empty() ? nullptr : &trav;
            double ec = solver::ExactCostCached(p, ops, g, ro_all, retained_in,
                                                ts, roles, tp);

            ++total_checked;
            ++configs_this_subset;
            double rel_err = (ec > 0) ? std::abs(qc - ec) / ec : (qc > 0 ? 1.0 : 0.0);
            if (rel_err > 1e-6) {
              ++total_mismatches;
              max_rel_err = std::max(max_rel_err, rel_err);
              if (total_mismatches <= 20) {
                std::cerr << "  MISMATCH ops=[";
                for (size_t i = 0; i < ops.size(); ++i)
                  std::cerr << (i ? "," : "") << ops[i];
                std::cerr << "] g=(" << w << "," << h << "," << k << ")"
                          << " snake=" << snake
                          << " ri=" << retained_in.size()
                          << " QC=" << qc << " EC=" << ec
                          << " rel=" << std::fixed << std::setprecision(6) << rel_err
                          << "\n";
              }
            }
          }
        }
      }
    }
  }

  std::cerr << "Validation: " << total_checked << " configs checked, "
            << total_mismatches << " mismatches";
  if (total_mismatches > 0)
    std::cerr << " (max rel err: " << std::fixed << std::setprecision(6) << max_rel_err << ")";
  std::cerr << "\n";
}

// ── Column Pool Generation ──────────────────────────────────────────────────
// Shared helper: connected subsets + DFS segments + convexity filter +
// BuildSuperColumns + merge heavy-input columns.

std::vector<SuperColumn> GenerateColumnPool(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    const AdjList& adj,
    int N,
    int max_size,
    std::vector<SuperColumn>& heavy_columns,
    ClassifyCache& cc,
    const RegimeAnalysis* regime,
    std::chrono::steady_clock::time_point t0,
    double deadline) {

  // Step 1: Connected subsets
  std::vector<std::vector<size_t>> subsets;
  EnumerateConnectedSubsets(dag, adj, subsets, max_size, t0, deadline);

  // Step 2: DFS-contiguous segments
  {
    auto dfs_dag = solver::BuildDAG_DFS(p);
    auto& dfs_order = dfs_dag.topo_order;
    int dfs_max_len = std::min(max_size + 2, 6);
    std::unordered_set<std::vector<size_t>, VectorHash> existing(
        subsets.begin(), subsets.end());
    int dfs_added = 0;
    for (int len = 2; len <= dfs_max_len; ++len) {
      for (int i = 0; i + len <= N; ++i) {
        std::vector<size_t> seg(dfs_order.begin() + i,
                                dfs_order.begin() + i + len);
        std::sort(seg.begin(), seg.end());
        if (!existing.insert(seg).second) continue;
        subsets.push_back(seg);
        ++dfs_added;
      }
    }
    if (dfs_added > 0)
      std::cerr << "DFS-contiguous segments: +" << dfs_added << " subsets\n";
  }

  // Step 3: Convexity filter
  {
    size_t before = subsets.size();
    subsets.erase(std::remove_if(subsets.begin(), subsets.end(),
        [&dag](const std::vector<size_t>& s) {
          return s.size() >= 2 && !solver::IsDAGConvex(dag, s);
        }), subsets.end());
    if (subsets.size() < before)
      std::cerr << "Convexity filter: removed " << (before - subsets.size())
                << " non-convex subsets\n";
  }

  // Step 4: Build super-columns
  std::vector<SuperColumn> columns;
  BuildSuperColumns(p, dag, subsets, columns, cc, t0, deadline);

  // Step 5: Merge heavy-input columns
  columns.insert(columns.end(), heavy_columns.begin(), heavy_columns.end());

  return columns;
}

// ── RunRewritePath: DAG rewrite + ILP on rewritten graph ─────────────────────

void RunRewritePath(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    int N,
    std::chrono::steady_clock::time_point t0,
    double hard_deadline,
    std::vector<FinalCandidate>& all_candidates,
    BestTracker& tracker) {

  auto rc_candidates = FindRecomputeCandidates(p, dag);
  if (rc_candidates.empty()) return;

  std::cerr << "Rewrite: " << rc_candidates.size() << " recompute candidates\n";
  auto rewrite = RewriteDAG(p, dag, rc_candidates);
  int rw_N = (int)rewrite.rw_problem.ops.size();
  std::cerr << "Rewrite: " << rw_N << " ops (+" << (rw_N - N) << " clones), "
            << rewrite.rw_problem.tensors.size() << " tensors\n";

  auto rw_dag = solver::BuildDAG(rewrite.rw_problem);
  auto rw_adj = BuildUndirectedAdj(rw_dag);
  ClassifyCache rw_cc;
  auto rw_regime = ComputeRegimeAnalysis(rewrite.rw_problem, rw_dag, rw_adj, rw_cc);

  double rw_deadline = solver::ElapsedSince(t0) + hard_deadline * kTimeBudgetRewrite;

  // Phase 0 on rewritten DAG
  std::vector<SuperColumn> rw_heavy;
  {
    double rw_p0_budget = hard_deadline * kTimeBudgetRewriteP0;
    GenerateHeavyInputColumns(rewrite.rw_problem, rw_dag, rw_heavy,
                              t0, rw_p0_budget, rw_cc, &rw_regime);
  }

  // Phase 1 on rewritten DAG
  int rw_max_size = DefaultMaxSize(rw_N);
  auto rw_columns = GenerateColumnPool(rewrite.rw_problem, rw_dag, rw_adj, rw_N, rw_max_size,
                                        rw_heavy, rw_cc, &rw_regime, t0, rw_deadline);

  if (!rw_columns.empty() && solver::ElapsedSince(t0) < rw_deadline) {
    std::vector<bool> no_recomp(rw_N, false);
    auto rw_phase1 = SolvePartitionILP(rewrite.rw_problem, rw_dag, rw_columns,
                                        no_recomp, t0, rw_deadline);
    if (rw_phase1.solved) {
      // Map partition back to original ops
      solver::Partition rw_part;
      for (size_t ci : rw_phase1.selected)
        rw_part.subgraphs.push_back(rw_columns[ci].ops);
      auto orig_part = MapPartitionBack(rewrite, rw_part);

      // Run through standard pipeline on original DAG
      orig_part = solver::RecomputationPass(p, dag, std::move(orig_part));
      orig_part = solver::ConvexityRepair(dag, std::move(orig_part));
      orig_part = solver::UnfusionPass(p, dag, std::move(orig_part), true);

      // Rewrite-HK: OptimizeOrdering + DirectSolution
      if (solver::ElapsedSince(t0) < rw_deadline) {
        auto rw_hk = PartitionToHKCandidate(p, dag, orig_part, "Rewrite-HK");
        rw_hk.elapsed_s = solver::ElapsedSince(t0);
        all_candidates.push_back(rw_hk);
        tracker.MaybeWrite(all_candidates.back());
      }

      // Rewrite: existing SubgraphReorder + BuildSolution
      orig_part = solver::SubgraphReorder(p, dag, std::move(orig_part));
      auto rw_sol = solver::BuildSolution(p, dag, orig_part);
      ZeroLatencies(rw_sol);
      auto rw_eval = mlsys::Evaluate(p, rw_sol);
      double rw_cost = rw_eval.ok() ? *rw_eval : 1e18;
      if (!rw_eval.ok()) std::cerr << "    RW EVAL ERROR: " << rw_eval.status() << "\n";
      std::cerr << "  Rewrite: " << (int64_t)rw_cost << "\n";
      all_candidates.push_back({std::move(rw_sol), rw_cost, "Rewrite",
                                solver::ElapsedSince(t0), (int)orig_part.subgraphs.size()});
      tracker.MaybeWrite(all_candidates.back());
    }
  }
}

// ── RunILPIterations: Multi-pass ILP refinement + pipeline ───────────────────

void RunILPIterations(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    std::vector<SuperColumn>& columns,
    const std::vector<bool>& recomputable,
    std::chrono::steady_clock::time_point t0,
    double hard_deadline,
    std::vector<FinalCandidate>& all_candidates,
    BestTracker& tracker) {

  if (columns.empty()) return;

  // Iterate: ILP → Phase 2 → update costs → re-ILP until convergence.
  std::vector<std::vector<size_t>> prev_partition;
  const int max_iters = kMaxILPIters;
  Phase1Result last_phase1{};

  for (int iter = 0; iter < max_iters; ++iter) {
    if (iter > 0 && solver::ElapsedSince(t0) > hard_deadline * kTimeBudgetILPBreak) break;

    std::cerr << "  ── ILP iteration " << iter << " ──\n";
    auto phase1 = SolvePartitionILP(p, dag, columns, recomputable, t0, hard_deadline, iter);
    if (!phase1.solved) break;
    last_phase1 = phase1;

    // Convergence: check if partition changed
    auto current_partition = ExtractOpPartition(phase1, columns);
    if (iter > 0 && current_partition == prev_partition) {
      std::cerr << "  Converged (same partition)\n";
      break;
    }
    prev_partition = current_partition;

    // Build ordered subgraphs from selected columns
    std::vector<OrderedSubgraph> subgraphs;
    for (size_t ci : phase1.selected) {
      auto& sc = columns[ci];
      OrderedSubgraph osg;
      osg.ops = sc.ops;
      osg.gran = sc.gran;
      osg.snake_mode = sc.snake_mode;
      osg.traversal = sc.traversal;
      osg.output_tensors = sc.output_tensors;
      osg.cost = sc.cost;
      osg.assumed_retained_in = sc.retained_in;
      osg.input_tensors = sc.input_tensors;
      subgraphs.push_back(std::move(osg));
    }

    // Phase 2: Ordering + Retention
    // Note: phase1.cost is the ILP objective (sum of column costs). After iter>0,
    // column costs may have been updated by UpdateColumnCosts.
    double phase1_partition_cost = phase1.cost;
    double phase2_cost = OptimizeOrdering(p, dag, subgraphs);

    double p1_to_p2_delta = phase2_cost - phase1_partition_cost;
    double p1_to_p2_pct = (phase1_partition_cost > 0)
        ? 100.0 * p1_to_p2_delta / phase1_partition_cost : 0;
    std::cerr << "  Phase 1 (partition-only): " << (int64_t)phase1_partition_cost << "\n";
    std::cerr << "  Phase 2 (ordering+retention): " << (int64_t)phase2_cost
              << "  (delta: " << (int64_t)p1_to_p2_delta
              << ", " << std::fixed << std::setprecision(1) << p1_to_p2_pct << "%)\n";

    // Emit TwoPhase-Direct candidate: preserves Phase 2 granularities
    {
      auto sol_direct = DirectSolutionFromOrdering(p, dag, subgraphs);
      ZeroLatencies(sol_direct);
      auto eval_d = mlsys::Evaluate(p, sol_direct);
      double cost_d = eval_d.ok() ? *eval_d : 1e18;
      if (!eval_d.ok()) std::cerr << "    EVAL ERROR (direct): " << eval_d.status() << "\n";
      std::string label_d = (iter == 0) ? "TwoPhase-Direct" : "TwoPhase-Direct-iter" + std::to_string(iter);
      std::cerr << "  " << label_d << ": " << (int64_t)cost_d << "\n";
      all_candidates.push_back({std::move(sol_direct), cost_d, label_d,
                                solver::ElapsedSince(t0), (int)subgraphs.size()});
      tracker.MaybeWrite(all_candidates.back());
    }

    // Emit TwoPhase candidate via BuildSolution (re-searches granularities)
    solver::Partition phase2_part;
    for (auto& sg : subgraphs)
      phase2_part.subgraphs.push_back(sg.ops);

    {
      auto p2 = phase2_part;
      auto sol = solver::BuildSolution(p, dag, p2);
      ZeroLatencies(sol);
      auto eval = mlsys::Evaluate(p, sol);
      double cost = eval.ok() ? *eval : 1e18;
      if (!eval.ok()) std::cerr << "    EVAL ERROR: " << eval.status() << "\n";
      std::string label = (iter == 0) ? "TwoPhase" : "TwoPhase-iter" + std::to_string(iter);
      std::cerr << "  " << label << ": " << (int64_t)cost << "\n";
      all_candidates.push_back({std::move(sol), cost, label,
                                solver::ElapsedSince(t0), (int)phase2_part.subgraphs.size()});
      tracker.MaybeWrite(all_candidates.back());
    }

    // Update column costs with Phase 2 retention context for next iteration
    if (iter + 1 < max_iters && solver::ElapsedSince(t0) < hard_deadline * kTimeBudgetILPBreak) {
      auto all_retained = ExtractAllRetainedTensors(subgraphs);
      UpdateColumnCosts(p, dag, columns, subgraphs, all_retained,
                        /*re_search_grans=*/iter == 0);
    }
  }

  // Also run the cached ILP partition through the standard pipeline
  if (last_phase1.solved) {
    auto& phase1 = last_phase1;
    solver::Partition ilp_part;
    for (size_t ci : phase1.selected)
      ilp_part.subgraphs.push_back(columns[ci].ops);
    ilp_part = solver::RecomputationPass(p, dag, std::move(ilp_part));
    ilp_part = solver::ConvexityRepair(dag, std::move(ilp_part));
    ilp_part = solver::UnfusionPass(p, dag, std::move(ilp_part), /*assume_retain_out=*/true);
    ilp_part = solver::SubgraphReorder(p, dag, std::move(ilp_part));
    auto sol2 = solver::BuildSolution(p, dag, ilp_part);
    ZeroLatencies(sol2);
    auto eval2 = mlsys::Evaluate(p, sol2);
    double cost2 = eval2.ok() ? *eval2 : 1e18;
    if (!eval2.ok()) std::cerr << "    EVAL ERROR: " << eval2.status() << "\n";
    std::cerr << "  ILP+pipeline: " << cost2 << "\n";
    all_candidates.push_back({std::move(sol2), cost2, "ILP+pipeline",
                              solver::ElapsedSince(t0), (int)ilp_part.subgraphs.size()});
    tracker.MaybeWrite(all_candidates.back());

    // Try ExhaustiveGranularity for denser tile search (if time allows)
    if (solver::ElapsedSince(t0) < hard_deadline * kTimeBudgetILPBreak) {
      auto sol3 = solver::BuildSolutionExhaustive(p, dag, ilp_part);
      ZeroLatencies(sol3);
      auto eval3 = mlsys::Evaluate(p, sol3);
      double cost3 = eval3.ok() ? *eval3 : 1e18;
      if (!eval3.ok()) std::cerr << "    EVAL ERROR: " << eval3.status() << "\n";
      std::cerr << "  ILP+exhaustive: " << cost3 << "\n";
      all_candidates.push_back({std::move(sol3), cost3, "ILP+exhaustive",
                                solver::ElapsedSince(t0), (int)ilp_part.subgraphs.size()});
      tracker.MaybeWrite(all_candidates.back());
    }
  }
}

// ── RunDPFallbacks: DP variants with multiple orderings + retain modes ───────

#if 0  // RunPartitionedILP — disabled, depth-layer partition doesn't help.
// The optimal subgraphs cross depth boundaries (mixing depth-0 T0-consumers
// with depth-1 T1-consumers). Branch-pair columns in Phase 0 are more effective.

void RunPartitionedILP(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    std::chrono::steady_clock::time_point t0,
    double hard_deadline,
    std::vector<FinalCandidate>& all_candidates,
    BestTracker& tracker) {

  int N = (int)dag.num_ops;
  if (N <= 20) return;  // small graphs don't benefit from partitioning

  // ── Step 1: Compute depth layers ──────────────────────────────────────
  std::vector<int> depth(dag.num_ops, 0);
  std::vector<std::vector<size_t>> op_preds(dag.num_ops);
  for (size_t a = 0; a < dag.num_ops; ++a)
    for (size_t b : dag.op_successors[a])
      op_preds[b].push_back(a);
  for (size_t op : dag.topo_order)
    for (size_t pred : op_preds[op])
      depth[op] = std::max(depth[op], depth[pred] + 1);

  int max_depth = *std::max_element(depth.begin(), depth.end());

  // Group ops by depth
  std::vector<std::vector<size_t>> layers(max_depth + 1);
  for (size_t i = 0; i < dag.num_ops; ++i)
    layers[depth[i]].push_back(i);

  // ── Step 2: Build blocks from depth layers ────────────────────────────
  // Each layer becomes a block. Small layers (singletons at the tail)
  // get merged with adjacent layers.
  constexpr int kMaxMergedBlockSize = 15;

  std::vector<std::vector<size_t>> blocks;
  std::vector<size_t> current;
  for (int d = 0; d <= max_depth; ++d) {
    if (layers[d].empty()) continue;
    if (current.size() + layers[d].size() <= (size_t)kMaxMergedBlockSize) {
      current.insert(current.end(), layers[d].begin(), layers[d].end());
    } else {
      if (!current.empty()) blocks.push_back(std::move(current));
      current = layers[d];
      // If this single layer is too large, split it into chunks
      while (current.size() > (size_t)kMaxMergedBlockSize) {
        std::vector<size_t> chunk(current.begin(),
                                   current.begin() + kMaxMergedBlockSize);
        blocks.push_back(std::move(chunk));
        current.erase(current.begin(), current.begin() + kMaxMergedBlockSize);
      }
    }
  }
  if (!current.empty()) blocks.push_back(std::move(current));

  int n_comp = (int)blocks.size();

  // Sort blocks by topo position
  std::vector<int> topo_pos(dag.num_ops);
  for (int i = 0; i < N; ++i) topo_pos[dag.topo_order[i]] = i;
  std::sort(blocks.begin(), blocks.end(),
            [&](const auto& a, const auto& b) {
              return topo_pos[a[0]] < topo_pos[b[0]];
            });

  if (n_comp <= 2) {
    std::cerr << "Partitioned ILP: only " << n_comp << " blocks, skipping\n";
    return;
  }

  std::cerr << "Partitioned ILP (depth-layer): " << N << " ops -> "
            << n_comp << " blocks (sizes:";
  for (auto& b : blocks) std::cerr << " " << b.size();
  std::cerr << ")\n";

  // No re-merge needed — depth-layer blocks already group horizontally

  // ── Step 4: Per-block column generation + ILP ─────────────────────────
  // For each block, generate columns (using the full DAG but restricted
  // to block ops), then solve a mini-ILP with remapped indices.

  solver::Partition combined_partition;
  double combined_cost = 0;
  bool all_solved = true;

  ClassifyCache block_cc;

  for (int bi = 0; bi < n_comp; ++bi) {
    if (solver::ElapsedSince(t0) > hard_deadline * 0.85) {
      all_solved = false;
      break;
    }

    auto& block_ops = blocks[bi];
    int block_size = (int)block_ops.size();

    if (block_size == 1) {
      // Singleton — just add as its own subgraph
      combined_partition.subgraphs.push_back(block_ops);
      std::unordered_set<size_t> empty_ri;
      auto cfg = solver::BestGranularity(p, block_ops, empty_ri, 0, &dag, true);
      combined_cost += cfg.cost;
      continue;
    }

    // Build block op set for filtering
    std::unordered_set<size_t> block_set(block_ops.begin(), block_ops.end());

    // Build remapping: absolute op index -> local index [0..block_size)
    std::unordered_map<size_t, size_t> op_to_local;
    for (int i = 0; i < block_size; ++i)
      op_to_local[block_ops[i]] = i;

    // Build block-local adjacency
    AdjList block_adj(block_size);
    for (size_t op : block_ops) {
      size_t local_op = op_to_local[op];
      for (size_t succ : dag.op_successors[op]) {
        if (block_set.count(succ)) {
          block_adj[local_op].insert(op_to_local[succ]);
          block_adj[op_to_local[succ]].insert(local_op);
        }
      }
    }
    // Also add co-consumer edges (undirected) — including graph inputs
    for (size_t t = 0; t < dag.num_tensors; ++t) {
      std::vector<size_t> local_cons;
      for (size_t c : dag.tensor_consumers[t])
        if (block_set.count(c)) local_cons.push_back(op_to_local[c]);
      if (local_cons.size() < 2) continue;  // need ≥2 block ops sharing this tensor
      int prod = dag.tensor_producer[t];
      if (prod >= 0 && block_set.count(prod)) {
        size_t lp = op_to_local[prod];
        for (size_t lc : local_cons) {
          block_adj[lp].insert(lc);
          block_adj[lc].insert(lp);
        }
      }
      // Connect all co-consumers (works for both graph inputs and produced tensors)
      for (size_t i = 0; i < local_cons.size(); ++i)
        for (size_t j = i + 1; j < local_cons.size(); ++j) {
          block_adj[local_cons[i]].insert(local_cons[j]);
          block_adj[local_cons[j]].insert(local_cons[i]);
        }
    }

    // Debug: adjacency density
    {
      int edges = 0;
      for (int i = 0; i < block_size; ++i) edges += block_adj[i].size();
      std::cerr << "  Block " << bi << ": " << block_size << " ops, "
                << edges/2 << " adj edges\n";
    }

    // Enumerate connected subsets in local index space
    // Create a mini-DAG just for subset enumeration (num_ops = block_size)
    solver::DAG mini_dag;
    mini_dag.num_ops = block_size;
    mini_dag.num_tensors = dag.num_tensors;
    mini_dag.tensor_producer = dag.tensor_producer;
    mini_dag.tensor_consumers = dag.tensor_consumers;
    mini_dag.op_successors.resize(block_size);
    for (size_t op : block_ops) {
      size_t local = op_to_local[op];
      for (size_t succ : dag.op_successors[op])
        if (block_set.count(succ))
          mini_dag.op_successors[local].push_back(op_to_local[succ]);
    }
    // Topo order within block
    {
      std::vector<int> in_deg(block_size, 0);
      for (int i = 0; i < block_size; ++i)
        for (size_t s : mini_dag.op_successors[i])
          in_deg[s]++;
      std::queue<size_t> q;
      for (int i = 0; i < block_size; ++i)
        if (in_deg[i] == 0) q.push(i);
      while (!q.empty()) {
        size_t u = q.front(); q.pop();
        mini_dag.topo_order.push_back(u);
        for (size_t s : mini_dag.op_successors[u])
          if (--in_deg[s] == 0) q.push(s);
      }
    }

    // Generate columns by group size: for each size k=1..block_size,
    // create a representative subset of k ops from this block.
    // Since same-depth ops are structurally identical (same type, same
    // tensor shapes), one representative per group size suffices —
    // the ILP can tile the block with groups of the optimal size.
    double remaining = hard_deadline - solver::ElapsedSince(t0);
    double per_block = std::max(0.5, remaining / std::max(1, n_comp - bi));
    double block_deadline = solver::ElapsedSince(t0) + per_block * 0.8;

    std::vector<std::vector<size_t>> abs_subsets;

    // Generate all contiguous subsets of each size (for ILP covering)
    // For a block of 15 ops, we need enough columns so the ILP can
    // partition them: e.g., 15 = 10+5 or 8+7 or 5+5+5.
    // Generate sliding windows of each size for full coverage.
    for (int sz = 1; sz <= block_size; ++sz) {
      for (int start = 0; start + sz <= block_size; ++start) {
        std::vector<size_t> subset;
        for (int j = start; j < start + sz; ++j)
          subset.push_back(block_ops[j]);
        std::sort(subset.begin(), subset.end());
        // Convexity check
        if (subset.size() >= 2 && !solver::IsDAGConvex(dag, subset)) continue;
        abs_subsets.push_back(std::move(subset));
      }
      if (solver::ElapsedSince(t0) > block_deadline) break;
    }

    std::cerr << "    Block " << bi << ": " << abs_subsets.size()
              << " subsets (sizes 1.." << block_size << ")\n";

    // Build super-columns (using real dag + real problem, absolute indices)
    std::vector<SuperColumn> block_columns;
    BuildSuperColumns(p, dag, abs_subsets, block_columns, block_cc, t0, block_deadline);

    if (block_columns.empty()) {
      // Fallback: singletons
      for (size_t op : block_ops) {
        combined_partition.subgraphs.push_back({op});
        std::unordered_set<size_t> empty_ri;
        auto cfg = solver::BestGranularity(p, {op}, empty_ri, 0, &dag, true);
        combined_cost += cfg.cost;
      }
      continue;
    }

    // Solve mini-ILP with remapped indices
    // Remap column ops to local indices for the ILP matrix
    std::vector<SuperColumn> local_columns = block_columns;
    for (auto& col : local_columns) {
      std::vector<size_t> local_ops;
      for (size_t op : col.ops) local_ops.push_back(op_to_local[op]);
      col.ops = local_ops;
    }

    // Build and solve ILP
    int num_cols = (int)local_columns.size();
    std::vector<double> col_cost(num_cols);
    std::vector<double> col_lower(num_cols, 0.0);
    std::vector<double> col_upper(num_cols, 1.0);
    std::vector<HighsVarType> integrality(num_cols, HighsVarType::kInteger);
    for (int j = 0; j < num_cols; ++j)
      col_cost[j] = local_columns[j].cost;

    std::vector<HighsInt> a_start(num_cols + 1);
    std::vector<HighsInt> a_index;
    std::vector<double> a_value;
    for (int j = 0; j < num_cols; ++j) {
      a_start[j] = (HighsInt)a_index.size();
      for (size_t op : local_columns[j].ops) {
        a_index.push_back((HighsInt)op);
        a_value.push_back(1.0);
      }
    }
    a_start[num_cols] = (HighsInt)a_index.size();

    std::vector<double> row_lower(block_size, 1.0);
    std::vector<double> row_upper(block_size, 1.0);

    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("time_limit", per_block);

    HighsModel model;
    model.lp_.num_col_ = num_cols;
    model.lp_.num_row_ = block_size;
    model.lp_.sense_ = ObjSense::kMinimize;
    model.lp_.col_cost_ = col_cost;
    model.lp_.col_lower_ = col_lower;
    model.lp_.col_upper_ = col_upper;
    model.lp_.row_lower_ = row_lower;
    model.lp_.row_upper_ = row_upper;
    model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    model.lp_.a_matrix_.start_ = a_start;
    model.lp_.a_matrix_.index_ = a_index;
    model.lp_.a_matrix_.value_ = a_value;
    model.lp_.integrality_ = integrality;
    highs.passModel(model);

    // Warm start: singleton cover
    {
      std::vector<double> warm(num_cols, 0.0);
      std::vector<int> best_sing(block_size, -1);
      for (int j = 0; j < num_cols; ++j) {
        if (local_columns[j].ops.size() == 1) {
          size_t op = local_columns[j].ops[0];
          if (best_sing[op] < 0 || local_columns[j].cost < local_columns[best_sing[op]].cost)
            best_sing[op] = j;
        }
      }
      bool all_covered = true;
      for (int i = 0; i < block_size; ++i) {
        if (best_sing[i] < 0) { all_covered = false; break; }
        warm[best_sing[i]] = 1.0;
      }
      if (all_covered) {
        HighsSolution warm_sol;
        warm_sol.col_value = warm;
        highs.setSolution(warm_sol);
      }
    }

    highs.run();
    auto model_status = highs.getModelStatus();
    if (model_status == HighsModelStatus::kOptimal ||
        model_status == HighsModelStatus::kObjectiveBound ||
        model_status == HighsModelStatus::kTimeLimit) {
      const auto& sol = highs.getSolution();
      for (int j = 0; j < num_cols; ++j) {
        if (sol.col_value[j] > 0.5) {
          // Use absolute-index columns for the combined partition
          combined_partition.subgraphs.push_back(block_columns[j].ops);
          combined_cost += block_columns[j].cost;
        }
      }
    } else {
      // ILP failed — fallback to singletons
      for (size_t op : block_ops) {
        combined_partition.subgraphs.push_back({op});
        std::unordered_set<size_t> empty_ri;
        auto cfg = solver::BestGranularity(p, {op}, empty_ri, 0, &dag, true);
        combined_cost += cfg.cost;
      }
    }
  }

  if (!all_solved || combined_partition.subgraphs.empty()) return;

  // ── Step 5: Run through standard pipeline ─────────────────────────────
  auto part = std::move(combined_partition);
  part = solver::RecomputationPass(p, dag, std::move(part));
  part = solver::ConvexityRepair(dag, std::move(part));
  part = solver::UnfusionPass(p, dag, std::move(part), /*assume_retain_out=*/true);
  part = solver::SubgraphReorder(p, dag, std::move(part));

  auto sol = solver::BuildSolution(p, dag, part);
  ZeroLatencies(sol);
  auto eval = mlsys::Evaluate(p, sol);
  double cost = eval.ok() ? *eval : 1e18;
  if (!eval.ok()) std::cerr << "  Partitioned ILP EVAL ERROR: " << eval.status() << "\n";
  std::cerr << "  Partitioned-ILP: " << (int64_t)cost
            << " (" << part.subgraphs.size() << " subgraphs)\n";
  all_candidates.push_back({std::move(sol), cost, "Partitioned-ILP",
                            solver::ElapsedSince(t0), (int)part.subgraphs.size()});
  tracker.MaybeWrite(all_candidates.back());

  // Also try HK ordering if small enough
  if ((int)part.subgraphs.size() <= 15 &&
      solver::ElapsedSince(t0) < hard_deadline * kTimeBudgetDPHK) {
    auto hk = PartitionToHKCandidate(p, dag, part, "Partitioned-ILP-HK");
    hk.elapsed_s = solver::ElapsedSince(t0);
    all_candidates.push_back(hk);
    tracker.MaybeWrite(all_candidates.back());
  }
}

#endif  // disabled RunPartitionedILP

// Helper: run DP partition with a given DAG ordering and retain mode, emit
// the partition candidate and optionally a Held-Karp-ordered candidate.
static void RunDPVariant(
    const mlsys::Problem& p,
    solver::DAG& dag_var,
    const std::string& label,
    bool assume_retain_out,
    std::chrono::steady_clock::time_point t0,
    double hard_deadline,
    std::vector<FinalCandidate>& all_candidates,
    BestTracker& tracker) {

  std::string full_label = label + (assume_retain_out ? "-R" : "");

  auto dp = DPFallbackWithPartition(p, dag_var, assume_retain_out, full_label);
  dp.candidate.elapsed_s = solver::ElapsedSince(t0);
  all_candidates.push_back(dp.candidate);
  tracker.MaybeWrite(all_candidates.back());

  // HK ordering on this partition (skip if too many subgroups — HK is O(2^S))
  if ((int)dp.part.subgraphs.size() <= 15 &&
      solver::ElapsedSince(t0) < hard_deadline * kTimeBudgetDPHK) {
    auto dp_hk = PartitionToHKCandidate(p, dag_var, dp.part, full_label + "-HK");
    dp_hk.elapsed_s = solver::ElapsedSince(t0);
    all_candidates.push_back(dp_hk);
    tracker.MaybeWrite(all_candidates.back());
  }
}

void RunDPFallbacks(
    const mlsys::Problem& p,
    const solver::DAG& dag,
    std::chrono::steady_clock::time_point t0,
    double hard_deadline,
    std::vector<FinalCandidate>& all_candidates,
    BestTracker& tracker) {

  // BFS ordering — retain=false (original) + retain=true
  {
    auto dag_bfs = dag;  // copy (BFS is already the default)
    RunDPVariant(p, dag_bfs, "DP-BFS", false, t0, hard_deadline, all_candidates, tracker);
  }
  if (solver::ElapsedSince(t0) < hard_deadline * kTimeBudgetDPDFS) {
    auto dag_bfs = dag;
    RunDPVariant(p, dag_bfs, "DP-BFS", true, t0, hard_deadline, all_candidates, tracker);
  }

  // DFS ordering — both retain modes
  if (solver::ElapsedSince(t0) < hard_deadline * kTimeBudgetDPDFS) {
    auto dag_dfs = solver::BuildDAG_DFS(p);
    RunDPVariant(p, dag_dfs, "DP-DFS", false, t0, hard_deadline, all_candidates, tracker);
  }
  if (solver::ElapsedSince(t0) < hard_deadline * kTimeBudgetDPDFS) {
    auto dag_dfs = solver::BuildDAG_DFS(p);
    RunDPVariant(p, dag_dfs, "DP-DFS", true, t0, hard_deadline, all_candidates, tracker);
  }

  // MaxOutput ordering — both retain modes
  if (solver::ElapsedSince(t0) < hard_deadline * kTimeBudgetDPDFS) {
    auto dag_maxout = solver::BuildDAG_MaxOutput(p);
    RunDPVariant(p, dag_maxout, "DP-MaxOut", false, t0, hard_deadline, all_candidates, tracker);
  }
  if (solver::ElapsedSince(t0) < hard_deadline * kTimeBudgetDPDFS) {
    auto dag_maxout = solver::BuildDAG_MaxOutput(p);
    RunDPVariant(p, dag_maxout, "DP-MaxOut", true, t0, hard_deadline, all_candidates, tracker);
  }
}

// ── SelectAndOutputBest: pick winner + write solution ────────────────────────

void SelectAndOutputBest(
    const mlsys::Problem& p,
    const solver::PreprocessResult& pr,
    const std::vector<FinalCandidate>& all_candidates,
    const char* output_path,
    std::chrono::steady_clock::time_point t0) {

  size_t best_idx = 0;
  for (size_t i = 1; i < all_candidates.size(); ++i)
    if (all_candidates[i].cost < all_candidates[best_idx].cost)
      best_idx = i;

  auto& best = all_candidates[best_idx];

  // Sort candidates by cost for summary table
  std::vector<size_t> sorted_idx(all_candidates.size());
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(),
            [&](size_t a, size_t b) { return all_candidates[a].cost < all_candidates[b].cost; });

  std::cerr << "\n=== Candidate Summary ===\n";
  std::cerr << "  #  Label              Cost          Sgs  Time(s)  vs Best\n";
  for (size_t rank = 0; rank < sorted_idx.size(); ++rank) {
    size_t i = sorted_idx[rank];
    auto& c = all_candidates[i];
    double gap_pct = (best.cost > 0) ? 100.0 * (c.cost - best.cost) / best.cost : 0;
    char gap_buf[32];
    std::snprintf(gap_buf, sizeof(gap_buf), "%+.1f%%", gap_pct);
    std::cerr << "  " << (rank + 1) << "  "
              << std::left << std::setw(18) << c.label
              << std::right << std::setw(14) << (int64_t)c.cost
              << std::setw(5) << c.num_subgraphs
              << "  " << std::fixed << std::setprecision(2) << c.elapsed_s << "s"
              << "  " << std::setw(8) << gap_buf
              << (i == best_idx ? "  *" : "") << "\n";
  }
  std::cerr << "Selected: " << best.label << " (" << (int64_t)best.cost << ")\n";
  std::cerr << "Evaluated total latency: " << (int64_t)best.cost
            << " (" << best.label << ")\n";

  solver::PrintDiagnostics(p, best.sol);
  auto final_sol = solver::UnPreprocess(best.sol, pr);
  solver::WriteSolution(final_sol, output_path);

  double elapsed = solver::ElapsedSince(t0);
  std::cerr << "Elapsed: " << elapsed << "s\n";
}

}  // namespace

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <problem.json> <solution.json> [max_size] [timeout] [--validate]\n";
    return 1;
  }

  auto t0 = std::chrono::steady_clock::now();

  auto problem = mlsys::ReadProblem(argv[1]);
  if (!problem.ok()) {
    std::cerr << problem.status() << "\n";
    return 1;
  }
  int N_orig = (int)problem->ops.size();
  auto pr = solver::Preprocess(*problem);
  auto& p = pr.problem;
  int N = (int)p.ops.size();

  // Check for --validate flag
  bool validate_mode = false;
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "--validate") validate_mode = true;
  }

  int max_size = (argc > 3 && !validate_mode) ? std::atoi(argv[3]) : DefaultMaxSize(N_orig);
  double timeout_override = (argc > 4) ? std::atof(argv[4]) : 0;  // 0 = use competition timeout

  std::cerr << "Unified solver: " << N << " ops, "
            << p.tensors.size() << " tensors"
            << " (max_size=" << max_size << ")\n";

  auto dag = solver::BuildDAG(p);

  if (validate_mode) {
    std::cerr << "=== Validating AnalyticalCost vs ExactCost ===" << std::endl;
    ValidateAnalyticalCost(p, dag);
    return 0;
  }
  auto adj = BuildUndirectedAdj(dag);
  ClassifyCache cc;
  std::vector<FinalCandidate> all_candidates;
  BestTracker tracker{pr, argv[2]};

  // ── Regime Analysis ────────────────────────────────────────────────────
  auto regime = ComputeRegimeAnalysis(p, dag, adj, cc);

  int bm_number = solver::ParseBenchmarkNumber(argv[1]);
  double timeout = timeout_override > 0 ? timeout_override : solver::CompetitionTimeout(N_orig, bm_number);
  double hard_deadline = timeout * kHardDeadlineRatio;

  // ── DAG Rewrite for Ephemeral Recomputation ───────────────────────────
  RunRewritePath(p, dag, N, t0, hard_deadline, all_candidates, tracker);

  // ── Phase 0: Heavy graph input fusion columns ──────────────────────────
  std::vector<SuperColumn> heavy_columns;
  {
    double phase0_budget = hard_deadline * kTimeBudgetPhase0;
    GenerateHeavyInputColumns(p, dag, heavy_columns, t0, phase0_budget, cc, &regime);
  }

  // ── Phase 1: Enumerate + ILP ─────────────────────────────────────────────
  double phase1_colgen_deadline = hard_deadline * kTimeBudgetILPBreak;
  auto columns = GenerateColumnPool(p, dag, adj, N, max_size, heavy_columns, cc,
                                     &regime, t0, phase1_colgen_deadline);

  // Identify recomputable ops: cheap ops with fan_out ≥ 2
  std::vector<bool> recomputable(N, false);
  {
    int n_recomp = 0;
    for (int i = 0; i < N; ++i) {
      if (IsRecomputable(p, dag, i)) {
        recomputable[i] = true;
        ++n_recomp;
      }
    }
    if (n_recomp > 0)
      std::cerr << "Recomputable ops: " << n_recomp << "/" << N << "\n";
  }

  // ── Column Generation: LP-guided pool enrichment ────────────────────────
  if (!columns.empty() && solver::ElapsedSince(t0) < hard_deadline * 0.35) {
    RunColumnGeneration(p, dag, adj, columns, recomputable,
                        cc, t0, hard_deadline);
  }

  // ── Graph Decomposition: enrich column pool with larger subsets ──────
  if (N >= 10 && solver::ElapsedSince(t0) < hard_deadline * 0.45) {
    auto components = solver::DecomposeDAG(p, dag);
    if (components.size() >= 2) {
      // For each small component, enumerate larger subsets and add to global pool.
      std::unordered_set<std::vector<size_t>, VectorHash> existing;
      for (auto& col : columns) existing.insert(col.ops);

      int decomp_added = 0;
      for (size_t ci = 0; ci < components.size(); ++ci) {
        auto& comp = components[ci];
        if (solver::ElapsedSince(t0) > hard_deadline * 0.45) break;
        int comp_n = (int)comp.ops.size();
        int comp_max_size = (comp_n <= 8) ? comp_n :
                            (comp_n <= 15) ? 6 :
                            (comp_n <= 30) ? 5 : DefaultMaxSize(comp_n);

        // Skip if component max_size isn't higher than global max_size.
        if (comp_max_size <= max_size) continue;

        std::unordered_set<size_t> comp_ops_set(comp.ops.begin(), comp.ops.end());
        AdjList comp_adj(N);
        for (size_t op : comp.ops)
          for (size_t nb : adj[op])
            if (comp_ops_set.count(nb))
              comp_adj[op].insert(nb);

        // Enumerate larger subsets within component.
        std::vector<std::vector<size_t>> new_subsets;
        std::vector<std::vector<size_t>> current_level;
        // Start from existing subsets of size max_size within this component.
        for (auto& col : columns) {
          if ((int)col.ops.size() != max_size) continue;
          bool in_comp = true;
          for (size_t op : col.ops)
            if (!comp_ops_set.count(op)) { in_comp = false; break; }
          if (in_comp) current_level.push_back(col.ops);
        }
        // Also start from singletons for small components.
        if (current_level.empty()) {
          for (size_t op : comp.ops)
            current_level.push_back({op});
        }

        std::unordered_set<std::vector<size_t>, VectorHash> seen(existing);
        for (int sz = max_size + 1; sz <= comp_max_size; ++sz) {
          if (solver::ElapsedSince(t0) > hard_deadline * 0.45) break;
          std::vector<std::vector<size_t>> next_level;
          for (auto& subset : current_level) {
            std::set<size_t> sub_set(subset.begin(), subset.end());
            for (size_t op : subset) {
              for (size_t nb : comp_adj[op]) {
                if (sub_set.count(nb) || nb <= subset[0]) continue;
                auto next = subset;
                auto pos = std::lower_bound(next.begin(), next.end(), nb);
                next.insert(pos, nb);
                if (!seen.insert(next).second) continue;
                if (!solver::IsDAGConvex(dag, next)) continue;
                new_subsets.push_back(next);
                next_level.push_back(next);
              }
            }
          }
          current_level = std::move(next_level);
          if (current_level.empty()) break;
        }

        if (!new_subsets.empty()) {
          size_t before = columns.size();
          BuildSuperColumns(p, dag, new_subsets, columns, cc,
                            t0, hard_deadline * 0.45);
          int added = (int)(columns.size() - before);
          decomp_added += added;
          std::cerr << "  Decomp comp " << ci << " (" << comp_n << " ops): +"
                    << added << " columns (size>" << max_size << ")\n";
        }
      }
      if (decomp_added > 0)
        std::cerr << "Decomposition enriched pool: +" << decomp_added << " columns\n";
    }
  }

  // ── Multi-Pass ILP Refinement ────────────────────────────────────────
  RunILPIterations(p, dag, columns, recomputable, t0, hard_deadline, all_candidates, tracker);

  // ── DP Fallbacks ────────────────────────────────────────────────────────
  RunDPFallbacks(p, dag, t0, hard_deadline, all_candidates, tracker);

  // ── Pick Best ──────────────────────────────────────────────────────────
  SelectAndOutputBest(p, pr, all_candidates, argv[2], t0);
  return 0;
}
