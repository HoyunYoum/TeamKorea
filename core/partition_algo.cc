// partition_algo.cc — Partition strategies + retention candidates + reorder.
//
// Extracted from solver_common.cc (Phase 3 refactor).

#include "partition_algo.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "cost.h"
#include "granularity.h"
#include "io_util.h"
#include "tensor_roles.h"

namespace solver {

namespace {
inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace

// ── Partition ────────────────────────────────────────────────────────────────

Partition InitialPartition(const DAG& dag) {
  Partition part;
  for (size_t op : dag.topo_order)
    part.subgraphs.push_back({op});
  return part;
}

// ── Subgraph signature for cost caching ──────────────────────────────────────

std::string SubgraphSig(const mlsys::Problem& p,
                        const std::vector<size_t>& ops, const DAG& dag) {
  auto ts = Classify(p, ops, &dag);
  std::string sig;
  sig.reserve(256);
  for (size_t op : ops) {
    const auto& o = p.ops[op];
    sig += o.op_type[0];
    sig += ',';
    sig += std::to_string(o.base_cost);
    for (size_t t : o.inputs) {
      sig += ts.ephemeral.count(t) ? ",e" : ",i";
      sig += std::to_string(p.tensors[t].width);
      sig += 'x';
      sig += std::to_string(p.tensors[t].height);
    }
    for (size_t t : o.outputs) {
      sig += ts.ephemeral.count(t) ? ">e" : ">o";
      sig += std::to_string(p.tensors[t].width);
      sig += 'x';
      sig += std::to_string(p.tensors[t].height);
    }
    sig += ';';
  }
  return sig;
}

// ── DP-optimal contiguous partition ──────────────────────────────────────────

Partition DPPartition(const mlsys::Problem& p, const DAG& dag,
                      Partition part,
                      bool assume_retain_out) {
  std::unordered_set<size_t> empty_ri;
  int N = (int)part.subgraphs.size();
  if (N <= 1) return part;

  // Signature cache: structurally identical subgraphs share cost
  std::unordered_map<std::string, double> sig_cache;
  auto CachedCost = [&](const std::vector<size_t>& ops) -> double {
    auto sig = SubgraphSig(p, ops, dag);
    auto it = sig_cache.find(sig);
    if (it != sig_cache.end()) return it->second;
    double c = BestGranularity(p, ops, empty_ri, 0, &dag, assume_retain_out).cost;
    sig_cache[sig] = c;
    return c;
  };

  // Dynamic interval cap: reduce for larger problems to stay within timeout
  const int MAX_INTERVAL = std::min(N, (N <= 20) ? 20 : (N <= 50) ? 15 : 12);

  // Precompute: which subgraph produces/consumes which tensors (for pruning)
  std::vector<std::unordered_set<size_t>> sg_produced(N), sg_consumed(N);
  for (int i = 0; i < N; ++i) {
    for (size_t op : part.subgraphs[i]) {
      for (size_t t : p.ops[op].outputs) sg_produced[i].insert(t);
      for (size_t t : p.ops[op].inputs) sg_consumed[i].insert(t);
    }
  }

  // dp[i] = min cost for scheduling initial subgraphs 0..i-1
  std::vector<double> dp(N + 1, kInf);
  std::vector<int> parent(N + 1, -1);
  dp[0] = 0;

  for (int i = 1; i <= N; ++i) {
    std::vector<size_t> merged_ops;
    // Track which tensors are produced/consumed in [j, i)
    std::unordered_set<size_t> interval_produced, interval_consumed;
    bool has_ephemerals = false;
    bool has_shared_inputs = false;

    for (int j = i - 1; j >= std::max(0, i - MAX_INTERVAL); --j) {
      // Grow interval [j, i): append ops from subgraph j
      auto& sg = part.subgraphs[j];
      merged_ops.insert(merged_ops.end(), sg.begin(), sg.end());

      // Update interval tensor tracking
      if (j < i - 1) {
        // Check if subgraph j creates new ephemerals with existing interval
        for (size_t t : sg_produced[j])
          if (interval_consumed.count(t)) has_ephemerals = true;
        for (size_t t : sg_consumed[j]) {
          if (interval_produced.count(t)) has_ephemerals = true;
          if (interval_consumed.count(t)) has_shared_inputs = true;
        }
      }
      for (size_t t : sg_produced[j]) interval_produced.insert(t);
      for (size_t t : sg_consumed[j]) interval_consumed.insert(t);

      if (dp[j] >= kInf) continue;  // unreachable prefix

      // Skip multi-op intervals with no IO savings potential
      if (j < i - 1 && !has_ephemerals && !has_shared_inputs) continue;

      // Feasibility: check minimum working set (h=1, w=1, k=1)
      if ((int)merged_ops.size() > 1) {
        auto ts_m = Classify(p, merged_ops, &dag);
        int64_t min_ws = 0;
        std::unordered_map<size_t, int64_t> tensor_min;
        for (size_t op : merged_ops) {
          const auto& o = p.ops[op];
          for (int pos = 0; pos < (int)o.inputs.size(); ++pos) {
            size_t t = o.inputs[pos];
            if (ts_m.produced.count(t)) continue;  // ephemeral
            int64_t slice = (o.op_type == "MatMul" && pos == 0)
                            ? p.tensors[t].width : 1;
            auto it2 = tensor_min.find(t);
            if (it2 == tensor_min.end()) tensor_min[t] = slice;
            else it2->second = std::max(it2->second, slice);
          }
        }
        for (auto& [tid, s] : tensor_min) min_ws += s;
        min_ws += (int64_t)ts_m.outputs.size();
        if (min_ws > p.fast_memory_capacity) continue;
      }

      double cost = CachedCost(merged_ops);
      if (cost >= kInf) continue;

      if (dp[j] + cost < dp[i]) {
        dp[i] = dp[j] + cost;
        parent[i] = j;
      }
    }
  }

  // Backtrack to recover partition
  if (dp[N] >= kInf) {
    std::cerr << "DP partition: no valid partition found, keeping original\n";
    return part;
  }

  Partition result;
  int i = N;
  while (i > 0) {
    int j = parent[i];
    std::vector<size_t> merged;
    for (int k = j; k < i; ++k)
      merged.insert(merged.end(), part.subgraphs[k].begin(),
                     part.subgraphs[k].end());
    result.subgraphs.push_back(std::move(merged));
    i = j;
  }
  std::reverse(result.subgraphs.begin(), result.subgraphs.end());

  std::cerr << "DP partition: " << N << " -> " << result.subgraphs.size()
            << " subgraphs (sig cache: " << sig_cache.size() << ")\n";
  return result;
}

// ── CanMergeSubgraphs ────────────────────────────────────────────────────────

bool CanMergeSubgraphs(const DAG& dag, const Partition& part,
                       int sg_a, int sg_b) {
  int S = (int)part.subgraphs.size();
  // Build op_to_sg mapping
  std::vector<int> op_to_sg(dag.num_ops, -1);
  for (int i = 0; i < S; ++i)
    for (size_t op : part.subgraphs[i])
      op_to_sg[op] = i;

  // Build subgraph-level DAG with sg_a and sg_b merged (both map to sg_a)
  std::vector<std::unordered_set<int>> sg_succs(S);
  std::vector<int> in_deg(S, 0);
  for (size_t a = 0; a < dag.num_ops; ++a) {
    int sa = op_to_sg[a];
    if (sa == sg_b) sa = sg_a;
    for (size_t b : dag.op_successors[a]) {
      int sb = op_to_sg[b];
      if (sb == sg_b) sb = sg_a;
      if (sa != sb && !sg_succs[sa].count(sb)) {
        sg_succs[sa].insert(sb);
        ++in_deg[sb];
      }
    }
  }
  // Kahn's algorithm — skip the merged-away node sg_b
  int count = 0;
  std::queue<int> q;
  for (int i = 0; i < S; ++i)
    if (i != sg_b && in_deg[i] == 0) q.push(i);
  while (!q.empty()) {
    int u = q.front(); q.pop(); ++count;
    for (int v : sg_succs[u])
      if (--in_deg[v] == 0) q.push(v);
  }
  return count == S - 1;  // S-1 because sg_b is skipped
}

// ── AgglomerativePartition ──────────────────────────────────────────────────

Partition AgglomerativePartition(const mlsys::Problem& p, const DAG& dag,
                                  bool assume_retain_out) {
  std::unordered_set<size_t> empty_ri;
  int N = (int)dag.num_ops;
  if (N <= 1) {
    Partition part;
    for (size_t op : dag.topo_order) part.subgraphs.push_back({op});
    return part;
  }

  // Signature cache for cost
  std::unordered_map<std::string, double> sig_cache;
  auto CachedCost = [&](const std::vector<size_t>& ops) -> double {
    auto sig = SubgraphSig(p, ops, dag);
    auto it = sig_cache.find(sig);
    if (it != sig_cache.end()) return it->second;
    double c = BestGranularity(p, ops, empty_ri, 0, &dag, assume_retain_out).cost;
    sig_cache[sig] = c;
    return c;
  };

  // Start with singleton partition
  Partition part;
  for (size_t op : dag.topo_order) part.subgraphs.push_back({op});
  int S = (int)part.subgraphs.size();

  // Compute initial costs
  std::vector<double> sg_cost(S);
  for (int i = 0; i < S; ++i)
    sg_cost[i] = CachedCost(part.subgraphs[i]);

  // Track which subgraphs are alive (not merged away)
  std::vector<bool> alive(S, true);

  // Build adjacency: pairs connected by tensor edges
  std::set<std::pair<int,int>> adjacent_pairs;
  {
    std::vector<int> op_to_sg(dag.num_ops, -1);
    for (int i = 0; i < S; ++i)
      for (size_t op : part.subgraphs[i]) op_to_sg[op] = i;
    for (size_t a = 0; a < dag.num_ops; ++a) {
      for (size_t b : dag.op_successors[a]) {
        int sa = op_to_sg[a], sb = op_to_sg[b];
        if (sa != sb) {
          int lo = std::min(sa, sb), hi = std::max(sa, sb);
          adjacent_pairs.insert({lo, hi});
        }
      }
    }
  }

  // Priority queue: (savings, sg_a, sg_b)
  using Entry = std::tuple<double, int, int>;
  std::priority_queue<Entry> pq;

  auto EvalPair = [&](int a, int b) {
    if (!alive[a] || !alive[b]) return;
    std::vector<size_t> merged;
    merged.insert(merged.end(), part.subgraphs[a].begin(), part.subgraphs[a].end());
    merged.insert(merged.end(), part.subgraphs[b].begin(), part.subgraphs[b].end());
    double mc = CachedCost(merged);
    if (mc >= kInf) return;
    double savings = sg_cost[a] + sg_cost[b] - mc;
    if (savings > 0)
      pq.push({savings, a, b});
  };

  // Initialize PQ with all adjacent pairs
  for (auto& [a, b] : adjacent_pairs)
    EvalPair(a, b);

  // Greedy merge loop
  while (!pq.empty()) {
    auto [sav, a, b] = pq.top(); pq.pop();
    if (!alive[a] || !alive[b]) continue;

    // Recompute savings (may be stale)
    std::vector<size_t> merged;
    merged.insert(merged.end(), part.subgraphs[a].begin(), part.subgraphs[a].end());
    merged.insert(merged.end(), part.subgraphs[b].begin(), part.subgraphs[b].end());
    double mc = CachedCost(merged);
    if (mc >= kInf) continue;
    double real_sav = sg_cost[a] + sg_cost[b] - mc;
    if (real_sav <= 0) continue;

    // Validate DAG acyclicity
    if (!CanMergeSubgraphs(dag, part, a, b)) continue;

    // Merge b into a
    part.subgraphs[a] = std::move(merged);
    sg_cost[a] = mc;
    alive[b] = false;
    part.subgraphs[b].clear();

    // Find new neighbors of merged subgraph and enqueue
    for (int i = 0; i < S; ++i) {
      if (i == a || !alive[i]) continue;
      // Check if i is adjacent to merged a (via any op edge)
      bool adj = false;
      for (size_t op : part.subgraphs[a]) {
        for (size_t succ : dag.op_successors[op]) {
          for (size_t op2 : part.subgraphs[i]) {
            if (succ == op2) { adj = true; break; }
          }
          if (adj) break;
        }
        if (adj) break;
        // Also check reverse: ops in i that are successors pointing to a
      }
      if (!adj) {
        for (size_t op : part.subgraphs[i]) {
          for (size_t succ : dag.op_successors[op]) {
            for (size_t op2 : part.subgraphs[a]) {
              if (succ == op2) { adj = true; break; }
            }
            if (adj) break;
          }
          if (adj) break;
        }
      }
      if (adj) EvalPair(a, i);
    }
  }

  // Compact: remove dead subgraphs
  Partition result;
  for (int i = 0; i < S; ++i)
    if (alive[i]) result.subgraphs.push_back(std::move(part.subgraphs[i]));

  std::cerr << "Agglomerative partition: " << N << " -> " << result.subgraphs.size()
            << " subgraphs (sig cache: " << sig_cache.size() << ")\n";
  return result;
}

// ── MaxFusionPartition ──────────────────────────────────────────────────────

Partition MaxFusionPartition(const mlsys::Problem& p, const DAG& dag,
                              bool sink_first, bool assume_retain_out) {
  std::unordered_set<size_t> empty_ri;
  int N = (int)dag.num_ops;

  // Processing order: reverse topo (sink-first) or forward topo (source-first)
  std::vector<size_t> order = dag.topo_order;
  if (sink_first) std::reverse(order.begin(), order.end());

  std::vector<bool> assigned(dag.num_ops, false);
  Partition result;

  // Build op predecessors for reverse traversal
  std::vector<std::vector<size_t>> op_predecessors(dag.num_ops);
  for (size_t a = 0; a < dag.num_ops; ++a)
    for (size_t b : dag.op_successors[a])
      op_predecessors[b].push_back(a);

  for (size_t seed_op : order) {
    if (assigned[seed_op]) continue;
    assigned[seed_op] = true;

    std::vector<size_t> sg_ops = {seed_op};

    // Greedily absorb neighbors
    bool grew = true;
    while (grew) {
      grew = false;
      // Collect candidate ops: unassigned neighbors of current subgraph
      std::vector<std::pair<int, size_t>> candidates;  // (ephemeral_priority, op)
      std::unordered_set<size_t> sg_set(sg_ops.begin(), sg_ops.end());

      for (size_t op : sg_ops) {
        // Successors
        for (size_t succ : dag.op_successors[op]) {
          if (!assigned[succ] && !sg_set.count(succ)) {
            // Check if absorbing creates an ephemeral
            // The tensor connecting op→succ becomes ephemeral if all consumers
            // are in the subgraph
            candidates.push_back({0, succ});
          }
        }
        // Predecessors
        for (size_t pred : op_predecessors[op]) {
          if (!assigned[pred] && !sg_set.count(pred)) {
            candidates.push_back({0, pred});
          }
        }
      }

      // Prioritize by ephemeral creation potential
      for (auto& [prio, cand_op] : candidates) {
        std::vector<size_t> trial = sg_ops;
        trial.push_back(cand_op);
        auto ts = Classify(p, trial, &dag);
        prio = (int)ts.ephemeral.size();  // more ephemerals = higher priority
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

      // Remove duplicates
      std::unordered_set<size_t> tried;
      for (auto& [prio, cand_op] : candidates) {
        if (tried.count(cand_op)) continue;
        tried.insert(cand_op);

        std::vector<size_t> trial = sg_ops;
        trial.push_back(cand_op);

        // Check feasibility via BestGranularity
        double cost = BestGranularity(p, trial, empty_ri, 0, &dag, assume_retain_out).cost;
        if (cost >= kInf) continue;

        // Absorb
        sg_ops.push_back(cand_op);
        sg_set.insert(cand_op);
        assigned[cand_op] = true;
        grew = true;
        break;  // Restart neighbor search with updated subgraph
      }
    }

    result.subgraphs.push_back(std::move(sg_ops));
  }

  std::cerr << "MaxFusion partition (" << (sink_first ? "sink" : "source")
            << "-first): " << N << " -> " << result.subgraphs.size() << " subgraphs\n";
  return result;
}

// ── RandomPartition ─────────────────────────────────────────────────────────

Partition RandomPartition(const mlsys::Problem& p, const DAG& dag,
                           uint32_t seed, double deadline_s,
                           bool assume_retain_out) {
  auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
  };

  std::unordered_set<size_t> empty_ri;
  int N = (int)dag.num_ops;
  if (N <= 1) {
    Partition part;
    for (size_t op : dag.topo_order) part.subgraphs.push_back({op});
    return part;
  }

  // Signature cache
  std::unordered_map<std::string, double> sig_cache;
  auto CachedCost = [&](const std::vector<size_t>& ops) -> double {
    auto sig = SubgraphSig(p, ops, dag);
    auto it = sig_cache.find(sig);
    if (it != sig_cache.end()) return it->second;
    double c = BestGranularity(p, ops, empty_ri, 0, &dag, assume_retain_out).cost;
    sig_cache[sig] = c;
    return c;
  };

  auto PartCost = [&](const Partition& part) -> double {
    double total = 0;
    for (auto& sg : part.subgraphs) {
      double c = CachedCost(sg);
      if (c >= kInf) return kInf;
      total += c;
    }
    return total;
  };

  // Build op predecessors
  std::vector<std::vector<size_t>> op_preds(dag.num_ops);
  for (size_t a = 0; a < dag.num_ops; ++a)
    for (size_t b : dag.op_successors[a])
      op_preds[b].push_back(a);

  Partition best_part;
  double best_cost = kInf;

  // Multiple restarts
  for (int restart = 0; restart < 3 && elapsed() < deadline_s * 0.95; ++restart) {
    std::mt19937 rng(seed + restart * 1000);

    // Generate random partition: assign each op to a random group
    int K = std::uniform_int_distribution<int>(
        std::max(1, N / 4), std::max(1, N))(rng);
    std::vector<int> op_group(dag.num_ops);
    for (size_t i = 0; i < dag.num_ops; ++i)
      op_group[i] = std::uniform_int_distribution<int>(0, K - 1)(rng);

    // Fix DAG violations: if op a → op b and group[a] == group[b], that's fine.
    // But if there's a cycle in the group-level DAG, merge groups.
    // Simple fix: for each edge a→b where group[a] != group[b],
    // check that it doesn't form a group-level cycle.
    // Actually, just build the partition and validate. If invalid, merge offending groups.
    bool fixed = false;
    for (int attempt = 0; attempt < 10 && !fixed; ++attempt) {
      // Build partition from groups
      std::unordered_map<int, std::vector<size_t>> groups;
      for (size_t i = 0; i < dag.num_ops; ++i)
        groups[op_group[i]].push_back(i);

      // Build group-level DAG
      std::unordered_map<int, std::unordered_set<int>> group_succs;
      std::unordered_map<int, int> group_indeg;
      for (auto& [g, _] : groups) group_indeg[g] = 0;
      for (size_t a = 0; a < dag.num_ops; ++a) {
        for (size_t b : dag.op_successors[a]) {
          int ga = op_group[a], gb = op_group[b];
          if (ga != gb && !group_succs[ga].count(gb)) {
            group_succs[ga].insert(gb);
            group_indeg[gb]++;
          }
        }
      }

      // Topo sort to check for cycles
      std::queue<int> q;
      for (auto& [g, deg] : group_indeg)
        if (deg == 0) q.push(g);
      int count = 0;
      std::vector<int> topo;
      while (!q.empty()) {
        int u = q.front(); q.pop(); topo.push_back(u); count++;
        for (int v : group_succs[u])
          if (--group_indeg[v] == 0) q.push(v);
      }

      if (count == (int)groups.size()) {
        fixed = true;
      } else {
        // Merge all cycle members: find groups not in topo, merge them all
        std::unordered_set<int> in_topo(topo.begin(), topo.end());
        int merge_target = -1;
        for (auto& [g, _] : groups) {
          if (!in_topo.count(g)) {
            if (merge_target < 0) merge_target = g;
            else {
              for (size_t op : groups[g]) op_group[op] = merge_target;
            }
          }
        }
      }
    }

    // Build partition
    Partition part;
    std::unordered_map<int, int> group_idx;
    for (size_t op : dag.topo_order) {
      int g = op_group[op];
      if (!group_idx.count(g)) {
        group_idx[g] = (int)part.subgraphs.size();
        part.subgraphs.push_back({});
      }
      part.subgraphs[group_idx[g]].push_back(op);
    }

    double cost = PartCost(part);
    if (cost >= kInf) continue;

    // Local search
    double restart_deadline = deadline_s * (restart + 1.0) / 3.0;
    int iters = 0;
    while (elapsed() < std::min(restart_deadline, deadline_s * 0.95)) {
      int S = (int)part.subgraphs.size();
      if (S <= 1) break;
      int move_type = std::uniform_int_distribution<int>(0, 2)(rng);

      if (move_type == 0 && S >= 2) {
        // Move: relocate random op to random other subgraph
        int from = std::uniform_int_distribution<int>(0, S - 1)(rng);
        if (part.subgraphs[from].size() <= 1) { iters++; continue; }
        int op_idx = std::uniform_int_distribution<int>(
            0, (int)part.subgraphs[from].size() - 1)(rng);
        size_t op = part.subgraphs[from][op_idx];
        int to = std::uniform_int_distribution<int>(0, S - 1)(rng);
        if (to == from) { iters++; continue; }

        // Try move
        double old_cost_from = CachedCost(part.subgraphs[from]);
        double old_cost_to = CachedCost(part.subgraphs[to]);

        std::vector<size_t> new_from, new_to;
        for (size_t o : part.subgraphs[from]) if (o != op) new_from.push_back(o);
        new_to = part.subgraphs[to];
        new_to.push_back(op);

        double new_cost_from = CachedCost(new_from);
        double new_cost_to = CachedCost(new_to);
        if (new_cost_from >= kInf || new_cost_to >= kInf) { iters++; continue; }

        double delta = (new_cost_from + new_cost_to) - (old_cost_from + old_cost_to);
        if (delta < -1e-6) {
          // Validate DAG
          Partition trial = part;
          trial.subgraphs[from] = new_from;
          trial.subgraphs[to] = new_to;
          // Quick DAG check: ensure no cycle by building topo sort
          // (simpler than CanMergeSubgraphs for move ops)
          part.subgraphs[from] = std::move(new_from);
          part.subgraphs[to] = std::move(new_to);
          cost += delta;
        }
      } else if (move_type == 1 && S >= 2) {
        // Merge: pick two random subgraphs
        int a = std::uniform_int_distribution<int>(0, S - 1)(rng);
        int b = std::uniform_int_distribution<int>(0, S - 2)(rng);
        if (b >= a) b++;

        std::vector<size_t> merged;
        merged.insert(merged.end(), part.subgraphs[a].begin(), part.subgraphs[a].end());
        merged.insert(merged.end(), part.subgraphs[b].begin(), part.subgraphs[b].end());

        double mc = CachedCost(merged);
        if (mc >= kInf) { iters++; continue; }
        double delta = mc - CachedCost(part.subgraphs[a]) - CachedCost(part.subgraphs[b]);
        if (delta < -1e-6 && CanMergeSubgraphs(dag, part, a, b)) {
          part.subgraphs[a] = std::move(merged);
          part.subgraphs.erase(part.subgraphs.begin() + b);
          cost += delta;
        }
      } else {
        // Split: pick random multi-op subgraph, split at midpoint
        int idx = std::uniform_int_distribution<int>(0, S - 1)(rng);
        if (part.subgraphs[idx].size() <= 1) { iters++; continue; }
        auto& sg = part.subgraphs[idx];
        int mid = std::uniform_int_distribution<int>(1, (int)sg.size() - 1)(rng);
        std::vector<size_t> left(sg.begin(), sg.begin() + mid);
        std::vector<size_t> right(sg.begin() + mid, sg.end());
        double sc = CachedCost(left) + CachedCost(right);
        if (sc >= kInf) { iters++; continue; }
        double delta = sc - CachedCost(sg);
        if (delta < -1e-6) {
          part.subgraphs[idx] = std::move(left);
          part.subgraphs.push_back(std::move(right));
          cost += delta;
        }
      }
      iters++;
    }

    if (cost < best_cost) {
      best_cost = cost;
      best_part = part;
    }
  }

  // Fallback: if nothing worked, return singleton partition
  if (best_part.subgraphs.empty()) {
    for (size_t op : dag.topo_order)
      best_part.subgraphs.push_back({op});
  }

  std::cerr << "Random partition (seed=" << seed << "): " << N << " -> "
            << best_part.subgraphs.size() << " subgraphs, cost="
            << (int64_t)best_cost << " (sig cache: " << sig_cache.size() << ")\n";
  return best_part;
}

// ── AntiChainPartition ──────────────────────────────────────────────────────

Partition AntiChainPartition(const mlsys::Problem& p, const DAG& dag,
                              bool assume_retain_out) {
  std::unordered_set<size_t> empty_ri;
  int N = (int)dag.num_ops;

  // Compute DAG levels: level[op] = 1 + max(level[pred])
  std::vector<int> level(dag.num_ops, 0);
  // Build predecessors
  std::vector<std::vector<size_t>> op_preds(dag.num_ops);
  for (size_t a = 0; a < dag.num_ops; ++a)
    for (size_t b : dag.op_successors[a])
      op_preds[b].push_back(a);

  int max_level = 0;
  for (size_t op : dag.topo_order) {
    for (size_t pred : op_preds[op])
      level[op] = std::max(level[op], level[pred] + 1);
    max_level = std::max(max_level, level[op]);
  }

  // Group ops by level
  std::vector<std::vector<size_t>> level_ops(max_level + 1);
  for (size_t op : dag.topo_order)
    level_ops[level[op]].push_back(op);

  // Within each level, pack into subgraphs respecting WS
  Partition result;
  for (int lv = 0; lv <= max_level; ++lv) {
    auto& ops = level_ops[lv];
    if (ops.empty()) continue;

    // Try to pack all ops at this level into one subgraph first
    double cost = BestGranularity(p, ops, empty_ri, 0, &dag, assume_retain_out).cost;
    if (cost < kInf) {
      result.subgraphs.push_back(ops);
    } else {
      // First-fit packing
      std::vector<std::vector<size_t>> bins;
      for (size_t op : ops) {
        bool placed = false;
        for (auto& bin : bins) {
          std::vector<size_t> trial = bin;
          trial.push_back(op);
          double tc = BestGranularity(p, trial, empty_ri, 0, &dag, assume_retain_out).cost;
          if (tc < kInf) {
            bin.push_back(op);
            placed = true;
            break;
          }
        }
        if (!placed) bins.push_back({op});
      }
      for (auto& bin : bins) result.subgraphs.push_back(std::move(bin));
    }
  }

  // Cross-level merge pass: try merging adjacent-level subgraphs
  // Signature cache for cost
  std::unordered_map<std::string, double> sig_cache;
  auto CachedCost = [&](const std::vector<size_t>& ops) -> double {
    auto sig = SubgraphSig(p, ops, dag);
    auto it = sig_cache.find(sig);
    if (it != sig_cache.end()) return it->second;
    double c = BestGranularity(p, ops, empty_ri, 0, &dag, assume_retain_out).cost;
    sig_cache[sig] = c;
    return c;
  };

  bool improved = true;
  while (improved) {
    improved = false;
    int S = (int)result.subgraphs.size();
    for (int i = 0; i < S - 1 && !improved; ++i) {
      for (int j = i + 1; j < S && !improved; ++j) {
        double ci = CachedCost(result.subgraphs[i]);
        double cj = CachedCost(result.subgraphs[j]);
        std::vector<size_t> merged;
        merged.insert(merged.end(), result.subgraphs[i].begin(), result.subgraphs[i].end());
        merged.insert(merged.end(), result.subgraphs[j].begin(), result.subgraphs[j].end());
        double cm = CachedCost(merged);
        if (cm < kInf && cm < ci + cj - 1e-6) {
          if (CanMergeSubgraphs(dag, result, i, j)) {
            result.subgraphs[i] = std::move(merged);
            result.subgraphs.erase(result.subgraphs.begin() + j);
            improved = true;
          }
        }
      }
    }
  }

  std::cerr << "AntiChain partition: " << N << " -> " << result.subgraphs.size()
            << " subgraphs\n";
  return result;
}

// ── Recomputation pass ───────────────────────────────────────────────────────

Partition RecomputationPass(const mlsys::Problem& p, const DAG& dag,
                            Partition part,
                            bool assume_retain_out) {
  std::unordered_set<size_t> empty_ri;
  bool improved = true;
  while (improved) {
    improved = false;
    for (auto& sg : part.subgraphs) {
      auto ts = Classify(p, sg, &dag);
      std::unordered_set<size_t> sg_set(sg.begin(), sg.end());

      for (size_t t : ts.inputs) {
        int prod = dag.tensor_producer[t];
        if (prod < 0) continue;           // graph input, no producer
        if (sg_set.count(prod)) continue;  // already in subgraph

        // Check: producer's inputs must all be graph inputs or already
        // available as inputs to this subgraph
        const auto& prod_op = p.ops[prod];
        bool can_recompute = true;
        for (size_t pt : prod_op.inputs) {
          if (dag.tensor_producer[pt] >= 0 && !ts.inputs.count(pt) &&
              !ts.produced.count(pt)) {
            can_recompute = false;
            break;
          }
        }
        if (!can_recompute) continue;

        // Check: recomputation only helps if the output tensor becomes
        // ephemeral (all consumers inside the extended subgraph). If external
        // consumers remain, there's no ephemeral benefit and adding the
        // producer can create cycles in the subgraph-level DAG.
        bool all_consumers_inside = true;
        for (size_t out_t : prod_op.outputs) {
          for (size_t cons : dag.tensor_consumers[out_t]) {
            if (!sg_set.count(cons) && cons != (size_t)prod) {
              all_consumers_inside = false;
              break;
            }
          }
          if (!all_consumers_inside) break;
        }
        if (!all_consumers_inside) continue;

        // Try adding the producer — check if it improves cost
        std::vector<size_t> extended = sg;
        extended.push_back(prod);
        auto cfg_old = BestGranularity(p, sg, empty_ri, 0, &dag, assume_retain_out);
        auto cfg_new = BestGranularity(p, extended, empty_ri, 0, &dag, assume_retain_out);
        if (cfg_new.cost < cfg_old.cost) {
          sg = extended;
          sg_set.insert(prod);
          improved = true;
          break;  // restart scan for this partition
        }
      }
      if (improved) break;  // restart outer loop

      // Deep recomputation: trace backward through cheap Pointwise chains
      for (size_t t : ts.inputs) {
        int prod = dag.tensor_producer[t];
        if (prod < 0 || sg_set.count(prod)) continue;
        // Skip if the direct producer is already a MatMul (handled above)
        if (p.ops[prod].op_type == "MatMul") continue;

        // Collect chain of Pointwise ops backward from tensor t
        std::vector<size_t> chain;
        std::unordered_set<size_t> chain_set;
        size_t cur_tensor = t;
        bool feasible = true;

        while (true) {
          int cprod = dag.tensor_producer[cur_tensor];
          if (cprod < 0) break;  // graph input — chain terminates OK
          if (sg_set.count(cprod)) break;  // already in subgraph
          if (chain_set.count(cprod)) { feasible = false; break; }
          if (p.ops[cprod].op_type != "Pointwise") { feasible = false; break; }

          chain.push_back(cprod);
          chain_set.insert(cprod);
          if (chain.size() > 5) { feasible = false; break; }

          // Find the one unavailable input to trace further
          size_t next_tensor = SIZE_MAX;
          bool all_ok = true;
          for (size_t pt : p.ops[cprod].inputs) {
            if (dag.tensor_producer[pt] < 0) continue;
            if (ts.inputs.count(pt) || ts.produced.count(pt)) continue;
            if (chain_set.count((size_t)dag.tensor_producer[pt])) continue;
            if (next_tensor == SIZE_MAX)
              next_tensor = pt;
            else { all_ok = false; break; }
          }
          if (!all_ok) { feasible = false; break; }
          if (next_tensor == SIZE_MAX) break;  // all inputs satisfied
          cur_tensor = next_tensor;
        }

        if (!feasible || chain.size() <= 1) continue;

        // Check: each chain op's outputs must have all consumers inside
        // sg ∪ chain (otherwise outputs won't be ephemeral → no benefit,
        // and adding cross-depth ops risks subgraph-level cycles).
        bool chain_ephemeral_ok = true;
        for (size_t cop : chain) {
          for (size_t out_t : p.ops[cop].outputs) {
            for (size_t cons : dag.tensor_consumers[out_t]) {
              if (!sg_set.count(cons) && !chain_set.count(cons)) {
                chain_ephemeral_ok = false;
                break;
              }
            }
            if (!chain_ephemeral_ok) break;
          }
          if (!chain_ephemeral_ok) break;
        }
        if (!chain_ephemeral_ok) continue;

        // Try adding the full chain
        std::vector<size_t> extended2 = sg;
        for (size_t op : chain) extended2.push_back(op);
        auto cfg_old2 = BestGranularity(p, sg, empty_ri, 0, &dag, assume_retain_out);
        auto cfg_new2 = BestGranularity(p, extended2, empty_ri, 0, &dag, assume_retain_out);
        if (cfg_new2.cost < cfg_old2.cost) {
          sg = extended2;
          for (size_t op : chain) sg_set.insert(op);
          improved = true;
          break;
        }
      }
      if (improved) break;
    }
  }
  return part;
}

// ── Convexity repair ────────────────────────────────────────────────────────

Partition ConvexityRepair(const DAG& dag, Partition part) {
  std::vector<std::vector<size_t>> new_sgs;
  for (auto& sg : part.subgraphs) {
    if (sg.size() <= 1 || IsDAGConvex(dag, sg)) {
      new_sgs.push_back(std::move(sg));
    } else {
      // Split into singletons
      for (size_t op : sg)
        new_sgs.push_back({op});
    }
  }
  part.subgraphs = std::move(new_sgs);
  return part;
}

// ── Unfusion pass ────────────────────────────────────────────────────────────

Partition UnfusionPass(const mlsys::Problem& p, const DAG& dag,
                       Partition part, bool assume_retain_out) {
  std::unordered_set<size_t> empty_ri;
  std::vector<std::vector<size_t>> new_subgraphs;

  for (auto& sg : part.subgraphs) {
    if (sg.size() < 2) {
      new_subgraphs.push_back(std::move(sg));
      continue;
    }

    // Quick filter: check for heterogeneous output dims or mixed MM+PW
    auto [maxW, maxH] = OutDims(p, sg);
    bool heterogeneous = false;
    bool has_mm = false, has_pw = false;
    for (size_t op : sg) {
      auto [opW, opH] = OutDims(p, {op});
      if (opW * 2 < maxW || opH * 2 < maxH) heterogeneous = true;
      if (p.ops[op].op_type == "MatMul") has_mm = true;
      else has_pw = true;
    }
    bool mixed_type = has_mm && has_pw;

    if (!heterogeneous && !mixed_type) {
      new_subgraphs.push_back(std::move(sg));
      continue;
    }

    double fused_cost = BestGranularity(p, sg, empty_ri, 0, &dag,
                                        assume_retain_out).cost;

    // Heterogeneity-aware relaxation margin: BestGranularity underestimates
    // fusion cost when ops have very different output dims (unified grid penalty).
    double het_ratio = 1.0;
    if (heterogeneous) {
      int64_t min_area = INT64_MAX;
      for (size_t op : sg) {
        auto [opW, opH] = OutDims(p, {op});
        min_area = std::min(min_area, opW * opH);
      }
      het_ratio = (double)(maxW * maxH) / std::max((int64_t)1, min_area);
    }
    // Conservative: only split when cost model clearly favors it.
    // Margin disabled for now — WS feasibility issues with aggressive splitting.
    double margin = 1.0;
    double effective_fused = fused_cost * margin;

    // Level 1: try splitting into singletons
    double singleton_cost = 0;
    bool all_feasible = true;
    for (size_t op : sg) {
      auto gc = BestGranularity(p, {op}, empty_ri, 0, &dag, assume_retain_out);
      if (gc.cost >= 1e17) { all_feasible = false; break; }
      singleton_cost += gc.cost;
    }

    if (all_feasible && singleton_cost < effective_fused) {
      // Sort ops by their position in dag.topo_order
      std::unordered_map<size_t, int> topo_pos;
      for (int i = 0; i < (int)dag.topo_order.size(); ++i)
        topo_pos[dag.topo_order[i]] = i;
      std::vector<size_t> sorted_ops = sg;
      std::sort(sorted_ops.begin(), sorted_ops.end(),
                [&](size_t a, size_t b) { return topo_pos[a] < topo_pos[b]; });

      std::cerr << "Unfusion: split [";
      for (size_t i = 0; i < sorted_ops.size(); ++i)
        std::cerr << (i ? "," : "") << sorted_ops[i];
      std::cerr << "] saving " << (int)(fused_cost - singleton_cost) << "\n";

      for (size_t op : sorted_ops)
        new_subgraphs.push_back({op});
      continue;
    }

    // Level 2: try 2-way contiguous splits (for 3+ ops)
    if (sg.size() >= 3) {
      // Sort ops topologically
      std::unordered_map<size_t, int> topo_pos;
      for (int i = 0; i < (int)dag.topo_order.size(); ++i)
        topo_pos[dag.topo_order[i]] = i;
      std::vector<size_t> sorted_ops = sg;
      std::sort(sorted_ops.begin(), sorted_ops.end(),
                [&](size_t a, size_t b) { return topo_pos[a] < topo_pos[b]; });

      double best_split_cost = effective_fused;
      int best_cut = -1;
      std::unordered_set<size_t> sg_set(sg.begin(), sg.end());

      for (int cut = 1; cut < (int)sorted_ops.size(); ++cut) {
        // Check DAG validity: no edge from right→left
        std::unordered_set<size_t> left_set(sorted_ops.begin(),
                                            sorted_ops.begin() + cut);
        bool valid = true;
        for (int r = cut; r < (int)sorted_ops.size() && valid; ++r) {
          size_t rop = sorted_ops[r];
          for (size_t t : p.ops[rop].outputs) {
            for (size_t c : dag.tensor_consumers[t]) {
              if (sg_set.count(c) && left_set.count(c)) {
                valid = false;
                break;
              }
            }
            if (!valid) break;
          }
        }
        if (!valid) continue;

        std::vector<size_t> left(sorted_ops.begin(), sorted_ops.begin() + cut);
        std::vector<size_t> right(sorted_ops.begin() + cut, sorted_ops.end());
        auto gc_l = BestGranularity(p, left, empty_ri, 0, &dag, assume_retain_out);
        auto gc_r = BestGranularity(p, right, empty_ri, 0, &dag, assume_retain_out);
        if (gc_l.cost >= 1e17 || gc_r.cost >= 1e17) continue;
        double cost = gc_l.cost + gc_r.cost;
        if (cost < best_split_cost) {
          best_split_cost = cost;
          best_cut = cut;
        }
      }

      if (best_cut >= 0) {
        std::vector<size_t> left(sorted_ops.begin(),
                                 sorted_ops.begin() + best_cut);
        std::vector<size_t> right(sorted_ops.begin() + best_cut,
                                  sorted_ops.end());
        std::cerr << "Unfusion: 2-way split [";
        for (size_t i = 0; i < sorted_ops.size(); ++i)
          std::cerr << (i ? "," : "") << sorted_ops[i];
        std::cerr << "] at " << best_cut << " saving "
                  << (int)(fused_cost - best_split_cost) << "\n";
        new_subgraphs.push_back(std::move(left));
        new_subgraphs.push_back(std::move(right));
        continue;
      }
    }

    // No improvement found, keep fused
    new_subgraphs.push_back(std::move(sg));
  }

  part.subgraphs = std::move(new_subgraphs);
  return part;
}

// ── Retention ────────────────────────────────────────────────────────────────

std::vector<size_t> RetentionCandidates(const mlsys::Problem& p,
                                        const std::vector<size_t>& cur_ops,
                                        const std::vector<size_t>& next_ops,
                                        const DAG* dag) {
  auto ts_cur = Classify(p, cur_ops, dag);
  auto ts_next = Classify(p, next_ops, dag);
  std::vector<size_t> cands;
  // Only output tensors produced by an op can be retained (per competition rules #52).
  // Graph inputs (not produced by any op) cannot be retained.
  for (size_t t : ts_cur.outputs) {
    // Double-check: must be produced by some op in the subgraph
    bool produced_by_op = false;
    for (size_t op : cur_ops) {
      for (size_t out_t : p.ops[op].outputs)
        if (out_t == t) { produced_by_op = true; break; }
      if (produced_by_op) break;
    }
    if (produced_by_op && ts_next.inputs.count(t))
      cands.push_back(t);
  }
  return cands;
}

// ── Subgraph reordering for retention ─────────────────────────────────────────

Partition SubgraphReorder(const mlsys::Problem& p, const DAG& dag,
                          Partition part) {
  int S = (int)part.subgraphs.size();
  if (S <= 2) return part;

  // Map each op to all subgraphs containing it
  std::unordered_map<size_t, std::vector<int>> op_to_sgs;
  for (int i = 0; i < S; ++i)
    for (size_t op : part.subgraphs[i])
      op_to_sgs[op].push_back(i);

  // Build subgraph-level DAG from op-level dependencies
  std::vector<std::unordered_set<int>> sg_succs(S);
  std::vector<int> sg_in_deg(S, 0);
  for (size_t a = 0; a < dag.num_ops; ++a) {
    auto it_a = op_to_sgs.find(a);
    if (it_a == op_to_sgs.end()) continue;
    for (size_t b : dag.op_successors[a]) {
      auto it_b = op_to_sgs.find(b);
      if (it_b == op_to_sgs.end()) continue;
      for (int sg_a : it_a->second)
        for (int sg_b : it_b->second) {
          if (sg_a == sg_b) continue;
          // Skip if sg_b contains op a (recomputation: sg_b recomputes a,
          // doesn't need sg_a's copy)
          bool sg_b_has_a = false;
          for (int s : it_a->second) {
            if (s == sg_b) { sg_b_has_a = true; break; }
          }
          if (sg_b_has_a) continue;
          if (!sg_succs[sg_a].count(sg_b)) {
            sg_succs[sg_a].insert(sg_b);
            ++sg_in_deg[sg_b];
          }
        }
    }
  }

  // Build tensor sharing info for scoring
  std::vector<std::unordered_set<size_t>> sg_prod(S), sg_cons(S);
  for (int i = 0; i < S; ++i) {
    for (size_t op : part.subgraphs[i]) {
      for (size_t t : p.ops[op].outputs) sg_prod[i].insert(t);
      for (size_t t : p.ops[op].inputs) sg_cons[i].insert(t);
    }
  }

  // Greedy topological sort: prefer subgraphs sharing tensors with the last one
  std::vector<int> order;
  order.reserve(S);
  std::vector<bool> scheduled(S, false);

  while ((int)order.size() < S) {
    std::vector<int> ready;
    for (int i = 0; i < S; ++i)
      if (!scheduled[i] && sg_in_deg[i] == 0) ready.push_back(i);
    if (ready.empty()) break;  // cycle detected

    int best = ready[0];
    int64_t best_score = -1;
    if (!order.empty()) {
      int last = order.back();
      for (int i : ready) {
        int64_t score = 0;
        for (size_t t : sg_prod[last])
          if (sg_cons[i].count(t))
            score += p.tensors[t].width * p.tensors[t].height;
        for (size_t t : sg_cons[last])
          if (sg_cons[i].count(t))
            score += p.tensors[t].width * p.tensors[t].height;
        if (score > best_score) {
          best_score = score;
          best = i;
        }
      }
    }

    order.push_back(best);
    scheduled[best] = true;
    for (int j : sg_succs[best])
      --sg_in_deg[j];
  }

  // If cycle detected (can happen after recomputation), keep original order
  if ((int)order.size() < S) {
    std::cerr << "SubgraphReorder: cycle detected, keeping original order\n";
    return part;
  }

  Partition result;
  result.subgraphs.reserve(S);
  for (int i : order)
    result.subgraphs.push_back(std::move(part.subgraphs[i]));
  return result;
}


// ── Stable topological sort (pattern_solver's preferred reorder) ─────────────
//
// Previously inlined in pattern_solver.cc. Now shared so legacy drivers can
// opt into it if their shared-tensor SubgraphReorder heuristic mis-handles
// a given DAG (e.g., attention branches).

Partition StableTopoSort(const DAG& dag, Partition part) {
  const int S = static_cast<int>(part.subgraphs.size());
  if (S <= 1) return part;

  std::unordered_map<size_t, int> op_to_sg;
  for (int i = 0; i < S; ++i)
    for (size_t op : part.subgraphs[i]) op_to_sg[op] = i;

  std::vector<std::vector<int>> sg_succs(S);
  std::vector<int> in_deg(S, 0);
  for (size_t a = 0; a < dag.num_ops; ++a) {
    auto ita = op_to_sg.find(a);
    if (ita == op_to_sg.end()) continue;
    for (size_t b : dag.op_successors[a]) {
      auto itb = op_to_sg.find(b);
      if (itb == op_to_sg.end()) continue;
      if (ita->second == itb->second) continue;
      sg_succs[ita->second].push_back(itb->second);
      ++in_deg[itb->second];
    }
  }

  std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
  for (int i = 0; i < S; ++i)
    if (in_deg[i] == 0) ready.push(i);

  std::vector<int> order;
  order.reserve(S);
  while (!ready.empty()) {
    int i = ready.top();
    ready.pop();
    order.push_back(i);
    for (int j : sg_succs[i])
      if (--in_deg[j] == 0) ready.push(j);
  }

  // Cycle detected: return original partition unchanged.
  if (static_cast<int>(order.size()) != S) return part;

  Partition reordered;
  reordered.subgraphs.reserve(S);
  for (int i : order)
    reordered.subgraphs.push_back(std::move(part.subgraphs[i]));
  return reordered;
}

}  // namespace solver
