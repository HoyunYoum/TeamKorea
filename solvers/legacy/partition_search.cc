// partition_search.cc — M4a SA framework implementation.

#include "partition_search.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace partition {

// ── SAPartition construction & conversion ────────────────────────────────────

SAPartition SAPartition::FromPartition(const solver::Partition& part,
                                        const solver::DAG& dag) {
  SAPartition sp;
  const int N = (int)dag.num_ops;
  sp.label.assign(N, -1);

  // Cache reverse adjacency once (DAG only provides op_successors).
  sp.op_predecessors.assign(N, {});
  for (int u = 0; u < N; ++u)
    for (size_t v : dag.op_successors[u])
      sp.op_predecessors[v].push_back((size_t)u);

  // Re-index to skip empty subgraphs.
  for (const auto& ops : part.subgraphs) {
    if (ops.empty()) continue;
    int sg = (int)sp.members.size();
    sp.members.emplace_back();
    for (size_t op : ops) {
      sp.label[op] = sg;
      sp.members.back().insert(op);
    }
  }
  sp.boundary.assign(sp.members.size(), {});
  for (int s = 0; s < (int)sp.members.size(); ++s)
    sp.RebuildBoundary(s, dag);
  return sp;
}

solver::Partition SAPartition::ToPartition() const {
  solver::Partition part;
  for (const auto& m : members) {
    if (m.empty()) continue;
    std::vector<size_t> ops(m.begin(), m.end());
    std::sort(ops.begin(), ops.end());
    part.subgraphs.push_back(std::move(ops));
  }
  return part;
}

void SAPartition::RebuildBoundary(int sg, const solver::DAG& dag) {
  boundary[sg].clear();
  for (size_t op : members[sg]) {
    RebuildBoundaryFor(op, dag);
  }
}

void SAPartition::RebuildBoundaryFor(size_t op, const solver::DAG& dag) {
  const int sg = label[op];
  bool on_boundary = false;
  for (size_t succ : dag.op_successors[op])
    if (label[succ] != sg) { on_boundary = true; break; }
  if (!on_boundary) {
    for (size_t pred : op_predecessors[op])
      if (label[pred] != sg) { on_boundary = true; break; }
  }
  if (on_boundary) boundary[sg].insert(op);
  else boundary[sg].erase(op);
}

// ── SAPartition Move/Merge/Split ─────────────────────────────────────────────

bool SAPartition::Move(size_t v, int target_sg, const solver::DAG& dag) {
  const int from = label[v];
  if (from == target_sg) return false;
  if ((int)members.size() <= target_sg || target_sg < 0) return false;
  if (members[from].size() <= 1) return false;  // would empty `from`

  // Convexity check: removing v from `from` keeps it convex, and adding v
  // to `target_sg` keeps it convex.
  std::vector<size_t> from_after(members[from].begin(), members[from].end());
  from_after.erase(std::remove(from_after.begin(), from_after.end(), v),
                   from_after.end());
  std::sort(from_after.begin(), from_after.end());
  if (from_after.size() >= 2 && !solver::IsDAGConvex(dag, from_after))
    return false;

  std::vector<size_t> to_after(members[target_sg].begin(),
                                members[target_sg].end());
  to_after.push_back(v);
  std::sort(to_after.begin(), to_after.end());
  if (to_after.size() >= 2 && !solver::IsDAGConvex(dag, to_after))
    return false;

  // Apply move.
  members[from].erase(v);
  members[target_sg].insert(v);
  label[v] = target_sg;

  // Incremental boundary update: v and its DAG neighbors in from/target_sg
  // may have changed boundary status.
  boundary[from].erase(v);
  RebuildBoundaryFor(v, dag);
  for (size_t nb : dag.op_successors[v]) RebuildBoundaryFor(nb, dag);
  for (size_t nb : op_predecessors[v]) RebuildBoundaryFor(nb, dag);

  // Some former-neighbors of v in `from` may lose boundary status.
  for (size_t op : members[from]) RebuildBoundaryFor(op, dag);
  return true;
}

bool SAPartition::Merge(int a, int b, const solver::DAG& dag) {
  if (a == b) return false;
  if (a < 0 || b < 0) return false;
  if (a >= (int)members.size() || b >= (int)members.size()) return false;
  if (members[a].empty() || members[b].empty()) return false;

  // Check subgraph-level acyclicity: merging a and b must not create a cycle
  // (there's no 3rd-party subgraph c with a→c→b or b→c→a).
  // Light check: construct merged ops and test convexity directly.
  std::vector<size_t> merged;
  merged.reserve(members[a].size() + members[b].size());
  for (size_t op : members[a]) merged.push_back(op);
  for (size_t op : members[b]) merged.push_back(op);
  std::sort(merged.begin(), merged.end());
  if (merged.size() >= 2 && !solver::IsDAGConvex(dag, merged))
    return false;

  // Commit.
  for (size_t op : members[b]) {
    label[op] = a;
    members[a].insert(op);
  }
  members[b].clear();
  boundary[b].clear();
  RebuildBoundary(a, dag);

  // Note: we DO NOT re-index subgraphs here (b becomes empty). ToPartition()
  // skips empties. Keeping the index stable simplifies bookkeeping; SA may
  // repopulate via Split later.
  return true;
}

bool SAPartition::Split(int s,
                        const std::unordered_set<size_t>& side_a,
                        const std::unordered_set<size_t>& side_b,
                        const solver::DAG& dag) {
  if (s < 0 || s >= (int)members.size()) return false;
  if (side_a.empty() || side_b.empty()) return false;
  if (side_a.size() + side_b.size() != members[s].size()) return false;
  for (size_t op : side_a) if (!members[s].count(op)) return false;
  for (size_t op : side_b) if (!members[s].count(op)) return false;

  // Convexity of both halves.
  std::vector<size_t> va(side_a.begin(), side_a.end()); std::sort(va.begin(), va.end());
  std::vector<size_t> vb(side_b.begin(), side_b.end()); std::sort(vb.begin(), vb.end());
  if (va.size() >= 2 && !solver::IsDAGConvex(dag, va)) return false;
  if (vb.size() >= 2 && !solver::IsDAGConvex(dag, vb)) return false;

  // Allocate a fresh subgraph id or reuse an empty one.
  int new_sg = -1;
  for (int i = 0; i < (int)members.size(); ++i)
    if (members[i].empty()) { new_sg = i; break; }
  if (new_sg < 0) {
    new_sg = (int)members.size();
    members.emplace_back();
    boundary.emplace_back();
  }

  members[s] = side_a;
  members[new_sg] = side_b;
  for (size_t op : side_b) label[op] = new_sg;

  RebuildBoundary(s, dag);
  RebuildBoundary(new_sg, dag);
  return true;
}

// ── Move proposal helpers ────────────────────────────────────────────────────

namespace {

struct Rng {
  std::mt19937 gen;
  explicit Rng(uint32_t seed) : gen(seed) {}
  int RandInt(int lo, int hi) {  // inclusive
    std::uniform_int_distribution<int> d(lo, hi);
    return d(gen);
  }
  double Rand01() {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    return d(gen);
  }
  template <typename T>
  const T& Pick(const std::vector<T>& v) { return v[RandInt(0, (int)v.size()-1)]; }
};

// Compute neighbor subgraphs of a given op: subgraphs that share a DAG edge
// with v (either v's successor or v's predecessor is in that subgraph, and
// that subgraph != v's current subgraph).
std::vector<int> NeighborSubgraphs(const SAPartition& sp, size_t v,
                                    const solver::DAG& dag) {
  const int sg = sp.label[v];
  std::unordered_set<int> nbs;
  for (size_t succ : dag.op_successors[v]) {
    int sg2 = sp.label[succ];
    if (sg2 != sg) nbs.insert(sg2);
  }
  for (size_t pred : sp.op_predecessors[v]) {
    int sg2 = sp.label[pred];
    if (sg2 != sg) nbs.insert(sg2);
  }
  return {nbs.begin(), nbs.end()};
}

// Propose a Move on a copy of sp. Returns (new_sp, success).
// Picks a random boundary op from a random non-empty subgraph, and moves it
// to a random neighbor subgraph.
bool ProposeMove(SAPartition* sp, const solver::DAG& dag, Rng* rng) {
  // Find subgraphs with non-empty boundary and size >= 2.
  std::vector<int> candidates;
  for (int s = 0; s < sp->num_subgraphs(); ++s)
    if (sp->members[s].size() >= 2 && !sp->boundary[s].empty())
      candidates.push_back(s);
  if (candidates.empty()) return false;

  int sg = rng->Pick(candidates);
  std::vector<size_t> bvec(sp->boundary[sg].begin(), sp->boundary[sg].end());
  if (bvec.empty()) return false;
  size_t v = rng->Pick(bvec);

  auto nbs = NeighborSubgraphs(*sp, v, dag);
  if (nbs.empty()) return false;
  int target = rng->Pick(nbs);

  return sp->Move(v, target, dag);
}

// Propose a Merge: pick two adjacent subgraphs (with a shared tensor edge)
// and merge them.
bool ProposeMerge(SAPartition* sp, const solver::DAG& dag, Rng* rng) {
  // Collect pairs (a, b) with at least one cross-subgraph DAG edge.
  std::vector<std::pair<int,int>> pairs;
  std::unordered_set<int64_t> seen;
  for (int s = 0; s < sp->num_subgraphs(); ++s) {
    if (sp->members[s].empty()) continue;
    for (size_t op : sp->members[s]) {
      for (size_t succ : dag.op_successors[op]) {
        int t = sp->label[succ];
        if (t == s) continue;
        int64_t key = (int64_t)std::min(s, t) * 1000000 + std::max(s, t);
        if (seen.insert(key).second) pairs.push_back({std::min(s,t), std::max(s,t)});
      }
    }
  }
  if (pairs.empty()) return false;

  auto pair = rng->Pick(pairs);
  return sp->Merge(pair.first, pair.second, dag);
}

// Propose a WS-guided Split: pick a subgraph with WS pressure (heuristic: larger
// op counts are candidates), pick the largest non-ephemeral tensor as cut.
// Falls back to random split if WS-guided path fails.
bool ProposeSplit(const mlsys::Problem& p, SAPartition* sp,
                  const solver::DAG& dag, Rng* rng, bool ws_guided) {
  // Candidates: subgraphs with ≥ 2 ops.
  std::vector<int> cands;
  for (int s = 0; s < sp->num_subgraphs(); ++s)
    if (sp->members[s].size() >= 2) cands.push_back(s);
  if (cands.empty()) return false;

  int sg = rng->Pick(cands);
  std::vector<size_t> ops(sp->members[sg].begin(), sp->members[sg].end());
  std::sort(ops.begin(), ops.end());
  const int n = (int)ops.size();

  // Build subgraph-internal adjacency.
  std::unordered_set<size_t> op_set(ops.begin(), ops.end());

  // Select cut edge.
  // WS-guided: pick edge with largest transfer tensor size.
  // Random: pick any internal edge at random.
  struct Edge { size_t u, v; int64_t size; };
  std::vector<Edge> edges;
  for (size_t u : ops) {
    for (size_t succ : dag.op_successors[u]) {
      if (!op_set.count(succ)) continue;
      // Tensor transferred u→succ is one of u's outputs that's in succ's inputs.
      int64_t tsize = 0;
      for (size_t t : p.ops[u].outputs) {
        bool consumed = false;
        for (size_t in : p.ops[succ].inputs) if (in == t) { consumed = true; break; }
        if (consumed)
          tsize = std::max(tsize, (int64_t)p.tensors[t].width * p.tensors[t].height);
      }
      edges.push_back({u, succ, tsize});
    }
  }
  if (edges.empty()) return false;

  Edge cut;
  if (ws_guided) {
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.size > b.size; });
    cut = edges.front();
  } else {
    cut = rng->Pick(edges);
  }

  // BFS from `cut.u`'s side (producer side) in reverse topo direction to
  // collect side_a (all ops not strictly-after cut.v).
  // We want: side_a contains cut.u and all its sub-ancestors within op_set.
  // side_b contains cut.v and all its sub-descendants.
  // Remaining ops: assign to the side from which they're reachable.
  // Simplest correct approach: 2-color via DAG reachability inside op_set.
  std::unordered_set<size_t> side_b;
  {
    std::queue<size_t> q; q.push(cut.v); side_b.insert(cut.v);
    while (!q.empty()) {
      size_t x = q.front(); q.pop();
      for (size_t s : dag.op_successors[x]) {
        if (op_set.count(s) && !side_b.count(s)) {
          side_b.insert(s);
          q.push(s);
        }
      }
    }
  }
  std::unordered_set<size_t> side_a;
  for (size_t op : ops) if (!side_b.count(op)) side_a.insert(op);

  if (side_a.empty() || side_b.empty()) return false;
  return sp->Split(sg, side_a, side_b, dag);
}

// Initial temperature heuristic: stddev of feasible costs across a few
// perturbations of the seed. Uses shared cache to avoid eating SA budget
// on uncached evaluations (especially costly on large partitions).
double AutoInitialTemp(const mlsys::Problem& p,
                       const solver::DAG& dag,
                       const solver::Partition& seed,
                       Rng* rng,
                       OptimalGranCache* cache) {
  auto eval_seed = EvaluatePartitionCached(p, dag, seed, cache,
                                            /*compute_baseline=*/false);
  if (!eval_seed.feasible) return 1.0;
  std::vector<double> costs = {eval_seed.cost};

  for (int attempt = 0; attempt < 10; ++attempt) {
    auto sp = SAPartition::FromPartition(seed, dag);
    bool moved = false;
    int op = rng->RandInt(0, 2);
    if (op == 0) moved = ProposeMove(&sp, dag, rng);
    else if (op == 1) moved = ProposeMerge(&sp, dag, rng);
    else moved = ProposeSplit(p, &sp, dag, rng, /*ws_guided=*/rng->Rand01() < 0.5);
    if (!moved) continue;
    auto part = sp.ToPartition();
    auto e = EvaluatePartitionCached(p, dag, part, cache,
                                      /*compute_baseline=*/false);
    if (e.feasible) costs.push_back(e.cost);
  }

  if (costs.size() < 2) return eval_seed.cost * 0.01;
  double mean = 0; for (double c : costs) mean += c; mean /= costs.size();
  double var = 0; for (double c : costs) var += (c - mean) * (c - mean);
  var /= costs.size();
  return std::max(std::sqrt(var) * 2.0, eval_seed.cost * 0.005);
}

}  // namespace

// ── Main SA loop ─────────────────────────────────────────────────────────────

SAResult RunSA(const mlsys::Problem& p,
               const solver::DAG& dag,
               const solver::Partition& seed,
               const SAConfig& cfg) {
  using clock = std::chrono::high_resolution_clock;
  SAResult res;

  auto t0 = clock::now();
  auto deadline = t0 + std::chrono::duration<double>(cfg.time_budget_s);

  // Cache shared across all iterations within this SA run — dominant speedup
  // on repeated subgraph queries.
  OptimalGranCache cache;

  // Evaluate seed.
  auto seed_eval = EvaluatePartitionCached(p, dag, seed, &cache,
                                            /*compute_baseline=*/false);
  if (!seed_eval.feasible) {
    res.best_eval = seed_eval;
    res.wall_time_s = 0;
    return res;
  }
  res.best_partition = seed;
  res.best_eval = seed_eval;
  res.seed_cost = seed_eval.cost;

  Rng rng(cfg.rng_seed);

  // Current state.
  auto cur_sp = SAPartition::FromPartition(seed, dag);
  double cur_cost = seed_eval.cost;

  // Temperature schedule.
  double T_init = cfg.initial_temp > 0
                  ? cfg.initial_temp
                  : AutoInitialTemp(p, dag, seed, &rng, &cache);
  double T_final = T_init * cfg.final_temp_ratio;

  // Cooling factor set up to reach T_final in max_iterations.
  int planned_iters = std::max(100, cfg.max_iterations);
  double cooling = std::pow(T_final / std::max(T_init, 1e-12),
                             1.0 / planned_iters);

  // Auto-adjust move weights by average subgraph size.
  double p_mv = cfg.p_move, p_mg = cfg.p_merge, p_sp = cfg.p_split;
  {
    double avg = (double)cur_sp.num_ops() /
                 std::max(1, cur_sp.num_subgraphs());
    if (avg > 5) { p_mv = 0.3; p_mg = 0.2; p_sp = 0.5; }
    else if (avg < 2) { p_mv = 0.2; p_mg = 0.6; p_sp = 0.2; }
  }
  double p_sum = p_mv + p_mg + p_sp;

  int last_improve_iter = 0;
  double T = T_init;

  for (int iter = 0; iter < cfg.max_iterations; ++iter) {
    if (clock::now() > deadline) break;
    if (iter - last_improve_iter > cfg.no_improvement_patience) break;

    // Sample move type.
    double r = rng.Rand01() * p_sum;
    int mv_type;
    if (r < p_mv) mv_type = 0;
    else if (r < p_mv + p_mg) mv_type = 1;
    else mv_type = 2;

    // Propose on a COPY (for rollback on reject).
    auto trial_sp = cur_sp;
    bool proposed = false;
    if (mv_type == 0) proposed = ProposeMove(&trial_sp, dag, &rng);
    else if (mv_type == 1) proposed = ProposeMerge(&trial_sp, dag, &rng);
    else {
      bool ws_guided = rng.Rand01() < cfg.ws_split_frac;
      proposed = ProposeSplit(p, &trial_sp, dag, &rng, ws_guided);
    }

    ++res.move_attempts[mv_type];
    if (!proposed) {
      ++res.move_constraint_rejects[mv_type];
      T *= cooling;
      continue;
    }

    // Evaluate full cost (cached path, no baseline for speed).
    auto trial_part = trial_sp.ToPartition();
    auto eval = EvaluatePartitionCached(p, dag, trial_part, &cache,
                                         /*compute_baseline=*/false);
    if (!eval.feasible) {
      ++res.move_constraint_rejects[mv_type];
      T *= cooling;
      continue;
    }

    double delta = eval.cost - cur_cost;
    bool accept = delta < 0 || (T > 0 && rng.Rand01() < std::exp(-delta / T));
    if (!accept) {
      ++res.move_cost_rejects[mv_type];
      T *= cooling;
      continue;
    }

    ++res.move_accepts[mv_type];
    cur_sp = std::move(trial_sp);
    cur_cost = eval.cost;

    if (cur_cost < res.best_eval.cost) {
      res.best_partition = std::move(trial_part);
      res.best_eval = std::move(eval);
      last_improve_iter = iter;
    }

    T *= cooling;
    res.iterations_done = iter + 1;
  }

  res.wall_time_s = std::chrono::duration<double>(clock::now() - t0).count();
  res.improved_over_seed = res.best_eval.cost < res.seed_cost - 1e-6;
  // Stash cache diagnostics onto the result via stderr log (keeps SAResult
  // minimal). Also report avg iteration time + cache miss per iter — useful
  // for judging whether delta evaluation would help.
  double ms_per_iter = res.iterations_done > 0
      ? 1000.0 * res.wall_time_s / res.iterations_done : 0.0;
  double misses_per_iter = res.iterations_done > 0
      ? (double)cache.misses / res.iterations_done : 0.0;
  std::fprintf(stderr,
               "    [cache] hits=%lld misses=%lld hit_rate=%.1f%%"
               "  ms/iter=%.2f  miss/iter=%.2f\n",
               (long long)cache.hits, (long long)cache.misses,
               cache.hit_rate() * 100.0, ms_per_iter, misses_per_iter);
  // Ensure returned partition is in execution order so callers can pass it
  // straight to solver::BuildSolution (retention decisions rely on list
  // order). This removes a class of bugs where callers forget to permute.
  if (res.best_eval.feasible)
    res.best_partition = ApplyOrderingToPartition(res.best_partition,
                                                   res.best_eval);
  return res;
}

// ── Multi-seed driver ────────────────────────────────────────────────────────

MultiSeedResult RunMultiSeedSA(const mlsys::Problem& p,
                                const solver::DAG& dag,
                                const std::vector<solver::Partition>& seeds,
                                double total_budget_s,
                                uint32_t base_seed) {
  using clock = std::chrono::high_resolution_clock;
  MultiSeedResult ms;
  ms.best_eval.cost = std::numeric_limits<double>::infinity();
  if (seeds.empty()) return ms;

  auto t0 = clock::now();
  double remaining = total_budget_s;

  for (size_t i = 0; i < seeds.size(); ++i) {
    if (remaining <= 0) break;
    double seed_budget = remaining / (seeds.size() - i);

    SAConfig cfg;
    cfg.time_budget_s = seed_budget;
    cfg.rng_seed = base_seed + (uint32_t)i * 1013;

    auto sa = RunSA(p, dag, seeds[i], cfg);
    remaining -= sa.wall_time_s;

    if (sa.best_eval.feasible && sa.best_eval.cost < ms.best_eval.cost) {
      ms.best_partition = sa.best_partition;
      ms.best_eval = sa.best_eval;
    }
    ms.per_seed.push_back(std::move(sa));
  }

  // ── Milestone 5: Recomputation post-pass on final best ───────────────
  if (ms.best_eval.feasible) {
    OptimalGranCache rc_cache;
    auto rc = ApplyRecomputation(p, dag, ms.best_partition, ms.best_eval,
                                  &rc_cache);
    if (rc.applied) {
      std::fprintf(stderr, "  [recompute] improved %.0f → %.0f\n",
                   ms.best_eval.cost, rc.evaluation.cost);
      ms.best_partition = std::move(rc.partition);
      ms.best_eval = std::move(rc.evaluation);
    }
  }

  ms.total_wall_time_s = std::chrono::duration<double>(clock::now() - t0).count();
  return ms;
}

}  // namespace partition
