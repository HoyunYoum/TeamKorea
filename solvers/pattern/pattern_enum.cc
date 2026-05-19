// pattern_enum.cc — Implementation of §8-based column enumeration.

#include "pattern_enum.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace pattern {

namespace {

// Helper: for a given candidate subgraph `ops`, run BestGranularity and
// build a Column. Returns infeasible Column (cost=inf) if WS-violating.
Column BuildColumn(const mlsys::Problem& p,
                   const solver::DAG& dag,
                   std::vector<size_t> ops,
                   PatternClass klass) {
  Column c;
  c.ops = std::move(ops);
  c.klass = klass;
  std::unordered_set<size_t> empty_retain;
  // assume_retain_out=false so column cost INCLUDES eviction of non-ephemeral
  // outputs — required for meaningful cost-per-op comparison in greedy pack
  // (§8 fusion benefit surfaces only when eviction is counted).
  auto cfg = solver::BestGranularity(p, c.ops, empty_retain, /*retained_size=*/0,
                                      &dag, /*assume_retain_out=*/false);
  c.gran = cfg.gran;
  c.traversal = cfg.traversal;
  c.analytic_cost = cfg.cost;
  // BestGranularity returns gran={0,0,0} with cost=inf when infeasible.
  bool feasible = std::isfinite(cfg.cost) &&
                  cfg.gran.width > 0 && cfg.gran.height > 0 && cfg.gran.depth > 0;
  if (feasible) {
    // ExactCost (Evaluate-faithful) at the gran that BestGranularity picked
    // under AnalyticalCost. Stored alongside analytic_cost so the driver
    // can run the MIP twice (once per cost metric) and pick the better GT
    // — the two rank patterns differently for heterogeneous C5 sgs, and
    // retention interactions sometimes favour the AnalyticalCost ranking.
    const std::vector<int64_t>* trav =
        c.traversal.empty() ? nullptr : &c.traversal;
    c.cost = solver::ExactCost(p, c.ops, c.gran, /*retain_out=*/{},
                                empty_retain, trav, &dag);
    c.ws = solver::WorkingSet(p, c.ops, c.gran, /*extra=*/0, &dag);

    // Third cost variant: α-retention — AnalyticalCost under retention-
    // optimistic gran (assume_retain_out=true drops all eviction, same as
    // legacy R) plus a residual-eviction add-back for outputs whose
    // retention is structurally uncertain. For each non-ephemeral output t:
    //   t has 1 external consumer:  α = 1.0 (single-hop retention likely,
    //                                        no add-back)
    //   t has 2 external consumers: α = 0.5 (half evict may still happen)
    //   t has 3+ external consumers: α = 0.2 (retention rarely covers all)
    //
    // cost_retain = cfg_r.cost + Σ (1 − α(t)) × evict(t) / bw
    //
    // α=1 reduces to the legacy R (full retention assumed); structural
    // α < 1 penalises the partition for fan-out cases where RetentionPass
    // cannot cover every consumer, biasing MIP away from "retention-
    // optimistic" partitions whose retention payout the simulator won't
    // actually deliver.
    {
      auto cfg_r = solver::BestGranularity(p, c.ops, empty_retain, 0, &dag,
                                            /*assume_retain_out=*/true);
      if (std::isfinite(cfg_r.cost)) {
        double bw = p.slow_memory_bandwidth;
        std::unordered_set<size_t> ops_set(c.ops.begin(), c.ops.end());
        double residual = 0;
        for (size_t op : c.ops) {
          if (op >= p.ops.size()) continue;
          for (size_t t : p.ops[op].outputs) {
            if (t >= dag.tensor_consumers.size()) continue;
            int n_ext = 0;
            for (size_t cons : dag.tensor_consumers[t]) {
              if (!ops_set.count(cons)) ++n_ext;
            }
            if (n_ext == 0) continue;  // ephemeral — no evict in the Analytical
            double alpha = (n_ext == 1) ? 1.0 : (n_ext == 2 ? 0.5 : 0.2);
            double evict = (double)p.tensors[t].width *
                           (double)p.tensors[t].height / bw;
            residual += (1.0 - alpha) * evict;
          }
        }
        c.cost_retain = cfg_r.cost + residual;
      }
    }
  } else {
    c.cost = cfg.cost;  // kInf
  }
  c.feasible = feasible && std::isfinite(c.cost);
  return c;
}

// Topological sort a subgraph's ops according to DAG order. Stable on
// equal positions; uses dag.topo_order as the canonical sequence.
void TopoSortSubgraph(const solver::DAG& dag, std::vector<size_t>& ops) {
  std::unordered_map<size_t, size_t> topo_pos;
  for (size_t i = 0; i < dag.topo_order.size(); ++i)
    topo_pos[dag.topo_order[i]] = i;
  std::sort(ops.begin(), ops.end(), [&](size_t a, size_t b) {
    return topo_pos[a] < topo_pos[b];
  });
}

}  // namespace

std::vector<Column> EnumerateColumns(const mlsys::Problem& p,
                                      const solver::DAG& dag) {
  std::vector<Column> out;

  // ── C0 + C1: singleton columns for every op ─────────────────────────────
  // Always feasible (single op fits within capacity unless the op itself is
  // pathological). Guarantees the greedy pack has a fallback.
  for (size_t op = 0; op < p.ops.size(); ++op) {
    PatternClass klass = (p.ops[op].op_type == "MatMul")
                             ? PatternClass::StandaloneMM
                             : PatternClass::StandalonePW;
    Column c = BuildColumn(p, dag, {op}, klass);
    if (c.feasible) out.push_back(std::move(c));
  }

  // ── Adjacent 2-pairs: C3 / C4 / C6 / PW-PW ─────────────────────────────
  // For each op A whose output has exactly ONE consumer B (F1 fan-out rule),
  // propose column {A, B}. Classify by op types + input position:
  //   MM→MM (A's out at B's LHS pos=0)    → LhsChain2 (C3)
  //   MM→MM (A's out at B's RHS pos>0)    → RhsChain2 (C4)
  //   MM→PW or PW→MM                      → PwMmKeqK (C6; valid at k=K)
  //   PW→PW                               → StandalonePW class (bundle cost)
  // BestGranularity internally applies the role-based fusion guards
  // (split_k_infeasible / uniform_k_required) so invalid pattern choices
  // come back with cost=inf and get filtered out.
  for (size_t A = 0; A < p.ops.size(); ++A) {
    if (p.ops[A].outputs.empty()) continue;
    size_t inter = p.ops[A].outputs[0];
    if (inter >= dag.tensor_consumers.size()) continue;
    // Dedup consumers: the DAG may list the same consuming op twice if its
    // inputs repeat a tensor (e.g., T @ T). F1 is about UNIQUE consumers.
    std::unordered_set<size_t> uc(dag.tensor_consumers[inter].begin(),
                                   dag.tensor_consumers[inter].end());
    if (uc.size() != 1) continue;  // F1 rule (single unique consumer)
    size_t B = *uc.begin();
    if (B == A) continue;
    // Classify.
    bool A_mm = p.ops[A].op_type == "MatMul";
    bool B_mm = p.ops[B].op_type == "MatMul";
    PatternClass klass = PatternClass::StandalonePW;  // default (PW-PW)
    if (A_mm && B_mm) {
      int pos = -1;
      for (size_t i = 0; i < p.ops[B].inputs.size(); ++i)
        if (p.ops[B].inputs[i] == inter) { pos = (int)i; break; }
      if (pos < 0) continue;
      klass = (pos == 0) ? PatternClass::LhsChain2 : PatternClass::RhsChain2;
    } else if (A_mm != B_mm) {
      klass = PatternClass::PwMmKeqK;
    }
    std::vector<size_t> ops{A, B};
    TopoSortSubgraph(dag, ops);
    Column c = BuildColumn(p, dag, std::move(ops), klass);
    if (c.feasible) out.push_back(std::move(c));
  }

  // ── Fork 3-op: A → {B, C} with A's output consumed by exactly 2 UNIQUE ops ─
  // Covers diamond (Ex3C-style: Op0 → Op1, Op0 → Op2) and parallel branches.
  // F1 rule satisfied trivially since all 2 consumers live in the subgraph.
  // B and C may have independent additional inputs (loaded from slow memory)
  // — only the shared A-output is ephemeral in the subgraph.
  // Dedup: T @ T patterns make a single downstream op appear twice in the
  // consumer list; treat such cases as single-consumer (handled by 2-pair).
  for (size_t A = 0; A < p.ops.size(); ++A) {
    if (p.ops[A].outputs.empty()) continue;
    size_t inter = p.ops[A].outputs[0];
    if (inter >= dag.tensor_consumers.size()) continue;
    std::unordered_set<size_t> uc(dag.tensor_consumers[inter].begin(),
                                   dag.tensor_consumers[inter].end());
    if (uc.size() != 2) continue;
    auto it = uc.begin();
    size_t B = *it++;
    size_t C = *it;
    if (B == A || C == A || B == C) continue;
    std::vector<size_t> ops{A, B, C};
    if (!solver::IsDAGConvex(dag, ops)) continue;
    TopoSortSubgraph(dag, ops);
    Column col = BuildColumn(p, dag, std::move(ops), PatternClass::F2aSingle);
    if (col.feasible) out.push_back(std::move(col));
  }

  // ── Branch-pair 2-op: parallel MMs sharing at least one input ─────────
  // Attention-critical: Q/K/V projections all consume x; fusing two of them
  // loads x once. Unified's Strategy D. Enumerate by shared input, not by
  // producer-consumer edge (which the linear 2-pair above handles).
  // Skip when A-B have producer-consumer relation (already covered by C3/C4).
  {
    std::unordered_map<size_t, std::vector<size_t>> input_consumers;
    for (size_t op = 0; op < p.ops.size(); ++op) {
      if (p.ops[op].op_type != "MatMul") continue;
      std::unordered_set<size_t> seen;
      for (size_t t : p.ops[op].inputs) {
        if (seen.count(t)) continue;
        seen.insert(t);
        input_consumers[t].push_back(op);
      }
    }
    std::unordered_set<uint64_t> done;  // dedupe pair keys
    for (auto& [t, cons] : input_consumers) {
      if (cons.size() < 2) continue;
      for (size_t i = 0; i + 1 < cons.size(); ++i) {
        for (size_t j = i + 1; j < cons.size(); ++j) {
          size_t A = std::min(cons[i], cons[j]);
          size_t B = std::max(cons[i], cons[j]);
          uint64_t key = (uint64_t)A << 32 | B;
          if (done.count(key)) continue;
          done.insert(key);
          // Skip if producer-consumer (already covered elsewhere).
          bool is_chain = false;
          for (size_t out_t : p.ops[A].outputs)
            for (size_t in_t : p.ops[B].inputs)
              if (out_t == in_t) { is_chain = true; break; }
          if (is_chain) continue;
          std::vector<size_t> ops{A, B};
          if (!solver::IsDAGConvex(dag, ops)) continue;
          TopoSortSubgraph(dag, ops);
          Column c = BuildColumn(p, dag, std::move(ops), PatternClass::F2aSingle);
          if (c.feasible) out.push_back(std::move(c));
        }
      }
    }
  }

  // ── N-way graph-input fan-out: 3+ consumers of a raw graph input ──────
  // Graph inputs cannot be retained (#52); fusing all (or many) consumers
  // is the only way to avoid per-subgraph reload. Bucket consumers by
  //   (out_W, out_H, K_op, role_of_shared_input)
  // so columns are granularity-compatible up front (unified-grid #28 +
  // uniform-K). BestGranularity still validates final capacity. Enumerate
  // sizes 3..min(|bucket|, cap) as sorted prefixes; full bucket dominates
  // the IO savings, prefixes let the MIP pick smaller packings when a
  // larger group OOMs.
  {
    constexpr size_t kMaxFanoutBucket = 8;
    struct BKey {
      int64_t ow, oh, k_op;
      int role;  // 0=LHS, 1=RHS, 2=PW/other
      bool operator==(const BKey& o) const {
        return ow == o.ow && oh == o.oh && k_op == o.k_op && role == o.role;
      }
    };
    struct BKeyHash {
      size_t operator()(const BKey& k) const {
        return std::hash<int64_t>()(k.ow) ^ (std::hash<int64_t>()(k.oh) << 1)
             ^ (std::hash<int64_t>()(k.k_op) << 2)
             ^ (std::hash<int>()(k.role) << 3);
      }
    };
    for (size_t t = 0; t < dag.num_tensors; ++t) {
      if ((size_t)t >= dag.tensor_producer.size()) break;
      if (dag.tensor_producer[t] >= 0) continue;  // only raw graph inputs
      if ((size_t)t >= dag.tensor_consumers.size()) continue;
      std::unordered_set<size_t> uniq(dag.tensor_consumers[t].begin(),
                                       dag.tensor_consumers[t].end());
      if (uniq.size() < 3) continue;  // pair case already covered above

      std::unordered_map<BKey, std::vector<size_t>, BKeyHash> buckets;
      for (size_t op : uniq) {
        if (op >= p.ops.size()) continue;
        const auto& o = p.ops[op];
        if (o.outputs.empty()) continue;
        int role = 2;
        int64_t k_op = 0;
        if (o.op_type == "MatMul") {
          for (size_t i = 0; i < o.inputs.size(); ++i) {
            if (o.inputs[i] == t) { role = (i == 0) ? 0 : 1; break; }
          }
          if (!o.inputs.empty())
            k_op = p.tensors[o.inputs[0]].width;
        }
        int64_t ow = p.tensors[o.outputs[0]].width;
        int64_t oh = p.tensors[o.outputs[0]].height;
        buckets[{ow, oh, k_op, role}].push_back(op);
      }

      for (auto& [key, bucket] : buckets) {
        if (bucket.size() < 3) continue;
        std::sort(bucket.begin(), bucket.end());
        size_t cap = std::min(bucket.size(), kMaxFanoutBucket);
        // Emit sizes 3..cap as sorted prefixes.
        for (size_t sz = 3; sz <= cap; ++sz) {
          std::vector<size_t> ops(bucket.begin(), bucket.begin() + sz);
          if (!solver::IsDAGConvex(dag, ops)) continue;
          TopoSortSubgraph(dag, ops);
          Column c = BuildColumn(p, dag, std::move(ops),
                                  PatternClass::F2aSingle);
          if (c.feasible) out.push_back(std::move(c));
        }
      }
    }
  }

  // ── Merge 3-op: {P_L, P_R, C} where P_L→C.LHS, P_R→C.RHS (F2a both-side) ─
  // C is a MatMul whose both inputs come from op outputs, each single-
  // consumer = C. BestGranularity guards enforce uniform-K = k=K; almost
  // always OOM on attention BMs but enumerate for topological completeness.
  for (size_t C = 0; C < p.ops.size(); ++C) {
    if (p.ops[C].op_type != "MatMul") continue;
    if (p.ops[C].inputs.size() < 2) continue;
    std::vector<size_t> producers;
    bool ok = true;
    for (size_t pos = 0; pos < p.ops[C].inputs.size() && ok; ++pos) {
      size_t t = p.ops[C].inputs[pos];
      if (t >= dag.tensor_producer.size()) { ok = false; break; }
      int prod = dag.tensor_producer[t];
      if (prod < 0) continue;  // graph input — single-side cases already covered by 2-pair
      if (t >= dag.tensor_consumers.size()) { ok = false; break; }
      if (dag.tensor_consumers[t].size() != 1) continue;  // not single-consumer, skip this side
      producers.push_back(static_cast<size_t>(prod));
    }
    if (!ok || producers.size() < 2) continue;
    std::sort(producers.begin(), producers.end());
    producers.erase(std::unique(producers.begin(), producers.end()), producers.end());
    if (producers.size() != 2) continue;
    std::vector<size_t> ops{producers[0], producers[1], C};
    if (!solver::IsDAGConvex(dag, ops)) continue;
    TopoSortSubgraph(dag, ops);
    Column c = BuildColumn(p, dag, std::move(ops), PatternClass::F2aSingle);
    if (c.feasible) out.push_back(std::move(c));
  }

  // ── 3-op linear: A → B → C ─────────────────────────────────────────────
  // F1 rule requires: A→B intermediate single-consumer, B→C intermediate
  // single-consumer. Classify by op types:
  //   MM-MM-MM → Chain3plusUniformK (C7; requires w=k=K uniform)
  //   MM-PW-MM / PW-MM-PW / MM-MM-PW / PW-PW-MM / etc. → Chain3plusUniformK
  //     when any MM present (still uniform K constraint via BestGranularity)
  //   PW-PW-PW → StandalonePW (bundled; likely collapsed by preprocess)
  // BestGranularity rejects infeasible K combinations; enumeration here
  // is permissive.
  for (size_t A = 0; A < p.ops.size(); ++A) {
    if (p.ops[A].outputs.empty()) continue;
    size_t interAB = p.ops[A].outputs[0];
    if (interAB >= dag.tensor_consumers.size()) continue;
    std::unordered_set<size_t> uc_ab(dag.tensor_consumers[interAB].begin(),
                                      dag.tensor_consumers[interAB].end());
    if (uc_ab.size() != 1) continue;
    size_t B = *uc_ab.begin();
    if (B == A || B >= p.ops.size()) continue;
    if (p.ops[B].outputs.empty()) continue;
    size_t interBC = p.ops[B].outputs[0];
    if (interBC >= dag.tensor_consumers.size()) continue;
    std::unordered_set<size_t> uc_bc(dag.tensor_consumers[interBC].begin(),
                                      dag.tensor_consumers[interBC].end());
    if (uc_bc.size() != 1) continue;
    size_t C = *uc_bc.begin();
    if (C == B || C == A || C >= p.ops.size()) continue;
    // Classify.
    int n_mm = 0;
    for (size_t op : {A, B, C})
      if (p.ops[op].op_type == "MatMul") ++n_mm;
    PatternClass klass = (n_mm > 0) ? PatternClass::Chain3plusUniformK
                                    : PatternClass::StandalonePW;
    std::vector<size_t> ops{A, B, C};
    TopoSortSubgraph(dag, ops);
    Column c = BuildColumn(p, dag, std::move(ops), klass);
    if (c.feasible) out.push_back(std::move(c));
  }

  return out;
}

solver::Partition GreedyPack(const std::vector<Column>& columns,
                              size_t num_ops) {
  // Build a per-op baseline (singleton cost) so pairs are ranked by SAVINGS
  // rather than raw cost-per-op.  A pair {A, B} is attractive iff
  //   savings = singleton(A) + singleton(B) − pair({A,B})
  // is positive and large.  Singletons themselves get savings 0.
  std::vector<double> op_singleton(num_ops, 0.0);
  for (const auto& c : columns) {
    if (c.ops.size() == 1 && c.ops[0] < num_ops) {
      op_singleton[c.ops[0]] = c.cost;
    }
  }

  std::vector<size_t> order(columns.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const auto& ca = columns[a];
    const auto& cb = columns[b];
    double sum_a = 0.0, sum_b = 0.0;
    for (size_t op : ca.ops) sum_a += op_singleton[op];
    for (size_t op : cb.ops) sum_b += op_singleton[op];
    // Savings per extra op (so 3-op fusion is compared fairly to 2-op).
    double sav_a = (ca.ops.size() > 1) ? (sum_a - ca.cost) / (ca.ops.size() - 1) : 0.0;
    double sav_b = (cb.ops.size() > 1) ? (sum_b - cb.cost) / (cb.ops.size() - 1) : 0.0;
    if (sav_a != sav_b) return sav_a > sav_b;
    if (ca.ops.size() != cb.ops.size())
      return ca.ops.size() > cb.ops.size();
    return a < b;
  });

  std::vector<bool> covered(num_ops, false);
  solver::Partition part;

  for (size_t idx : order) {
    const Column& c = columns[idx];
    if (c.ops.size() == 1) continue;  // singletons handled as fill-in below
    double sum = 0.0;
    for (size_t op : c.ops) sum += op_singleton[op];
    if (c.cost >= sum) break;  // no more beneficial fusions
    bool conflict = false;
    for (size_t op : c.ops) {
      if (op >= num_ops || covered[op]) { conflict = true; break; }
    }
    if (conflict) continue;
    for (size_t op : c.ops) covered[op] = true;
    part.subgraphs.push_back(c.ops);
  }

  // Fill remaining ops with singleton subgraphs.
  for (size_t op = 0; op < num_ops; ++op) {
    if (!covered[op]) part.subgraphs.push_back({op});
  }

  return part;
}

}  // namespace pattern
