// preprocess.cc — Dead-op elimination + PW chain collapse + Un/Preprocess driver.

#include "preprocess.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace solver {

TensorAnalysis AnalyzeTensors(const mlsys::Problem& p, const DAG& dag) {
  const size_t nt = p.tensors.size();
  TensorAnalysis a;
  a.use_count.assign(nt, 0);
  a.last_consumer_op.assign(nt, -1);
  a.is_graph_output.assign(nt, false);
  a.is_single_use.assign(nt, false);

  for (size_t i = 0; i < p.ops.size(); ++i)
    for (size_t t : p.ops[i].inputs)
      ++a.use_count[t];

  std::vector<int> topo_rank(dag.num_ops, 0);
  for (int r = 0; r < (int)dag.topo_order.size(); ++r)
    topo_rank[dag.topo_order[r]] = r;
  for (size_t i = 0; i < p.ops.size(); ++i)
    for (size_t t : p.ops[i].inputs)
      a.last_consumer_op[t] = std::max(a.last_consumer_op[t], topo_rank[i]);

  for (size_t t = 0; t < nt; ++t) {
    a.is_graph_output[t] = (a.use_count[t] == 0);
    a.is_single_use[t] = (a.use_count[t] == 1);
  }
  return a;
}

mlsys::Problem EliminateDeadOps(const mlsys::Problem& p,
                                 const TensorAnalysis& analysis) {
  const size_t no = p.ops.size();
  const size_t nt = p.tensors.size();

  std::vector<bool> dead(no, false);
  std::vector<int> live_use_count(nt);
  for (size_t t = 0; t < nt; ++t)
    live_use_count[t] = analysis.use_count[t];

  DAG dag = BuildDAG(p);
  for (int r = (int)dag.topo_order.size() - 1; r >= 0; --r) {
    size_t op = dag.topo_order[r];
    bool all_dead = true;
    for (size_t t : p.ops[op].outputs) {
      if (live_use_count[t] > 0 || analysis.use_count[t] == 0) {
        // use_count == 0 means graph output (no consumers) — keep it.
        // live_use_count > 0 means some live op still consumes it.
        all_dead = false;
        break;
      }
    }
    if (!all_dead) continue;

    dead[op] = true;
    for (size_t t : p.ops[op].inputs)
      --live_use_count[t];
  }

  int n_dead = 0;
  for (size_t i = 0; i < no; ++i)
    if (dead[i]) ++n_dead;
  if (n_dead == 0) return p;

  std::vector<int> tensor_map(nt, -1);
  std::vector<int> op_map(no, -1);

  std::vector<bool> tensor_live(nt, false);
  for (size_t i = 0; i < no; ++i) {
    if (dead[i]) continue;
    for (size_t t : p.ops[i].inputs) tensor_live[t] = true;
    for (size_t t : p.ops[i].outputs) tensor_live[t] = true;
  }
  for (size_t t = 0; t < nt; ++t)
    if (dag.tensor_producer[t] < 0) tensor_live[t] = true;

  size_t new_t = 0;
  for (size_t t = 0; t < nt; ++t)
    if (tensor_live[t]) tensor_map[t] = (int)new_t++;

  size_t new_o = 0;
  for (size_t i = 0; i < no; ++i)
    if (!dead[i]) op_map[i] = (int)new_o++;

  mlsys::Problem np;
  np.fast_memory_capacity = p.fast_memory_capacity;
  np.slow_memory_bandwidth = p.slow_memory_bandwidth;
  np.native_granularity = p.native_granularity;
  np.tensors.resize(new_t);
  for (size_t t = 0; t < nt; ++t)
    if (tensor_map[t] >= 0) np.tensors[tensor_map[t]] = p.tensors[t];
  np.ops.resize(new_o);
  for (size_t i = 0; i < no; ++i) {
    if (dead[i]) continue;
    auto& nop = np.ops[op_map[i]];
    nop.op_type = p.ops[i].op_type;
    nop.base_cost = p.ops[i].base_cost;
    for (size_t t : p.ops[i].inputs) nop.inputs.push_back(tensor_map[t]);
    for (size_t t : p.ops[i].outputs) nop.outputs.push_back(tensor_map[t]);
  }

  std::cerr << "[Preprocess] EliminateDeadOps: removed " << n_dead << " ops, "
            << (nt - new_t) << " tensors\n";
  return np;
}

int CollapsePWChains(mlsys::Problem& p, const TensorAnalysis& analysis) {
  const size_t no = p.ops.size();

  DAG dag = BuildDAG(p);

  std::vector<int> chain_next(no, -1);
  std::vector<int> chain_prev(no, -1);

  for (size_t i = 0; i < no; ++i) {
    if (p.ops[i].op_type != "Pointwise") continue;
    if (p.ops[i].outputs.size() != 1) continue;
    size_t out_t = p.ops[i].outputs[0];
    if (!analysis.is_single_use[out_t]) continue;
    if (dag.tensor_consumers[out_t].size() != 1) continue;
    size_t j = dag.tensor_consumers[out_t][0];
    if (j == i || p.ops[j].op_type != "Pointwise") continue;
    if (chain_prev[j] >= 0) continue;
    chain_next[i] = (int)j;
    chain_prev[j] = (int)i;
  }

  std::vector<bool> removed(no, false);
  int n_removed = 0;

  for (size_t i = 0; i < no; ++i) {
    if (chain_next[i] < 0 || chain_prev[i] >= 0) continue;

    std::vector<size_t> chain;
    for (int cur = (int)i; cur >= 0; cur = chain_next[cur])
      chain.push_back(cur);
    if (chain.size() < 2) continue;

    size_t head = chain[0];
    size_t tail = chain.back();

    std::unordered_set<size_t> intermediates;
    for (size_t k = 0; k < chain.size() - 1; ++k)
      intermediates.insert(p.ops[chain[k]].outputs[0]);

    std::vector<size_t> merged_inputs;
    for (size_t c : chain)
      for (size_t t : p.ops[c].inputs)
        if (!intermediates.count(t))
          merged_inputs.push_back(t);

    p.ops[head].inputs = std::move(merged_inputs);
    p.ops[head].outputs = p.ops[tail].outputs;
    int64_t total_cost = 0;
    for (size_t c : chain) total_cost += p.ops[c].base_cost;
    p.ops[head].base_cost = total_cost;

    for (size_t k = 1; k < chain.size(); ++k) {
      removed[chain[k]] = true;
      ++n_removed;
    }
  }

  if (n_removed == 0) return 0;

  std::vector<mlsys::Op> new_ops;
  new_ops.reserve(no - n_removed);
  for (size_t i = 0; i < no; ++i)
    if (!removed[i])
      new_ops.push_back(std::move(p.ops[i]));
  p.ops = std::move(new_ops);

  std::cerr << "[Preprocess] CollapsePWChains: collapsed " << n_removed
            << " ops (" << p.ops.size() << " ops remain)\n";
  return n_removed;
}

PreprocessResult Preprocess(mlsys::Problem p) {
  PreprocessResult result;
  result.orig_problem = p;
  const size_t orig_nops = p.ops.size();
  const size_t orig_ntensors = p.tensors.size();

  result.op_expansion.resize(orig_nops);
  for (size_t i = 0; i < orig_nops; ++i)
    result.op_expansion[i] = {i};

  result.tensor_map.resize(orig_ntensors);
  for (size_t t = 0; t < orig_ntensors; ++t)
    result.tensor_map[t] = t;

  DAG dag = BuildDAG(p);
  TensorAnalysis analysis = AnalyzeTensors(p, dag);

  // Pass 2: dead-op elimination (no-op on contest benchmarks, but safe).
  mlsys::Problem p2 = EliminateDeadOps(p, analysis);
  if (p2.ops.size() < p.ops.size()) {
    dag = BuildDAG(p);
    std::vector<bool> dead(p.ops.size(), false);
    std::vector<int> live_uc(p.tensors.size());
    for (size_t t = 0; t < p.tensors.size(); ++t)
      live_uc[t] = analysis.use_count[t];
    for (int r = (int)dag.topo_order.size() - 1; r >= 0; --r) {
      size_t op = dag.topo_order[r];
      bool all_dead = true;
      for (size_t t : p.ops[op].outputs) {
        if (live_uc[t] > 0 || analysis.use_count[t] == 0)
          { all_dead = false; break; }
      }
      if (!all_dead) continue;
      dead[op] = true;
      for (size_t t : p.ops[op].inputs) --live_uc[t];
    }

    for (size_t i = 0; i < p.ops.size(); ++i)
      if (dead[i]) result.dead_ops.push_back(i);

    std::vector<bool> tensor_live(p.tensors.size(), false);
    for (size_t i = 0; i < p.ops.size(); ++i) {
      if (dead[i]) continue;
      for (size_t t : p.ops[i].inputs) tensor_live[t] = true;
      for (size_t t : p.ops[i].outputs) tensor_live[t] = true;
    }
    for (size_t t = 0; t < p.tensors.size(); ++t)
      if (dag.tensor_producer[t] < 0) tensor_live[t] = true;

    std::vector<size_t> new_tensor_map;
    for (size_t t = 0; t < p.tensors.size(); ++t)
      if (tensor_live[t]) new_tensor_map.push_back(t);
    result.tensor_map = std::move(new_tensor_map);

    std::vector<std::vector<size_t>> new_expansion;
    for (size_t i = 0; i < p.ops.size(); ++i)
      if (!dead[i]) new_expansion.push_back(result.op_expansion[i]);
    result.op_expansion = std::move(new_expansion);
    p = std::move(p2);
  }

  // Pass 3: PW chain collapsing (fixpoint).
  //
  // Each sub-pass picks, for every fan-in PW, at most one predecessor to
  // chain in (first-wins). Multi-input PWs — e.g. moe_small's op 18 with
  // predecessors op 5 AND op 9 — leave residual chainable edges after
  // one sweep. Iterate until no edges remain; op_expansion is accumulated
  // consistently because each sub-pass resolves only the chains it found.
  while (true) {
    dag = BuildDAG(p);
    analysis = AnalyzeTensors(p, dag);

    const size_t no = p.ops.size();
    std::vector<int> chain_next(no, -1), chain_prev(no, -1);
    for (size_t i = 0; i < no; ++i) {
      if (p.ops[i].op_type != "Pointwise") continue;
      if (p.ops[i].outputs.size() != 1) continue;
      size_t out_t = p.ops[i].outputs[0];
      if (!analysis.is_single_use[out_t]) continue;
      if (dag.tensor_consumers[out_t].size() != 1) continue;
      size_t j = dag.tensor_consumers[out_t][0];
      if (j == i || p.ops[j].op_type != "Pointwise") continue;
      if (chain_prev[j] >= 0) continue;
      chain_next[i] = (int)j;
      chain_prev[j] = (int)i;
    }

    std::unordered_map<size_t, std::vector<size_t>> chains;
    std::vector<bool> is_removed(no, false);
    for (size_t i = 0; i < no; ++i) {
      if (chain_next[i] < 0 || chain_prev[i] >= 0) continue;
      std::vector<size_t> chain;
      for (int cur = (int)i; cur >= 0; cur = chain_next[cur])
        chain.push_back(cur);
      if (chain.size() < 2) continue;
      for (size_t k = 1; k < chain.size(); ++k)
        is_removed[chain[k]] = true;
      chains[i] = std::move(chain);
    }

    int n_collapsed = CollapsePWChains(p, analysis);
    if (n_collapsed == 0) break;

    std::vector<std::vector<size_t>> new_expansion;
    for (size_t i = 0; i < no; ++i) {
      if (is_removed[i]) continue;
      if (chains.count(i)) {
        std::vector<size_t> expanded;
        for (size_t m : chains[i])
          for (size_t orig : result.op_expansion[m])
            expanded.push_back(orig);
        new_expansion.push_back(std::move(expanded));
      } else {
        new_expansion.push_back(std::move(result.op_expansion[i]));
      }
    }
    result.op_expansion = std::move(new_expansion);
  }

  result.problem = std::move(p);
  return result;
}

mlsys::Solution UnPreprocess(const mlsys::Solution& sol,
                              const PreprocessResult& pr) {
  mlsys::Solution out;
  out.subgraphs.reserve(sol.subgraphs.size() + pr.dead_ops.size());
  for (const auto& sg : sol.subgraphs) {
    mlsys::Subgraph nsg;
    for (size_t op : sg.ops)
      for (size_t orig : pr.op_expansion[op])
        nsg.ops.push_back(orig);
    nsg.granularity = sg.granularity;
    nsg.traversal_order = sg.traversal_order;
    nsg.subgraph_latency = sg.subgraph_latency;
    for (size_t t : sg.tensors_to_retain)
      nsg.tensors_to_retain.push_back(pr.tensor_map[t]);
    out.subgraphs.push_back(std::move(nsg));
  }

  if (!pr.dead_ops.empty()) {
    const auto& orig = pr.orig_problem;
    for (size_t orig_op : pr.dead_ops) {
      mlsys::Subgraph dsg;
      dsg.ops = {orig_op};
      int64_t k = 1;
      if (orig.ops[orig_op].op_type == "MatMul") {
        k = orig.tensors[orig.ops[orig_op].inputs[0]].width;
      }
      dsg.granularity = {orig.native_granularity.width,
                         orig.native_granularity.height, k};
      dsg.subgraph_latency = 0;
      out.subgraphs.push_back(std::move(dsg));
    }
  }

  return out;
}

}  // namespace solver
