// dag.cc — DAG construction + convexity check + decomposition.

#include "dag.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_set>

namespace solver {

DAG BuildDAG(const mlsys::Problem& p) {
  DAG dag;
  dag.num_ops = p.ops.size();
  dag.num_tensors = p.tensors.size();
  dag.tensor_producer.assign(dag.num_tensors, -1);
  dag.tensor_consumers.resize(dag.num_tensors);
  for (size_t i = 0; i < dag.num_ops; ++i) {
    for (size_t t : p.ops[i].outputs) dag.tensor_producer[t] = (int)i;
    for (size_t t : p.ops[i].inputs) dag.tensor_consumers[t].push_back(i);
  }
  std::vector<std::vector<size_t>> op_preds(dag.num_ops);
  dag.op_successors.resize(dag.num_ops);
  for (size_t i = 0; i < dag.num_ops; ++i) {
    for (size_t t : p.ops[i].inputs) {
      int prod = dag.tensor_producer[t];
      if (prod >= 0 && (size_t)prod != i) {
        op_preds[i].push_back(prod);
        dag.op_successors[prod].push_back(i);
      }
    }
  }
  // BFS topological sort (Kahn's)
  std::vector<int> in_deg(dag.num_ops, 0);
  for (size_t i = 0; i < dag.num_ops; ++i)
    in_deg[i] = (int)op_preds[i].size();
  std::vector<size_t> q;
  for (size_t i = 0; i < dag.num_ops; ++i)
    if (in_deg[i] == 0) q.push_back(i);
  dag.topo_order.reserve(dag.num_ops);
  for (size_t h = 0; h < q.size(); ++h) {
    size_t op = q[h];
    dag.topo_order.push_back(op);
    for (size_t s : dag.op_successors[op])
      if (--in_deg[s] == 0) q.push_back(s);
  }
  return dag;
}

DAG BuildDAG_DFS(const mlsys::Problem& p) {
  DAG dag = BuildDAG(p);
  std::vector<std::vector<size_t>> op_preds(dag.num_ops);
  for (size_t i = 0; i < dag.num_ops; ++i)
    for (size_t t : p.ops[i].inputs) {
      int prod = dag.tensor_producer[t];
      if (prod >= 0 && (size_t)prod != i) op_preds[i].push_back(prod);
    }
  std::vector<int> in_deg(dag.num_ops, 0);
  for (size_t i = 0; i < dag.num_ops; ++i)
    in_deg[i] = (int)op_preds[i].size();
  std::vector<bool> visited(dag.num_ops, false);
  std::vector<size_t> post_order;
  post_order.reserve(dag.num_ops);
  std::function<void(size_t)> dfs = [&](size_t op) {
    if (visited[op]) return;
    visited[op] = true;
    for (size_t succ : dag.op_successors[op])
      if (--in_deg[succ] == 0) dfs(succ);
    post_order.push_back(op);
  };
  for (size_t i = 0; i < dag.num_ops; ++i)
    if (in_deg[i] == 0) dfs(i);
  dag.topo_order.assign(post_order.rbegin(), post_order.rend());
  return dag;
}

DAG BuildDAG_Random(const mlsys::Problem& p, uint32_t seed) {
  DAG dag = BuildDAG(p);
  std::mt19937 rng(seed);
  std::vector<int> in_deg(dag.num_ops, 0);
  for (size_t i = 0; i < dag.num_ops; ++i)
    for (size_t t : p.ops[i].inputs) {
      int prod = dag.tensor_producer[t];
      if (prod >= 0 && (size_t)prod != i) ++in_deg[i];
    }
  std::vector<size_t> ready;
  for (size_t i = 0; i < dag.num_ops; ++i)
    if (in_deg[i] == 0) ready.push_back(i);
  dag.topo_order.clear();
  dag.topo_order.reserve(dag.num_ops);
  while (!ready.empty()) {
    std::uniform_int_distribution<size_t> dist(0, ready.size() - 1);
    size_t idx = dist(rng);
    size_t op = ready[idx];
    ready[idx] = ready.back();
    ready.pop_back();
    dag.topo_order.push_back(op);
    for (size_t s : dag.op_successors[op])
      if (--in_deg[s] == 0) ready.push_back(s);
  }
  return dag;
}

DAG BuildDAG_MaxOutput(const mlsys::Problem& p) {
  DAG dag = BuildDAG(p);
  std::vector<int64_t> output_size(dag.num_ops, 0);
  for (size_t i = 0; i < dag.num_ops; ++i)
    for (size_t t : p.ops[i].outputs)
      output_size[i] += p.tensors[t].width * p.tensors[t].height;
  std::vector<int> in_deg(dag.num_ops, 0);
  for (size_t i = 0; i < dag.num_ops; ++i)
    for (size_t t : p.ops[i].inputs) {
      int prod = dag.tensor_producer[t];
      if (prod >= 0 && (size_t)prod != i) ++in_deg[i];
    }
  auto cmp = [&](size_t a, size_t b) { return output_size[a] < output_size[b]; };
  std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> pq(cmp);
  for (size_t i = 0; i < dag.num_ops; ++i)
    if (in_deg[i] == 0) pq.push(i);
  dag.topo_order.clear();
  while (!pq.empty()) {
    size_t op = pq.top(); pq.pop();
    dag.topo_order.push_back(op);
    for (size_t s : dag.op_successors[op])
      if (--in_deg[s] == 0) pq.push(s);
  }
  return dag;
}

bool IsDAGConvex(const DAG& dag, const std::vector<size_t>& ops) {
  if (ops.size() <= 1) return true;
  std::unordered_set<size_t> op_set(ops.begin(), ops.end());

  // For each op in the set, follow its successors that are OUTSIDE the set.
  // If any such path reaches another op IN the set, the subset is non-convex.
  for (size_t a : ops) {
    for (size_t succ : dag.op_successors[a]) {
      if (op_set.count(succ)) continue;  // direct successor inside set, OK
      std::queue<size_t> q;
      std::unordered_set<size_t> visited;
      q.push(succ);
      visited.insert(succ);
      while (!q.empty()) {
        size_t cur = q.front(); q.pop();
        for (size_t next : dag.op_successors[cur]) {
          if (op_set.count(next)) return false;
          if (visited.insert(next).second)
            q.push(next);
        }
      }
    }
  }
  return true;
}

std::vector<Component> DecomposeDAG(const mlsys::Problem& p, const DAG& dag) {
  int N = (int)dag.num_ops;
  if (N <= 1) return {{{0}, {}}};

  auto& topo = dag.topo_order;
  std::vector<int> topo_pos(N);
  for (int i = 0; i < N; ++i)
    topo_pos[topo[i]] = i;

  struct TensorLiveness {
    int produce_pos = -1;
    int last_consume = -1;
    bool is_graph_input;
  };

  std::vector<TensorLiveness> tlive(dag.num_tensors);
  for (size_t t = 0; t < dag.num_tensors; ++t) {
    tlive[t].is_graph_input = (dag.tensor_producer[t] < 0);
    if (dag.tensor_producer[t] >= 0)
      tlive[t].produce_pos = topo_pos[dag.tensor_producer[t]];
    for (size_t cons : dag.tensor_consumers[t])
      tlive[t].last_consume = std::max(tlive[t].last_consume, topo_pos[cons]);
  }

  std::vector<int> live(N, 0);
  std::vector<int> starts(N, 0), ends(N, 0);
  for (size_t t = 0; t < dag.num_tensors; ++t) {
    if (tlive[t].is_graph_input) continue;
    if (tlive[t].produce_pos < 0) continue;
    if (tlive[t].last_consume < 0) continue;
    starts[tlive[t].produce_pos]++;
    if (tlive[t].last_consume + 1 < N)
      ends[tlive[t].last_consume + 1]++;
  }

  int running = 0;
  for (int i = 0; i < N; ++i) {
    running += starts[i] - ends[i];
    live[i] = running;
  }

  int min_comp = std::max(3, N / 10);
  std::vector<std::pair<int, int>> candidates;
  for (int i = min_comp - 1; i < N - min_comp; ++i)
    candidates.push_back({live[i], i});
  std::sort(candidates.begin(), candidates.end());

  std::vector<int> split_points;
  for (auto& [lc, pos] : candidates) {
    if (lc > 3) break;
    bool too_close = false;
    for (int sp : split_points)
      if (std::abs(sp - pos) < min_comp) { too_close = true; break; }
    if (too_close) continue;
    split_points.push_back(pos);
    if ((int)split_points.size() >= 5) break;
  }

  if (split_points.empty()) {
    Component comp;
    comp.ops.resize(N);
    std::iota(comp.ops.begin(), comp.ops.end(), 0);
    return {comp};
  }

  std::sort(split_points.begin(), split_points.end());

  std::vector<Component> components;
  int start_pos = 0;
  for (int sp : split_points) {
    Component comp;
    for (int i = start_pos; i <= sp; ++i)
      comp.ops.push_back(topo[i]);
    std::sort(comp.ops.begin(), comp.ops.end());
    components.push_back(std::move(comp));
    start_pos = sp + 1;
  }
  {
    Component comp;
    for (int i = start_pos; i < N; ++i)
      comp.ops.push_back(topo[i]);
    std::sort(comp.ops.begin(), comp.ops.end());
    components.push_back(std::move(comp));
  }

  for (auto& comp : components) {
    std::unordered_set<size_t> ops_set(comp.ops.begin(), comp.ops.end());
    for (size_t op : comp.ops) {
      for (size_t t : p.ops[op].inputs) {
        if (dag.tensor_producer[t] >= 0 && !ops_set.count(dag.tensor_producer[t]))
          comp.cut_tensors.push_back(t);
      }
      for (size_t t : p.ops[op].outputs) {
        for (size_t cons : dag.tensor_consumers[t])
          if (!ops_set.count(cons))
            comp.cut_tensors.push_back(t);
      }
    }
    std::sort(comp.cut_tensors.begin(), comp.cut_tensors.end());
    comp.cut_tensors.erase(
        std::unique(comp.cut_tensors.begin(), comp.cut_tensors.end()),
        comp.cut_tensors.end());
  }

  components.erase(
      std::remove_if(components.begin(), components.end(),
                     [](const Component& c) { return c.ops.empty(); }),
      components.end());

  std::cerr << "Graph decomposition: " << components.size() << " components (";
  for (size_t i = 0; i < components.size(); ++i) {
    if (i > 0) std::cerr << ", ";
    std::cerr << components[i].ops.size();
    if (!components[i].cut_tensors.empty())
      std::cerr << "/" << components[i].cut_tensors.size() << "cut";
  }
  std::cerr << " ops)\n";

  return components;
}

}  // namespace solver
