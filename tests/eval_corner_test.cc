#include <iostream>
#include <string>
#include <vector>

#include "mlsys.h"

struct CornerTest {
  std::string name;
  mlsys::Problem problem;
  mlsys::Solution solution;
  bool expect_ok;
  std::string expect_substr;  // if !expect_ok, error message should contain this
  double expected_value;       // if expect_ok
};

mlsys::Problem MakeBaseProblem(
    std::vector<std::pair<int64_t, int64_t>> tensor_dims,
    std::vector<std::string> op_types,
    std::vector<std::vector<size_t>> inputs,
    std::vector<std::vector<size_t>> outputs,
    std::vector<int64_t> base_costs, int64_t capacity = 50000,
    int64_t bandwidth = 10, int64_t native_w = 128, int64_t native_h = 128) {
  mlsys::Problem p;
  for (auto [w, h] : tensor_dims) {
    p.tensors.push_back({.width = w, .height = h});
  }
  for (size_t i = 0; i < op_types.size(); ++i) {
    mlsys::Op op;
    op.op_type = op_types[i];
    op.inputs = inputs[i];
    op.outputs = outputs[i];
    op.base_cost = base_costs[i];
    p.ops.push_back(std::move(op));
  }
  p.fast_memory_capacity = capacity;
  p.slow_memory_bandwidth = bandwidth;
  p.native_granularity = {.width = native_w, .height = native_h, .depth = 1};
  return p;
}

mlsys::Solution MakeSolution(
    std::vector<std::vector<size_t>> subgraphs,
    std::vector<std::array<int64_t, 3>> granularities,
    std::vector<std::vector<size_t>> tensors_to_retain,
    std::vector<double> latencies,
    std::vector<std::optional<mlsys::TraversalOrder>> traversals = {}) {
  mlsys::Solution s;
  for (size_t i = 0; i < subgraphs.size(); ++i) {
    mlsys::Subgraph sg;
    sg.ops = subgraphs[i];
    sg.granularity = {.width = granularities[i][0],
                      .height = granularities[i][1],
                      .depth = granularities[i][2]};
    sg.tensors_to_retain = tensors_to_retain[i];
    sg.subgraph_latency = latencies[i];
    if (i < traversals.size() && traversals[i].has_value()) {
      sg.traversal_order = traversals[i];
    }
    s.subgraphs.push_back(std::move(sg));
  }
  return s;
}

std::vector<CornerTest> BuildTests() {
  std::vector<CornerTest> tests;

  // 1. MM dimension mismatch: LHS 128x64, RHS 128x128 → LHS.width(64) != RHS.height(128)
  {
    auto p = MakeBaseProblem(
        {{64, 128}, {128, 128}, {128, 128}},  // T0=64w×128h, T1=128×128, T2=128×128
        {"MatMul"}, {{0, 1}}, {{2}}, {1000});
    auto s = MakeSolution({{0}}, {{{128, 128, 64}}}, {{}}, {0});
    tests.push_back({"MM_dim_mismatch", p, s, false, "LHS.width", 0});
  }

  // 2. Tile exceeds native granularity (w=256 > native=128)
  {
    auto p = MakeBaseProblem(
        {{256, 256}, {256, 256}}, {"Pointwise"}, {{0}}, {{1}}, {1000});
    auto s = MakeSolution({{0}}, {{{256, 128, 1}}}, {{}}, {0});
    tests.push_back({"tile_exceeds_native", p, s, false, "native", 0});
  }

  // 3. MM→PW (epilogue) fusion with split-K — VALID under issue #71.
  //    PW consumes MM's fully-formed output tile; it does not participate in
  //    the k-accumulation, so split-K is compute-neutral for the epilogue.
  {
    // Op0: MM (T0@T1→T2), Op1: PW (T2→T3). Fused with k=64 < K=128.
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"MatMul", "Pointwise"}, {{0, 1}, {2}}, {{2}, {3}}, {1000, 100});
    auto s = MakeSolution({{0, 1}}, {{{128, 128, 64}}}, {{}}, {0});
    tests.push_back({"mm_pw_split_k_epilogue", p, s, true, "", 4915.2});
  }

  // 3a. PW→MM(LHS) split-K with w < MM.K — FORBIDDEN under issue #71.
  //     granule alignment requires w ≥ MM.K (one PW tile covers full LHS strip).
  {
    // Op0: PW (T0→T1), Op1: MM (T1@T2→T3). K = T1.width = 128. w=64 < 128.
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"Pointwise", "MatMul"}, {{0}, {1, 2}}, {{1}, {3}}, {100, 1000});
    auto s = MakeSolution({{0, 1}}, {{{64, 128, 64}}}, {{}}, {0});
    tests.push_back({"pw_to_mm_lhs_split_k_misaligned", p, s, false,
                     "issue #71", 0});
  }

  // 3b. PW→MM(RHS) split-K with h < MM.K — FORBIDDEN under issue #71.
  //     granule alignment requires h ≥ MM.K for PW feeding MM's RHS.
  {
    // Op0: PW (T0→T1 used as MM RHS), Op1: MM (T2@T1→T3). K = T1.height = 128.
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"Pointwise", "MatMul"}, {{0}, {2, 1}}, {{1}, {3}}, {100, 1000});
    auto s = MakeSolution({{0, 1}}, {{{128, 64, 64}}}, {{}}, {0});
    tests.push_back({"pw_to_mm_rhs_split_k_misaligned", p, s, false,
                     "issue #71", 0});
  }

  // 4. 3+ MM chain with split-K (k < K)
  {
    // A(128×128) @ B(128×128) → T3, T3 @ C(128×128) → T4, T4 @ D(128×128) → T5
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"MatMul", "MatMul", "MatMul"},
        {{0, 1}, {3, 2}, {4, 2}},   // Op0:T0@T1→T3, Op1:T3@T2→T4, Op2:T4@T2→T5
        {{3}, {4}, {5}},
        {1000, 1000, 1000});
    auto s = MakeSolution({{0, 1, 2}}, {{{128, 128, 64}}}, {{}}, {0});
    tests.push_back({"3mm_split_k", p, s, false, "3+", 0});
  }

  // 5. Retain a graph input tensor
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}}, {"Pointwise"}, {{0}}, {{1}}, {1000});
    auto s = MakeSolution({{0}}, {{{128, 128, 1}}}, {{0}}, {0});  // retain T0 (graph input)
    tests.push_back({"retain_graph_input", p, s, false, "graph input", 0});
  }

  // 5b. Retain a graph output tensor (must materialize to slow memory).
  //     Attempting to skip eviction via retention is a scoring loophole.
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}}, {"Pointwise"}, {{0}}, {{1}}, {1000});
    auto s = MakeSolution({{0}}, {{{128, 128, 1}}}, {{1}}, {0});  // retain T1 (graph output)
    tests.push_back({"retain_graph_output", p, s, false, "graph output", 0});
  }

  // 6. OOM: working set exceeds capacity
  {
    // MM with 128×128 tensors, capacity only 10000.
    // WS = LHS(128×128) + RHS(128×128) + OUT(128×128) = 49152 >> 10000
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}},
        {"MatMul"}, {{0, 1}}, {{2}}, {1000}, /*capacity=*/10000);
    auto s = MakeSolution({{0}}, {{{128, 128, 128}}}, {{}}, {0});
    tests.push_back({"oom_working_set", p, s, false, "working set", 0});
  }

  // 7. 3+ MM chain with k=K but w != k (should fail)
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"MatMul", "MatMul", "MatMul"},
        {{0, 1}, {3, 2}, {4, 2}},
        {{3}, {4}, {5}},
        {1000, 1000, 1000},
        /*capacity=*/500000);
    // k=128=K but w=64 != k → should reject
    auto s = MakeSolution({{0, 1, 2}}, {{{64, 128, 128}}}, {{}}, {0});
    tests.push_back({"3mm_k_eq_K_w_ne_k", p, s, false, "w == k", 0});
  }

  // 8. Single PW op, simple valid case — sanity
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}}, {"Pointwise"}, {{0}}, {{1}}, {500});
    auto s = MakeSolution({{0}}, {{{128, 128, 1}}}, {{}}, {0});
    // IO: load T0 (128*128/10=1638.4) + evict T1 (1638.4) = 3276.8
    // compute = 500. latency = max(500, 3276.8) = 3276.8
    tests.push_back({"single_pw_valid", p, s, true, "", 3276.8});
  }

  // 9. Retained tensor from prior subgraph: no IO for retained input
  {
    // Op0: PW T0→T1 (retain T1), Op1: PW T1→T2
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}},
        {"Pointwise", "Pointwise"}, {{0}, {1}}, {{1}, {2}}, {1000, 1000});
    auto s = MakeSolution(
        {{0}, {1}},
        {{{128, 128, 1}}, {{128, 128, 1}}},
        {{1}, {}},   // retain T1 from sg0
        {0, 0});
    // sg0: load T0 (1638.4), compute 1000, no evict T1 (retained).
    //   latency = max(1000, 1638.4) = 1638.4
    // sg1: T1 retained (no load). compute 1000. evict T2 (1638.4).
    //   latency = max(1000, 1638.4) = 1638.4
    // total = 3276.8
    tests.push_back({"retained_no_io", p, s, true, "", 3276.8});
  }

  // 10. Non-square MM→MM chain: A(128×64) @ B(64×256) → T(128×256),
  //     T(128×256) @ C(256×128) → Out(128×128)
  //     K_up=64, K_down=256, k=64 → nk=4
  {
    auto p = MakeBaseProblem(
        {{64, 128}, {256, 64}, {128, 256}, {256, 128}, {128, 128}},
        //  T0: w=64,h=128 (A: 128×64)
        //  T1: w=256,h=64 (B: 64×256)
        //  T2: w=128,h=256 (C: 256×128)
        //  T3: w=256,h=128 (intermediate: 128×256)  — ephemeral
        //  T4: w=128,h=128 (final output: 128×128)
        {"MatMul", "MatMul"},
        {{0, 1}, {3, 2}},
        {{3}, {4}},
        {2000, 4000},
        /*capacity=*/100000, /*bandwidth=*/10);

    auto s = MakeSolution({{0, 1}}, {{{128, 128, 64}}}, {{}}, {0});

    // K_up = T0.width = 64, K_down = T3.width = 256, k=64, nk = ceil(256/64) = 4
    // Output: T4 (128×128). nw=1, nh=1. 1 tile.
    //
    // Per step compute:
    //   Op0 (chain up): base=2000. K_denom = K_down = 256. k_eff = 64.
    //     compute = 2000 * 64 / 256 = 500
    //   Op1 (chain down): base=4000. K_denom = K_down = 256. k_eff = 64.
    //     compute = 4000 * 64 / 256 = 1000
    //   total compute per step = 1500
    //
    // IO per step:
    //   Step 0 (ks=0):
    //     Op0 LHS (T0, chain up): h_eff=128, K_up=64. Load 128*64=8192. IO=819.2
    //     Op0 RHS (T1, chain up): K_up*k_eff = 64*64 = 4096. IO=409.6
    //     Op1 RHS (T2, chain down): k_eff*w_eff = 64*128 = 8192. IO=819.2
    //     total IO_in = 2048.0
    //   Steps 1-2 (ks=1,2):
    //     Op0 LHS: same h_tile → reuse. IO=0
    //     Op0 RHS: new k_step → load 64*64=4096. IO=409.6
    //     Op1 RHS: new k_step → load 64*128=8192. IO=819.2
    //     total IO_in = 1228.8
    //   Step 3 (ks=3, last):
    //     same as step 1-2 for io_in: 1228.8
    //     + evict T4: 128*128/10 = 1638.4
    //     total IO = 1228.8 + 1638.4 = 2867.2
    //
    // Latencies:
    //   step 0: max(1500, 2048.0) = 2048.0
    //   step 1: max(1500, 1228.8) = 1500
    //   step 2: max(1500, 1228.8) = 1500
    //   step 3: max(1500, 2867.2) = 2867.2
    //   total = 7915.2
    tests.push_back({"nonsquare_mm_chain", p, s, true, "", 7915.2});
  }

  // 11. Two standalone MMs (not chained) with k=K (nk=1) — valid fusion at k=K
  {
    // Op0: T0@T1→T2, Op1: T3@T4→T5. No chain dependency. k >= all K's.
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128},
         {128, 128}, {128, 128}, {128, 128}},
        {"MatMul", "MatMul"},
        {{0, 1}, {3, 4}},
        {{2}, {5}},
        {1000, 1000},
        /*capacity=*/200000);
    auto s = MakeSolution({{0, 1}}, {{{128, 128, 128}}}, {{}}, {0});
    // nk=1 (k=128=K for both). Both standalone.
    // 1 tile, 1 step.
    // Compute: (1000*128/128) + (1000*128/128) = 2000
    // IO: load T0(16384/10=1638.4) + T1(1638.4) + T3(1638.4) + T4(1638.4)
    //     + evict T2(1638.4) + T5(1638.4) = 9830.4
    // latency = max(2000, 9830.4) = 9830.4
    tests.push_back({"two_standalone_mm_k_eq_K", p, s, true, "", 9830.4});
  }

  // 12. PW with sub-native granularity and multiple tiles
  {
    // 256×256 tensor, gran 64×64, native 128×128.
    // 16 tiles, each pays base_cost=800.
    auto p = MakeBaseProblem(
        {{256, 256}, {256, 256}}, {"Pointwise"}, {{0}}, {{1}}, {800},
        /*capacity=*/50000, /*bandwidth=*/10, /*native_w=*/128, /*native_h=*/128);
    auto s = MakeSolution({{0}}, {{{64, 64, 1}}}, {{}}, {0});
    // 16 tiles. Each tile: load T0 slice (64*64/10=409.6), evict T1 slice (409.6).
    // IO per tile = 819.2. Compute = 800.
    // latency per tile = max(800, 819.2) = 819.2
    // No cross-tile reuse for PW.
    // Total = 16 * 819.2 = 13107.2
    tests.push_back({"pw_sub_native_16_tiles", p, s, true, "", 13107.2});
  }

  // 13. Chain upstream with k_step remainder: K_down not divisible by k
  {
    // A(100×100) @ B(100×100) → T(100×100), T @ C(100×100) → Out(100×100)
    // native 128×128. gran [100, 100, 32]. K_down=100, k=32, nk=ceil(100/32)=4.
    // Steps: k_eff = 32, 32, 32, 4.
    auto p = MakeBaseProblem(
        {{100, 100}, {100, 100}, {100, 100}, {100, 100}, {100, 100}},
        {"MatMul", "MatMul"},
        {{0, 1}, {3, 2}},
        {{3}, {4}},
        {2000, 2000},
        /*capacity=*/100000, /*bandwidth=*/10,
        /*native_w=*/128, /*native_h=*/128);
    auto s = MakeSolution({{0, 1}}, {{{100, 100, 32}}}, {{}}, {0});

    // K_up=100, K_down=100, k=32, nk=4. 1 tile.
    // k_eff: 32, 32, 32, 4
    //
    // Compute per step:
    //   Op0: 2000 * k_eff / 100
    //   Op1: 2000 * k_eff / 100
    //   Step 0-2: (2000*32/100 + 2000*32/100) = 640+640 = 1280
    //   Step 3: (2000*4/100 + 2000*4/100) = 80+80 = 160
    //
    // IO:
    //   Step 0 (ks=0):
    //     Op0 LHS (T0): 100*100=10000. IO=1000
    //     Op0 RHS (T1): 100*32=3200. IO=320
    //     Op1 RHS (T2): 32*100=3200. IO=320
    //     IO_in = 1640
    //   Steps 1-2:
    //     Op0 LHS: reuse. IO=0
    //     Op0 RHS: 100*32=3200. IO=320
    //     Op1 RHS: 32*100=3200. IO=320
    //     IO_in = 640
    //   Step 3 (last, k_eff=4):
    //     Op0 LHS: reuse. IO=0
    //     Op0 RHS: 100*4=400. IO=40
    //     Op1 RHS: 4*100=400. IO=40
    //     IO_in = 80
    //     Evict T4: 100*100/10 = 1000
    //     IO = 80 + 1000 = 1080
    //
    // Latencies:
    //   step 0: max(1280, 1640) = 1640
    //   step 1: max(1280, 640) = 1280
    //   step 2: max(1280, 640) = 1280
    //   step 3: max(160, 1080) = 1080
    //   total = 5280
    tests.push_back({"chain_k_remainder", p, s, true, "", 5280.0});
  }

  // 14. Recomputation: same op in two subgraphs (Example 3B pattern)
  {
    // Op0: PW T0→T1, Op1: PW T1→T2, Op2: PW T1,T2→T3.
    // Schedule: sg0=[Op0,Op1] retain T2, sg1=[Op0,Op2] retain nothing.
    // T1 in sg0: consumed by Op1 in sg → ephemeral.
    // T1 in sg1: consumed by Op2 in sg → ephemeral.
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"Pointwise", "Pointwise", "Pointwise"},
        {{0}, {1}, {1, 2}},
        {{1}, {2}, {3}},
        {1500, 1500, 1500});
    auto s = MakeSolution(
        {{0, 1}, {0, 2}},
        {{{128, 128, 1}}, {{128, 128, 1}}},
        {{2}, {}},
        {0, 0});
    // sg0: load T0(1638.4). compute=3000. T1 ephemeral. T2 retained (no evict).
    //   latency = max(3000, 1638.4) = 3000
    // sg1: load T0(1638.4). compute=3000. T1 ephemeral. T2 retained from sg0 (no load).
    //   Evict T3(1638.4).
    //   latency = max(3000, 1638.4+1638.4) = max(3000, 3276.8) = 3276.8
    // total = 6276.8
    tests.push_back({"recomputation", p, s, true, "", 6276.8});
  }

  // 15. Standalone MM with split-K, multi-tile, raster order
  {
    // A(256×128) @ B(128×256) → C(256×256). native 128×128. gran [128,128,64].
    // K_op=128, k=64, nk=2. nw=2, nh=2. 4 tiles × 2 k-steps = 8 cells.
    auto p = MakeBaseProblem(
        {{128, 256}, {256, 128}, {256, 256}},
        // T0: w=128,h=256 (A: 256rows × 128cols)
        // T1: w=256,h=128 (B: 128rows × 256cols)
        // T2: w=256,h=256 (C: 256rows × 256cols)
        {"MatMul"}, {{0, 1}}, {{2}}, {2000},
        /*capacity=*/100000, /*bandwidth=*/10);
    auto s = MakeSolution({{0}}, {{{128, 128, 64}}}, {{}}, {0});

    // K_op = T0.width = 128. k=64. nk=2.
    // Output T2: 256×256. nw=2, nh=2. 4 tiles. Raster: (0,0),(0,1),(1,0),(1,1).
    //
    // Compute per step = 2000 * 64 / 128 = 1000.
    //
    // Tile (0,0) = ht=0, wt=0:
    //   ks=0: LHS key=(0,-1,0) → load T0[0:128, 0:64] = 128*64=8192. IO=819.2
    //         RHS key=(-1,0,0) → load T1[0:64, 0:128] = 64*128=8192. IO=819.2
    //         IO_in=1638.4. max(1000, 1638.4)=1638.4
    //   ks=1: LHS key=(0,-1,1) → load T0[0:128, 64:128] = 128*64=8192. IO=819.2
    //         RHS key=(-1,0,1) → load T1[64:128, 0:128] = 64*128=8192. IO=819.2
    //         evict C tile 128*128/10=1638.4
    //         IO=1638.4+1638.4=3276.8. max(1000, 3276.8)=3276.8
    //
    // Tile (0,1) = ht=0, wt=1:
    //   ks=0: LHS key=(0,-1,0) → different from (0,-1,1) → load! IO=819.2
    //         RHS key=(-1,1,0) → different from (-1,0,1) → load! IO=819.2
    //         IO_in=1638.4. max(1000,1638.4)=1638.4
    //   ks=1: LHS key=(0,-1,1) → different from (0,-1,0) → load! IO=819.2
    //         RHS key=(-1,1,1) → different from (-1,1,0) → load! IO=819.2
    //         evict 1638.4. IO=1638.4+1638.4=3276.8. max(1000,3276.8)=3276.8
    //
    // Tile (1,0) = ht=1, wt=0:
    //   ks=0: LHS key=(1,-1,0) → diff from (0,-1,1) → load! IO=819.2
    //         RHS key=(-1,0,0) → diff from (-1,1,1) → load! IO=819.2
    //         IO_in=1638.4. 1638.4
    //   ks=1: LHS key=(1,-1,1) → diff → load! RHS key=(-1,0,1) → diff → load!
    //         IO_in=1638.4+evict 1638.4=3276.8. 3276.8
    //
    // Tile (1,1) = ht=1, wt=1:
    //   ks=0: LHS key=(1,-1,0) → diff from (1,-1,1) → load! RHS → load!
    //         1638.4
    //   ks=1: LHS → load! RHS → load!
    //         IO_in=1638.4+evict 1638.4=3276.8
    //
    // All tiles identical: 1638.4 + 3276.8 = 4915.2 per tile.
    // Total = 4 * 4915.2 = 19660.8
    tests.push_back({"standalone_mm_split_k_multi_tile", p, s, true, "", 19660.8});
  }

  // 16. 3+ MM chain k=K, w=k, but one op has K_op != k → reject
  {
    // Op0: T0(128×128) @ T1(128×128) → T3(128×128). K_op0=128.
    // Op1: T3(128×128) @ T2(128×64) → T4(128×64). K_op1=128.
    // Op2: T4(128×64) @ T6(64×128) → T5(128×128). K_op2=64 != k=128.
    // All MM dims valid. gran [128,128,128]. w=k=128 ✓. But K_op2=64.
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {64, 128}, {128, 128},
         {64, 128}, {128, 128}, {128, 64}},
        //  T0: w=128,h=128 (A)
        //  T1: w=128,h=128 (B)
        //  T2: w=64,h=128  (C: 128×64)
        //  T3: w=128,h=128 (intermediate Op0→Op1)
        //  T4: w=64,h=128  (intermediate Op1→Op2: 128×64)
        //  T5: w=128,h=128 (final output)
        //  T6: w=128,h=64  (D: 64×128)
        {"MatMul", "MatMul", "MatMul"},
        {{0, 1}, {3, 2}, {4, 6}},
        {{3}, {4}, {5}},
        {1000, 1000, 1000},
        /*capacity=*/500000);
    auto s = MakeSolution({{0, 1, 2}}, {{{128, 128, 128}}}, {{}}, {0});
    tests.push_back({"3mm_k_eq_K_Kop_mismatch", p, s, false, "reduction dims", 0});
  }

  // --- Ephemeral global check tests (Issue #51) ---
  // Diamond graph: Op0: PW T0→T1, Op1: PW T1→T2, Op2: PW T1,T2→T3.
  // T1 consumed by Op1 and Op2.

  // 17. Invalid ephemeral: [[0,1],[2]] — T1 ephemeral in sg0 but Op2 needs it
  //     and Op2's subgraph does NOT contain Op0 (no recompute).
  //     T1 should be forced non-ephemeral (evicted to slow memory).
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"Pointwise", "Pointwise", "Pointwise"},
        {{0}, {1}, {1, 2}},
        {{1}, {2}, {3}},
        {1000, 1000, 1000},
        /*capacity=*/100000);
    auto s = MakeSolution(
        {{0, 1}, {2}},
        {{{128, 128, 1}}, {{128, 128, 1}}},
        {{2}, {}},
        {0, 0});
    // T1 would be ephemeral in sg0 but Op2 (external) can't recompute Op0 → error.
    tests.push_back({"ephemeral_ext_consumer_no_recompute", p, s, false,
                      "external consumer", 0});
  }

  // 18. Valid ephemeral with recompute: [[0,1],[0,2]] — Op0 in both subgraphs.
  //     T1 is ephemeral in both (external consumers covered by recompute).
  //     This is the same as test 14 (recomputation), verifying global check works.
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"Pointwise", "Pointwise", "Pointwise"},
        {{0}, {1}, {1, 2}},
        {{1}, {2}, {3}},
        {1000, 1000, 1000});
    auto s = MakeSolution(
        {{0, 1}, {0, 2}},
        {{{128, 128, 1}}, {{128, 128, 1}}},
        {{2}, {}},
        {0, 0});
    // sg0: T1 ephemeral (Op2 external, but Op0 is in Op2's sg → covered).
    //   load T0 (1638.4). compute 2000. T2 retained.
    //   latency = max(2000, 1638.4) = 2000  // WAIT — no eviction of T1!
    // sg1: T1 ephemeral (Op1 external, but Op0 is in Op1's sg → covered).
    //   load T0 (1638.4). compute 2000. T2 retained (no load). evict T3 (1638.4).
    //   IO = 1638.4 + 1638.4 = 3276.8
    //   latency = max(2000, 3276.8) = 3276.8
    // total = 5276.8
    tests.push_back({"ephemeral_recompute_covered", p, s, true, "", 5276.8});
  }

  // 19. Op missing from schedule entirely.
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}},
        {"Pointwise", "Pointwise"},
        {{0}, {1}},
        {{1}, {2}},
        {1000, 1000});
    // Only schedule Op0, skip Op1.
    auto s = MakeSolution({{0}}, {{{128, 128, 1}}}, {{}}, {0});
    tests.push_back({"op_missing_from_schedule", p, s, false, "not scheduled", 0});
  }

  // 20. WS bug: 2-chain with K_down < K_up, k >= K_down → nk=1, standalone.
  //     Op0's K_op=256, but WS must use k_eff=K_op=256, not min(k,K_op)=128.
  //     Op0: MM (T0:128×256 @ T1:256×128) → T3:128×128 (K_up=256... wait)
  //     T0: w=256,h=128 means matrix is 128 rows × 256 cols. K = LHS.width = 256.
  //     T1: w=128,h=256 means matrix is 256 rows × 128 cols. RHS.height must = K = 256. ✓
  //     T3 (output): h×w of output = 128×128. w=128,h=128.
  //     Op1: MM (T3:128×128 @ T2:128×128) → T4:128×128. K_down = T3.width = 128.
  //     Fused [Op0, Op1], k=128. K_slicing=K_down=128, nk=1.
  //     Both become standalone. T3 ephemeral.
  //     Op0 LHS (T0): h*k_eff = 128*256 = 32768 (not 128*128=16384!)
  //     Op0 RHS (T1): k_eff*w = 256*128 = 32768
  //     Op1 RHS (T2): k_eff*w = 128*128 = 16384
  //     Op1 OUT (T4): 128*128 = 16384
  //     WS = 32768 + 32768 + 16384 + 16384 = 98304
  //     With capacity=80000 → should OOM.
  {
    auto p = MakeBaseProblem(
        {{256, 128}, {128, 256}, {128, 128}, {128, 128}, {128, 128}},
        // T0: w=256,h=128 (LHS of Op0, K=256)
        // T1: w=128,h=256 (RHS of Op0, height=256=K) ✓
        // T2: w=128,h=128 (RHS of Op1)
        // T3: w=128,h=128 (intermediate, ephemeral)
        // T4: w=128,h=128 (output)
        {"MatMul", "MatMul"},
        {{0, 1}, {3, 2}},
        {{3}, {4}},
        {2000, 2000},
        /*capacity=*/80000);
    auto s = MakeSolution({{0, 1}}, {{{128, 128, 128}}}, {{}}, {0});
    tests.push_back({"ws_standalone_nk1_oom", p, s, false, "working set", 0});
  }

  // 22. MM output shape mismatch: LHS(128×128) @ RHS(128×128) should yield
  //     output of (w=128, h=128), but we declare it as (w=64, h=64).
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {64, 64}},
        {"MatMul"}, {{0, 1}}, {{2}}, {1000});
    auto s = MakeSolution({{0}}, {{{64, 64, 128}}}, {{}}, {0});
    tests.push_back({"mm_output_shape_mismatch", p, s, false,
                      "output shape", 0});
  }

  // 23. Retain tensor not produced in subgraph and not carried over.
  //     sg0=[Op0] retain T3 (produced by Op1, not Op0).
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"Pointwise", "Pointwise"},
        {{0}, {1}},
        {{1}, {2}},
        {1000, 1000});
    // Add one more op to give T3 a producer, then schedule only Op0 retaining T3.
    // Easier: just reuse existing setup and retain a tensor from a later op.
    // Here: sg0 = [Op0], retain T2 (produced by Op1). T2's producer Op1 is not in sg0.
    auto s = MakeSolution({{0}, {1}},
                           {{{128, 128, 1}}, {{128, 128, 1}}},
                           {{2}, {}},    // retain T2 in sg0 — but T2 is not produced yet!
                           {0, 0});
    tests.push_back({"retain_not_produced", p, s, false,
                      "not produced by any op", 0});
  }

  // 23b. Pass-through retention: sg produces T1, retains T1. sg1 does NOT
  //      produce T1 but re-lists T1 in tensors_to_retain to "carry" it to sg2.
  //      Per spec #34/#52 this is illegal — intermediate sg must recompute
  //      the producer to legally re-retain.
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}},
        {"Pointwise", "Pointwise", "Pointwise"},
        {{0}, {1}, {1}},     // Op2 also consumes T1 (multi-hop target)
        {{1}, {2}, {3}},
        {1000, 1000, 1000});
    // sg0=[Op0] produces+retains T1; sg1=[Op1] passes T1 through without
    // producing it; sg2=[Op2] consumes T1 from sg1's retention.
    auto s = MakeSolution({{0}, {1}, {2}},
                           {{{128, 128, 1}}, {{128, 128, 1}}, {{128, 128, 1}}},
                           {{1}, {1}, {}},
                           {0, 0, 0});
    tests.push_back({"retain_passthrough_no_recompute", p, s, false,
                      "not produced by any op", 0});
  }

  // 24. Unified grid violation: two PW ops with different output dims in same sg.
  {
    // Op0: PW T0(128×128) → T1(128×128). Op1: PW T2(64×64) → T3(64×64).
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {64, 64}, {64, 64}},
        {"Pointwise", "Pointwise"},
        {{0}, {2}},
        {{1}, {3}},
        {1000, 1000},
        /*capacity=*/100000);
    auto s = MakeSolution({{0, 1}}, {{{64, 64, 1}}}, {{}}, {0});
    tests.push_back({"unified_grid_violation", p, s, false,
                      "inconsistent shapes", 0});
  }

  // 25. Traversal order wrong size.
  {
    auto p = MakeBaseProblem(
        {{256, 256}, {256, 256}}, {"Pointwise"}, {{0}}, {{1}}, {800});
    // 256×256 with gran 128×128 → 4 tiles. Provide only 3 entries.
    auto s = MakeSolution({{0}}, {{{128, 128, 1}}}, {{}}, {0},
                           {mlsys::TraversalOrder{0, 1, 2}});
    tests.push_back({"traversal_wrong_size", p, s, false,
                      "traversal_order size", 0});
  }

  // 26. Traversal order duplicate index.
  {
    auto p = MakeBaseProblem(
        {{256, 256}, {256, 256}}, {"Pointwise"}, {{0}}, {{1}}, {800});
    auto s = MakeSolution({{0}}, {{{128, 128, 1}}}, {{}}, {0},
                           {mlsys::TraversalOrder{0, 1, 1, 2}});
    tests.push_back({"traversal_duplicate", p, s, false,
                      "duplicate", 0});
  }

  // 27. Traversal order out-of-range index.
  {
    auto p = MakeBaseProblem(
        {{256, 256}, {256, 256}}, {"Pointwise"}, {{0}}, {{1}}, {800});
    auto s = MakeSolution({{0}}, {{{128, 128, 1}}}, {{}}, {0},
                           {mlsys::TraversalOrder{0, 1, 2, 5}});
    tests.push_back({"traversal_out_of_range", p, s, false,
                      "out-of-range", 0});
  }

  // 21. Same setup but capacity=100000 → should fit and evaluate correctly.
  {
    auto p = MakeBaseProblem(
        {{256, 128}, {128, 256}, {128, 128}, {128, 128}, {128, 128}},
        {"MatMul", "MatMul"},
        {{0, 1}, {3, 2}},
        {{3}, {4}},
        {2000, 2000},
        /*capacity=*/100000);
    auto s = MakeSolution({{0, 1}}, {{{128, 128, 128}}}, {{}}, {0});
    // nk=1, 1 tile. Both standalone.
    // Op0: compute = 2000*256/256 = 2000. Op1: compute = 2000*128/128 = 2000.
    //   total compute = 4000.
    // IO: Op0 LHS(T0): 128*256/10 = 3276.8
    //     Op0 RHS(T1): 256*128/10 = 3276.8
    //     Op1 RHS(T2): 128*128/10 = 1638.4
    //     Evict T4: 128*128/10 = 1638.4
    //     total IO = 9830.4
    // latency = max(4000, 9830.4) = 9830.4
    tests.push_back({"ws_standalone_nk1_ok", p, s, true, "", 9830.4});
  }

  // 28. Middle MatMul in 3-chain at nk=1 with uniform K = w = 128 (native)
  //
  // Chain: Op0 (Head) → Op1 (Middle) → Op2 (Tail)
  //   Op0: T0 (128×128) @ T1 (128×128) → T3 (128×128, ephemeral)
  //   Op1: T3 @ T2 (128×128) → T4 (128×128, ephemeral)     [LHS & out ephemeral]
  //   Op2: T4 @ T5 (128×128) → T6 (128×128, graph output)
  // k = w = K_op = 128 → nk = 1. One tile, one k-step.
  // Validates that the evaluator accepts a 3+ MM chain when the uniform
  // w = k = K_op rule holds and that Middle MM dispatches correctly.
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128},
         {128, 128}, {128, 128}},
        {"MatMul", "MatMul", "MatMul"},
        {{0, 1}, {3, 2}, {4, 5}},
        {{3}, {4}, {6}},
        {1000, 1000, 1000},
        /*capacity=*/200000);
    auto s = MakeSolution({{0, 1, 2}}, {{{128, 128, 128}}}, {{}}, {0});
    // Compute: 3 × 1000 = 3000 (each MM at k=K=128, one step).
    // IO loads (once): T0, T1, T2, T5 + evict T6 → 5 × 16384/10 = 8192.
    // latency = max(3000, 8192) = 8192.
    tests.push_back({"middle_mm_nk1_uniform_k", p, s, true, "", 8192.0});
  }

  // 29. Middle MatMul at nk=1 with NON-uniform K — must be rejected
  //
  // Same chain shape as #28 but the middle op's reduction width (128) does
  // not match Op0's (64) → uniform-K rule is violated.
  {
    auto p = MakeBaseProblem(
        {{64, 128}, {128, 64}, {128, 128}, {128, 128}, {128, 128},
         {128, 128}, {128, 128}},
        {"MatMul", "MatMul", "MatMul"},
        {{0, 1}, {3, 2}, {4, 5}},
        {{3}, {4}, {6}},
        {1000, 1000, 1000},
        /*capacity=*/200000);
    auto s = MakeSolution({{0, 1, 2}}, {{{128, 128, 128}}}, {{}}, {0});
    tests.push_back({"middle_mm_nk1_nonuniform_k", p, s, false,
                     "all reduction dims == k", 0});
  }

  // 30. Middle MatMul with split-K (nk>1) — must be rejected
  //
  // Any Middle MM requires nk=1; the evaluator should refuse split-K.
  {
    auto p = MakeBaseProblem(
        {{128, 128}, {128, 128}, {128, 128}, {128, 128}, {128, 128},
         {128, 128}, {128, 128}},
        {"MatMul", "MatMul", "MatMul"},
        {{0, 1}, {3, 2}, {4, 5}},
        {{3}, {4}, {6}},
        {1000, 1000, 1000},
        /*capacity=*/200000);
    auto s = MakeSolution({{0, 1, 2}}, {{{128, 128, 64}}}, {{}}, {0});
    tests.push_back({"middle_mm_splitk_forbidden", p, s, false,
                     "Middle MatMul", 0});
  }

  return tests;
}

int main() {
  auto tests = BuildTests();
  int passed = 0, failed = 0;

  for (const auto& tc : tests) {
    auto result = mlsys::Evaluate(tc.problem, tc.solution);

    if (tc.expect_ok) {
      if (!result.ok()) {
        std::cerr << tc.name << ": FAIL (expected OK, got error: "
                  << result.status() << ")\n";
        ++failed;
      } else if (std::abs(*result - tc.expected_value) > 0.1) {
        std::cerr << tc.name << ": FAIL (got " << *result << ", expected "
                  << tc.expected_value << ")\n";
        ++failed;
      } else {
        std::cout << tc.name << ": PASS (" << *result << ")\n";
        ++passed;
      }
    } else {
      if (result.ok()) {
        std::cerr << tc.name << ": FAIL (expected error containing '"
                  << tc.expect_substr << "', got OK with value " << *result
                  << ")\n";
        ++failed;
      } else {
        std::string msg(result.status().message());
        if (msg.find(tc.expect_substr) != std::string::npos) {
          std::cout << tc.name << ": PASS (error: " << msg << ")\n";
          ++passed;
        } else {
          std::cerr << tc.name << ": FAIL (error '" << msg
                    << "' does not contain '" << tc.expect_substr << "')\n";
          ++failed;
        }
      }
    }
  }

  std::cout << "\n" << passed << " passed, " << failed << " failed out of "
            << tests.size() << "\n";
  return failed > 0 ? 1 : 0;
}
