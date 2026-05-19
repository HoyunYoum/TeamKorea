// solver_common.h — Shared infrastructure for all MLSys solvers
//
// Extracted from io_solver.cc and optimal_solver.cc to eliminate ~700 lines
// of duplication per solver. All new solvers should #include this header.

#ifndef SOLVER_COMMON_H_
#define SOLVER_COMMON_H_

#include "cost.h"
#include "dag.h"
#include "granularity.h"
#include "io_util.h"
#include "mlsys.h"
#include "partition_algo.h"
#include "preprocess.h"
#include "solution_build.h"
#include "tensor_roles.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

namespace solver {

// ── Constants & Helpers ──────────────────────────────────────────────────────

int64_t CeilDiv(int64_t a, int64_t b);

// DAG types + builders live in dag.h (included above).
// Tensor role types + classifiers live in tensor_roles.h (included below).
// Cost model (ComputeTileWS, OutDims, ExactCost, AnalyticalCost,
// SnakeRow/Col, DimCandidates, KCandidates, WorkingSet, DominantRole)
// and `kInf` live in cost.h (included below).

// Granularity search (GranConfig, BestGranularity, BoundResult,
// ExhaustiveGranularity) lives in granularity.h (included below).
//
// Partition strategies (Partition, InitialPartition, SubgraphSig, DPPartition,
// CanMergeSubgraphs, Agglomerative, MaxFusion, Random, AntiChain,
// RecomputationPass, ConvexityRepair, UnfusionPass, RetentionCandidates,
// SubgraphReorder) live in partition_algo.h (included below).
//
// Solution construction (BuildSolution, BuildSolutionExhaustive,
// PrintDiagnostics) lives in solution_build.h (included below).
//
// WriteSolution lives in io_util.h (included below).

// Preprocessing passes (TensorAnalysis, AnalyzeTensors, EliminateDeadOps,
// CollapsePWChains, PreprocessResult, Preprocess, UnPreprocess) live in
// preprocess.h (included below).

// ── Common Pipeline ──────────────────────────────────────────────────────────

// Timing helpers (ElapsedSince, CompetitionTimeout, ParseBenchmarkNumber)
// live in io_util.h (included below).

// Component + DecomposeDAG live in dag.h (included above).

}  // namespace solver

#endif  // SOLVER_COMMON_H_
