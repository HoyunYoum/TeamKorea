// solver_common.cc — Shared solver infrastructure
//
// Extracted from io_solver.cc and optimal_solver.cc to eliminate duplication.
// All functions live in namespace solver{}.

#include "solver_common.h"

namespace solver {

// ── Constants & Helpers ──────────────────────────────────────────────────────

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// DAG, BuildDAG*, IsDAGConvex, DecomposeDAG live in dag.cc/h.

// Tensor classification + role types (Classify, BuildRoles,
// ClassifyMMRoles, CountMMRoles, CountFanoutEphemerals, TensorRoleInfo,
// MMRole, MMRoleCounts) live in tensor_roles.cc/h.


// Cost model + WS + traversals + candidates (ComputeTileWS,
// DominantRole, OutDims, WorkingSet, SnakeRow/Col, DimCandidates,
// KCandidates, ExactCost{,Cached}, AnalyticalCost) live in cost.cc/h.



// Granularity search (BestGranularity, ExhaustiveGranularity) lives
// in granularity.cc/h.
//
// Partition strategies (InitialPartition, DPPartition, Agglomerative,
// MaxFusion, Random, AntiChain, RecomputationPass, ConvexityRepair,
// UnfusionPass, RetentionCandidates, SubgraphReorder, SubgraphSig,
// CanMergeSubgraphs, Partition) live in partition_algo.cc/h.
//
// Solution construction (BuildSolution, BuildSolutionExhaustive,
// PrintDiagnostics) lives in solution_build.cc/h.

// WriteSolution + timing helpers (ElapsedSince, CompetitionTimeout,
// ParseBenchmarkNumber) live in io_util.cc/h.


// Preprocessing passes (AnalyzeTensors, EliminateDeadOps,
// CollapsePWChains, Preprocess, UnPreprocess) live in preprocess.cc/h.


}  // namespace solver
