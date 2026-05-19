// local_search.h — Two-phase local search over partition moves.
//
// Runs after the E/A/R parallel pipeline when wall-clock budget remains.
// Three adjacent-pair moves underlie both phases:
//   Merge(i, i+1)           — combine sg[i] and sg[i+1] into one subgraph
//   Split(i, k)              — split sg[i] at topo-sorted index k
//   Repartition(i, i+1, k)   — combine sg[i] ∪ sg[i+1] and re-split at k
//
// Phase A — First-improvement hill-climb. Sweeps Merge → Split → Repartition;
// restarts on each accept; exits when a full sweep yields no improvement.
//
// Phase B — Simulated annealing. Starts from the hill-climb's local optimum
// (schedule: T₀ = best·5e-3, T_min = best·1e-6, cool = 0.9998,
// move mix 25% Merge / 25% Split / 50% Repartition). Accepts under the
// Metropolis criterion (Δ<0 always, else exp(-Δ/T)). All-time-best tracked
// separately from the SA random walker; only new all-time-bests fire the
// ImproveCallback, and the walker is rolled back to best on exit so a
// mid-walk deadline hit never regresses the returned solution.
//
// Each candidate move:
//   1. Reconstructs only the affected subgraph(s), drops stale retention
//      across broken boundaries, keeps retention on unchanged neighbours.
//   2. Runs solver::BestGranularity on each affected subgraph, carrying the
//      previous sg's tensors_to_retain as retained_in (same rule as
//      retention_pass::reopt_range).
//   3. Runs mlsys::Evaluate on the full solution.
//
// Convexity and partition-level topological validity are preserved by:
//   - Topo-sorting combined ops before split points.
//   - Rejecting any move whose new subgraph fails solver::IsDAGConvex.
//
// On every new all-time best the optional ImproveCallback fires; the
// pattern_solver uses this to push the new best through the existing
// try_write helper (tmp→rename), keeping the atomic on-disk invariant so
// SIGTERM mid-SA-walk still leaves a complete valid solution behind.

#ifndef LOCAL_SEARCH_H_
#define LOCAL_SEARCH_H_

#include <functional>

#include "mlsys.h"
#include "solver_common.h"

namespace pattern {

// Fires on every strictly-better solution LocalSearch finds.
using ImproveCallback = std::function<void(const mlsys::Solution&, double)>;

struct LocalSearchResult {
  mlsys::Solution solution;
  int n_improvements = 0;   // new all-time bests (Phase A + Phase B)
  int n_moves_tried = 0;    // total move trials (Phase A + Phase B)
  int n_hc_moves = 0;       // Phase A: hill-climb trials
  int n_sa_moves = 0;       // Phase B: SA trials
  int n_sa_accepts = 0;     // Phase B: Metropolis accepts (incl. non-best)
  double cost_before = 0.0;
  double cost_after = 0.0;
};

LocalSearchResult LocalSearch(const mlsys::Problem& p,
                               const solver::DAG& dag,
                               mlsys::Solution sol,
                               double deadline_s,
                               ImproveCallback on_improve = {});

}  // namespace pattern

#endif  // LOCAL_SEARCH_H_
