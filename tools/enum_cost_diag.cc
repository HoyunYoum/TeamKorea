// enum_cost_diag.cc — Per-column AnalyticalCost vs ExactCost audit.
//
// Step 1 of the #71 Stage 2 cost-model audit: run the pattern enumerator on a
// problem, print each emitted column's (AnalyticalCost, ExactCost, Δ%) at the
// gran BestGranularity picked under AnalyticalCost, plus the metadata that
// classifies *why* the column might be mis-ranked:
//   - klass (C0/C1/C3/C4/C5/C6/C7) + n_mm / n_pw
//   - pw→mm edge kinds (lhs / rhs) within the subgraph
//   - is_pure_standalone, total_mm
//   - split-K flag (k < maxK)
//   - granule-alignment flags (w ≥ pw_to_mm_lhs_K, h ≥ pw_to_mm_rhs_K)
//
// Usage:  enum_cost_diag <problem.json> [--top N] [--only-diverge %]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mlsys.h"
#include "pattern_enum.h"
#include "solver_common.h"

namespace {

// ── ExactCost-preferred-gran probe ────────────────────────────────────────
// For the outlier columns, scan all feasible (w,h,k,snake) with the same
// guards BestGranularity applies and pick the gran that minimises ExactCost.
// Lets us quantify whether AnalyticalCost's mis-ranking actually costs us
// anything: if E_at_A_pick == E_at_E_pick, the divergence is harmless.
struct ExactBest {
  mlsys::Granularity gran{0, 0, 0};
  double exact_cost = solver::kInf;
  double analytic_cost = solver::kInf;
  int snake_mode = 0;
};

ExactBest ExactCostBestGran(const mlsys::Problem& p,
                            const solver::DAG& dag,
                            const std::vector<size_t>& ops) {
  // Replicate BestGranularity's guard prelude so the candidate space is
  // identical — the only change is the ranking cost.
  auto ts = solver::Classify(p, ops, &dag);
  auto tensor_roles = solver::BuildRoles(p, ops, ts);
  int64_t n_outputs = (int64_t)ts.outputs.size();
  int64_t n_fanout_eph = solver::CountFanoutEphemerals(p, ops, ts);
  auto [oW, oH] = solver::OutDims(p, ops);
  int64_t natW = p.native_granularity.width;
  int64_t natH = p.native_granularity.height;
  int64_t C = p.fast_memory_capacity;
  int64_t maxK = 0;
  for (size_t op : ops)
    if (p.ops[op].op_type == "MatMul")
      maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);
  auto mm_roles = solver::ClassifyMMRoles(p, ops, ts);
  auto rc = solver::CountMMRoles(mm_roles, ops, p);
  const bool split_k_infeasible =
      rc.n_middle > 0 ||
      (rc.total_mm() >= 2 && !rc.is_pure_chain_pair() && !rc.is_pure_standalone()) ||
      (rc.is_pure_standalone() && rc.total_mm() >= 2);
  const bool is_mm_pw_mm_3op = (ops.size() == 3 && rc.total_mm() == 2 &&
                                 rc.n_pw == 1 && rc.n_standalone == 2);
  const bool uniform_k_required =
      rc.n_middle > 0 || rc.total_mm() >= 3 ||
      (rc.total_mm() >= 2 && rc.n_pw > 0 && !is_mm_pw_mm_3op);

  int64_t pw_to_mm_lhs_K = 0, pw_to_mm_rhs_K = 0;
  for (size_t pw_op : ops) {
    if (p.ops[pw_op].op_type != "Pointwise") continue;
    for (size_t t : p.ops[pw_op].outputs) {
      for (size_t mm_op : ops) {
        if (mm_op == pw_op) continue;
        if (p.ops[mm_op].op_type != "MatMul") continue;
        const auto& mins = p.ops[mm_op].inputs;
        int64_t mm_K = p.tensors[mins[0]].width;
        for (int pos = 0; pos < (int)mins.size(); ++pos) {
          if (mins[pos] != t) continue;
          if (pos == 0) pw_to_mm_lhs_K = std::max(pw_to_mm_lhs_K, mm_K);
          else pw_to_mm_rhs_K = std::max(pw_to_mm_rhs_K, mm_K);
        }
      }
    }
  }

  auto k_cands = (split_k_infeasible && maxK > 0)
                     ? std::vector<int64_t>{maxK}
                     : solver::KCandidates(maxK);
  auto w_base = solver::DimCandidates(oW, natW);
  auto h_base = solver::DimCandidates(oH, natH);
  if (uniform_k_required) {
    bool uniform_K = true;
    for (size_t op : ops) {
      if (p.ops[op].op_type != "MatMul") continue;
      if (p.tensors[p.ops[op].inputs[0]].width != maxK) { uniform_K = false; break; }
    }
    if (!uniform_K || maxK > natW) return {};
    w_base = {maxK};
  }

  ExactBest best;
  std::unordered_set<size_t> empty_retain;
  auto Try = [&](int64_t w, int64_t h, int64_t k) {
    if (w <= 0 || h <= 0 || k <= 0) return;
    if (h > natH) h = natH;
    if (w > natW) w = natW;
    if (maxK > 0 && k < maxK) {
      if (pw_to_mm_lhs_K > 0 && w < pw_to_mm_lhs_K) return;
      if (pw_to_mm_rhs_K > 0 && h < pw_to_mm_rhs_K) return;
    }
    int64_t ws = solver::ComputeTileWS(p, tensor_roles, n_outputs, empty_retain,
                                        w, h, k, n_fanout_eph, /*retained_size=*/0);
    if (ws > C) return;
    mlsys::Granularity g{w, h, k};
    for (int snake = 0; snake < 3; ++snake) {
      // ExactCost is snake-invariant when no chain residency drives tile
      // reuse beyond the DK_* caching, so we don't loop over snakes here —
      // but still call it once per (w,h,k). Keep the signature for a future
      // trav-aware extension if needed.
      (void)snake;
      double e = solver::ExactCostCached(p, ops, g, /*retain_out=*/{},
                                          empty_retain, ts, tensor_roles,
                                          /*trav=*/nullptr);
      if (e < best.exact_cost) {
        double a = solver::AnalyticalCost(p, ops, g, 0, empty_retain, ts,
                                           tensor_roles, maxK, n_outputs,
                                           /*assume_retain_out=*/false);
        best = {g, e, a, 0};
      }
      break;  // snake-invariant
    }
  };
  auto CeilDivLocal = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  (void)CeilDivLocal;
  for (int64_t k : k_cands) {
    for (int64_t w : w_base) {
      for (int64_t h : h_base) Try(w, h, k);
      // Analytic h_max: same equation BestGranularity uses (granularity.cc:193-210).
      int64_t coeff_h = n_outputs * w;
      int64_t const_wk = 0;
      for (auto& [tid, role] : tensor_roles) {
        int64_t lhs_k_ws = role.head_LHS_K > 0 ? role.head_LHS_K : k;
        if (role.K_lhs > 0 && !role.is_rhs)
          coeff_h += role.is_pw ? std::max(lhs_k_ws, w) : lhs_k_ws;
        else if (role.K_lhs > 0 && role.is_rhs) coeff_h += lhs_k_ws;
        else if (role.is_rhs && !role.is_pw) const_wk += k * w;
        else if (role.is_rhs && role.is_pw) coeff_h += w;
        else if (role.is_pw) coeff_h += w;
      }
      if (coeff_h > 0) {
        int64_t h_max = (C - const_wk) / coeff_h;
        if (h_max > 0) {
          Try(w, h_max, k);
          if (natH > 1) Try(w, (h_max / natH) * natH, k);
        }
      }
    }
    for (int64_t h : h_base) {
      int64_t coeff_w = n_outputs * h;
      int64_t const_hk = 0;
      for (auto& [tid, role] : tensor_roles) {
        int64_t lhs_k_ws = role.head_LHS_K > 0 ? role.head_LHS_K : k;
        if (role.K_lhs > 0 && !role.is_rhs) const_hk += h * lhs_k_ws;
        else if (role.K_lhs > 0 && role.is_rhs) const_hk += h * lhs_k_ws;
        else if (role.is_rhs && !role.is_pw) coeff_w += k;
        else if (role.is_rhs && role.is_pw) coeff_w += std::max(k, h);
        else if (role.is_pw) coeff_w += h;
      }
      if (coeff_w > 0) {
        int64_t w_max = (C - const_hk) / coeff_w;
        if (w_max > 0) {
          Try(w_max, h, k);
          if (natW > 1) Try((w_max / natW) * natW, h, k);
        }
      }
    }
  }
  return best;
}

struct ColMeta {
  double analytic, exact;
  double rel_delta;         // (A - E) / max(|E|, eps)
  int n_mm, n_pw;
  int total_mm;
  bool pure_standalone;
  bool has_chain;
  int pw_to_mm_lhs_K;
  int pw_to_mm_rhs_K;
  int64_t maxK;
  int64_t w, h, k;
  bool split_k;
  bool lhs_aligned;         // for split-K: w >= pw_to_mm_lhs_K
  bool rhs_aligned;         // for split-K: h >= pw_to_mm_rhs_K
  pattern::PatternClass klass;
  std::vector<size_t> ops;
  std::string op_sig;       // "MM,PW,MM" etc.
};

const char* KlassName(pattern::PatternClass k) {
  switch (k) {
    case pattern::PatternClass::StandaloneMM:      return "C0-MM";
    case pattern::PatternClass::StandalonePW:      return "C1-PW";
    case pattern::PatternClass::LhsChain2:         return "C3-Lhs2";
    case pattern::PatternClass::RhsChain2:         return "C4-Rhs2";
    case pattern::PatternClass::F2aSingle:         return "C5-F2a";
    case pattern::PatternClass::PwMmKeqK:          return "C6-PwMm";
    case pattern::PatternClass::Chain3plusUniformK:return "C7-3+";
  }
  return "?";
}

ColMeta BuildMeta(const mlsys::Problem& p, const solver::DAG& dag,
                  const pattern::Column& c) {
  ColMeta m{};
  m.analytic = c.analytic_cost;
  m.exact = c.cost;
  double denom = std::max(std::abs(m.exact), 1.0);
  m.rel_delta = (m.analytic - m.exact) / denom;
  m.klass = c.klass;
  m.ops = c.ops;
  m.w = c.gran.width; m.h = c.gran.height; m.k = c.gran.depth;

  std::ostringstream sig;
  m.n_mm = m.n_pw = 0;
  for (size_t i = 0; i < c.ops.size(); ++i) {
    bool is_mm = p.ops[c.ops[i]].op_type == "MatMul";
    if (i) sig << ",";
    sig << (is_mm ? "MM" : "PW");
    if (is_mm) ++m.n_mm; else ++m.n_pw;
  }
  m.op_sig = sig.str();

  // MM role counts (same logic as BestGranularity guard).
  auto ts = solver::Classify(p, c.ops, &dag);
  auto mm_roles = solver::ClassifyMMRoles(p, c.ops, ts);
  auto rc = solver::CountMMRoles(mm_roles, c.ops, p);
  m.total_mm = rc.total_mm();
  m.pure_standalone = rc.is_pure_standalone();
  m.has_chain = rc.has_chain();

  // PW→MM edges within subgraph.
  int64_t lhs_K = 0, rhs_K = 0;
  for (size_t pw_op : c.ops) {
    if (p.ops[pw_op].op_type != "Pointwise") continue;
    for (size_t t : p.ops[pw_op].outputs) {
      for (size_t mm_op : c.ops) {
        if (mm_op == pw_op) continue;
        if (p.ops[mm_op].op_type != "MatMul") continue;
        const auto& mins = p.ops[mm_op].inputs;
        int64_t mm_K = p.tensors[mins[0]].width;
        for (int pos = 0; pos < (int)mins.size(); ++pos) {
          if (mins[pos] != t) continue;
          if (pos == 0) lhs_K = std::max(lhs_K, mm_K);
          else rhs_K = std::max(rhs_K, mm_K);
        }
      }
    }
  }
  m.pw_to_mm_lhs_K = static_cast<int>(lhs_K);
  m.pw_to_mm_rhs_K = static_cast<int>(rhs_K);

  int64_t maxK = 0;
  for (size_t op : c.ops)
    if (p.ops[op].op_type == "MatMul")
      maxK = std::max(maxK, p.tensors[p.ops[op].inputs[0]].width);
  m.maxK = maxK;
  m.split_k = (maxK > 0 && m.k > 0 && m.k < maxK);
  m.lhs_aligned = (lhs_K == 0) || (m.w >= lhs_K);
  m.rhs_aligned = (rhs_K == 0) || (m.h >= rhs_K);
  return m;
}

void PrintRow(const ColMeta& m) {
  std::cout << std::fixed << std::setprecision(2);
  char flag = ' ';
  if (std::abs(m.rel_delta) > 0.05) flag = '*';
  else if (std::abs(m.rel_delta) > 0.01) flag = '.';
  std::cout << flag << "  "
            << std::setw(9) << KlassName(m.klass)
            << "  ops=[" << std::setw(16) << m.op_sig << "]"
            << "  g=" << std::setw(4) << m.w << "x"
            << std::setw(4) << m.h << "/" << std::setw(4) << m.k
            << "  K=" << std::setw(5) << m.maxK;
  std::cout << "  " << (m.split_k ? "splitK" : "fullK ");
  std::cout << "  pure_sa=" << (m.pure_standalone ? 1 : 0)
            << "  tot_mm=" << m.total_mm
            << "  pw2mm(L,R)=(" << m.pw_to_mm_lhs_K << "," << m.pw_to_mm_rhs_K << ")";
  if (m.split_k && (m.pw_to_mm_lhs_K || m.pw_to_mm_rhs_K)) {
    std::cout << "  align=(" << (m.lhs_aligned ? 1 : 0)
              << "," << (m.rhs_aligned ? 1 : 0) << ")";
  }
  std::cout << "  A=" << std::setw(11) << m.analytic
            << "  E=" << std::setw(11) << m.exact
            << "  Δ=" << std::setw(7) << std::setprecision(2)
            << (m.rel_delta * 100.0) << "%";
  std::cout << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <problem.json> [--top N] "
                                         "[--only-diverge PCT] [--all]\n";
    return 1;
  }
  int top_n = 40;
  double diverge_pct = 1.0;   // show rows with |Δ|>1% in the Outliers table
  bool show_all = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--top" && i + 1 < argc) top_n = std::atoi(argv[++i]);
    else if (a == "--only-diverge" && i + 1 < argc)
      diverge_pct = std::atof(argv[++i]);
    else if (a == "--all") show_all = true;
  }

  auto pr = mlsys::ReadProblem(argv[1]);
  if (!pr.ok()) { std::cerr << pr.status() << "\n"; return 1; }
  mlsys::Problem p = *pr;
  // Mirror the solver pipeline: preprocess first so enumerator sees the same
  // ops as pattern_solver. Keep the unprocessed copy only if we ever need to
  // map columns back to the original op numbering (we don't here).
  auto pre = solver::Preprocess(p);
  const auto& pp = pre.problem;
  auto dag = solver::BuildDAG(pp);

  auto columns = pattern::EnumerateColumns(pp, dag);

  std::vector<ColMeta> metas;
  metas.reserve(columns.size());
  for (const auto& c : columns) metas.push_back(BuildMeta(pp, dag, c));

  // Sort by |Δ| desc.
  std::sort(metas.begin(), metas.end(),
            [](const ColMeta& a, const ColMeta& b) {
              return std::abs(a.rel_delta) > std::abs(b.rel_delta);
            });

  // Aggregate stats.
  int n_total = (int)metas.size();
  int n_diverge = 0, n_pw2mm_lhs = 0, n_pw2mm_rhs = 0;
  int n_splitk = 0, n_splitk_diverge = 0;
  int n_pure_sa_2mm = 0, n_pure_sa_2mm_diverge = 0;
  double sum_abs = 0, max_abs = 0;
  for (const auto& m : metas) {
    double a = std::abs(m.rel_delta);
    sum_abs += a; max_abs = std::max(max_abs, a);
    if (a * 100 > diverge_pct) ++n_diverge;
    if (m.pw_to_mm_lhs_K) ++n_pw2mm_lhs;
    if (m.pw_to_mm_rhs_K) ++n_pw2mm_rhs;
    if (m.split_k) { ++n_splitk; if (a * 100 > diverge_pct) ++n_splitk_diverge; }
    if (m.pure_standalone && m.total_mm == 2) {
      ++n_pure_sa_2mm;
      if (a * 100 > diverge_pct) ++n_pure_sa_2mm_diverge;
    }
  }

  std::cout << "──────────────────────────────────────────────────────────────\n";
  std::cout << "Problem: " << argv[1] << "\n";
  std::cout << "Columns enumerated: " << n_total << "\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Mean |Δ|%: " << (n_total ? sum_abs / n_total * 100 : 0)
            << "   Max |Δ|%: " << max_abs * 100 << "\n";
  std::cout << "|Δ|>" << diverge_pct << "% diverging: " << n_diverge
            << " / " << n_total << "\n";
  std::cout << "  with PW→MM(LHS) edge:  " << n_pw2mm_lhs << " cols\n";
  std::cout << "  with PW→MM(RHS) edge:  " << n_pw2mm_rhs << " cols\n";
  std::cout << "  split-K (k<maxK):      " << n_splitk_diverge << " diverging / "
            << n_splitk << " total\n";
  std::cout << "  pure_sa && tot_mm==2:  " << n_pure_sa_2mm_diverge
            << " diverging / " << n_pure_sa_2mm << " total\n";
  std::cout << "──────────────────────────────────────────────────────────────\n";

  std::cout << "Top " << std::min(top_n, n_total)
            << " |Δ| outliers (klass / ops / gran / maxK / split? / metadata):\n";
  std::cout << "flag  klass      ops                 gran              "
               "K    mode    pure_sa  tot_mm pw→mm(L,R)     costs\n";
  int printed = 0;
  for (const auto& m : metas) {
    if (!show_all && printed >= top_n) break;
    PrintRow(m);
    ++printed;
  }

  // ── Impact probe: for every diverging column (|Δ|>1%), compare the gran
  // ── AnalyticalCost chose vs the gran ExactCost would choose. If they
  // ── coincide, the divergence is harmless (downstream c.cost is optimal).
  int probe_limit_env = 0;
  if (const char* s = std::getenv("PROBE_LIMIT")) probe_limit_env = std::atoi(s);
  int probe_limit = (probe_limit_env > 0) ? probe_limit_env : 10;
  std::cout << "\nImpact probe (|Δ|>1%, limit " << probe_limit
            << "): does A's gran match E's gran?\n";
  std::cout << "      klass       ops            gran_A (w,h,k)        "
               "E_at_A         gran_E (w,h,k)        E_at_E         gain%\n";
  int probed = 0;
  int n_actionable = 0;
  double max_actionable_gain = 0;
  for (size_t i = 0; i < metas.size() && probed < probe_limit; ++i) {
    const auto& m = metas[i];
    if (std::abs(m.rel_delta) < 0.01) break;
    auto best_e = ExactCostBestGran(pp, dag, m.ops);
    double e_at_a = m.exact;
    double e_at_e = best_e.exact_cost;
    double gain = (e_at_a - e_at_e) / std::max(e_at_a, 1.0) * 100.0;
    std::cout << std::fixed << std::setprecision(2);
    char flag = (gain > 1.0) ? '!' : ' ';
    if (gain > 1.0) {
      ++n_actionable;
      max_actionable_gain = std::max(max_actionable_gain, gain);
    }
    std::cout << flag << "  " << std::setw(9) << KlassName(m.klass)
              << "  [" << std::setw(13) << m.op_sig << "]"
              << "  (" << std::setw(4) << m.w << "," << std::setw(4) << m.h
              << "," << std::setw(5) << m.k << ")"
              << "  " << std::setw(12) << e_at_a
              << "    (" << std::setw(4) << best_e.gran.width
              << "," << std::setw(4) << best_e.gran.height
              << "," << std::setw(5) << best_e.gran.depth << ")"
              << "  " << std::setw(12) << e_at_e
              << "  " << std::setw(6) << gain << "%\n";
    ++probed;
  }
  std::cout << "Probe summary: probed=" << probed
            << "  actionable(gain>1%)=" << n_actionable
            << "  max_gain=" << max_actionable_gain << "%\n";
  return 0;
}
