// mip_pack.cc — HiGHS set-cover implementation.

#include "mip_pack.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "Highs.h"

namespace pattern {

MipResult MipPack(const std::vector<Column>& columns,
                  size_t num_ops,
                  double time_limit_s,
                  CostVariant variant) {
  MipResult res;
  if (columns.empty() || num_ops == 0) return res;

  auto t0 = std::chrono::steady_clock::now();

  const int num_cols = static_cast<int>(columns.size());
  const int num_rows = static_cast<int>(num_ops);

  // Sparse column-major A matrix: for each column, non-zeros = its ops.
  std::vector<HighsInt> a_start;
  std::vector<HighsInt> a_index;
  std::vector<double> a_value;
  a_start.reserve(num_cols + 1);
  a_index.reserve(num_cols * 3);
  a_value.reserve(num_cols * 3);

  for (int j = 0; j < num_cols; ++j) {
    a_start.push_back(static_cast<HighsInt>(a_index.size()));
    // Sort by op index to ensure row indices are ascending per HiGHS convention.
    std::vector<size_t> sorted = columns[j].ops;
    std::sort(sorted.begin(), sorted.end());
    for (size_t op : sorted) {
      if (op >= num_ops) continue;
      a_index.push_back(static_cast<HighsInt>(op));
      a_value.push_back(1.0);
    }
  }
  a_start.push_back(static_cast<HighsInt>(a_index.size()));

  std::vector<double> col_cost(num_cols);
  std::vector<double> col_lower(num_cols, 0.0);
  std::vector<double> col_upper(num_cols, 1.0);
  std::vector<HighsVarType> integrality(num_cols, HighsVarType::kInteger);
  for (int j = 0; j < num_cols; ++j) {
    double c;
    switch (variant) {
      case CostVariant::A: c = columns[j].analytic_cost; break;
      case CostVariant::R: c = columns[j].cost_retain; break;
      case CostVariant::E: default: c = columns[j].cost; break;
    }
    col_cost[j] = std::isfinite(c) ? c : 1e18;
  }

  std::vector<double> row_lower(num_rows, 1.0);
  std::vector<double> row_upper(num_rows, 1.0);

  HighsModel model;
  model.lp_.num_col_ = num_cols;
  model.lp_.num_row_ = num_rows;
  model.lp_.sense_ = ObjSense::kMinimize;
  model.lp_.col_cost_ = col_cost;
  model.lp_.col_lower_ = col_lower;
  model.lp_.col_upper_ = col_upper;
  model.lp_.row_lower_ = row_lower;
  model.lp_.row_upper_ = row_upper;
  model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  model.lp_.a_matrix_.start_ = a_start;
  model.lp_.a_matrix_.index_ = a_index;
  model.lp_.a_matrix_.value_ = a_value;
  model.lp_.integrality_ = integrality;

  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("time_limit", std::max(0.1, time_limit_s));
  // Tight gap: column count is O(10·N) from §8 enumeration, MIP solves in
  // <5ms on all tested BMs — prove true optimum, not a 0.5% neighbourhood.
  highs.setOptionValue("mip_rel_gap", 1e-9);
  highs.setOptionValue("mip_abs_gap", 1e-9);
  highs.passModel(model);
  highs.run();

  res.solve_time_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  res.is_optimal = highs.getModelStatus() == HighsModelStatus::kOptimal;

  const auto& sol = highs.getSolution();
  if (sol.col_value.empty()) return res;

  // Extract selected columns.
  std::vector<bool> covered(num_ops, false);
  for (int j = 0; j < num_cols; ++j) {
    if (sol.col_value[j] > 0.5) {
      res.partition.subgraphs.push_back(columns[j].ops);
      ++res.n_selected;
      for (size_t op : columns[j].ops) {
        if (op < num_ops) covered[op] = true;
      }
    }
  }
  // Defensive: fill any uncovered ops with singletons (should not happen
  // if singleton columns are included in the library).
  for (size_t op = 0; op < num_ops; ++op) {
    if (!covered[op]) {
      res.partition.subgraphs.push_back({op});
    }
  }
  res.objective = highs.getObjectiveValue();
  return res;
}

}  // namespace pattern
