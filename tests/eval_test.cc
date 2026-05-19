#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "mlsys.h"

struct TestCase {
  std::string name;
  std::string problem_file;
  std::string solution_file;
  double expected;
  double tolerance;
};

int main() {
  std::vector<TestCase> tests = {
      {"Ex1A", "tests/examples/ex1a_problem.json",
       "tests/examples/ex1a_solution.json", 6553.6, 0.1},
      {"Ex1B", "tests/examples/ex1b_problem.json",
       "tests/examples/ex1b_solution.json", 3276.8, 0.1},
      {"Ex1C", "tests/examples/ex1c_problem.json",
       "tests/examples/ex1c_solution.json", 4400.0, 0.1},
      {"Ex2A", "tests/examples/ex2a_problem.json",
       "tests/examples/ex2a_solution.json", 26214.4, 0.1},
      {"Ex2B", "tests/examples/ex2b_problem.json",
       "tests/examples/ex2b_solution.json", 13107.2, 0.1},
      {"Ex3A", "tests/examples/ex3a_problem.json",
       "tests/examples/ex3a_solution.json", 11468.8, 0.1},
      {"Ex3B", "tests/examples/ex3b_problem.json",
       "tests/examples/ex3b_solution.json", 6276.8, 0.1},
      {"Ex3C", "tests/examples/ex3c_problem.json",
       "tests/examples/ex3c_solution.json", 4638.4, 0.1},
      {"Ex4A", "tests/examples/ex4a_problem.json",
       "tests/examples/ex4a_solution.json", 7096.0, 0.1},
      {"Ex4B", "tests/examples/ex4b_problem.json",
       "tests/examples/ex4b_solution.json", 6548.0, 0.1},
      {"Ex5B", "tests/examples/ex5b_problem.json",
       "tests/examples/ex5b_solution.json", 6915.2, 0.1},
      // RHS-chain mirror of Ex5B: A @ (B @ C). Validates §8 symmetry —
      // evaluator must score identically at 6915.2 given matching dims.
      {"Ex5B_RHS", "tests/examples/ex5b_rhs_problem.json",
       "tests/examples/ex5b_rhs_solution.json", 6915.2, 0.1},
  };

  int passed = 0, failed = 0;
  for (const auto& tc : tests) {
    auto problem = mlsys::ReadProblem(tc.problem_file);
    if (!problem.ok()) {
      std::cerr << tc.name << ": FAIL (read problem: " << problem.status()
                << ")\n";
      ++failed;
      continue;
    }
    auto solution = mlsys::ReadSolution(tc.solution_file);
    if (!solution.ok()) {
      std::cerr << tc.name << ": FAIL (read solution: " << solution.status()
                << ")\n";
      ++failed;
      continue;
    }
    auto result = mlsys::Evaluate(*problem, *solution);
    if (!result.ok()) {
      std::cerr << tc.name << ": FAIL (evaluate: " << result.status() << ")\n";
      ++failed;
      continue;
    }
    double diff = std::abs(*result - tc.expected);
    if (diff <= tc.tolerance) {
      std::cout << tc.name << ": PASS (" << *result << ")\n";
      ++passed;
    } else {
      std::cerr << tc.name << ": FAIL (got " << *result << ", expected "
                << tc.expected << ", diff " << diff << ")\n";
      ++failed;
    }
  }
  std::cout << "\n" << passed << " passed, " << failed << " failed out of "
            << tests.size() << "\n";
  return failed > 0 ? 1 : 0;
}
