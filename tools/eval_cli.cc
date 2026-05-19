#include <iostream>
#include "mlsys.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <problem.json> <solution.json>\n";
    return 1;
  }
  auto problem = mlsys::ReadProblem(argv[1]);
  if (!problem.ok()) {
    std::cerr << "Error reading problem: " << problem.status() << "\n";
    return 1;
  }
  auto solution = mlsys::ReadSolution(argv[2]);
  if (!solution.ok()) {
    std::cerr << "Error reading solution: " << solution.status() << "\n";
    return 1;
  }
  auto result = mlsys::Evaluate(*problem, *solution);
  if (!result.ok()) {
    std::cerr << "Error: " << result.status() << "\n";
    return 1;
  }
  std::cout << *result << "\n";
  return 0;
}
