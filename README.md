# TeamKorea — MLSys 2026 Scheduling Contest

TeamKorea's pattern-based solution for the [MLSys 2026 scheduling contest](https://github.com/yarongmu-google/MLSys).

TeamKorea's agent-based solution, submitted to Track B, is available at [chnlee/TeamKorea](https://github.com/chnlee/TeamKorea).

## Build

Requires cmake ≥ 3.16, ninja, g++ with C++20.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mlsys
```

Dependencies (abseil, HiGHS, nlohmann/json) are fetched automatically via CMake FetchContent. The binary has no runtime dependencies beyond libc/libm.

## Run

```bash
./build/mlsys benchmarks/mlsys-2026-17.json /tmp/out.json
./build/eval_cli benchmarks/mlsys-2026-17.json /tmp/out.json   # evaluate cost
./build/lower_bound benchmarks/mlsys-2026-17.json              # compute/IO lower bound
```

## Tests

```bash
./build/eval_test
./build/eval_corner_test
```

## Benchmark Costs

Pattern solver results on all 24 benchmarks.

| Benchmark | \# Nodes | \# Edges | Timeout | Pattern Solver |
| :---- | :---- | :---- | :---- | ----: |
| [mlsys-2026-1](./benchmarks/mlsys-2026-1.json) | 5 | 9 | 2s | 322,008 |
| [mlsys-2026-2](./benchmarks/mlsys-2026-2.json) | 5 | 7 | 2s | 48,738 |
| [mlsys-2026-3](./benchmarks/mlsys-2026-3.json) | 4 | 6 | 2s | 65,982 |
| [mlsys-2026-4](./benchmarks/mlsys-2026-4.json) | 5 | 10 | 2s | 22,281 |
| [mlsys-2026-5](./benchmarks/mlsys-2026-5.json) | 19 | 34 | 5s | 492,614 |
| [mlsys-2026-6](./benchmarks/mlsys-2026-6.json) | 17 | 29 | 5s | 166,579 |
| [mlsys-2026-7](./benchmarks/mlsys-2026-7.json) | 15 | 21 | 5s | 115,853 |
| [mlsys-2026-8](./benchmarks/mlsys-2026-8.json) | 20 | 37 | 5s | 74,470 |
| [mlsys-2026-9](./benchmarks/mlsys-2026-9.json) | 32 | 56 | 15s | 20,411,920 |
| [mlsys-2026-10](./benchmarks/mlsys-2026-10.json) | 28 | 47 | 15s | 6,224,041 |
| [mlsys-2026-11](./benchmarks/mlsys-2026-11.json) | 26 | 38 | 15s | 1,040,921 |
| [mlsys-2026-12](./benchmarks/mlsys-2026-12.json) | 31 | 46 | 15s | 2,766,694 |
| [mlsys-2026-13](./benchmarks/mlsys-2026-13.json) | 63 | 126 | 30s | 15,297,786 |
| [mlsys-2026-14](./benchmarks/mlsys-2026-14.json) | 63 | 96 | 30s | 2,654,400 |
| [mlsys-2026-15](./benchmarks/mlsys-2026-15.json) | 61 | 97 | 30s | 1,314,179 |
| [mlsys-2026-16](./benchmarks/mlsys-2026-16.json) | 63 | 85 | 30s | 4,348,356 |
| [mlsys-2026-17](./benchmarks/mlsys-2026-17.json) | 103 | 198 | 60s | 4,966,400 |
| [mlsys-2026-18](./benchmarks/mlsys-2026-18.json) | 96 | 176 | 60s | 1,417,600 |
| [mlsys-2026-19](./benchmarks/mlsys-2026-19.json) | 103 | 154 | 60s | 1,993,600 |
| [mlsys-2026-20](./benchmarks/mlsys-2026-20.json) | 103 | 178 | 60s | 5,228,800 |
| [mlsys-2026-21](./benchmarks/mlsys-2026-21.json) | 152 | 280 | 120s | 2,102,400 |
| [mlsys-2026-22](./benchmarks/mlsys-2026-22.json) | 150 | 240 | 120s | 3,096,000 |
| [mlsys-2026-23](./benchmarks/mlsys-2026-23.json) | 121 | 186 | 120s | 3,435,200 |
| [mlsys-2026-24](./benchmarks/mlsys-2026-24.json) | 112 | 192 | 120s | 2,526,400 |

## Problem

See [PROBLEM.md](./PROBLEM.md) for the full problem specification.
