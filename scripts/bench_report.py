#!/usr/bin/env python3
"""Report generator for scripts/bench.sh runs.

Reads a run directory (results/<run-name>/), re-evaluates saved solutions
with eval_cli to get ground-truth GTs, and prints a comparison table.

Usage:
  bench_report.py <run_dir> <eval_cli_path> <repo_root>
  bench_report.py --diff <run_dir_a> <run_dir_b> <eval_cli_path> <repo_root>
"""

import os
import subprocess
import sys
from collections import defaultdict, OrderedDict


def load_meta(run_dir):
    """meta.tsv → dict[(bm, solver)] = (status, elapsed_s)."""
    meta = {}
    p = os.path.join(run_dir, "meta.tsv")
    if not os.path.exists(p):
        return meta
    with open(p) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            bm, solver, status, elapsed = parts
            meta[(bm, solver)] = (status, int(elapsed))
    return meta


def load_lb(run_dir):
    lbs = {}
    p = os.path.join(run_dir, "lb.tsv")
    if not os.path.exists(p):
        return lbs
    with open(p) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            try:
                lbs[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return lbs


def bm_to_problem_path(bm, repo_root):
    if bm.startswith("public-"):
        n = bm.replace("public-", "")
        return os.path.join(repo_root, "benchmarks", f"mlsys-2026-{n}.json")
    return os.path.join(repo_root, "synthetic", "generated", f"{bm}.json")


def eval_gt(eval_cli, problem, solution):
    """Run eval_cli on (problem, solution); returns float GT or None."""
    if not os.path.exists(solution):
        return None
    try:
        out = subprocess.run(
            [eval_cli, problem, solution],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.strip().splitlines()[::-1]:
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    return None


def gather_gts(run_dir, eval_cli, repo_root):
    """Re-evaluate all solutions in run_dir; returns {(bm, solver): gt_or_status}."""
    meta = load_meta(run_dir)
    results = {}
    solutions_dir = os.path.join(run_dir, "solutions")
    for (bm, solver), (status, elapsed) in meta.items():
        if status != "OK":
            results[(bm, solver)] = status
            continue
        sol_path = os.path.join(solutions_dir, solver, f"{bm}.json")
        problem = bm_to_problem_path(bm, repo_root)
        gt = eval_gt(eval_cli, problem, sol_path)
        if gt is None:
            results[(bm, solver)] = "INVALID"
        else:
            results[(bm, solver)] = gt
    return results, meta


def print_table(run_dir, eval_cli, repo_root):
    lbs = load_lb(run_dir)
    results, meta = gather_gts(run_dir, eval_cli, repo_root)

    # Collect solvers and BMs.
    solvers = sorted({s for _, s in results})
    # Pin canonical solver order if present.
    canon = ["pattern", "io", "partition", "unified"]
    ordered = [s for s in canon if s in solvers] + [s for s in solvers if s not in canon]
    solvers = ordered

    bms_public = sorted(
        [b for b in lbs if b.startswith("public-")],
        key=lambda x: int(x.replace("public-", "")),
    )
    bms_synth = sorted([b for b in lbs if not b.startswith("public-")])
    bms = bms_public + bms_synth

    # Metadata header.
    meta_path = os.path.join(run_dir, "metadata.txt")
    if os.path.exists(meta_path):
        print(f"# {os.path.basename(run_dir)}")
        with open(meta_path) as f:
            for line in f:
                print(f"#   {line.rstrip()}")
        print()

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, str):
            return v
        return f"{v:,.0f}"

    # Column widths.
    w_bm = max(len("BM"), max(len(b) for b in bms)) if bms else 4
    w_lb = 14
    w_cell = 14
    header = f"{'BM':<{w_bm}}  {'LB':>{w_lb}}"
    for s in solvers:
        header += f"  {s:>{w_cell}}"
    header += "  winner"
    print(header)
    print("-" * len(header))

    wins = defaultdict(int)
    solo = defaultdict(int)

    for bm in bms:
        lb = lbs.get(bm, 0.0)
        row_vals = {}
        numeric = {}
        for s in solvers:
            v = results.get((bm, s))
            row_vals[s] = v
            if isinstance(v, (int, float)):
                numeric[s] = v
        if numeric:
            best = min(numeric.values())
            winners = [s for s, v in numeric.items() if v == best]
        else:
            winners = []

        for w in winners:
            wins[w] += 1
        if len(winners) == 1:
            solo[winners[0]] += 1

        line = f"{bm:<{w_bm}}  {lb:>{w_lb},.0f}"
        for s in solvers:
            cell = row_vals[s]
            if cell is None:
                cell_str = "MISS"
            elif isinstance(cell, str):
                cell_str = cell
            else:
                cell_str = f"{cell:,.0f}"
            line += f"  {cell_str:>{w_cell}}"
        if winners:
            if len(solvers) > 1 and len(winners) == len(solvers):
                line += "  tie"
            else:
                line += f"  {','.join(winners)}"
        else:
            line += "  none"
        print(line)

    # Summary.
    print()
    print(f"{'solver':<12}  {'solved/total':>14}  {'wins+ties':>11}  {'solo wins':>10}")
    total = len(bms)
    for s in solvers:
        solved = sum(
            1 for bm in bms
            if isinstance(results.get((bm, s)), (int, float))
        )
        print(f"  {s:<10}  {solved:>6}/{total:>6}  {wins[s]:>10}  {solo[s]:>10}")

    # Aggregate Σ GT across BMs each solver could solve (only for comparability).
    print()
    for s in solvers:
        total_gt = sum(
            results[(bm, s)] for bm in bms
            if isinstance(results.get((bm, s)), (int, float))
        )
        print(f"  {s:<10}  Σ GT over solved BMs: {total_gt:,.0f}")


def print_diff(run_a, run_b, eval_cli, repo_root):
    lbs_a = load_lb(run_a)
    lbs_b = load_lb(run_b)
    ra, _ = gather_gts(run_a, eval_cli, repo_root)
    rb, _ = gather_gts(run_b, eval_cli, repo_root)

    bms = sorted(set([bm for bm, _ in ra] + [bm for bm, _ in rb]))
    # Per-BM: find best GT in each run across solvers.
    best_a = {}
    best_b = {}
    for run_data, best in ((ra, best_a), (rb, best_b)):
        per_bm = defaultdict(list)
        for (bm, s), v in run_data.items():
            if isinstance(v, (int, float)):
                per_bm[bm].append((s, v))
        for bm, lst in per_bm.items():
            lst.sort(key=lambda x: x[1])
            best[bm] = lst[0] if lst else None

    w_bm = max(4, max((len(b) for b in bms), default=4))
    print(f"{'BM':<{w_bm}}  {'A best':>20}  {'B best':>20}  {'Δ (B-A)':>14}  {'Δ%':>7}")
    print("-" * (w_bm + 72))
    wins_a = wins_b = ties = 0
    for bm in bms:
        a = best_a.get(bm)
        b = best_b.get(bm)
        a_s = f"{a[1]:,.0f} ({a[0]})" if a else "-"
        b_s = f"{b[1]:,.0f} ({b[0]})" if b else "-"
        if a and b:
            d = b[1] - a[1]
            dp = 100 * d / a[1] if a[1] else 0
            d_str = f"{d:+,.0f}"
            dp_str = f"{dp:+.2f}%"
            if d < 0: wins_b += 1
            elif d > 0: wins_a += 1
            else: ties += 1
        else:
            d_str = "-"
            dp_str = "-"
        print(f"{bm:<{w_bm}}  {a_s:>20}  {b_s:>20}  {d_str:>14}  {dp_str:>7}")
    print()
    print(f"B better on {wins_b} BMs, A better on {wins_a}, tie on {ties}")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--diff":
        print_diff(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) >= 4:
        print_table(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
