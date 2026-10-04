#!/usr/bin/env python3
"""Turn benchmark CSVs/logs into resume-ready numbers. Nothing is hard-coded: every figure
printed comes from the files you pass in.

  python3 scripts/summarize.py --name SIFT1M --cpp results/sift_cpp.csv \
      --lib results/sift_hnswlib.csv --log results/sift_cpp.log
"""
import argparse, csv, math, re


def load(path):
    rows = list(csv.DictReader(open(path)))
    return sorted((float(r["recall"]), float(r["qps_1t"])) for r in rows), rows


def qps_at(pts, target):
    for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
        if r0 <= target <= r1 and r1 > r0:
            t = (target - r0) / (r1 - r0)
            return math.exp(math.log(q0) + t * (math.log(q1) - math.log(q0)))
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--cpp", required=True)
ap.add_argument("--lib", required=True)
ap.add_argument("--log", help="bench stdout log (for build time / thread speedup)")
ap.add_argument("--target", type=float, default=0.95)
a = ap.parse_args()

cpp, cpp_rows = load(a.cpp)
lib, _ = load(a.lib)
qc, ql = qps_at(cpp, a.target), qps_at(lib, a.target)
if not (qc and ql):
    raise SystemExit(f"sweep never reaches recall {a.target}; widen --ef (max recall: cpp {cpp[-1][0]:.3f}, lib {lib[-1][0]:.3f})")
ratio = qc / ql
pct = abs(1 - ratio) * 100
rel = "within %.0f%% of" % pct if pct >= 1 else "on par with"
p99 = next((float(r["p99_us"]) for r in cpp_rows if float(r["recall"]) >= a.target), None)

speed = ""
if a.log:
    m = re.search(r"speedup with (\d+) threads = ([\d.]+)x", open(a.log).read())
    if m:
        speed = f", {float(m.group(2)):.1f}x build speedup on {m.group(1)} threads"

print(f"\n=== {a.name}: numbers ===")
print(f"QPS @ recall@10={a.target}: hnsw-cpp {qc:,.0f} | hnswlib {ql:,.0f} | ratio {ratio:.2f}x")
print(f"\n=== {a.name}: resume bullets (edit to taste) ===")
print(f"- Implemented HNSW approximate nearest-neighbour search from scratch in C++17 (layered graph, heuristic neighbour "
      f"selection, multithreaded lock-per-node build{speed}); reaches {a.target:.2f} recall@10 on {a.name} at "
      f"{qc:,.0f} queries/s/thread, {rel} hnswlib ({ql:,.0f} q/s).")
print("- Validated with ASan/UBSan/TSan; diagnosed and fixed graph orphaning caused by concurrent inserts "
      "(unreachable nodes after parallel builds), restoring full layer-0 connectivity.")
if p99:
    print(f"  (p99 single-query latency at that recall: {p99:.0f} us)")
