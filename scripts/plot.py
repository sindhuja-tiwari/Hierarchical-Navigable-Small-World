#!/usr/bin/env python3
"""Plot recall@k (x) vs queries/sec (y, log) for one or more result CSVs, ann-benchmarks style.

  python3 scripts/plot.py results/hnsw_cpp.csv results/hnswlib.csv -o results/recall_vs_qps.png \
      --title "SIFT-128, M=16, efC=200, k=10"
"""
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+")
ap.add_argument("-o", "--out", default="results/recall_vs_qps.png")
ap.add_argument("--title", default="recall@10 vs QPS")
ap.add_argument("--col", default="qps_1t", choices=["qps_1t", "qps_mt"])
ap.add_argument("--xmin", type=float, default=0.90)
ap.add_argument("--targets", default="0.90,0.95,0.99", help="recall levels at which to print interpolated QPS")
args = ap.parse_args()

import math
series = {}
styles = {"hnsw-cpp": ("o-", "#1f77b4"), "hnswlib": ("s--", "#d62728")}
fig, ax = plt.subplots(figsize=(7, 4.6))
for path in args.csvs:
    rows = list(csv.DictReader(open(path)))
    impl = rows[0]["impl"]
    pts = sorted((float(r["recall"]), float(r[args.col])) for r in rows)
    series[impl] = pts
    pts = [p for p in pts if p[0] >= args.xmin]
    fmt, color = styles.get(impl, ("^-", None))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], fmt, label=impl, color=color, ms=4)
ax.set_yscale("log")
ax.set_xlabel("recall@10")
ax.set_ylabel("queries / second" + (" (1 thread)" if args.col == "qps_1t" else " (all threads)"))
ax.set_title(args.title)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print("wrote", args.out)


def qps_at(pts, target):
    """Log-linear interpolation of QPS at a recall target (None if the sweep never reaches it)."""
    for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
        if r0 <= target <= r1 and r1 > r0:
            t = (target - r0) / (r1 - r0)
            return math.exp(math.log(q0) + t * (math.log(q1) - math.log(q0)))
    return None


targets = [float(t) for t in args.targets.split(",")]
impls = list(series)
print("\nQPS at fixed recall@10 (interpolated, %s):" % args.col)
print("%-10s" % "recall" + "".join("%12s" % i for i in impls) + ("%18s" % ("ratio" ) if len(impls) == 2 else ""))
for t in targets:
    vals = [qps_at(series[i], t) for i in impls]
    row = "%-10.2f" % t + "".join("%12s" % ("%.0f" % v if v else "-") for v in vals)
    if len(vals) == 2 and all(vals):
        row += "%17.2fx" % (vals[0] / vals[1])
    print(row)
