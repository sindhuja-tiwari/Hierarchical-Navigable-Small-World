#!/usr/bin/env python3
"""Run the same benchmark with hnswlib (the reference implementation) and write a CSV in
the same schema as ./bench, so scripts/plot.py can overlay the two curves.

  python3 scripts/bench_hnswlib.py --base B.fvecs --query Q.fvecs --gt GT.ivecs \
      --M 16 --efc 200 --threads 8 --out results/hnswlib.csv [--metric angular]

Query timing is single-threaded over the whole query batch (no per-query Python overhead).
"""
import argparse, csv, os, time
import numpy as np
import hnswlib


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    return a.reshape(-1, a[0] + 1)[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", required=True, help="ivecs ground truth written by ./bench --gt")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--metric", default="l2", choices=["l2", "angular"])
    ap.add_argument("--ef", default="10,20,40,60,80,120,200,300,500,800")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/hnswlib.csv")
    args = ap.parse_args()

    base, q, gt = read_fvecs(args.base), read_fvecs(args.query), read_ivecs(args.gt)
    if args.metric == "angular":  # same preprocessing as ./bench: normalise + L2
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    n, d = base.shape
    idx = hnswlib.Index(space="l2", dim=d)
    idx.init_index(max_elements=n, ef_construction=args.efc, M=args.M, random_seed=100)
    t0 = time.perf_counter()
    idx.add_items(base, np.arange(n), num_threads=args.threads)
    build_s = time.perf_counter() - t0
    print(f"hnswlib build: M={args.M} efC={args.efc} threads={args.threads} -> {build_s:.2f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"{'ef':>6} {'recall@k':>9} {'QPS (1 thr)':>12} {'QPS (all)':>12}")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "M", "efc", "ef", "recall", "qps_1t", "qps_mt", "p50_us", "p99_us", "build_s"])
        for ef in [int(x) for x in args.ef.split(",")]:
            idx.set_ef(max(ef, args.k))
            best = min(_timed(idx, q, args.k, 1) for _ in range(args.reps))
            labels, _ = idx.knn_query(q, k=args.k, num_threads=1)
            hit = sum(len(set(labels[i]) & set(gt[i, : args.k])) for i in range(len(q)))
            recall = hit / (len(q) * args.k)
            best_mt = min(_timed(idx, q, args.k, args.threads) for _ in range(args.reps))
            qps1, qpsm = len(q) / best, len(q) / best_mt
            print(f"{ef:6d} {recall:9.4f} {qps1:12.0f} {qpsm:12.0f}")
            w.writerow(["hnswlib", args.M, args.efc, ef, recall, qps1, qpsm, "", "", build_s])
    print("wrote", args.out)


def _timed(idx, q, k, threads):
    t = time.perf_counter()
    idx.knn_query(q, k=k, num_threads=threads)
    return time.perf_counter() - t


if __name__ == "__main__":
    main()
