# Hierarchical-Navigable-Small-World (cpp)

A from-scratch **HNSW** (Hierarchical Navigable Small World) approximate nearest-neighbour index in
C++17, built from the original paper (Malkov & Yashunin). Header-only, no dependencies, with a
multithreaded build, lock-free queries, a correctness/sanitizer test suite, and a benchmark harness
that measures **recall@10 vs queries-per-second** against [hnswlib](https://github.com/nmslib/hnswlib).

```
include/hnsw.h        the index (~550 lines)
include/dataset.h     fvecs/ivecs IO, synthetic data, parallel brute-force ground truth
src/bench.cpp         recall@k vs QPS sweep -> CSV
tests/test_hnsw.cpp   unit/property tests (also run under ASan, UBSan, TSan)
scripts/              hnswlib baseline, plotting, SIFT/GloVe download + conversion
```

## Quick start

```bash
make            # builds build/bench and build/test_hnsw  (-O3 -march=native)
make test       # correctness suite
make asan tsan  # same suite under AddressSanitizer+UBSan / ThreadSanitizer
```

```cpp
#include "hnsw.h"
hnsw::Index::Params p; p.dim = 128; p.max_elements = 1'000'000; p.M = 16; p.ef_construction = 200;
hnsw::Index index(p);
index.add_batch(vectors, n, /*threads=*/8);            // row-major float*, labels = row number
auto hits = index.search(query, /*k=*/10, /*ef=*/100); // vector<pair<squared_dist, label>>, ascending
```

## Reproducing the benchmark on the standard datasets

```bash
scripts/fetch_data.sh                                    # SIFT1M (fvecs) + GloVe-100 (hdf5)
python3 scripts/hdf5_to_fvecs.py data/glove-100-angular.hdf5 data/glove

# SIFT-128 (L2)
build/bench --base data/sift_base.fvecs --query data/sift_query.fvecs --gt data/sift_gt.ivecs \
            --M 16 --efc 200 --threads 8 --compare-build --out results/sift_cpp.csv
python3 scripts/bench_hnswlib.py --base data/sift_base.fvecs --query data/sift_query.fvecs \
            --gt data/sift_gt.ivecs --M 16 --efc 200 --threads 8 --out results/sift_hnswlib.csv
python3 scripts/plot.py results/sift_cpp.csv results/sift_hnswlib.csv -o results/sift.png --title "SIFT-128, M=16, efC=200"

# GloVe-100 (angular; vectors are L2-normalised, which preserves the ordering)
build/bench --base data/glove_base.fvecs --query data/glove_query.fvecs --gt data/glove_gt.ivecs --metric angular ...
```

`bench` computes exact ground truth by brute force the first time and caches it to `--gt`; the
hnswlib script reads the same file, so both implementations are scored against identical truth.
Always run hnswlib and `bench` with the same `M`, `efC` and thread count.

## Results so far

> **These are not SIFT/GloVe numbers.** They come from a 200k x 128-d synthetic Gaussian-mixture
> dataset, run in a sandbox with **1 CPU core** (so no multithreaded-speedup figure yet), because
> the standard datasets weren't downloadable there. They validate the implementation and the
> methodology; the SIFT-1M / GloVe-100 runs above are what belong in a write-up.

200k x 128-d, 1000 queries, M=16, efC=200, k=10, single-threaded queries, same data and ground truth:

| target recall@10 | hnsw-cpp QPS | hnswlib 0.8.0 QPS | ratio |
|---|---|---|---|
| 0.90 | 8,608 | 8,708 | 0.99x |
| 0.95 | 6,420 | 6,667 | 0.96x |
| 0.99 | 4,414 | 4,300 | 1.03x |

(QPS interpolated at fixed recall, since the two graphs reach slightly different recall at the same
`ef`; raw sweeps in `results/`.) Single-thread build: 48.6 s vs 54.5 s for hnswlib.
Plot: `results/recall_vs_qps.png`.

Caveats worth stating when you quote this: synthetic data, one machine, timing harnesses differ
(my per-query loop in C++ vs hnswlib's batched call from Python), and hnswlib dispatches
AVX/AVX512 at runtime while this uses AVX2+FMA.

## Design notes (interview material)

**Structure.** Each point gets a max layer `floor(-ln(U) / ln M)`. Layer 0 holds everyone with up to
`2M` links; layers >= 1 hold exponentially fewer points with up to `M` links. Level assignment is a
pure function of `(seed, internal id)` (splitmix64), so there is no shared RNG to contend on.

**Search (Alg. 5).** Greedy 1-NN descent from the entry point through the upper layers, then a
best-first search on layer 0 keeping the `ef` closest. `ef` is the recall/speed knob.

**Neighbour selection (Alg. 4).** The heuristic keeps a candidate only if it is closer to the new
point than to every neighbour already kept. This spreads edges across directions rather than
picking the M nearest (which all sit in the same cluster), and is what keeps clustered data
navigable. Same heuristic shrinks a neighbour's list when a reverse edge overflows it.

**Memory layout.** Vectors are one contiguous array. Layer-0 links are a flat `[count, ids...]`
block per node (cache-friendly, no pointer chasing); upper layers are per-node vectors since few
nodes have them. Neighbour data is software-prefetched before the distance loop. Visited-set
reset is O(1) via an epoch counter in a `thread_local` array.

**Concurrency.**
- One mutex per node guards that node's neighbour lists; a thread holds at most one node lock at a
  time, so lock-order deadlock is impossible by construction.
- A global mutex is taken only by an insert that will raise the top layer (rare: ~1/M per layer).
- A node's vector/level are written before it becomes reachable; it links itself first, then
  neighbours link back. Queries after the build take no locks at all.
- Verified with ThreadSanitizer, ASan and UBSan on the full test suite.

**A real finding: concurrent inserts orphan nodes.** Two inserts running at the same moment can't
see each other. One can be pruned out of every neighbour list by the heuristic, leaving it (and
anything only reachable through it) with no path from the entry point. In an 8-thread build of 20k
points this was 0.1-0.5% of nodes (serial: ~0-0.02%), and it grows with thread count. I confirmed
it is an algorithmic effect, not a data race: TSan is clean, and running 8 threads with every
`add()` behind one big lock gives the serial result. `repair_connectivity()` runs at the end of
`add_batch`: for each orphan it finds the nearest reachable nodes and adds an edge from the first
one that can spare a slot without orphaning someone else (target in-degree > 1). After repair the
parallel build has 0 unreachable nodes and recall within ~0.5% of serial.

## Known limitations / next steps

- L2 only (cosine via normalisation). No inner-product metric, deletion, update, or persistence.
- Fixed capacity (`max_elements`); no resizing.
- Parallel builds are not bit-reproducible (internal ids follow thread arrival order).
- Distance kernel is AVX2+FMA or portable scalar; no AVX-512 / NEON path.
- Ideas with measurable payoffs: `keepPrunedConnections`, int8/SQ vectors (memory vs recall),
  a `save()/load()` format, and a profile-guided look at the remaining gap to hnswlib.
