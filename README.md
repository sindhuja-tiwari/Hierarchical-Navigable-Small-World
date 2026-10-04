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


## Known limitations / next steps

- L2 only (cosine via normalisation). No inner-product metric, deletion, update, or persistence.
- Fixed capacity (`max_elements`); no resizing.
- Parallel builds are not bit-reproducible (internal ids follow thread arrival order).
- Distance kernel is AVX2+FMA or portable scalar; no AVX-512 / NEON path.
- Ideas with measurable payoffs: `keepPrunedConnections`, int8/SQ vectors (memory vs recall),
  a `save()/load()` format, and a profile-guided look at the remaining gap to hnswlib.
