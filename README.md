# HNSW in C++17

A from-scratch **HNSW** (Hierarchical Navigable Small World) approximate nearest-neighbour index,
built from the original paper (Malkov & Yashunin). Header-only, no dependencies, with a
multithreaded build, lock-free queries, an AVX2 distance kernel, a correctness suite that also runs
under ASan / UBSan / TSan, and a benchmark harness that measures **recall@10 vs queries-per-second**
against [hnswlib](https://github.com/nmslib/hnswlib) on the same data and ground truth.

![recall vs QPS](results/recall_vs_qps.png)

```
include/hnsw.h        the index (~550 lines)
include/dataset.h     fvecs/ivecs IO, synthetic data, parallel brute-force ground truth
src/bench.cpp         recall@k vs QPS sweep -> CSV
tests/test_hnsw.cpp   unit/property tests
scripts/              dataset download, hnswlib baseline, plotting, result summary, run_all.sh
results/              CSVs, logs and plots produced by the benchmarks
```

## Highlights

* C++17, header-only HNSW index
* No runtime dependencies for the core index
* L2 distance with an AVX2+FMA fast path and portable scalar fallback
* Multithreaded index construction
* Lock-free queries after construction
* Deterministic level generation from `(seed, internal_id)`
* Flat layer-0 adjacency storage for better locality
* Best-first search with configurable `ef`
* Parallel brute-force ground truth generation
* Correctness and invariant tests
* Benchmark harness comparing against `hnswlib`
* Recall@10, single-thread QPS, multi-thread QPS, p50 and p99 latency
* SIFT1M and GloVe-100 benchmark datasets

## Quick start

### Build

Requires a C++17 compiler and `make`.

```bash
make
```

On macOS, Apple Clang works out of the box:

```bash
make CXX=clang++
```

### Run tests

```bash
make test
```

The test suite covers:

* tiny and edge-case indexes
* exact self matches
* recall against brute-force ground truth
* graph invariants
* result ordering
* parallel-build quality

Example output:

```text
tiny / edge cases
recall + invariants (single thread, 5k x 32d)
  recall@10 ef=10: 0.9320   ef=100: 1.0000  levels=2  avg deg0=18.0
exact self match
parallel build (8 threads) matches serial quality
result ordering

all tests passed
```

## Using the index

The core API is intentionally small:

```cpp
hnsw::Index::Params p;
p.dim = 128;
p.max_elements = 1'000'000;
p.M = 16;
p.ef_construction = 200;

hnsw::Index index(p);

index.add_batch(vectors, n, /*threads=*/8);

auto hits = index.search(
    query,
    /*k=*/10,
    /*ef=*/100
);
```

Search results are returned in ascending distance order.

For cosine/angular similarity, normalize vectors to unit length before indexing. The resulting nearest-neighbor ordering is equivalent to angular similarity.

## Benchmarking

The repository includes a reproducible benchmark harness for comparing this implementation with [`hnswlib`](https://github.com/nmslib/hnswlib).

The benchmark configuration used for the reported results is:

| Parameter        |  Value |
| ---------------- | -----: |
| `M`              |     16 |
| `efConstruction` |    200 |
| `k`              |     10 |
| Query `ef`       | 10–800 |
| Build threads    |      4 |
| Queries          | 10,000 |

Datasets:

* **SIFT1M:** 1,000,000 vectors × 128 dimensions, L2 distance
* **GloVe-100:** 1,183,514 vectors × 100 dimensions, angular distance

The benchmark records:

* recall@10
* single-thread query QPS
* multi-thread query QPS
* p50 latency
* p99 latency
* build time
* memory usage

Raw CSVs, logs, plots, and the recorded environment are available under [`results/`](results/).

### Reproducing the benchmark

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install numpy matplotlib h5py hnswlib
```

Then run:

```bash
scripts/run_all.sh 4
```

The script:

1. builds the C++ implementation
2. runs the correctness tests
3. downloads the benchmark datasets when needed
4. converts GloVe HDF5 data to `fvecs`
5. runs hnsw-cpp
6. runs hnswlib with the same HNSW parameters
7. generates CSV results and plots
8. writes benchmark logs and summary information

> **Runtime note:** the full benchmark can take many hours on a laptop because `--compare-build` also measures a single-thread build of the full datasets. The reported run was performed on a 4-core Intel Mac.

Datasets are intentionally excluded from Git via `.gitignore`.

## Measured results

The following results were measured on an Intel Mac, x86_64, with Apple Clang 16, using four build/query threads.

### SIFT1M

Configuration:

* 1,000,000 × 128-d vectors
* L2 distance
* `M=16`
* `efConstruction=200`
* 10,000 queries
* 4-thread build

#### Build

| Implementation      |   Build time |
| ------------------- | -----------: |
| hnsw-cpp, 4 threads | **415.96 s** |
| hnsw-cpp, 1 threa   |              |

