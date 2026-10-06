#!/usr/bin/env bash
# One-shot: build, download SIFT1M + GloVe-100, benchmark hnsw-cpp and hnswlib on both,
# plot, and print resume bullets filled from the measured numbers.
#   scripts/run_all.sh [threads]        (default: all cores)
# Needs: g++, make, python3 with numpy, matplotlib, h5py, hnswlib (pip install numpy matplotlib h5py hnswlib)
# Time: roughly 30-60 min on 8 cores (the --compare-build single-thread build of 1M points dominates).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "${1:-}" ]; then
  T="$1"
elif command -v nproc >/dev/null 2>&1; then
  T="$(nproc)"
else
  T="$(sysctl -n hw.ncpu)"
fi

CXX="${CXX:-c++}"

make CXX="$CXX" all
make CXX="$CXX" test

if [ ! -f data/sift_base.fvecs ] || [ ! -f data/sift_query.fvecs ]; then
  scripts/fetch_data.sh
fi

[ -f data/glove_base.fvecs ] || python3 scripts/hdf5_to_fvecs.py data/glove-100-angular.hdf5 data/glove
mkdir -p results
EF=10,20,30,40,60,80,120,200,300,500,800

# ---- SIFT-128 (L2) ----
build/bench --base data/sift_base.fvecs --query data/sift_query.fvecs --gt data/sift_gt.ivecs \
    --metric l2 --M 16 --efc 200 --threads "$T" --ef $EF --compare-build --out results/sift_cpp.csv | tee results/sift_cpp.log
python3 scripts/bench_hnswlib.py --base data/sift_base.fvecs --query data/sift_query.fvecs --gt data/sift_gt.ivecs \
    --metric l2 --M 16 --efc 200 --threads "$T" --ef $EF --out results/sift_hnswlib.csv | tee results/sift_hnswlib.log
python3 scripts/plot.py results/sift_cpp.csv results/sift_hnswlib.csv -o results/sift.png \
    --title "SIFT1M (128-d), M=16, efC=200, k=10, 1 thread" | tee results/sift_match.txt

# ---- GloVe-100 (angular) ----
build/bench --base data/glove_base.fvecs --query data/glove_query.fvecs --gt data/glove_gt.ivecs \
    --metric angular --M 16 --efc 200 --threads "$T" --ef $EF --compare-build --out results/glove_cpp.csv | tee results/glove_cpp.log
python3 scripts/bench_hnswlib.py --base data/glove_base.fvecs --query data/glove_query.fvecs --gt data/glove_gt.ivecs \
    --metric angular --M 16 --efc 200 --threads "$T" --ef $EF --out results/glove_hnswlib.csv | tee results/glove_hnswlib.log
python3 scripts/plot.py results/glove_cpp.csv results/glove_hnswlib.csv -o results/glove.png \
    --title "GloVe-100 (angular), M=16, efC=200, k=10, 1 thread" | tee results/glove_match.txt

python3 scripts/summarize.py --name "SIFT1M" --cpp results/sift_cpp.csv --lib results/sift_hnswlib.csv --log results/sift_cpp.log
python3 scripts/summarize.py --name "GloVe-100" --cpp results/glove_cpp.csv --lib results/glove_hnswlib.csv --log results/glove_cpp.log
