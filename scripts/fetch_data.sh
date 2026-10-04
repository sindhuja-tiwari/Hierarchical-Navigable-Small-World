#!/usr/bin/env bash
# Download the standard ANN benchmark datasets (needs network; not available in every sandbox).
#   SIFT1M  : 1,000,000 x 128-d, L2        (corpus-texmex.irisa.fr, fvecs)
#   GloVe-100: 1,183,514 x 100-d, angular   (ann-benchmarks.com, hdf5)
set -euo pipefail
mkdir -p data && cd data
if [ ! -f sift_base.fvecs ]; then
  curl -L -o sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  tar xzf sift.tar.gz --strip-components=1 sift/sift_base.fvecs sift/sift_query.fvecs
  rm sift.tar.gz
fi
if [ ! -f glove-100-angular.hdf5 ]; then
  curl -L -O http://ann-benchmarks.com/glove-100-angular.hdf5
fi
echo "done. For GloVe: python3 ../scripts/hdf5_to_fvecs.py glove-100-angular.hdf5 glove"
echo "SIFT ground truth is computed (and cached) by ./bench --gt data/sift_gt.ivecs"
