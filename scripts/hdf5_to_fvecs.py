#!/usr/bin/env python3
"""Convert an ann-benchmarks .hdf5 file (train/test/neighbors) to fvecs/ivecs for ./bench.

  python3 scripts/hdf5_to_fvecs.py data/glove-100-angular.hdf5 data/glove
  -> data/glove_base.fvecs  data/glove_query.fvecs  data/glove_gt.ivecs
Requires: pip install h5py numpy
"""
import sys
import h5py
import numpy as np


def write(path, arr, dtype):
    arr = np.ascontiguousarray(arr.astype(dtype))
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.int32 if dtype == np.int32 else np.float32)
    head = np.full((n,), d, dtype=np.int32)
    if dtype == np.int32:
        out[:, 0] = head
    else:
        out[:, 0] = head.view(np.float32)
    out[:, 1:] = arr
    out.tofile(path)


src, prefix = sys.argv[1], sys.argv[2]
with h5py.File(src, "r") as f:
    write(prefix + "_base.fvecs", np.array(f["train"]), np.float32)
    write(prefix + "_query.fvecs", np.array(f["test"]), np.float32)
    write(prefix + "_gt.ivecs", np.array(f["neighbors"]), np.int32)
print("wrote", prefix + "_{base,query}.fvecs and", prefix + "_gt.ivecs")
