// dataset.h - fvecs/ivecs IO, synthetic data, parallel brute-force ground truth.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hnsw.h"

namespace ds {

struct Matrix {
    size_t n = 0, d = 0;
    std::vector<float> v;
    const float* row(size_t i) const { return &v[i * d]; }
};

// fvecs: repeated [int32 dim][dim x float32]. ivecs: same with int32 payload.
inline Matrix read_fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    Matrix m;
    int32_t dim;
    while (f.read(reinterpret_cast<char*>(&dim), 4)) {
        if (m.d == 0) m.d = static_cast<size_t>(dim);
        if (static_cast<size_t>(dim) != m.d) throw std::runtime_error("ragged fvecs: " + path);
        size_t off = m.v.size();
        m.v.resize(off + m.d);
        if (!f.read(reinterpret_cast<char*>(&m.v[off]), static_cast<std::streamsize>(m.d * 4)))
            throw std::runtime_error("truncated fvecs: " + path);
        ++m.n;
    }
    return m;
}

inline void write_fvecs(const std::string& path, const Matrix& m) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    int32_t dim = static_cast<int32_t>(m.d);
    for (size_t i = 0; i < m.n; ++i) {
        f.write(reinterpret_cast<const char*>(&dim), 4);
        f.write(reinterpret_cast<const char*>(m.row(i)), static_cast<std::streamsize>(m.d * 4));
    }
}

struct IMatrix {
    size_t n = 0, w = 0;
    std::vector<int32_t> v;
    const int32_t* row(size_t i) const { return &v[i * w]; }
};

inline bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

inline IMatrix read_ivecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    IMatrix m;
    int32_t w;
    while (f.read(reinterpret_cast<char*>(&w), 4)) {
        if (m.w == 0) m.w = static_cast<size_t>(w);
        size_t off = m.v.size();
        m.v.resize(off + m.w);
        if (!f.read(reinterpret_cast<char*>(&m.v[off]), static_cast<std::streamsize>(m.w * 4)))
            throw std::runtime_error("truncated ivecs: " + path);
        ++m.n;
    }
    return m;
}

inline void write_ivecs(const std::string& path, const IMatrix& m) {
    std::ofstream f(path, std::ios::binary);
    int32_t w = static_cast<int32_t>(m.w);
    for (size_t i = 0; i < m.n; ++i) {
        f.write(reinterpret_cast<const char*>(&w), 4);
        f.write(reinterpret_cast<const char*>(m.row(i)), static_cast<std::streamsize>(m.w * 4));
    }
}

inline void normalize(Matrix& m) {
    for (size_t i = 0; i < m.n; ++i) {
        float* r = &m.v[i * m.d];
        double s = 0;
        for (size_t j = 0; j < m.d; ++j) s += double(r[j]) * r[j];
        float inv = s > 0 ? static_cast<float>(1.0 / std::sqrt(s)) : 0.f;
        for (size_t j = 0; j < m.d; ++j) r[j] *= inv;
    }
}

// Gaussian-mixture data: `clusters` centres ~ N(0, I), points ~ centre + N(0, I) (heavily
// overlapping clusters, so nearest neighbours are genuinely hard to separate).
// Clustered data is a much better stand-in for real embeddings than i.i.d. noise.
// Same `seed` => same centres; different `stream` => independent points (use stream=1 for queries).
inline Matrix synthetic(size_t n, size_t d, size_t clusters, uint64_t seed, uint64_t stream = 0) {
    std::mt19937_64 crng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> centres(clusters * d);
    for (auto& x : centres) x = g(crng);
    std::mt19937_64 rng(seed * 1000003ULL + stream + 1);
    Matrix m;
    m.n = n;
    m.d = d;
    m.v.resize(n * d);
    for (size_t i = 0; i < n; ++i) {
        const float* c = &centres[(rng() % clusters) * d];
        for (size_t j = 0; j < d; ++j) m.v[i * d + j] = c[j] + g(rng);
    }
    return m;
}

// Exact k-NN by brute force, `threads` workers. Row q holds the ids of the k nearest base points.
inline IMatrix brute_force_knn(const Matrix& base, const Matrix& queries, size_t k, size_t threads) {
    IMatrix gt;
    gt.n = queries.n;
    gt.w = k;
    gt.v.assign(queries.n * k, 0);
    std::atomic<size_t> next{0};
    auto work = [&] {
        std::vector<std::pair<float, int32_t>> dist(base.n);
        for (;;) {
            size_t q = next.fetch_add(1);
            if (q >= queries.n) break;
            for (size_t i = 0; i < base.n; ++i) dist[i] = {hnsw::l2_sq(queries.row(q), base.row(i), base.d), static_cast<int32_t>(i)};
            std::partial_sort(dist.begin(), dist.begin() + static_cast<std::ptrdiff_t>(k), dist.end());
            for (size_t j = 0; j < k; ++j) gt.v[q * k + j] = dist[j].second;
        }
    };
    std::vector<std::thread> pool;
    for (size_t t = 0; t < std::max<size_t>(1, threads); ++t) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    return gt;
}

}  // namespace ds
