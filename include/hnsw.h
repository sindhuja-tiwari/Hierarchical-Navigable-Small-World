#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// SIMD backend: AVX2+FMA (x86-64), NEON (Apple Silicon / AArch64), else portable scalar.
// Define HNSW_NO_SIMD to force the scalar kernel (useful for apples-to-apples comparisons).
#if !defined(HNSW_NO_SIMD) && defined(__AVX2__) && defined(__FMA__)
#define HNSW_USE_AVX2 1
#include <immintrin.h>
#elif !defined(HNSW_NO_SIMD) && defined(__aarch64__) && defined(__ARM_NEON)
#define HNSW_USE_NEON 1
#include <arm_neon.h>
#endif

namespace hnsw {

using node_t = uint32_t;   // internal id (insertion order of arrival)
using label_t = uint64_t;  // user-visible id

// Which distance kernel this translation unit was compiled with (printed by bench/tests/CI).
inline const char* simd_backend() {
#if defined(HNSW_USE_AVX2)
    return "AVX2+FMA";
#elif defined(HNSW_USE_NEON)
    return "NEON";
#else
    return "scalar";
#endif
}

// ---------------------------------------------------------------------------
// Distance: squared L2. AVX2+FMA (x86, -march=native) or NEON (arm64) when available.
// ---------------------------------------------------------------------------
inline float l2_sq(const float* a, const float* b, size_t d) {
#if defined(HNSW_USE_AVX2)
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    for (; i + 8 <= d; i += 8) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < d; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
#elif defined(HNSW_USE_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
    }
    for (; i + 4 <= d; i += 4) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = vfmaq_f32(acc0, d0, d0);
    }
    float s = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < d; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
#else
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float d0 = a[i] - b[i], d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2], d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0; s1 += d1 * d1; s2 += d2 * d2; s3 += d3 * d3;
    }
    for (; i < d; ++i) {
        float t = a[i] - b[i];
        s0 += t * t;
    }
    return (s0 + s1) + (s2 + s3);
#endif
}

// ---------------------------------------------------------------------------
class Index {
public:
    static constexpr size_t kMaxM = 128;
    static constexpr int kMaxLevel = 32;

    struct Params {
        size_t dim = 0;
        size_t max_elements = 0;
        size_t M = 16;                 // max out-degree on layers >= 1 (layer 0 uses 2*M)
        size_t ef_construction = 200;  // candidate-list size while building
        uint64_t seed = 100;           // level assignment is a pure function of (seed, id)
    };

    explicit Index(const Params& p) : p_(p) {
        if (p.dim == 0 || p.max_elements == 0) throw std::invalid_argument("dim and max_elements must be > 0");
        if (p.M < 2 || p.M > kMaxM) throw std::invalid_argument("M must be in [2, 128]");
        if (p.max_elements >= UINT32_MAX) throw std::invalid_argument("max_elements too large");
        M_ = p.M;
        M0_ = 2 * p.M;
        mult_ = 1.0 / std::log(static_cast<double>(M_));
        data_.assign(p.max_elements * p.dim, 0.0f);
        labels_.assign(p.max_elements, 0);
        levels_.assign(p.max_elements, 0);
        l0_.assign(p.max_elements * (M0_ + 1), 0);
        upper_.resize(p.max_elements);
        locks_.reset(new std::mutex[p.max_elements]);
    }

    size_t size() const { return count_.load(); }
    size_t dim() const { return p_.dim; }
    const Params& params() const { return p_; }

    // Insert one vector. Thread-safe with respect to other add() calls.
    void add(const float* v, label_t label) {
        const size_t id = count_.fetch_add(1);
        if (id >= p_.max_elements) {
            count_.fetch_sub(1);
            throw std::length_error("hnsw::Index is full");
        }
        const int level = random_level(id);
        std::memcpy(&data_[id * p_.dim], v, p_.dim * sizeof(float));
        labels_[id] = label;
        levels_[id] = level;
        upper_[id].assign(static_cast<size_t>(level) * (M_ + 1), 0);

        // Only an insert that may raise the top layer serialises on the global lock.
        std::unique_lock<std::mutex> glock(global_, std::defer_lock);
        int64_t ep = entry_.load();
        if (ep < 0 || level > levels_[ep]) {
            glock.lock();
            ep = entry_.load();  // may have changed while we waited
        }
        if (ep < 0) {  // very first element
            entry_.store(static_cast<int64_t>(id));
            return;
        }

        const int top = levels_[ep];
        node_t cur = static_cast<node_t>(ep);
        float cur_d = l2_sq(v, vec(cur), p_.dim);

        // Phase 1: greedy descent through layers above the new node's level.
        for (int l = top; l > level; --l) cur = greedy<true>(v, cur, cur_d, l);

        // Phase 2: from min(level, top) down to 0, find ef_construction candidates,
        // choose M neighbours with the heuristic and wire up bidirectional links.
        for (int l = std::min(level, top); l >= 0; --l) {
            MaxHeap heap = search_layer<true>(v, cur, cur_d, p_.ef_construction, l);
            std::vector<Cand> cands(heap.size());
            for (size_t i = heap.size(); i-- > 0; heap.pop()) cands[i] = heap.top();  // ascending
            cur = cands.front().second;  // closest becomes entry point for next layer
            cur_d = cands.front().first;
            std::vector<Cand> sel = select_neighbors(cands, M_, static_cast<node_t>(id));
            connect(static_cast<node_t>(id), sel, l);
        }

        if (level > top) entry_.store(static_cast<int64_t>(id));
    }

    // Insert n row-major vectors with `threads` workers. Labels are first_label + row.
    // Returns the number of nodes still unreachable after repair (0 when the graph is whole).
    size_t add_batch(const float* data, size_t n, size_t threads, label_t first_label = 0) {
        if (n == 0) return 0;
        const bool seeded = (size() == 0);
        if (seeded) add(data, first_label);  // seed the graph before going parallel
        std::atomic<size_t> next{seeded ? 1u : 0u};
        if (threads <= 1) {
            for (size_t i = next; i < n; ++i) add(data + i * p_.dim, first_label + i);
            return repair_connectivity();
        }
        std::vector<std::thread> pool;
        std::vector<std::exception_ptr> errs(threads);
        for (size_t t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                try {
                    for (;;) {
                        size_t i = next.fetch_add(1);
                        if (i >= n) break;
                        add(data + i * p_.dim, first_label + i);
                    }
                } catch (...) {
                    errs[t] = std::current_exception();
                }
            });
        }
        for (auto& th : pool) th.join();
        for (auto& e : errs)
            if (e) std::rethrow_exception(e);
        return repair_connectivity();  // concurrent inserts can orphan a few nodes; see below
    }

    // Re-attach nodes that have no layer-0 path from the entry point. Two inserts that run
    // at the same moment cannot see each other, so one of them can be pruned out of every
    // neighbour list and (with its local island) become unreachable. Typically 0.1-0.5% of
    // nodes after a parallel build; serial builds almost never need this.
    // For each orphan we find its nearest reachable nodes and give the first one that can
    // spare an edge (an edge to a node with in-degree > 1, so nobody else is orphaned) a
    // link to it. Single-threaded; call after the build. Returns nodes still unreachable.
    size_t repair_connectivity(size_t max_rounds = 5) {
        const size_t n = size();
        for (size_t round = 0; round < max_rounds; ++round) {
            std::vector<node_t> un = unreachable_ids();
            if (un.empty()) return 0;
            std::vector<uint32_t> indeg(n, 0);
            for (node_t u = 0; u < n; ++u) {
                const node_t* L = links(u, 0);
                for (size_t i = 0; i < L[0]; ++i) ++indeg[L[1 + i]];
            }
            std::vector<char> orphan(n, 0);
            for (node_t u : un) orphan[u] = 1;
            const node_t ep = static_cast<node_t>(entry_.load());
            for (node_t u : un) {
                if (!orphan[u]) continue;  // reached through an earlier repair this round
                float d0 = l2_sq(vec(u), vec(ep), p_.dim);
                node_t cur = ep;
                for (int l = levels_[ep]; l > 0; --l) cur = greedy<false>(vec(u), cur, d0, l);
                MaxHeap heap = search_layer<false>(vec(u), cur, d0, p_.ef_construction, 0);
                std::vector<Cand> cands(heap.size());
                for (size_t i = heap.size(); i-- > 0; heap.pop()) cands[i] = heap.top();
                bool attached = false;
                for (const Cand& c : cands) {
                    node_t* L = links(c.second, 0);
                    size_t cnt = L[0];
                    if (orphan[c.second]) continue;
                    if (cnt < M0_) {
                        L[1 + cnt] = u;
                        L[0] = static_cast<node_t>(cnt + 1);
                        attached = true;
                    } else {  // replace the farthest edge whose target keeps another in-edge
                        size_t worst = cnt;
                        float wd = -1;
                        for (size_t i = 0; i < cnt; ++i) {
                            if (indeg[L[1 + i]] <= 1) continue;
                            float d = l2_sq(vec(c.second), vec(L[1 + i]), p_.dim);
                            if (d > wd) { wd = d; worst = i; }
                        }
                        if (worst == cnt) continue;
                        --indeg[L[1 + worst]];
                        L[1 + worst] = u;
                        attached = true;
                    }
                    if (attached) { ++indeg[u]; break; }
                }
                if (!attached) continue;
                // u (and any orphans downstream of it) are reachable now.
                std::vector<node_t> stack{u};
                orphan[u] = 0;
                while (!stack.empty()) {
                    node_t x = stack.back();
                    stack.pop_back();
                    const node_t* L = links(x, 0);
                    for (size_t i = 0; i < L[0]; ++i)
                        if (orphan[L[1 + i]]) { orphan[L[1 + i]] = 0; stack.push_back(L[1 + i]); }
                }
            }
        }
        return unreachable_ids().size();
    }

    // k-NN query (paper Alg. 5). Returns (squared distance, label), ascending. Lock-free.
    std::vector<std::pair<float, label_t>> search(const float* q, size_t k, size_t ef) const {
        std::vector<std::pair<float, label_t>> out;
        int64_t ep = entry_.load();
        if (ep < 0 || k == 0) return out;
        node_t cur = static_cast<node_t>(ep);
        float cur_d = l2_sq(q, vec(cur), p_.dim);
        for (int l = levels_[ep]; l > 0; --l) cur = greedy<false>(q, cur, cur_d, l);
        MaxHeap heap = search_layer<false>(q, cur, cur_d, std::max(ef, k), 0);
        while (heap.size() > k) heap.pop();
        out.resize(heap.size());
        for (size_t i = heap.size(); i-- > 0; heap.pop()) out[i] = {heap.top().first, labels_[heap.top().second]};
        return out;
    }

    // ---- diagnostics --------------------------------------------------------
    // Structural invariants: degree bounds, valid ids, no self loops, no duplicates,
    // neighbours exist on the layer they are linked on.
    bool check_invariants(std::string* err = nullptr) const {
        const size_t n = size();
        auto fail = [&](const std::string& m) {
            if (err) *err = m;
            return false;
        };
        for (node_t u = 0; u < n; ++u) {
            for (int l = 0; l <= levels_[u]; ++l) {
                const node_t* L = links(u, l);
                size_t c = L[0];
                if (c > cap(l)) return fail("degree exceeds cap at node " + std::to_string(u));
                std::vector<node_t> seen(L + 1, L + 1 + c);
                std::sort(seen.begin(), seen.end());
                if (std::adjacent_find(seen.begin(), seen.end()) != seen.end())
                    return fail("duplicate neighbour at node " + std::to_string(u));
                for (size_t i = 0; i < c; ++i) {
                    node_t v = L[1 + i];
                    if (v >= n) return fail("neighbour id out of range");
                    if (v == u) return fail("self loop");
                    if (levels_[v] < l) return fail("neighbour below layer");
                }
            }
        }
        return true;
    }

    // Fraction of nodes reachable from the entry point over layer-0 edges.
    double layer0_reachable_fraction() const {
        const size_t n = size();
        if (n == 0) return 1.0;
        return 1.0 - static_cast<double>(unreachable_ids().size()) / static_cast<double>(n);
    }

    // Internal ids with no layer-0 path from the entry point (ascending).
    std::vector<node_t> unreachable_ids() const {
        const size_t n = size();
        std::vector<node_t> out;
        if (n == 0) return out;
        std::vector<char> seen(n, 0);
        std::vector<node_t> stack{static_cast<node_t>(entry_.load())};
        seen[stack[0]] = 1;
        while (!stack.empty()) {
            node_t u = stack.back();
            stack.pop_back();
            const node_t* L = links(u, 0);
            for (size_t i = 0; i < L[0]; ++i)
                if (!seen[L[1 + i]]) {
                    seen[L[1 + i]] = 1;
                    stack.push_back(L[1 + i]);
                }
        }
        for (node_t u = 0; u < n; ++u)
            if (!seen[u]) out.push_back(u);
        return out;
    }

    double avg_degree_layer0() const {
        const size_t n = size();
        if (n == 0) return 0;
        size_t s = 0;
        for (node_t u = 0; u < n; ++u) s += links(u, 0)[0];
        return static_cast<double>(s) / static_cast<double>(n);
    }

    int max_level() const {
        int64_t ep = entry_.load();
        return ep < 0 ? -1 : levels_[ep];
    }

    // Approximate resident bytes (vectors + graph).
    size_t memory_bytes() const {
        size_t b = size() * p_.dim * sizeof(float) + size() * (M0_ + 1) * sizeof(node_t);
        for (size_t i = 0; i < size(); ++i) b += upper_[i].size() * sizeof(node_t);
        return b;
    }

private:
    using Cand = std::pair<float, node_t>;
    using MaxHeap = std::priority_queue<Cand>;  // farthest on top
    using MinHeap = std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>>;  // nearest on top

    Params p_;
    size_t M_ = 0, M0_ = 0;
    double mult_ = 0;
    std::vector<float> data_;
    std::vector<label_t> labels_;
    std::vector<int> levels_;
    std::vector<node_t> l0_;                  // per node: [count, id0..id(M0-1)]
    std::vector<std::vector<node_t>> upper_;  // per node: (level) blocks of [count, id0..id(M-1)]
    std::unique_ptr<std::mutex[]> locks_;
    std::atomic<size_t> count_{0};
    std::atomic<int64_t> entry_{-1};
    std::mutex global_;

    const float* vec(node_t i) const { return &data_[static_cast<size_t>(i) * p_.dim]; }
    size_t cap(int level) const { return level == 0 ? M0_ : M_; }
    node_t* links(node_t n, int level) {
        return level == 0 ? &l0_[static_cast<size_t>(n) * (M0_ + 1)] : &upper_[n][static_cast<size_t>(level - 1) * (M_ + 1)];
    }
    const node_t* links(node_t n, int level) const {
        return level == 0 ? &l0_[static_cast<size_t>(n) * (M0_ + 1)] : &upper_[n][static_cast<size_t>(level - 1) * (M_ + 1)];
    }

    // l = floor(-ln(U) * mL), mL = 1/ln(M). Pure function of (seed, id): no shared RNG state.
    static uint64_t mix64(uint64_t z) {  // splitmix64 finaliser
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    int random_level(uint64_t id) const {
        const uint64_t z = mix64(p_.seed + (id + 1) * 0x9E3779B97F4A7C15ULL);
        double u = static_cast<double>((z >> 11) + 1) * (1.0 / 9007199254740993.0);  // (0,1)
        return std::min(static_cast<int>(-std::log(u) * mult_), kMaxLevel);
    }

    // Per-thread "visited" set using an epoch counter: reset is O(1), not O(n).
    struct Visited {
        std::vector<uint32_t> mark;
        uint32_t epoch = 0;
        void reset(size_t n) {
            if (mark.size() < n) {
                mark.assign(n, 0);
                epoch = 0;
            }
            if (++epoch == 0) {
                std::fill(mark.begin(), mark.end(), 0);
                epoch = 1;
            }
        }
        bool test_and_set(node_t i) {
            if (mark[i] == epoch) return true;
            mark[i] = epoch;
            return false;
        }
    };
    static Visited& visited() {
        static thread_local Visited v;
        return v;
    }

    // Copy a node's neighbour list into buf (under the node lock when Locked). Returns count.
    template <bool Locked>
    size_t read_links(node_t n, int level, node_t* buf) const {
        if constexpr (Locked) {
            std::lock_guard<std::mutex> g(locks_[n]);
            const node_t* L = links(n, level);
            size_t c = L[0];
            std::memcpy(buf, L + 1, c * sizeof(node_t));
            return c;
        } else {
            const node_t* L = links(n, level);
            size_t c = L[0];
            std::memcpy(buf, L + 1, c * sizeof(node_t));
            return c;
        }
    }

    // Greedy 1-NN walk on a single layer (used for the upper layers).
    template <bool Locked>
    node_t greedy(const float* q, node_t ep, float& ep_d, int level) const {
        node_t buf[2 * kMaxM];
        for (bool improved = true; improved;) {
            improved = false;
            size_t c = read_links<Locked>(ep, level, buf);
            for (size_t i = 0; i < c; ++i) {
                float d = l2_sq(q, vec(buf[i]), p_.dim);
                if (d < ep_d) {
                    ep_d = d;
                    ep = buf[i];
                    improved = true;
                }
            }
        }
        return ep;
    }

    // Paper Alg. 2. Best-first search keeping the ef closest elements seen on `level`.
    template <bool Locked>
    MaxHeap search_layer(const float* q, node_t ep, float ep_d, size_t ef, int level) const {
        Visited& vis = visited();
        vis.reset(p_.max_elements);
        MaxHeap top;
        MinHeap cand;
        top.emplace(ep_d, ep);
        cand.emplace(ep_d, ep);
        vis.test_and_set(ep);
        float worst = ep_d;
        node_t buf[2 * kMaxM];
        while (!cand.empty()) {
            Cand c = cand.top();
            if (c.first > worst && top.size() >= ef) break;  // closest candidate is worse than all results
            cand.pop();
            const size_t cnt = read_links<Locked>(c.second, level, buf);
#if defined(__GNUC__)
            for (size_t i = 0; i < cnt; ++i) __builtin_prefetch(vec(buf[i]));
#endif
            for (size_t i = 0; i < cnt; ++i) {
                node_t nb = buf[i];
                if (vis.test_and_set(nb)) continue;
                float d = l2_sq(q, vec(nb), p_.dim);
                if (top.size() < ef || d < worst) {
                    cand.emplace(d, nb);
                    top.emplace(d, nb);
                    if (top.size() > ef) top.pop();
                    worst = top.top().first;
                }
            }
        }
        return top;
    }

    // Paper Alg. 4 (without keepPrunedConnections / extendCandidates). `cands` must be
    // sorted by ascending distance to the base point. A candidate is kept only if it is
    // closer to the base point than to every neighbour already kept; this favours
    // neighbours in different "directions" and keeps clustered data connected.
    //
    // Exact duplicates of the base point (distance 0) need special handling. A duplicate
    // sits at the same distance from every other candidate as the base point does, so it
    // carries no direction and must not be used to prune anyone. And because they sort
    // first, without a cap they would fill the whole list and turn each group of copies
    // into an island. So at most a quarter of the slots go to duplicates up front, the
    // heuristic runs on the rest, and any slots still free are topped up with duplicates.
    // Which duplicates a node keeps is a hash of (base, duplicate) rather than lowest id
    // first; otherwise every list would keep the same oldest copies and newer copies
    // would lose all their in-edges. Data without exact duplicates is unaffected.
    std::vector<Cand> select_neighbors(const std::vector<Cand>& cands, size_t m, node_t base) const {
        if (cands.size() <= m) return cands;
        const size_t max_dups = std::max<size_t>(1, m / 4);
        size_t ndup = 0;
        while (ndup < cands.size() && cands[ndup].first == 0.0f) ++ndup;
        std::vector<Cand> reordered;  // only built when there are duplicates to shuffle
        if (ndup > 1) {
            reordered = cands;
            auto rank = [&](node_t x) { return mix64((static_cast<uint64_t>(base) << 32) | x); };
            std::sort(reordered.begin(), reordered.begin() + ndup,
                      [&](const Cand& a, const Cand& b) { return rank(a.second) < rank(b.second); });
        }
        const std::vector<Cand>& order = ndup > 1 ? reordered : cands;
        std::vector<Cand> sel, spare_dups;
        sel.reserve(m);
        size_t dups = 0;
        for (const Cand& c : order) {
            if (sel.size() == m) break;
            if (c.first == 0.0f) {
                if (dups < max_dups) {
                    sel.push_back(c);
                    ++dups;
                } else {
                    spare_dups.push_back(c);
                }
                continue;
            }
            bool keep = true;
            for (const Cand& s : sel) {
                if (s.first == 0.0f) continue;  // a duplicate of the base point prunes nothing
                if (l2_sq(vec(c.second), vec(s.second), p_.dim) < c.first) {
                    keep = false;
                    break;
                }
            }
            if (keep) sel.push_back(c);
        }
        for (size_t i = 0; i < spare_dups.size() && sel.size() < m; ++i) sel.push_back(spare_dups[i]);
        return sel;
    }

    // Set id's own list, then add reverse edges, shrinking a neighbour's list with the
    // same heuristic when it overflows. One node lock held at a time.
    void connect(node_t id, const std::vector<Cand>& sel, int level) {
        {
            std::lock_guard<std::mutex> g(locks_[id]);
            node_t* L = links(id, level);
            L[0] = static_cast<node_t>(sel.size());
            for (size_t i = 0; i < sel.size(); ++i) L[1 + i] = sel[i].second;
        }
        const size_t capn = cap(level);
        for (const Cand& s : sel) {
            const node_t n = s.second;
            std::lock_guard<std::mutex> g(locks_[n]);
            node_t* NL = links(n, level);
            const size_t c = NL[0];
            if (c < capn) {
                NL[1 + c] = id;
                NL[0] = static_cast<node_t>(c + 1);
            } else {
                std::vector<Cand> cs;
                cs.reserve(c + 1);
                cs.emplace_back(s.first, id);  // dist(n, id) == dist(id, n)
                for (size_t j = 0; j < c; ++j) cs.emplace_back(l2_sq(vec(n), vec(NL[1 + j]), p_.dim), NL[1 + j]);
                std::sort(cs.begin(), cs.end());
                std::vector<Cand> keep = select_neighbors(cs, capn, n);
                NL[0] = static_cast<node_t>(keep.size());
                for (size_t j = 0; j < keep.size(); ++j) NL[1 + j] = keep[j].second;
            }
        }
    }
};

}  // namespace hnsw
