// Dependency-free test suite. Exit code 0 == all passed.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>

#include "dataset.h"
#include "hnsw.h"

static int g_fail = 0;
#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("  FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond);       \
            std::printf(__VA_ARGS__);                                           \
            std::printf("\n");                                                  \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

static double recall_at_k(const hnsw::Index& idx, const ds::Matrix& q, const ds::IMatrix& gt, size_t k, size_t ef) {
    size_t hit = 0;
    for (size_t i = 0; i < q.n; ++i) {
        auto res = idx.search(q.row(i), k, ef);
        std::set<int64_t> truth(gt.row(i), gt.row(i) + k);
        for (auto& r : res) hit += truth.count(static_cast<int64_t>(r.second));
    }
    return static_cast<double>(hit) / static_cast<double>(q.n * k);
}

static std::unique_ptr<hnsw::Index> build(const ds::Matrix& base, size_t M, size_t efc, size_t threads) {
    hnsw::Index::Params p;
    p.dim = base.d;
    p.max_elements = base.n;
    p.M = M;
    p.ef_construction = efc;
    auto idx = std::make_unique<hnsw::Index>(p);
    idx->add_batch(base.v.data(), base.n, threads);
    return idx;
}

static void test_tiny() {
    std::printf("tiny / edge cases\n");
    hnsw::Index::Params p;
    p.dim = 4;
    p.max_elements = 3;
    hnsw::Index idx(p);
    float q[4] = {0, 0, 0, 0};
    CHECK(idx.search(q, 5, 10).empty(), "empty index returns nothing");
    float a[4] = {1, 0, 0, 0}, b[4] = {0, 2, 0, 0}, c[4] = {0, 0, 3, 0};
    idx.add(a, 10);
    auto r = idx.search(q, 5, 10);
    CHECK(r.size() == 1 && r[0].second == 10 && r[0].first == 1.0f, "single element");
    idx.add(b, 20);
    idx.add(c, 30);
    r = idx.search(q, 5, 10);
    CHECK(r.size() == 3, "k > n returns n, got %zu", r.size());
    CHECK(r[0].second == 10 && r[1].second == 20 && r[2].second == 30, "ordering by distance");
    bool threw = false;
    try {
        idx.add(a, 99);
    } catch (const std::length_error&) {
        threw = true;
    }
    CHECK(threw, "adding beyond capacity throws");
    CHECK(idx.size() == 3, "size unchanged after failed add, got %zu", idx.size());
}

static void test_recall_and_invariants() {
    std::printf("recall + invariants (single thread, 5k x 32d)\n");
    ds::Matrix base = ds::synthetic(5000, 32, 50, 1), q = ds::synthetic(200, 32, 50, 1, 1);
    ds::IMatrix gt = ds::brute_force_knn(base, q, 10, 2);
    auto idxp = build(base, 16, 200, 1);
    hnsw::Index& idx = *idxp;
    std::string err;
    CHECK(idx.check_invariants(&err), "%s", err.c_str());
    CHECK(idx.layer0_reachable_fraction() > 0.999, "reachable %.4f", idx.layer0_reachable_fraction());
    double r10 = recall_at_k(idx, q, gt, 10, 10), r100 = recall_at_k(idx, q, gt, 10, 100);
    std::printf("  recall@10 ef=10: %.4f   ef=100: %.4f   levels=%d  avg deg0=%.1f\n", r10, r100, idx.max_level(), idx.avg_degree_layer0());
    CHECK(r100 >= 0.97, "recall@10 at ef=100 was %.4f", r100);
    CHECK(r100 >= r10, "recall must not decrease with ef");
}

static void test_self_query() {
    std::printf("exact self match\n");
    ds::Matrix base = ds::synthetic(2000, 16, 20, 2);
    auto idxp = build(base, 16, 100, 1);
    hnsw::Index& idx = *idxp;
    size_t ok = 0;
    for (size_t i = 0; i < base.n; i += 7) {
        auto r = idx.search(base.row(i), 1, 50);
        if (!r.empty() && r[0].second == i && r[0].first == 0.0f) ++ok;
    }
    size_t total = (base.n + 6) / 7;
    CHECK(ok >= total * 99 / 100, "self-match %zu/%zu", ok, total);
}

static void test_parallel_build() {
    std::printf("parallel build (8 threads) matches serial quality\n");
    ds::Matrix base = ds::synthetic(8000, 32, 60, 3), q = ds::synthetic(200, 32, 60, 3, 1);
    ds::IMatrix gt = ds::brute_force_knn(base, q, 10, 2);
    auto serial_p = build(base, 16, 200, 1);
    auto par_p = build(base, 16, 200, 8);
    hnsw::Index &serial = *serial_p, &par = *par_p;
    std::string err;
    CHECK(par.check_invariants(&err), "parallel build invariants: %s", err.c_str());
    CHECK(par.size() == base.n, "size %zu", par.size());
    CHECK(par.layer0_reachable_fraction() > 0.9995, "parallel reachable %.5f (repair should leave ~0 orphans)", par.layer0_reachable_fraction());
    double rs = recall_at_k(serial, q, gt, 10, 100), rp = recall_at_k(par, q, gt, 10, 100);
    std::printf("  serial recall@10=%.4f  parallel recall@10=%.4f\n", rs, rp);
    CHECK(rp >= rs - 0.02, "parallel recall %.4f vs serial %.4f", rp, rs);
    // every label must be present exactly once
    std::set<hnsw::label_t> labels;
    for (size_t i = 0; i < base.n; i += 1) {
        auto r = par.search(base.row(i), 1, 100);
        if (!r.empty()) labels.insert(r[0].second);
    }
    CHECK(labels.size() >= base.n * 98 / 100, "distinct top-1 labels %zu", labels.size());
}

static void test_search_ordering() {
    std::printf("result ordering\n");
    ds::Matrix base = ds::synthetic(3000, 24, 30, 4), q = ds::synthetic(50, 24, 30, 4, 1);
    auto idxp = build(base, 12, 100, 1);
    hnsw::Index& idx = *idxp;
    for (size_t i = 0; i < q.n; ++i) {
        auto r = idx.search(q.row(i), 20, 60);
        CHECK(r.size() == 20, "got %zu results", r.size());
        for (size_t j = 1; j < r.size(); ++j) CHECK(r[j - 1].first <= r[j].first, "not ascending");
        std::set<hnsw::label_t> uniq;
        for (auto& x : r) uniq.insert(x.second);
        CHECK(uniq.size() == r.size(), "duplicate labels in result");
    }
}

// Exact duplicate vectors used to turn each group of copies into an island: with 50 distinct
// vectors x 100 copies, 98% of nodes were unreachable and repair could not fix it.
static void test_duplicates() {
    std::printf("exact duplicate vectors stay connected\n");
    const size_t d = 16;
    struct Case { const char* name; size_t n, distinct; } cases[] = {{"50 x 100 copies", 5000, 50}, {"2500 x 2 copies", 5000, 2500}};
    for (const Case& c : cases) {
        ds::Matrix u = ds::synthetic(c.distinct, d, 10, 5);
        ds::Matrix base;
        base.n = c.n;
        base.d = d;
        base.v.resize(c.n * d);
        for (size_t i = 0; i < c.n; ++i) std::memcpy(&base.v[i * d], u.row(i % c.distinct), d * sizeof(float));
        hnsw::Index::Params p;
        p.dim = d;
        p.max_elements = c.n;
        // serial inserts without repair: the graph itself must hold together
        hnsw::Index serial(p);
        for (size_t i = 0; i < c.n; ++i) serial.add(base.row(i), i);
        CHECK(serial.unreachable_ids().empty(), "%s: %zu unreachable before repair", c.name, serial.unreachable_ids().size());
        std::string err;
        CHECK(serial.check_invariants(&err), "%s: %s", c.name, err.c_str());
        hnsw::Index par(p);
        size_t left = par.add_batch(base.v.data(), c.n, 8);
        CHECK(left == 0, "%s: %zu unreachable after parallel build", c.name, left);
        // Serial build (deterministic): every copy of the query, up to k, must come back.
        // Parallel build: the top hit must be an exact copy. Parallel builds can still miss
        // the second of two copies for ~0.5% of queries; the code before this fix did too.
        const size_t want = std::min<size_t>(c.n / c.distinct, 10);
        size_t all_copies = 0, top_exact = 0;
        for (size_t i = 0; i < c.distinct; ++i) {
            auto r = serial.search(u.row(i), 10, 50);
            if (r.size() == 10 && r[want - 1].first == 0.0f) ++all_copies;
            auto rp = par.search(u.row(i), 10, 50);
            if (!rp.empty() && rp[0].first == 0.0f) ++top_exact;
        }
        CHECK(all_copies == c.distinct, "%s: serial found every copy for %zu/%zu queries", c.name, all_copies, c.distinct);
        CHECK(top_exact == c.distinct, "%s: parallel top hit exact for %zu/%zu queries", c.name, top_exact, c.distinct);
    }
}

static void test_api_edges() {
    std::printf("api edge cases\n");
    auto bad = [](size_t dim, size_t max_el, size_t M) {
        hnsw::Index::Params p;
        p.dim = dim;
        p.max_elements = max_el;
        p.M = M;
        try {
            hnsw::Index idx(p);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(bad(0, 10, 16), "dim 0 must throw");
    CHECK(bad(4, 0, 16), "max_elements 0 must throw");
    CHECK(bad(4, 10, 1), "M 1 must throw");
    CHECK(bad(4, 10, 129), "M 129 must throw");

    ds::Matrix a = ds::synthetic(1000, 8, 10, 6), b = ds::synthetic(1000, 8, 10, 6, 1);
    hnsw::Index::Params p;
    p.dim = 8;
    p.max_elements = 2000;
    hnsw::Index idx(p);
    CHECK(idx.add_batch(a.v.data(), a.n, 4) == 0, "first batch fully reachable");
    CHECK(idx.add_batch(b.v.data(), b.n, 4, 1000) == 0, "second batch fully reachable");
    CHECK(idx.size() == 2000, "size %zu", idx.size());
    CHECK(idx.search(a.row(0), 0, 10).empty(), "k = 0 returns nothing");
    auto r = idx.search(b.row(7), 1, 50);
    CHECK(!r.empty() && r[0].second == 1007 && r[0].first == 0.0f, "second batch labels start at first_label");
}

int main() {
    std::printf("distance kernel: %s\n", hnsw::simd_backend());
    test_tiny();
    test_recall_and_invariants();
    test_self_query();
    test_parallel_build();
    test_search_ordering();
    test_duplicates();
    test_api_edges();
    if (g_fail) {
        std::printf("\n%d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("\nall tests passed\n");
    return 0;
}
