// bench.cpp - recall@k vs queries-per-second benchmark, ann-benchmarks style.
//
//   Real data (fvecs, e.g. SIFT1M):
//     ./bench --base sift_base.fvecs --query sift_query.fvecs --gt sift_gt.ivecs --metric l2
//   GloVe-100 (convert the ann-benchmarks .hdf5 first: scripts/hdf5_to_fvecs.py):
//     ./bench --base glove_base.fvecs --query glove_query.fvecs --gt glove_gt.ivecs --metric angular
//   Synthetic smoke test (no downloads):
//     ./bench --synthetic 100000 128 1000 --dump data/syn
//
// Query throughput is measured single-threaded (one query at a time, like ann-benchmarks'
// default) and, separately, with all threads.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include "dataset.h"
#include "hnsw.h"

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static std::vector<size_t> parse_list(const std::string& s) {
    std::vector<size_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::stoul(tok));
    return v;
}

int main(int argc, char** argv) {
    std::map<std::string, std::vector<std::string>> a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        std::vector<std::string> vals;
        while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) vals.push_back(argv[++i]);
        a[k] = vals;
    }
    auto get = [&](const char* k, const char* def) { return a.count(k) && !a[k].empty() ? a[k][0] : std::string(def); };
    const size_t M = std::stoul(get("--M", "16"));
    const size_t efc = std::stoul(get("--efc", "200"));
    const size_t k = std::stoul(get("--k", "10"));
    const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t threads = std::stoul(get("--threads", std::to_string(hw).c_str()));
    const std::string metric = get("--metric", "l2");
    const std::string out_path = get("--out", "results/hnsw_cpp.csv");
    const std::vector<size_t> efs = parse_list(get("--ef", "10,20,40,60,80,120,200,300,500,800"));
    const bool compare_build = a.count("--compare-build") > 0;
    const size_t reps = std::stoul(get("--reps", "3"));

    // ---- data --------------------------------------------------------------
    ds::Matrix base, queries;
    if (a.count("--synthetic")) {
        auto& s = a["--synthetic"];
        if (s.size() < 3) { std::cerr << "--synthetic N DIM NQ\n"; return 2; }
        size_t n = std::stoul(s[0]), d = std::stoul(s[1]), nq = std::stoul(s[2]);
        base = ds::synthetic(n, d, 100, 42, 0);
        queries = ds::synthetic(nq, d, 100, 42, 1);
        if (a.count("--dump")) {
            ds::write_fvecs(a["--dump"][0] + "_base.fvecs", base);
            ds::write_fvecs(a["--dump"][0] + "_query.fvecs", queries);
        }
    } else if (a.count("--base") && a.count("--query")) {
        base = ds::read_fvecs(a["--base"][0]);
        queries = ds::read_fvecs(a["--query"][0]);
    } else {
        std::cerr << "usage: bench (--base F --query F | --synthetic N DIM NQ) [--gt F] [--metric l2|angular]\n"
                     "             [--M 16] [--efc 200] [--k 10] [--threads T] [--ef 10,20,...] [--out csv] [--compare-build]\n";
        return 2;
    }
    if (metric == "angular") {
        ds::normalize(base);
        ds::normalize(queries);
    }
    std::printf("data: %zu base x %zu dim, %zu queries, metric=%s, threads=%zu\n", base.n, base.d, queries.n, metric.c_str(), threads);

    // ---- ground truth (cached to --gt if given) ------------------------------
    ds::IMatrix gt;
    const std::string gt_path = get("--gt", "");
    const size_t kgt = std::max<size_t>(k, 100);
    if (!gt_path.empty() && ds::file_exists(gt_path)) {
        gt = ds::read_ivecs(gt_path);
        std::printf("ground truth: loaded %s (%zu x %zu)\n", gt_path.c_str(), gt.n, gt.w);
    } else {
        auto t0 = Clock::now();
        gt = ds::brute_force_knn(base, queries, std::min(kgt, base.n), threads);
        std::printf("ground truth: brute force in %.1fs\n", secs(t0, Clock::now()));
        if (!gt_path.empty()) ds::write_ivecs(gt_path, gt);
    }
    if (gt.n != queries.n || gt.w < k) { std::cerr << "ground truth does not match queries/k\n"; return 2; }

    // ---- build ---------------------------------------------------------------
    hnsw::Index::Params p;
    p.dim = base.d;
    p.max_elements = base.n;
    p.M = M;
    p.ef_construction = efc;
    auto index = std::make_unique<hnsw::Index>(p);
    auto t0 = Clock::now();
    index->add_batch(base.v.data(), base.n, threads);
    double build_s = secs(t0, Clock::now());
    std::printf("build: M=%zu efC=%zu threads=%zu -> %.2fs (%.0f inserts/s), %.1f MB, levels=%d, avg deg0=%.1f, unreachable=%zu\n", M, efc,
                threads, build_s, base.n / build_s, index->memory_bytes() / 1e6, index->max_level() + 1, index->avg_degree_layer0(),
                index->unreachable_ids().size());
    if (compare_build && threads > 1) {
        hnsw::Index single(p);
        auto t1 = Clock::now();
        single.add_batch(base.v.data(), base.n, 1);
        double s1 = secs(t1, Clock::now());
        std::printf("build: 1 thread -> %.2fs, speedup with %zu threads = %.2fx\n", s1, threads, s1 / build_s);
    }

    // ---- query sweep ---------------------------------------------------------
    std::ofstream csv;
    {
        auto parent = std::filesystem::path(out_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        csv.open(out_path);
        csv << "impl,M,efc,ef,recall,qps_1t,qps_mt,p50_us,p99_us,build_s\n";
    }
    std::printf("\n%6s %9s %12s %12s %9s %9s\n", "ef", "recall@k", "QPS (1 thr)", "QPS (all)", "p50 us", "p99 us");
    const size_t nq = queries.n;
    std::vector<hnsw::label_t> got(nq * k);
    for (size_t ef : efs) {
        // single-thread, one query at a time; keep the fastest of `reps` passes
        double best = 1e18;
        std::vector<double> lat(nq), best_lat;
        for (size_t r = 0; r < reps; ++r) {
            auto ts = Clock::now();
            for (size_t q = 0; q < nq; ++q) {
                auto q0 = Clock::now();
                auto res = index->search(queries.row(q), k, ef);
                lat[q] = secs(q0, Clock::now()) * 1e6;
                for (size_t j = 0; j < k; ++j) got[q * k + j] = j < res.size() ? res[j].second : ~0ULL;
            }
            double el = secs(ts, Clock::now());
            if (el < best) { best = el; best_lat = lat; }
        }
        size_t hit = 0;
        for (size_t q = 0; q < nq; ++q) {
            std::set<int64_t> truth(gt.row(q), gt.row(q) + k);
            for (size_t j = 0; j < k; ++j) hit += truth.count(static_cast<int64_t>(got[q * k + j]));
        }
        double recall = static_cast<double>(hit) / static_cast<double>(nq * k);
        std::sort(best_lat.begin(), best_lat.end());
        double p50 = best_lat[nq / 2], p99 = best_lat[std::min(nq - 1, static_cast<size_t>(nq * 0.99))];

        // all threads
        double best_mt = 1e18;
        for (size_t r = 0; r < reps; ++r) {
            std::atomic<size_t> next{0};
            auto ts = Clock::now();
            std::vector<std::thread> pool;
            for (size_t t = 0; t < threads; ++t)
                pool.emplace_back([&] {
                    for (;;) {
                        size_t q = next.fetch_add(1);
                        if (q >= nq) break;
                        auto res = index->search(queries.row(q), k, ef);
                        asm volatile("" : : "r"(res.data()) : "memory");
                    }
                });
            for (auto& th : pool) th.join();
            best_mt = std::min(best_mt, secs(ts, Clock::now()));
        }
        double qps1 = nq / best, qpsm = nq / best_mt;
        std::printf("%6zu %9.4f %12.0f %12.0f %9.1f %9.1f\n", ef, recall, qps1, qpsm, p50, p99);
        csv << "hnsw-cpp," << M << "," << efc << "," << ef << "," << recall << "," << qps1 << "," << qpsm << "," << p50 << "," << p99 << ","
            << build_s << "\n";
    }
    std::printf("\nwrote %s\n", out_path.c_str());
    return 0;
}
