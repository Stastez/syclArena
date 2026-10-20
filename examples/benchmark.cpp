#include <syclarena/syclarena.hpp>
#include <chrono>
#include <vector>
#include <format>
#include <iostream>
#include <algorithm>

#define SEP_SIZE 84

#define CSECTION  "\x1B[1;34m"
#define CBENCH    "\x1B[1;36m"
#define CFASTER   "\x1B[1;32m"
#define CSLOWER   "\x1B[1;31m"

using Clock = std::chrono::high_resolution_clock;

struct Stats {
    double mean;
    double median;
    double min;
    double max;
};

struct Config {
    static constexpr size_t ITERS  = 500;
    static constexpr size_t WARMUP = 50;
    static constexpr size_t SIZE_SMALL  = 256;
    static constexpr size_t SIZE_MEDIUM = 256 * syclArena::KB;
    static constexpr size_t SIZE_LARGE  = 2   * syclArena::MB;
    static constexpr size_t BATCH_SIZE = 512;
};

// Baseline: SYCL's own allocators. sycl::malloc_device/malloc_shared have no
// stream-ordered variant, so they are the only native baseline.
static void base_alloc(void** p, size_t bytes, sycl::queue* stream, bool is_device) {
    try {
        *p = is_device ? sycl::malloc_device(bytes, syclArena::resolve_queue(stream))
                       : sycl::malloc_shared(bytes, syclArena::resolve_queue(stream));
    }
    catch (const sycl::exception& e) {
        throw syclArena::gpu_error(e.what());
    }
}

static void base_free(void* p, sycl::queue* stream) {
    try {
        sycl::free(p, syclArena::resolve_queue(stream));
    }
    catch (const sycl::exception& e) {
        throw syclArena::gpu_error(e.what());
    }
}

inline double elapsed(const Clock::time_point& start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

inline Stats compute_stats(std::vector<double> times) {
    if (times.empty()) return Stats{0, 0, 0, 0};
    Stats s;
    std::sort(times.begin(), times.end());
    s.min    = times.front();
    s.max    = times.back();
    s.median = times[times.size() / 2];
    double sum  = 0;
    for (const auto& t : times) sum += t;
    s.mean   = sum / times.size();
    return s;
}

inline void print_separator(const char& sep = '-') {
    std::cout << std::format("{}\n", std::string(SEP_SIZE, sep));
}

inline void print_header(const char* title) {
    std::cout << std::format(CBENCH "{}" SYCLAR_CNORMAL "\n", title);
    print_separator('=');
    std::cout << std::format("{:<34}  {:>10}  {:>11}  {:>10}  {:>10}\n",
        "allocator", "mean (us)", "median (us)", "min (us)", "max (us)");
    print_separator();
}

inline void print_row(const char* label, const Stats& s) {
    std::cout << std::format("{:<34}  {:10.2f}  {:11.2f}  {:10.2f}  {:10.2f}\n",
        label, s.mean, s.median, s.min, s.max);
}

inline void print_speedup(const char* fast, double fast_mean,
                           const char* slow, double slow_mean) {
    const bool faster = fast_mean < slow_mean;
    const double ratio = faster ? slow_mean / fast_mean : fast_mean / slow_mean;
    std::cout << std::format("syclArena vs {:<34}  {}{:.2f}x {}" SYCLAR_CNORMAL "\n",
        slow, faster ? CFASTER : CSLOWER, ratio, faster ? "faster" : "slower");
}

inline void print_footer(const Stats& base, const Stats& arena, const char* baseLabel) {
    print_separator();
    print_speedup("syclArena", arena.mean, baseLabel, base.mean);
    print_separator('=');
    std::cout << '\n';
}

inline void print_stats(const std::vector<double>& t_base,
                        const std::vector<double>& t_arena,
                        const char* memType, const char* areType) {
    auto s_base  = compute_stats(t_base);
    auto s_arena = compute_stats(t_arena);
    print_row(memType, s_base);
    print_row(areType, s_arena);
    print_footer(s_base, s_arena, memType);
}

void run_benchmark(syclArena::DeviceArena& alloc, sycl::queue* stream = nullptr);

int main() {
    syclArena::Logger::set_level(0);

    // syclArena benchmark (device pool)
    {
        sycl::queue stream;
        syclArena::DeviceArena alloc;
        alloc.create_gpu_pool(4 * syclArena::GB, syclArena::GPUMemoryType::Device, &stream);
        SYCLARENA_SYNC(&stream);
        std::cout << std::format(CSECTION "syclArena benchmark (device)  —  {} iterations, {} warmup:" SYCLAR_CNORMAL "\n\n", Config::ITERS, Config::WARMUP);
        run_benchmark(alloc, &stream);
        alloc.destroy_gpu_pool();
    }
    std::cout << std::endl;
    // syclArena benchmark (managed pool)
    {
        syclArena::DeviceArena alloc;
        alloc.create_gpu_pool(4 * syclArena::GB, syclArena::GPUMemoryType::Managed);
        SYCLARENA_SYNCALL;
        std::cout << std::format(CSECTION "syclArena benchmark (managed)  —  {} iterations, {} warmup:" SYCLAR_CNORMAL "\n\n", Config::ITERS, Config::WARMUP);
        run_benchmark(alloc);
    }

    return 0;
}

void run_benchmark(syclArena::DeviceArena& alloc, sycl::queue* stream) {
    const bool is_device = alloc.gpu_memory_type() == syclArena::GPUMemoryType::Device;
    std::string memType = is_device ? "sycl::malloc_device  + sycl::free"
                                    : "sycl::malloc_shared  + sycl::free";
    std::string areType = is_device ? "syclArena alloc   " : "syclArena alloc      ";
    areType += "+ dealloc";
    // --------------------------------------
    // Benchmark 1: single large alloc + free
    // --------------------------------------
    print_header("Benchmark 1: single large alloc + free  (8 MB per iteration)");
    {
        float* p = nullptr;
        // warmup with native single
        for (size_t i = 0; i < Config::WARMUP; i++) {
            base_alloc(reinterpret_cast<void**>(&p), Config::SIZE_LARGE * sizeof(float), stream, is_device);
            base_free(p, stream);
        }
        // timed runs with native single
        std::vector<double> t_base(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            auto start = Clock::now();
            base_alloc(reinterpret_cast<void**>(&p), Config::SIZE_LARGE * sizeof(float), stream, is_device);
            base_free(p, stream);
            t_base[i] = elapsed(start);
        }

        // warmup with arena single
        for (size_t i = 0; i < Config::WARMUP; i++) {
            auto q = alloc.allocate<float>(Config::SIZE_LARGE);
            alloc.deallocate(q);
        }
        // timed runs with arena single
        std::vector<double> t_arena(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            auto start = Clock::now();
            auto q  = alloc.allocate<float>(Config::SIZE_LARGE);
            alloc.deallocate(q);
            t_arena[i] = elapsed(start);
        }

        print_stats(t_base, t_arena, memType.c_str(), areType.c_str());
    }
    alloc.reset_gpu_pool();

    // -------------------------------------------------------
    // Benchmark 2: batch allocate 512 x 1 KB, then free all
    // -------------------------------------------------------
    print_header("Benchmark 2: batch alloc 512 x 1 KB, then free all  (per round)");
    {
        std::vector<float*> ptrs(Config::BATCH_SIZE, nullptr);

        // warmup with native batch
        for (size_t i = 0; i < Config::WARMUP; i++) {
            for (size_t j = 0; j < Config::BATCH_SIZE; j++)
                base_alloc(reinterpret_cast<void**>(&ptrs[j]), Config::SIZE_SMALL * sizeof(float), stream, is_device);
            for (size_t j = 0; j < Config::BATCH_SIZE; j++)
                base_free(ptrs[j], stream);
        }

        // timed runs with native batch
        std::vector<double> t_base(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            auto start = Clock::now();
            for (size_t j = 0; j < Config::BATCH_SIZE; j++)
                base_alloc(reinterpret_cast<void**>(&ptrs[j]), Config::SIZE_SMALL * sizeof(float), stream, is_device);
            for (size_t j = 0; j < Config::BATCH_SIZE; j++)
                base_free(ptrs[j], stream);
            t_base[i] = elapsed(start);
        }

        // warmup with arena batch
        for (size_t i = 0; i < Config::WARMUP; i++) {
            for (size_t j = 0; j < Config::BATCH_SIZE; j++)
                ptrs[j] = alloc.allocate<float>(Config::SIZE_SMALL);
            for (size_t j = 0; j < Config::BATCH_SIZE; j++)
                alloc.deallocate(ptrs[j]);
        }

        // timed runs with arena batch
        std::vector<double> t_arena(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            auto start = Clock::now();
            for (size_t j = 0; j < Config::BATCH_SIZE; j++)
                ptrs[j] = alloc.allocate<float>(Config::SIZE_SMALL);
            for (size_t j = 0; j < Config::BATCH_SIZE; j++)
                alloc.deallocate(ptrs[j]);
            t_arena[i] = elapsed(start);
        }

        print_stats(t_base, t_arena, memType.c_str(), areType.c_str());
    }
    alloc.reset_gpu_pool();

    // -----------------------------------------------------------------------
    // Benchmark 3: interleaved ping-pong (4 live buffers cycling, 1 MB each)
    // -----------------------------------------------------------------------
    print_header("Benchmark 3: interleaved ping-pong  (4 live x 1 MB, cycling)");
    {
        constexpr size_t LIVE = 4;
        std::vector<float*> live_base(LIVE, nullptr);
        std::vector<float*> live_arena(LIVE, nullptr);

        // warmup with native ping-pong
        for (size_t j = 0; j < LIVE; j++)
            base_alloc(reinterpret_cast<void**>(&live_base[j]), Config::SIZE_MEDIUM * sizeof(float), stream, is_device);
        for (size_t i = 0; i < Config::WARMUP; i++) {
            size_t slot = i % LIVE;
            base_free(live_base[slot], stream);
            base_alloc(reinterpret_cast<void**>(&live_base[slot]), Config::SIZE_MEDIUM * sizeof(float), stream, is_device);
        }
        // timed runs with native ping-pong
        std::vector<double> t_base(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            size_t slot = i % LIVE;
            auto start = Clock::now();
            base_free(live_base[slot], stream);
            base_alloc(reinterpret_cast<void**>(&live_base[slot]), Config::SIZE_MEDIUM * sizeof(float), stream, is_device);
            t_base[i] = elapsed(start);
        }
        for (size_t j = 0; j < LIVE; j++) base_free(live_base[j], stream);

        // warmup with arena ping-pong
        for (size_t j = 0; j < LIVE; j++)
            live_arena[j] = alloc.allocate<float>(Config::SIZE_MEDIUM);
        for (size_t i = 0; i < Config::WARMUP; i++) {
            size_t slot = i % LIVE;
            alloc.deallocate(live_arena[slot]);
            live_arena[slot] = alloc.allocate<float>(Config::SIZE_MEDIUM);
        }
        // timed runs with arena ping-pong
        std::vector<double> t_arena(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            size_t slot = i % LIVE;
            auto start = Clock::now();
            alloc.deallocate(live_arena[slot]);
            live_arena[slot] = alloc.allocate<float>(Config::SIZE_MEDIUM);
            t_arena[i] = elapsed(start);
        }
        for (size_t j = 0; j < LIVE; j++) alloc.deallocate(live_arena[j]);

        print_stats(t_base, t_arena, memType.c_str(), areType.c_str());
    }
    alloc.reset_gpu_pool();

    // -----------------------------------------------------------------------
    // Benchmark 4: resize chain  (1 MB to 8 MB in 8 doublings, per iteration)
    // -----------------------------------------------------------------------
    print_header("Benchmark 4: resize chain  (1 MB to 8 MB in 8 doublings)");
    {
        constexpr size_t STEPS = 8;
        // warmup with native resize chain
        for (size_t i = 0; i < Config::WARMUP; i++) {
            float* p = nullptr;
            size_t sz = Config::SIZE_MEDIUM;
            for (size_t s = 0; s < STEPS; s++) {
                base_free(p, stream);
                base_alloc(reinterpret_cast<void**>(&p), sz * sizeof(float), stream, is_device); sz *= 2;
            }
            base_free(p, stream);
        }
        // timed runs with native resize chain
        std::vector<double> t_base(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            float* p = nullptr;
            size_t sz = Config::SIZE_MEDIUM;
            auto start = Clock::now();
            for (size_t s = 0; s < STEPS; s++) {
                base_free(p, stream);
                base_alloc(reinterpret_cast<void**>(&p), sz * sizeof(float), stream, is_device); sz *= 2;
            }
            base_free(p, stream);
            t_base[i] = elapsed(start);
        }
        // warmup with arena resize chain
        for (size_t i = 0; i < Config::WARMUP; i++) {
            float* p = nullptr;
            size_t sz = Config::SIZE_MEDIUM;
            for (size_t s = 0; s < STEPS; s++) { alloc.resize<float>(p, sz); sz *= 2; }
            alloc.deallocate(p);
        }
        // timed runs with arena resize chain
        std::vector<double> t_arena(Config::ITERS);
        for (size_t i = 0; i < Config::ITERS; i++) {
            float* p = nullptr;
            size_t sz = Config::SIZE_MEDIUM;
            auto start = Clock::now();
            for (size_t s = 0; s < STEPS; s++) { alloc.resize<float>(p, sz); sz *= 2; }
            alloc.deallocate(p);
            t_arena[i] = elapsed(start);
        }

        print_stats(t_base, t_arena, memType.c_str(), areType.c_str());
    }
    alloc.reset_gpu_pool();

    // -----------------------------------------------------------------------
    // Benchmark 5: fragmentation + compact cycle
    //   8 x 32 MB allocs, free alternating => 4 x 32 MB holes
    //   Measure: compact latency, then 64 MB alloc that only fits after compact
    // -----------------------------------------------------------------------
    if (is_device) {
        print_header("Benchmark 5: fragmentation + compact cycle  (8 x 32 MB, free alternating)");
        {
            constexpr size_t NBLOCKS  = 8;
            constexpr size_t BLK_ELEM = 32 * syclArena::MB / sizeof(float);

            std::vector<double> t_compact(Config::ITERS);
            std::vector<double> t_alloc_after(Config::ITERS);

            for (size_t i = 0; i < Config::ITERS + Config::WARMUP; i++) {
                // Build fragmented region
                float* ptrs[NBLOCKS];
                for (size_t b = 0; b < NBLOCKS; b++)
                    ptrs[b] = alloc.allocate<float>(BLK_ELEM);
                for (size_t b = 0; b < NBLOCKS; b += 2)
                    alloc.deallocate(ptrs[b]);

                // Time compact. 32 MB blocks live in the high region (>= LARGE_MIN),
                // so pack the high end.
                auto start0 = Clock::now();
                alloc.compact_gpu_dynamic([](syclArena::addr_t, syclArena::addr_t, size_t) {}, stream, syclArena::CompactRegion::High);
                SYCLARENA_SYNC(stream);
                double ct = elapsed(start0);

                // Time the large alloc that only succeeds after compact
                auto start1 = Clock::now();
                float* big = alloc.allocate<float>(64 * syclArena::MB / sizeof(float));
                double at = elapsed(start1);

                if (i >= Config::WARMUP) {
                    t_compact[i - Config::WARMUP]     = ct;
                    t_alloc_after[i - Config::WARMUP] = at;
                }

                alloc.reset_gpu_pool();
            }

            auto sc = compute_stats(t_compact);
            auto sa = compute_stats(t_alloc_after);
            print_row("compaction (256 MB active)", sc);
            print_row("alloc 64 MB after compact ",          sa);
            print_separator();
            std::cout << std::format("  {}{:<56}  {:.2f} us{}\n",
                CBENCH, "compact + alloc total (mean)", sc.mean + sa.mean, SYCLAR_CNORMAL);
            print_separator('=');
            std::cout << '\n';
        }
    }

    // -----------------------------------------------------------------------
    // Benchmark 6: stable vs dynamic allocation latency
    // -----------------------------------------------------------------------
    if (is_device) {
        print_header("Benchmark 6: stable vs dynamic alloc latency  (varied sizes)");
        {
            // Re-create pool with a stable region
            alloc.destroy_gpu_pool();
            alloc.create_gpu_pool(4 * syclArena::GB, syclArena::GPUMemoryType::Device, stream,
                                  512 * syclArena::MB);
            SYCLARENA_SYNC(stream);

            const size_t sizes[] = { Config::SIZE_SMALL, Config::SIZE_MEDIUM, Config::SIZE_LARGE };
            const char*  labels[] = { "256 B", "256 KB", "2 MB" };

            for (size_t si = 0; si < 3; si++) {
                const size_t ELEM = sizes[si] / sizeof(float);
                std::vector<double> t_stable(Config::ITERS);
                std::vector<double> t_dynamic(Config::ITERS);

                // Warmup stable
                for (size_t i = 0; i < Config::WARMUP; i++) {
                    auto p = alloc.allocate<float>(ELEM, syclArena::Region::Stable);
                    alloc.deallocate(p);
                }
                // Timed stable
                for (size_t i = 0; i < Config::ITERS; i++) {
                    auto start = Clock::now();
                    auto p = alloc.allocate<float>(ELEM, syclArena::Region::Stable);
                    alloc.deallocate(p);
                    t_stable[i] = elapsed(start);
                }

                // Warmup dynamic
                for (size_t i = 0; i < Config::WARMUP; i++) {
                    auto p = alloc.allocate<float>(ELEM, syclArena::Region::Dynamic);
                    alloc.deallocate(p);
                }
                // Timed dynamic
                for (size_t i = 0; i < Config::ITERS; i++) {
                    auto start = Clock::now();
                    auto p = alloc.allocate<float>(ELEM, syclArena::Region::Dynamic);
                    alloc.deallocate(p);
                    t_dynamic[i] = elapsed(start);
                }

                auto ss = compute_stats(t_stable);
                auto sd = compute_stats(t_dynamic);
                std::string ls = std::string("stable  ") + labels[si];
                std::string ld = std::string("dynamic ") + labels[si];
                print_row(ls.c_str(),  ss);
                print_row(ld.c_str(),  sd);
                print_separator();
                const bool dyn_faster = sd.mean < ss.mean;
                const double ratio = dyn_faster ? ss.mean / sd.mean : sd.mean / ss.mean;
                std::cout << std::format("dynamic vs stable [{:<6}]         {}  {:.2f}x {}" SYCLAR_CNORMAL "\n",
                    labels[si], dyn_faster ? CFASTER : CSLOWER, ratio, dyn_faster ? "faster" : "slower");
                print_separator('=');
                std::cout << '\n';
            }
        }
    }
}
