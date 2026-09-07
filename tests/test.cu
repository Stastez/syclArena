
#include <cuarena/cuarena.cuh>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

#define CHEADER "\x1B[1;36m"
#define CTEST "\x1B[1;34m"
#define CPASS "\x1B[1;32m"
#define CFAIL "\x1B[1;31m"

struct TestFailure : std::runtime_error {
    explicit TestFailure(const char* expr, int line)
        : std::runtime_error(std::string("line ") + std::to_string(line) + ": " + expr) 
        { }
};

#define TCHECK(expr) \
    if (!(expr)) throw TestFailure(#expr, __LINE__)

inline int total = 0, passed = 0;

inline void section(const char* title) {
    std::cout << std::format("\n{}{}{}\n", CHEADER, title, CUAR_CNORMAL);
    std::cout << std::string(60, '-') << '\n';
}

inline void gpu_cleanup() {
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess)
        cudaMemPoolTrimTo(pool, 0);
}

template<typename Func>
void run_test(const char* name, Func func) {
    ++total;
    try {
        func();
        std::cout << std::format("  {}{}{} {}PASS{}\n",
            CTEST, name, CUAR_CNORMAL, CPASS, CUAR_CNORMAL);
        ++passed;
    }
    catch (const TestFailure& e) {
        std::cerr << std::format("  {}FAIL  {}  ({}){}\n", CFAIL, name, e.what(), CUAR_CNORMAL);
    }
    catch (const std::exception& e) {
        std::cerr << std::format("  {}FAIL  {}  (unexpected exception: {}){}\n", CFAIL, name, e.what(), CUAR_CNORMAL);
    }
    gpu_cleanup();
}

void test_gpu_device(cudaStream_t stream) {
    section("GPU Device pool");

    run_test("create / destroy", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        TCHECK(a.gpu_capacity() == 256 * cuArena::MB);
        TCHECK(a.gpu_used()     == 0);
        TCHECK(a.destroy_gpu_pool());
        TCHECK(a.gpu_capacity() == 0);
    });

    run_test("allocate / deallocate", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* p = a.allocate<float>(1024);
        TCHECK(p != nullptr);
        TCHECK(a.gpu_used() > 0);
        a.deallocate(p);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("available tracks capacity", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        size_t before = a.gpu_available();
        float* p = a.allocate<float>(1024);
        TCHECK(a.gpu_available() < before);
        a.deallocate(p);
        TCHECK(a.gpu_available() == before);
    });

    run_test("resize buffer", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* p = a.allocate<float>(1024);
        TCHECK(p != nullptr);
        a.resize<float>(p, 4096);
        TCHECK(p != nullptr);
        TCHECK(a.gpu_used() > 0);
        a.deallocate(p);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("resize to zero deallocates", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* p = a.allocate<float>(1024);
        a.resize<float>(p, 0);
        TCHECK(p == nullptr);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("resize bail when already large enough", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* p    = a.allocate<float>(4096);
        float* orig = p;
        a.resize<float>(p, 1024);
        TCHECK(p == orig);
        a.deallocate(p);
    });

    run_test("free-list coalescing reclaims block", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* p1      = a.allocate<float>(1024);
        float* p2      = a.allocate<float>(1024);
        size_t used_two = a.gpu_used();
        a.deallocate(p1);
        a.deallocate(p2);
        TCHECK(a.gpu_used() == 0);
        float* p3 = a.allocate<float>(2048);
        TCHECK(p3 != nullptr);
        TCHECK(a.gpu_used() <= used_two);
        a.deallocate(p3);
    });

    run_test("pool exhaustion throws", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(4 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        bool threw = false;
        try { a.allocate<float>(8 * cuArena::MB); }
        catch (const cuArena::gpu_memory_error&) { threw = true; }
        TCHECK(threw);
    });

    run_test("reset invalidates allocations", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        a.allocate<float>(1024);
        a.allocate<float>(2048);
        a.reset_gpu_pool();
        TCHECK(a.gpu_used()      == 0);
        TCHECK(a.gpu_available() == a.gpu_capacity());
    });

    run_test("null / zero guard", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        TCHECK(a.allocate<float>(0) == nullptr);
        a.deallocate<float>(nullptr);
    });
}

void test_gpu_managed() {
    section("GPU Managed pool");

    run_test("create / destroy", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        TCHECK(a.gpu_memory_type() == cuArena::GPUMemoryType::Managed);
        TCHECK(a.gpu_capacity()    == 256 * cuArena::MB);
        TCHECK(a.destroy_gpu_pool());
    });

    run_test("allocate / deallocate", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        float* p = a.allocate<float>(1024);
        TCHECK(p != nullptr);
        TCHECK(a.gpu_used() > 0);
        a.deallocate(p);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("host-accessible read/write", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        cudaDeviceSynchronize();
        int* p = a.allocate<int>(4);
        TCHECK(p != nullptr);
        p[0] = 42; p[1] = 7; p[2] = -1; p[3] = 0;
        TCHECK(p[0] == 42 && p[1] == 7 && p[2] == -1 && p[3] == 0);
        a.deallocate(p);
    });

    run_test("resize buffer", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        float* p = a.allocate<float>(1024);
        a.resize<float>(p, 8192);
        TCHECK(p != nullptr);
        a.deallocate(p);
        TCHECK(a.gpu_used() == 0);
    });

    run_test("free-list coalescing reclaims block", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        float* p1 = a.allocate<float>(1024);
        float* p2 = a.allocate<float>(1024);
        a.deallocate(p1);
        a.deallocate(p2);
        TCHECK(a.gpu_used() == 0);
        float* p3 = a.allocate<float>(2048);
        TCHECK(p3 != nullptr);
        a.deallocate(p3);
    });

    run_test("pool exhaustion throws", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(4 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        bool threw = false;
        try { a.allocate<float>(8 * cuArena::MB); }
        catch (const cuArena::gpu_memory_error&) { threw = true; }
        TCHECK(threw);
    });

    run_test("reset", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        a.allocate<float>(1024);
        a.reset_gpu_pool();
        TCHECK(a.gpu_used()      == 0);
        TCHECK(a.gpu_available() == a.gpu_capacity());
    });
}

void test_cpu_pinned() {
    section("CPU Pinned pool");

    run_test("create / destroy", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pinned));
        TCHECK(a.cpu_memory_type() == cuArena::CPUMemoryType::Pinned);
        TCHECK(a.cpu_capacity()    == 128 * cuArena::MB);
        TCHECK(a.destroy_cpu_pool());
        TCHECK(a.cpu_capacity()    == 0);
    });

    run_test("allocate / deallocate", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pinned));
        float* p = a.allocate_pinned<float>(1024);
        TCHECK(p != nullptr);
        TCHECK(a.cpu_used() > 0);
        a.deallocate_pinned(p);
        TCHECK(a.cpu_used() == 0);
    });

    run_test("read/write", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pinned));
        int* p = a.allocate_pinned<int>(4);
        p[0] = 1; p[1] = 2; p[2] = 3; p[3] = 4;
        TCHECK(p[0] == 1 && p[1] == 2 && p[2] == 3 && p[3] == 4);
        a.deallocate_pinned(p);
    });

    run_test("resize buffer", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pinned));
        float* p = a.allocate_pinned<float>(1024);
        a.resize_pinned<float>(p, 8192);
        TCHECK(p != nullptr);
        a.deallocate_pinned(p);
        TCHECK(a.cpu_used() == 0);
    });

    run_test("free-list coalescing", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pinned));
        float* p1 = a.allocate_pinned<float>(1024);
        float* p2 = a.allocate_pinned<float>(1024);
        a.deallocate_pinned(p1);
        a.deallocate_pinned(p2);
        TCHECK(a.cpu_used() == 0);
        float* p3 = a.allocate_pinned<float>(2048);
        TCHECK(p3 != nullptr);
        a.deallocate_pinned(p3);
    });

    run_test("pool exhaustion throws", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(4 * cuArena::MB, cuArena::CPUMemoryType::Pinned));
        bool threw = false;
        try { a.allocate_pinned<float>(8 * cuArena::MB); }
        catch (const cuArena::cpu_memory_error&) { threw = true; }
        TCHECK(threw);
    });

    run_test("reset", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pinned));
        a.allocate_pinned<float>(1024);
        a.reset_cpu_pool();
        TCHECK(a.cpu_used()      == 0);
        TCHECK(a.cpu_available() == a.cpu_capacity());
    });

    run_test("null / zero guard", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pinned));
        TCHECK(a.allocate_pinned<float>(0) == nullptr);
        a.deallocate_pinned<float>(nullptr);
    });
}

void test_cpu_pageable() {
    section("CPU Pageable pool");

    run_test("create / destroy", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pageable));
        TCHECK(a.cpu_memory_type() == cuArena::CPUMemoryType::Pageable);
        TCHECK(a.cpu_capacity()    == 128 * cuArena::MB);
        TCHECK(a.destroy_cpu_pool());
        TCHECK(a.cpu_capacity()    == 0);
    });

    run_test("allocate / deallocate", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pageable));
        float* p = a.allocate_pinned<float>(1024);
        TCHECK(p != nullptr);
        TCHECK(a.cpu_used() > 0);
        a.deallocate_pinned(p);
        TCHECK(a.cpu_used() == 0);
    });

    run_test("read/write", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pageable));
        int* p = a.allocate_pinned<int>(4);
        p[0] = 10; p[1] = 20; p[2] = 30; p[3] = 40;
        TCHECK(p[0] == 10 && p[1] == 20 && p[2] == 30 && p[3] == 40);
        a.deallocate_pinned(p);
    });

    run_test("resize buffer", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pageable));
        float* p = a.allocate_pinned<float>(1024);
        a.resize_pinned<float>(p, 8192);
        TCHECK(p != nullptr);
        a.deallocate_pinned(p);
        TCHECK(a.cpu_used() == 0);
    });

    run_test("free-list coalescing", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pageable));
        float* p1 = a.allocate_pinned<float>(1024);
        float* p2 = a.allocate_pinned<float>(1024);
        a.deallocate_pinned(p1);
        a.deallocate_pinned(p2);
        TCHECK(a.cpu_used() == 0);
        float* p3 = a.allocate_pinned<float>(2048);
        TCHECK(p3 != nullptr);
        a.deallocate_pinned(p3);
    });

    run_test("pool exhaustion throws", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(4 * cuArena::MB, cuArena::CPUMemoryType::Pageable));
        bool threw = false;
        try { a.allocate_pinned<float>(8 * cuArena::MB); }
        catch (const cuArena::cpu_memory_error&) { threw = true; }
        TCHECK(threw);
    });

    run_test("reset", [] {
        cuArena::DeviceArena a;
        TCHECK(a.create_cpu_pool(128 * cuArena::MB, cuArena::CPUMemoryType::Pageable));
        a.allocate_pinned<float>(1024);
        a.reset_cpu_pool();
        TCHECK(a.cpu_used()      == 0);
        TCHECK(a.cpu_available() == a.cpu_capacity());
    });
}

void test_stable_region(cudaStream_t stream) {
    section("Stable region");

    run_test("pointer within in stable bounds", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream,
                                 64 * cuArena::MB));
        float* p = a.allocate<float>(1024, cuArena::Region::Stable);
        TCHECK(p != nullptr);
        // pointer must be within the first 64 MB of the pool
        auto base  = reinterpret_cast<uintptr_t>(a.gpu_base());
        auto uptr  = reinterpret_cast<uintptr_t>(p);
        TCHECK(uptr >= base && uptr < base + 64 * cuArena::MB);
        a.deallocate(p);
    });

    run_test("stable and dynamic pointers do not overlap", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream,
                                 64 * cuArena::MB));
        float* sp = a.allocate<float>(1024, cuArena::Region::Stable);
        float* dp = a.allocate<float>(1024, cuArena::Region::Dynamic);
        TCHECK(sp != nullptr && dp != nullptr && sp != dp);
        auto base  = reinterpret_cast<uintptr_t>(a.gpu_base());
        auto usp   = reinterpret_cast<uintptr_t>(sp);
        auto udp   = reinterpret_cast<uintptr_t>(dp);
        TCHECK(usp < base + 64 * cuArena::MB);  // stable ptr in stable region
        TCHECK(udp >= base + 64 * cuArena::MB); // dynamic ptr beyond stable region
        a.deallocate(sp);
        a.deallocate(dp);
    });

    run_test("stable free + realloc coalesces", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream,
                                 64 * cuArena::MB));
        float* p1 = a.allocate<float>(1024, cuArena::Region::Stable);
        float* p2 = a.allocate<float>(1024, cuArena::Region::Stable);
        a.deallocate(p1);
        a.deallocate(p2);
        TCHECK(a.gpu_stable_used() == 0);
        // After coalescing, the full 64 MB should be reclaimable
        float* p3 = a.allocate<float>(2048, cuArena::Region::Stable);
        TCHECK(p3 != nullptr);
        a.deallocate(p3);
    });

    run_test("stable out-of-memory throws", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(16 * cuArena::MB, cuArena::GPUMemoryType::Device, stream,
                                 4 * cuArena::MB));
        bool threw = false;
        try { a.allocate<float>(8 * cuArena::MB, cuArena::Region::Stable); }
        catch (const cuArena::gpu_memory_error&) { threw = true; }
        TCHECK(threw);
    });

    run_test("resize on stable pointer throws", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream,
                                 64 * cuArena::MB));
        float* p = a.allocate<float>(1024, cuArena::Region::Stable);
        TCHECK(p != nullptr);
        bool threw = false;
        try { a.resize<float>(p, 4096); }
        catch (const cuArena::gpu_memory_error&) { threw = true; }
        TCHECK(threw);
        a.deallocate(p);
    });

    run_test("fallback to dynamic when no stable region", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* p = a.allocate<float>(1024, cuArena::Region::Stable);
        TCHECK(p != nullptr);
        a.deallocate(p);
    });
}

void test_dynamic_bins(cudaStream_t stream) {
    section("Dynamic bins + coalescing");

    run_test("fragmentation then coalesce", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        constexpr size_t N = 8;
        constexpr size_t BLOCK = 8 * cuArena::MB;
        float* ptrs[N];
        for (size_t i = 0; i < N; i++)
            ptrs[i] = a.allocate<float>(BLOCK / sizeof(float));
        // Free even-indexed blocks; create N/2 holes of BLOCK each
        for (size_t i = 0; i < N; i += 2)
            a.deallocate(ptrs[i]);
        // Free odd-indexed blocks; adjacent to already-freed; coalescing must merge them
        for (size_t i = 1; i < N; i += 2)
            a.deallocate(ptrs[i]);
        TCHECK(a.gpu_used() == 0);
        // After full coalescing the entire pool should be available again
        TCHECK(a.gpu_available() == a.gpu_capacity());
    });

    run_test("adjacent frees coalesce into one large block", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(64 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* p1 = a.allocate<float>(8 * cuArena::MB / sizeof(float));
        float* p2 = a.allocate<float>(8 * cuArena::MB / sizeof(float));
        a.deallocate(p1);
        a.deallocate(p2);
        // The two 8 MB blocks should coalesce into one 16 MB block
        float* big = a.allocate<float>(16 * cuArena::MB / sizeof(float));
        TCHECK(big != nullptr);
        a.deallocate(big);
    });

    run_test("Out-of-memory after fragmentation, before compact", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(32 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        // Fill with 4 x 8 MB blocks
        float* p[4];
        for (int i = 0; i < 4; i++)
            p[i] = a.allocate<float>(8 * cuArena::MB / sizeof(float));
        // Leave 2 x 8 MB holes but no contiguous 16 MB
        a.deallocate(p[0]);
        a.deallocate(p[2]);
        // 16 MB request should fail (largest contiguous free is 8 MB)
        bool threw = false;
        try { a.allocate<float>(16 * cuArena::MB / sizeof(float)); }
        catch (const cuArena::gpu_memory_error&) { threw = true; }
        TCHECK(threw);
        a.deallocate(p[1]);
        a.deallocate(p[3]);
    });

    run_test("different size classes go to correct bins", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        // Allocate one block per size class boundary
        float* small  = a.allocate<float>(128  / sizeof(float));           // bin 0  [128B, 256B)
        float* medium = a.allocate<float>(256 * cuArena::KB / sizeof(float)); // bin 11 [256KB, 512KB)
        float* large  = a.allocate<float>(4 * cuArena::MB / sizeof(float));  // bin 18 [4MB, 8MB)
        TCHECK(small != nullptr && medium != nullptr && large != nullptr);
        a.deallocate(small);
        a.deallocate(medium);
        a.deallocate(large);
        TCHECK(a.gpu_used() == 0);
    });
}

void test_compact(cudaStream_t stream) {
    section("Compact");

    run_test("compact with no allocs resets dynamic region", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(64 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* p = a.allocate<float>(8 * cuArena::MB / sizeof(float));
        a.deallocate(p);
        // Fragmentation hole exists in free list; compact should discard it
        size_t before = a.gpu_available();
        int calls = 0;
        a.compact_gpu_dynamic([&](cuArena::addr_t, cuArena::addr_t, size_t) { ++calls; }, stream);
        CUARENA_CHECK(cudaStreamSynchronize(stream));
        TCHECK(calls == 0);
        TCHECK(a.gpu_available() >= before);
        TCHECK(a.gpu_available() == a.gpu_capacity());
    });

    run_test("compact packs blocks and reclaims hole", [&] {
        cuArena::DeviceArena a;
        // Fill entire pool with 8 x 8 MB blocks, then free alternating ones.
        // This leaves 4 x 8 MB holes but no remaining space,
        // so a 16 MB request would fail before compact.
        TCHECK(a.create_gpu_pool(64 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        constexpr int N = 8;
        float* ptrs[N];
        for (int i = 0; i < N; i++)
            ptrs[i] = a.allocate<float>(8 * cuArena::MB / sizeof(float));
        for (int i = 0; i < N; i += 2)
            a.deallocate(ptrs[i]);
        size_t avail_before = a.gpu_available();
        bool threw = false;
        try { a.allocate<float>(16 * cuArena::MB / sizeof(float)); }
        catch (const cuArena::gpu_memory_error&) { threw = true; }
        TCHECK(threw);

        // Compact: 4 active blocks pack to the front, freeing a 32 MB contiguous tail
        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        }, stream);
        CUARENA_CHECK(cudaStreamSynchronize(stream));

        // Available space must be at least as much as before compact
        TCHECK(a.gpu_available() >= avail_before);

        // 16 MB must now succeed
        float* big = a.allocate<float>(16 * cuArena::MB / sizeof(float));
        TCHECK(big != nullptr);
        a.deallocate(big);

        // Free active blocks using updated pointers from the callback
        for (int i = 1; i < N; i += 2) {
            auto it = moved.find(ptrs[i]);
            float* np = (it != moved.end()) ? static_cast<float*>(it->second) : ptrs[i];
            a.deallocate(np);
        }
    });

    run_test("callback not fired for unmoved blocks", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(64 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        // Single allocation at the front for which compact should leave in place
        float* p = a.allocate<float>(8 * cuArena::MB / sizeof(float));
        bool fired = false;
        a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            if (old_p != new_p) fired = true;
        }, stream);
        CUARENA_CHECK(cudaStreamSynchronize(stream));
        TCHECK(!fired);
        a.deallocate(p);
    });

    run_test("compact preserves GPU data", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(64 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        CUARENA_CHECK(cudaDeviceSynchronize());
        int* p1 = a.allocate<int>(4, cuArena::Region::Dynamic);
        int* p2 = a.allocate<int>(4, cuArena::Region::Dynamic);
        int* p3 = a.allocate<int>(4, cuArena::Region::Dynamic);
        // Write some values via managed memory
        p1[0] = 111; p2[0] = 222; p3[0] = 333;
        CUARENA_CHECK(cudaDeviceSynchronize());
        // Free p2 to create a hole
        a.deallocate(p2);
        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        });
        CUARENA_CHECK(cudaDeviceSynchronize());
        int* np1 = moved.count(p1) ? static_cast<int*>(moved[p1]) : p1;
        int* np3 = moved.count(p3) ? static_cast<int*>(moved[p3]) : p3;
        TCHECK(np1[0] == 111);
        TCHECK(np3[0] == 333);
        a.deallocate(np1);
        a.deallocate(np3);
    });
}

void test_compact_high(cudaStream_t stream) {
    section("Compact (high region)");

    run_test("compact high reclaims a stranded hole", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        // Large blocks carve down from the pool end, so A is on top, then B, then C.
        float* A = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        float* B = a.allocate<float>(64 * cuArena::MB / sizeof(float));
        float* C = a.allocate<float>(64 * cuArena::MB / sizeof(float));
        TCHECK(A > B && B > C);
        // Freeing the top block strands its 32 MB. Only a free run at the region
        // bottom rewinds, and C is sitting there.
        a.deallocate(A);
        const size_t free_before = a.gpu_available();
        const size_t largest_before = a.gpu_largest_free_block();
        // 128 MB free, but split 96 and 32, so 120 MB cannot be served.
        TCHECK(free_before > largest_before);
        bool threw = false;
        try { a.allocate<float>(120 * cuArena::MB / sizeof(float)); }
        catch (const cuArena::gpu_memory_error&) { threw = true; }
        TCHECK(threw);

        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        const size_t n = a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        }, stream, cuArena::CompactRegion::High);
        CUARENA_CHECK(cudaStreamSynchronize(stream));

        // Both blocks moved up, and all the free space is one run now.
        TCHECK(n == 2);
        TCHECK(a.gpu_available() >= free_before);
        TCHECK(a.gpu_largest_free_block() == a.gpu_available());

        float* nB = static_cast<float*>(moved[B]);
        float* nC = static_cast<float*>(moved[C]);
        TCHECK(nB > B && nC > C);

        // The request that failed above now fits.
        float* big = a.allocate<float>(120 * cuArena::MB / sizeof(float));
        TCHECK(big != nullptr);
        a.deallocate(big);

        a.deallocate(nB);
        a.deallocate(nC);
    });

    run_test("compact high leaves the low region alone", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* small = a.allocate<float>(8 * cuArena::MB / sizeof(float));
        float* A = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        float* B = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        a.deallocate(A);

        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        }, stream, cuArena::CompactRegion::High);
        CUARENA_CHECK(cudaStreamSynchronize(stream));

        TCHECK(moved.find(small) == moved.end());
        float* nB = moved.count(B) ? static_cast<float*>(moved[B]) : B;
        a.deallocate(nB);
        a.deallocate(small);
    });

    // Packing up makes a block overlap itself when it moves less than its own size.
    // The gap here is too small to bounce through, so the chunked path runs.
    run_test("compact high preserves data through an overlapping move", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        CUARENA_CHECK(cudaDeviceSynchronize());

        const size_t nfloat = 64 * cuArena::MB / sizeof(float);
        float* A = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        float* B = a.allocate<float>(nfloat);
        float* C = a.allocate<float>(nfloat);
        // 48 MB of small blocks leave the gap smaller than one large block.
        float* s1 = a.allocate<float>(12 * cuArena::MB / sizeof(float));
        float* s2 = a.allocate<float>(12 * cuArena::MB / sizeof(float));
        float* s3 = a.allocate<float>(12 * cuArena::MB / sizeof(float));
        float* s4 = a.allocate<float>(12 * cuArena::MB / sizeof(float));
        a.deallocate(A);

        // This pattern comes out wrong if any chunk lands out of order.
        const size_t stride = 4096;
        for (size_t i = 0; i < nfloat; i += stride) {
            B[i] = (float)(i + 1);
            C[i] = -(float)(i + 1);
        }
        CUARENA_CHECK(cudaDeviceSynchronize());

        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        }, nullptr, cuArena::CompactRegion::High);
        CUARENA_CHECK(cudaDeviceSynchronize());

        float* nB = static_cast<float*>(moved[B]);
        float* nC = static_cast<float*>(moved[C]);
        // Confirm both moved up by less than their own size.
        TCHECK((size_t)((char*)nB - (char*)B) < 64 * cuArena::MB);
        TCHECK((size_t)((char*)nC - (char*)C) < 64 * cuArena::MB);
        bool ok = true;
        for (size_t i = 0; i < nfloat; i += stride) {
            if (nB[i] != (float)(i + 1)) ok = false;
            if (nC[i] != -(float)(i + 1)) ok = false;
        }
        TCHECK(ok);

        // The reclaimed space is one usable block.
        float* big = a.allocate<float>(64 * cuArena::MB / sizeof(float));
        TCHECK(big != nullptr);
        a.deallocate(big);

        a.deallocate(nB);
        a.deallocate(nC);
        a.deallocate(s1);
        a.deallocate(s2);
        a.deallocate(s3);
        a.deallocate(s4);
    });
}

void test_compact_high_corners(cudaStream_t stream) {
    section("Compact (high region, corners)");

    run_test("compact high moves many blocks in the right order", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(320 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        CUARENA_CHECK(cudaDeviceSynchronize());

        const size_t nfloat = 32 * cuArena::MB / sizeof(float);
        float* L[8];
        for (int i = 0; i < 8; i++) L[i] = a.allocate<float>(nfloat);
        for (int i = 0; i < 8; i += 2) a.deallocate(L[i]);

        const size_t stride = 4096;
        for (int i = 1; i < 8; i += 2)
            for (size_t k = 0; k < nfloat; k += stride) L[i][k] = (float)(i * 1000 + (k / stride));
        CUARENA_CHECK(cudaDeviceSynchronize());

        const size_t used_before = a.gpu_used();
        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        const size_t n = a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        }, nullptr, cuArena::CompactRegion::High);
        CUARENA_CHECK(cudaDeviceSynchronize());

        TCHECK(n == 4);
        TCHECK(a.gpu_used() == used_before);

        bool ok = true;
        for (int i = 1; i < 8; i += 2) {
            float* np = moved.count(L[i]) ? static_cast<float*>(moved[L[i]]) : L[i];
            for (size_t k = 0; k < nfloat; k += stride)
                if (np[k] != (float)(i * 1000 + (k / stride))) ok = false;
        }
        TCHECK(ok);
        TCHECK(a.gpu_largest_free_block() == a.gpu_available());

        for (int i = 1; i < 8; i += 2) {
            float* np = moved.count(L[i]) ? static_cast<float*>(moved[L[i]]) : L[i];
            a.deallocate(np);
        }
    });

    // A small hole above a big block means many chunks, not the two a same size pair gives.
    run_test("compact high copies a block in many chunks", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        CUARENA_CHECK(cudaDeviceSynchronize());

        const size_t nfloat = 128 * cuArena::MB / sizeof(float);
        float* A = a.allocate<float>(16 * cuArena::MB / sizeof(float));
        float* B = a.allocate<float>(nfloat);
        a.deallocate(A);

        const size_t stride = 4096;
        for (size_t k = 0; k < nfloat; k += stride) B[k] = (float)(k + 7);
        CUARENA_CHECK(cudaDeviceSynchronize());

        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        }, nullptr, cuArena::CompactRegion::High);
        CUARENA_CHECK(cudaDeviceSynchronize());

        // 128 MB moving up by 16 MB, with a 112 MB gap, so it goes in 8 chunks.
        float* nB = static_cast<float*>(moved[B]);
        TCHECK((size_t)((char*)nB - (char*)B) == 16 * cuArena::MB);
        bool ok = true;
        for (size_t k = 0; k < nfloat; k += stride)
            if (nB[k] != (float)(k + 7)) ok = false;
        TCHECK(ok);
        a.deallocate(nB);
    });

    run_test("compact high on a packed region moves nothing", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* A = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        float* B = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        const size_t used_before = a.gpu_used();
        const size_t avail_before = a.gpu_available();

        int fired = 0;
        const size_t n = a.compact_gpu_dynamic([&](cuArena::addr_t, cuArena::addr_t, size_t) {
            ++fired;
        }, stream, cuArena::CompactRegion::High);
        CUARENA_CHECK(cudaStreamSynchronize(stream));

        TCHECK(n == 0);
        TCHECK(fired == 0);
        TCHECK(a.gpu_used() == used_before);
        TCHECK(a.gpu_available() == avail_before);
        a.deallocate(A);
        a.deallocate(B);
    });

    run_test("compact high with no large blocks is a no-op", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Device, stream));
        float* small = a.allocate<float>(8 * cuArena::MB / sizeof(float));
        const size_t avail_before = a.gpu_available();

        const size_t n = a.compact_gpu_dynamic([&](cuArena::addr_t, cuArena::addr_t, size_t) {},
                                               stream, cuArena::CompactRegion::High);
        CUARENA_CHECK(cudaStreamSynchronize(stream));

        TCHECK(n == 0);
        TCHECK(a.gpu_available() == avail_before);
        float* big = a.allocate<float>(200 * cuArena::MB / sizeof(float));
        TCHECK(big != nullptr);
        a.deallocate(big);
        a.deallocate(small);
    });

    run_test("stable allocs survive compact high", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed, nullptr,
                                 32 * cuArena::MB));
        CUARENA_CHECK(cudaDeviceSynchronize());
        int* sp = a.allocate<int>(4, cuArena::Region::Stable);
        float* A = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        float* B = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        sp[0] = 4242;
        CUARENA_CHECK(cudaDeviceSynchronize());
        a.deallocate(A);

        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        }, nullptr, cuArena::CompactRegion::High);
        CUARENA_CHECK(cudaDeviceSynchronize());

        TCHECK(moved.find(sp) == moved.end());
        TCHECK(sp[0] == 4242);
        TCHECK(a.gpu_stable_used() > 0);
        float* nB = moved.count(B) ? static_cast<float*>(moved[B]) : B;
        a.deallocate(nB);
        a.deallocate(sp);
    });

    // Both ends have to stay consistent after a pack, so allocate from each and check
    // the two do not land on top of each other.
    run_test("both regions still allocate after compact high", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(256 * cuArena::MB, cuArena::GPUMemoryType::Managed));
        CUARENA_CHECK(cudaDeviceSynchronize());
        float* A = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        float* B = a.allocate<float>(32 * cuArena::MB / sizeof(float));
        a.deallocate(A);

        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        }, nullptr, cuArena::CompactRegion::High);
        CUARENA_CHECK(cudaDeviceSynchronize());
        float* nB = moved.count(B) ? static_cast<float*>(moved[B]) : B;

        const size_t nbig = 64 * cuArena::MB / sizeof(float);
        const size_t nsmall = 8 * cuArena::MB / sizeof(float);
        float* big = a.allocate<float>(nbig);
        float* small = a.allocate<float>(nsmall);
        for (size_t k = 0; k < nbig; k += 4096) big[k] = 1.0f;
        for (size_t k = 0; k < nsmall; k += 4096) small[k] = 2.0f;
        CUARENA_CHECK(cudaDeviceSynchronize());

        bool ok = true;
        for (size_t k = 0; k < nbig; k += 4096) if (big[k] != 1.0f) ok = false;
        for (size_t k = 0; k < nsmall; k += 4096) if (small[k] != 2.0f) ok = false;
        TCHECK(ok);

        // And a low pack still works with large blocks live.
        a.compact_gpu_dynamic([&](cuArena::addr_t, cuArena::addr_t, size_t) {},
                              nullptr, cuArena::CompactRegion::Low);
        CUARENA_CHECK(cudaDeviceSynchronize());
        for (size_t k = 0; k < nbig; k += 4096) if (big[k] != 1.0f) ok = false;
        TCHECK(ok);

        a.deallocate(big);
        a.deallocate(nB);
    });
}

void test_mixed_stable_dynamic(cudaStream_t stream) {
    section("Mixed stable + dynamic");

    run_test("stable allocs survive compact", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(128 * cuArena::MB, cuArena::GPUMemoryType::Device, stream,
                                 32 * cuArena::MB));
        float* sp = a.allocate<float>(4 * cuArena::MB / sizeof(float), cuArena::Region::Stable);
        float* dp1 = a.allocate<float>(8 * cuArena::MB / sizeof(float), cuArena::Region::Dynamic);
        float* dp2 = a.allocate<float>(8 * cuArena::MB / sizeof(float), cuArena::Region::Dynamic);
        a.deallocate(dp1); // create hole in dynamic region
        std::unordered_map<cuArena::addr_t, cuArena::addr_t> moved;
        a.compact_gpu_dynamic([&](cuArena::addr_t old_p, cuArena::addr_t new_p, size_t) {
            moved[old_p] = new_p;
        }, stream);
        CUARENA_CHECK(cudaStreamSynchronize(stream));
        // Stable pointer must not appear in the moved map
        TCHECK(moved.find(sp) == moved.end());
        // Stable used must be unchanged
        TCHECK(a.gpu_stable_used() > 0);
        float* np2 = moved.count(dp2) ? static_cast<float*>(moved[dp2]) : dp2;
        a.deallocate(sp);
        a.deallocate(np2);
    });

    run_test("used memory includes both regions", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(128 * cuArena::MB, cuArena::GPUMemoryType::Device, stream,
                                 32 * cuArena::MB));
        float* sp = a.allocate<float>(1024, cuArena::Region::Stable);
        float* dp = a.allocate<float>(1024, cuArena::Region::Dynamic);
        size_t total = a.gpu_used();
        TCHECK(total >= a.gpu_stable_used());
        TCHECK(total > 0);
        a.deallocate(sp);
        a.deallocate(dp);
        TCHECK(a.gpu_used() == 0);
        TCHECK(a.gpu_stable_used() == 0);
    });

    run_test("reset clears both stable and dynamic", [&] {
        cuArena::DeviceArena a;
        TCHECK(a.create_gpu_pool(128 * cuArena::MB, cuArena::GPUMemoryType::Device, stream,
                                 32 * cuArena::MB));
        a.allocate<float>(1024, cuArena::Region::Stable);
        a.allocate<float>(1024, cuArena::Region::Dynamic);
        a.reset_gpu_pool();
        TCHECK(a.gpu_used()        == 0);
        TCHECK(a.gpu_stable_used() == 0);
        TCHECK(a.gpu_available()   == a.gpu_dynamic_capacity());
    });
}

void test_constructor_combinations(cudaStream_t stream) {
    section("Constructor combinations");

    run_test("Device + Pinned", [&] {
        cuArena::DeviceArena a(128 * cuArena::MB, 256 * cuArena::MB,
                               cuArena::GPUMemoryType::Device,
                               cuArena::CPUMemoryType::Pinned, stream);
        TCHECK(a.gpu_memory_type() == cuArena::GPUMemoryType::Device);
        TCHECK(a.cpu_memory_type() == cuArena::CPUMemoryType::Pinned);
        TCHECK(a.gpu_capacity()    >  0);
        TCHECK(a.cpu_capacity()    >  0);
    });

    run_test("Device + Pageable", [&] {
        cuArena::DeviceArena a(128 * cuArena::MB, 256 * cuArena::MB,
                               cuArena::GPUMemoryType::Device,
                               cuArena::CPUMemoryType::Pageable, stream);
        TCHECK(a.gpu_memory_type() == cuArena::GPUMemoryType::Device);
        TCHECK(a.cpu_memory_type() == cuArena::CPUMemoryType::Pageable);
        TCHECK(a.gpu_capacity()    >  0);
        TCHECK(a.cpu_capacity()    >  0);
    });

    run_test("Managed + Pinned", [] {
        cuArena::DeviceArena a(128 * cuArena::MB, 256 * cuArena::MB,
                               cuArena::GPUMemoryType::Managed,
                               cuArena::CPUMemoryType::Pinned);
        TCHECK(a.gpu_memory_type() == cuArena::GPUMemoryType::Managed);
        TCHECK(a.cpu_memory_type() == cuArena::CPUMemoryType::Pinned);
        TCHECK(a.gpu_capacity()    >  0);
        TCHECK(a.cpu_capacity()    >  0);
    });

    run_test("Managed + Pageable", [] {
        cuArena::DeviceArena a(128 * cuArena::MB, 256 * cuArena::MB,
                               cuArena::GPUMemoryType::Managed,
                               cuArena::CPUMemoryType::Pageable);
        TCHECK(a.gpu_memory_type() == cuArena::GPUMemoryType::Managed);
        TCHECK(a.cpu_memory_type() == cuArena::CPUMemoryType::Pageable);
        TCHECK(a.gpu_capacity()    >  0);
        TCHECK(a.cpu_capacity()    >  0);
    });
}

int main() {
    cuArena::Logger::set_level(0);

    cudaStream_t stream;
    CUARENA_CHECK(cudaStreamCreate(&stream));

    test_gpu_device(stream);
    test_gpu_managed();
    test_cpu_pinned();
    test_cpu_pageable();
    test_constructor_combinations(stream);
    test_stable_region(stream);
    test_dynamic_bins(stream);
    test_compact(stream);
    test_compact_high(stream);
    test_compact_high_corners(stream);
    test_mixed_stable_dynamic(stream);

    CUARENA_CHECK(cudaStreamDestroy(stream));

    std::cout << '\n' << std::string(60, '=') << '\n';
    std::cout << std::format("  {}{}{} / {} tests passed{}\n",
        passed == total ? CPASS : CFAIL, passed, CHEADER, total, CUAR_CNORMAL);
    std::cout << std::string(60, '=') << "\n\n";

    return (passed == total) ? 0 : 1;
}
