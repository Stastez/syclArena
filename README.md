![GitHub License](https://img.shields.io/github/license/muhos/cuArena)
[![Build Status](https://github.com/muhos/cuArena/actions/workflows/test-build.yml/badge.svg)](https://github.com/muhos/cuArena/actions/workflows/test-build.yml)

# syclArena

Pool-based SYCL 2020 memory allocator with isolated-bin allocation, address-ordered coalescing free lists, stable/dynamic memory regions, and on-demand defragmentation. muhos built cuArena for their own SYCL projects, now ported to SYCL. All my allocators will be migrating to it as their memory backend. If it fits your use case, you're welcome to use it too.

---

## Overview

syclArena pre-allocates a contiguous block of GPU memory and a block of CPU memory at startup (one `sycl::malloc_device`/`sycl::malloc_shared` + one `sycl::malloc_host` call), then services all subsequent allocations entirely in host-side C++ — no `sycl::malloc_device` call is made per allocation.

The GPU pool is divided into two regions:

- **Stable region** (front of pool), for long-term allocations that persist across iterations. Uses a linear pointer with coalescing-only free list. Compact never touches it.
- **Dynamic region** (rest of pool), for temporary allocations that are frequently created and destroyed. Uses isolated size-class bins for $O(1)$ allocation in the common case, with coalescing on every free. Supports on-demand compaction to eliminate fragmentation.

Inside the dynamic region, requests at or above a large-block threshold (16 MB by default, override with the `SYCLARENA_LARGE_MIN_MB` environment variable) are carved from a **high sub-region** that grows down from the pool end, so large blocks cannot strand the contiguous space smaller allocations need. `compact_gpu_dynamic()` can pack either end: `CompactRegion::Low` slides the small blocks to the front, `CompactRegion::High` slides the large blocks to the pool end.

The design is suited for SYCL applications that perform many repeated allocations of varying sizes in performance-critical paths, where the overhead of `sycl::malloc_device` and `sycl::free` becomes measurable. AI frameworks like PyTorch and TensorFlow ship their own pool allocators for this reason but those allocators are inseparable from their runtimes. syclArena offers a similar approach as a lightweight, embeddable library for any SYCL application.

---

## Designed for Performance

- **Zero runtime overhead**, every allocation and deallocation is a pure host-side operation; no `sycl::malloc_device` per call
- **$O(1)$ allocation**, 32 isolated size-class bins covering $[128\,\text{B},\,256\,\text{GB})$; the right bin is found in one step, no scanning
- **Immediate coalescing**, adjacent free blocks merge on every `deallocate()`, keeping the free list compact without a separate GC pass
- **On-demand defragmentation**, `compact_gpu_dynamic()` eliminates fragmentation in one async pass when needed; a callback delivers the updated pointers

---

## Requirements

| Dependency | Notes |
|---|---|
| AdaptiveCpp (`acpp`) | SYCL 2020 compiler; the library is built as plain C++ through `acpp` |
| SYCL device runtime | e.g. ROCm on AMD GPUs |
| C++ standard | C++20 |
| CMake | 3.18 |
| GCC | 13 (for `std::format` in examples) |

---

## Build

```bash
git clone https://github.com/stastez/syclArena.git
cd syclArena
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the usage example:
```bash
./build/syclArena_usage
```

Run the benchmark:
```bash
./build/syclArena_benchmark
```

Run the tests:
```bash
./build/syclArena_test
```

### Options

| Option | Default | Description |
|---|---|---|
| `SYCLARENA_BUILD_EXAMPLES` | `ON` | Build `syclArena_usage` and `syclArena_benchmark` |
| `SYCLARENA_BUILD_TESTS` | `ON` | Build `syclArena_test` |

---

## Integration

After building, `build/` contains the static library (`libsyclarena.a`) and headers are under `include/syclarena/`. Point your build system at both and compile with `acpp` (the SYCL runtime is linked automatically by the `acpp` driver). Alternatively, install into a system prefix:

```bash
cmake --install build --prefix /usr/local
```

---

## Quick start

```cpp
#include <syclarena/syclarena.hpp>

int main() {
    syclArena::Logger::set_level(1);

    sycl::queue stream;

    syclArena::DeviceArena alloc;

    // 4 GB GPU pool: 512 MB stable for long-term buffers, rest dynamic for temporaries
    alloc.create_gpu_pool(4 * syclArena::GB, syclArena::GPUMemoryType::Device, &stream, 512 * syclArena::MB);
    alloc.create_cpu_pool(512 * syclArena::MB, syclArena::CPUMemoryType::Pinned);
    SYCLARENA_SYNC(&stream);

    // Stable allocation which remains alive for the duration of the session
    float* persistent = alloc.allocate<float>(128 * syclArena::MB / sizeof(float), syclArena::Region::Stable);

    // Dynamic allocations that are frequently created and destroyed
    float* scratch = alloc.allocate<float>(256 * syclArena::MB / sizeof(float));
    alloc.deallocate(scratch);

    // Compact the dynamic region when fragmentation builds up
    std::unordered_map<syclArena::addr_t, syclArena::addr_t> map;
    alloc.compact_gpu_dynamic(
        [&](syclArena::addr_t old_p, syclArena::addr_t new_p, size_t) {
            map[old_p] = new_p;
        },
        &stream);
    SYCLARENA_SYNC(&stream);

    // Pinned CPU buffer for host <-> device transfers
    float* h_stage = alloc.allocate_pinned<float>(32 * syclArena::MB / sizeof(float));

    alloc.deallocate(persistent);
    alloc.deallocate_pinned(h_stage);
    alloc.destroy_gpu_pool();
    alloc.destroy_cpu_pool();
}
```

A null `queue` argument (the default everywhere) selects a process-wide default queue on the default SYCL device.

---

## API

### Types

```cpp
enum class GPUMemoryType { Device, Managed };   // sycl::malloc_device / sycl::malloc_shared
enum class CPUMemoryType { Pinned, Pageable };  // sycl::malloc_host   / std::malloc
enum class Region        { Stable, Dynamic };
enum class CompactRegion { Low, High };         // which end of the dynamic pool to pack

// Callback fired by compact_gpu_dynamic for each relocated block.
// Runs after all copies are queued on the queue; wait on the queue before
// accessing data through the new pointers.
using RelocateCallback = std::function<void(addr_t old_ptr, addr_t new_ptr, size_t bytes)>;
```

### Pool lifecycle

```cpp
// Construct and create both pools immediately.
// bypass = true: no pools are created; allocate/deallocate pass
// straight through to SYCL USM (see "Bypass mode" below).
DeviceArena(size_t cpu_limit, size_t gpu_limit,
            GPUMemoryType gtype  = GPUMemoryType::Device,
            CPUMemoryType ctype  = CPUMemoryType::Pinned,
            sycl::queue*  queue  = nullptr,
            bool          bypass = false);

// Create pools manually (limit = 0 uses all available memory minus a safety margin)
// stable_bytes: size of the stable region at the front of the GPU pool (0 = no stable region)
bool create_gpu_pool(size_t limit        = 0,
                     GPUMemoryType type  = GPUMemoryType::Device,
                     sycl::queue*  queue = nullptr,
                     size_t stable_bytes = 0);
bool create_cpu_pool(size_t limit        = 0,
                     CPUMemoryType type  = CPUMemoryType::Pinned);

// Destroy pools and release memory to the runtime
bool destroy_gpu_pool();
bool destroy_cpu_pool();

// Resize (invalidates all existing allocations, preserves type and stable proportion)
bool resize_gpu_pool(size_t new_size, sycl::queue* queue = nullptr);
bool resize_cpu_pool(size_t new_size);

// Reset free lists (invalidates all existing allocations, no GPU free/malloc)
void reset_gpu_pool();
void reset_cpu_pool();
```

### GPU allocation

```cpp
// region defaults to Dynamic; pass Region::Stable for long-term allocations
template<class T> T*   allocate  (size_t count, Region region = Region::Dynamic);
template<class T> void deallocate(T* ptr);
template<class T> void resize    (T*& ptr, size_t new_count);  // Dynamic only
```

### Compaction

```cpp
// Compact the dynamic allocations to one end of the dynamic region
// (Low = front, the default; High = pool end, for the large-block region).
// Queues copies on `queue`; caller must wait on it before using new pointers.
size_t compact_gpu_dynamic(RelocateCallback callback, sycl::queue* queue = nullptr,
                           CompactRegion which = CompactRegion::Low);
```

### CPU pinned allocation

```cpp
template<class T> T*   allocate_pinned  (size_t count);
template<class T> void deallocate_pinned(T* ptr);
template<class T> void resize_pinned    (T*& ptr, size_t new_count);
```

### Bypass mode

```cpp
syclArena::DeviceArena alloc(128 * syclArena::MB, 256 * syclArena::MB,
                             syclArena::GPUMemoryType::Device,
                             syclArena::CPUMemoryType::Pinned, &stream,
                             /*bypass=*/true);
```

With `bypass = true` the arena makes no pool allocation at construction and all arena functionality is off:

- `allocate`/`deallocate` pass straight through to `sycl::malloc_device`/`sycl::malloc_shared` (selected by `GPUMemoryType`) and `sycl::free`
- `allocate_pinned`/`deallocate_pinned` pass through to `sycl::malloc_host` and `sycl::free` (`std::malloc`/`std::free` for `CPUMemoryType::Pageable`)
- `resize`/`resize_pinned` free the old buffer and allocate a new one
- the `Region::Stable` argument is ignored
- pool lifecycle methods (`create_*_pool`, `resize_*_pool`) are rejected, `compact_gpu_dynamic()` is a no-op, and the stats queries report zero

Use it to keep the syclArena allocator interface while deferring all allocation to the SYCL runtime.

### Stats

```cpp
addr_t gpu_base             ();  // raw base pointer of the GPU pool
size_t gpu_capacity         ();  // total GPU pool size
size_t gpu_stable_capacity  ();  // size of the stable region
size_t gpu_dynamic_capacity ();  // size of the dynamic region
size_t gpu_used             ();  // bytes used across both regions
size_t gpu_stable_used      ();  // bytes used in the stable region
size_t gpu_available        ();  // free bytes in the dynamic region

size_t cpu_capacity ();
size_t cpu_used     ();
size_t cpu_available();
```

### Logger

```cpp
syclArena::Logger::set_level(0);  // silent (default)
syclArena::Logger::set_level(1);  // info: pool creation, destruction, resize
syclArena::Logger::set_level(2);  // debug
```

### Error handling

```cpp
try {
    float* p = alloc.allocate<float>(too_many);
}
catch (const syclArena::gpu_memory_error& e) { ... }  // pool exhausted
catch (const syclArena::cpu_memory_error& e) { ... }  // pinned pool exhausted
catch (const syclArena::gpu_error& e)       { ... }   // SYCL runtime error
```

---

## Benchmark results

Measured on AMD Radeon RX 6800 XT (gfx1030, 16 GB), AdaptiveCpp (`acpp`, clang 20), Linux. 500 iterations, 50 warmup. All times in microseconds. Baselines are SYCL's own allocators; `sycl::malloc_device`/`sycl::malloc_shared` have no stream-ordered variant, so there is no async baseline to compare against.

### Device pool

**Benchmark 1: single large alloc + free (8 MB)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `sycl::malloc_device` + `sycl::free` | 314.98 | 322.18 | 252.36 | 408.88 |
| syclArena alloc + dealloc | **0.08** | **0.08** | **0.07** | **0.14** |

syclArena is **$4108{\times}$ faster** than `sycl::malloc_device`.

**Benchmark 2: batch alloc 512 $\times$ 1 KB, then free all**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `sycl::malloc_device` + `sycl::free` | 901.70 | 888.92 | 872.84 | 1265.72 |
| syclArena alloc + dealloc | **64.66** | **61.51** | **59.29** | **114.57** |

syclArena is **$14.0{\times}$ faster** than `sycl::malloc_device`.

**Benchmark 3: interleaved ping-pong (4 live $\times$ 1 MB, cycling)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `sycl::malloc_device` + `sycl::free` | 1.42 | 1.41 | 1.35 | 1.67 |
| syclArena alloc + dealloc | **0.08** | **0.08** | **0.07** | **0.11** |

syclArena is **$18.0{\times}$ faster** than `sycl::malloc_device`.

**Benchmark 4: resize chain (1 MB to 8 MB in 8 doublings)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `sycl::malloc_device` + `sycl::free` | 2258.62 | 2228.70 | 2103.10 | 2655.82 |
| syclArena resize | **0.65** | **0.65** | **0.64** | **0.81** |

syclArena is **$3478{\times}$ faster** than `sycl::malloc_device`.

**Benchmark 5: fragmentation + compact cycle (8 $\times$ 32 MB, free alternating)**

| operation | mean | median | min | max |
|---|---|---|---|---|
| `compact_gpu_dynamic` (256 MB active) | 543.70 | 546.72 | 479.51 | 562.34 |
| alloc 64 MB after compact | **0.12** | **0.06** | **0.03** | **1.87** |

Compact + subsequent alloc total: **543.82 $\mu s$**. The 32 MB blocks live in the high (large-block) region, so the cycle compacts `CompactRegion::High`; the timing window includes the queue wait, so the compact cost is purely GPU memcpy bandwidth. After compaction, the large allocation completes in a fraction of a microsecond.

**Benchmark 6: stable vs dynamic allocation latency**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| stable 256 B | **0.06** | **0.06** | **0.05** | **0.11** |
| dynamic 256 B | 0.08 | 0.08 | 0.07 | 0.10 |
| stable 256 KB | **0.06** | **0.06** | **0.05** | **0.07** |
| dynamic 256 KB | 0.11 | 0.11 | 0.10 | 0.12 |
| stable 2 MB | **0.06** | **0.06** | **0.05** | **0.07** |
| dynamic 2 MB | 0.11 | 0.11 | 0.10 | 0.11 |

Stable is $1.36–1.89{\times}$ faster than dynamic across all size classes. The gap reflects the bin-lookup overhead in the dynamic allocator vs. the stable region's simple linear pointer.

### Managed pool

**Benchmark 1: single large alloc + free (8 MB)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `sycl::malloc_shared` + `sycl::free` | 20272.07 | 3271.48 | 2699.28 | 121392.25 |
| syclArena alloc + dealloc | **0.08** | **0.08** | **0.07** | **0.10** |

syclArena is **$>40000{\times}$ faster** than `sycl::malloc_shared` on a median basis (the baseline mean is inflated by occasional spikes over 120 ms).

**Benchmark 2: batch alloc 512 $\times$ 1 KB, then free all**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `sycl::malloc_shared` + `sycl::free` | 47615.27 | 47382.45 | 46465.33 | 87616.33 |
| syclArena alloc + dealloc | **60.19** | **60.02** | **58.47** | **93.25** |

syclArena is **$791{\times}$ faster** than `sycl::malloc_shared`.

**Benchmark 3: interleaved ping-pong (4 live $\times$ 1 MB, cycling)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `sycl::malloc_shared` + `sycl::free` | 836.46 | 477.31 | 399.82 | 2390.00 |
| syclArena alloc + dealloc | **0.10** | **0.10** | **0.10** | **0.25** |

syclArena is **$8080{\times}$ faster** than `sycl::malloc_shared`.

**Benchmark 4: resize chain (1 MB to 8 MB in 8 doublings)**

| allocator | mean | median | min | max |
|---|---|---|---|---|
| `sycl::malloc_shared` + `sycl::free` | 25551.34 | 24278.30 | 22899.69 | 156012.62 |
| syclArena resize | **0.57** | **0.57** | **0.56** | **0.71** |

syclArena is **$44910{\times}$ faster** than `sycl::malloc_shared`.

### Remarks

`sycl::malloc_shared` on this platform carries large, unpredictable per-call latency (max values of 87–156 ms in the batch/resize benchmarks) on top of a 0.3–2.7 ms median. syclArena's managed pool amortizes all of that to a single allocation at creation time; the suballocations run at 0.08–0.6 $\mu s$ regardless.

The resize benchmark captures the compounding cost most clearly: 8 sequential `sycl::malloc_device` calls accumulate to 2.3 ms per chain, while syclArena's free-list reuse keeps the entire chain under $1\,\mu s$ for both pool types.

syclArena's max values stay within roughly $2\times$ of its median across all benchmarks, reflecting the deterministic nature of pure host-side allocation.
