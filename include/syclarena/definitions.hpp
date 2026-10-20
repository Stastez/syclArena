#pragma once

#include <cassert>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "types.hpp"
#include "logger.hpp"

namespace syclArena {

struct gpu_error : std::runtime_error {
    explicit gpu_error(const char* msg) : std::runtime_error(msg) {}
};

// A null queue selects a process-wide default queue on the default device,
// mirroring CUDA's implicit default stream / current device.
inline sycl::queue& default_queue() noexcept {
    static sycl::queue q;
    return q;
}

inline sycl::queue& resolve_queue(sycl::queue* queue) noexcept {
    return queue ? *queue : default_queue();
}

// Total and currently free GPU memory on `dev`.
// The free query is an AdaptiveCpp vendor extension; on implementations
// without it the total is reported as free and pool capping degrades
// gracefully (allocation fails later instead of being capped up front).
inline void gpu_mem_info(const sycl::device& dev, size_t& free_bytes, size_t& total_bytes) {
    total_bytes = static_cast<size_t>(dev.get_info<sycl::info::device::global_mem_size>());
    free_bytes  = total_bytes;
    try {
        free_bytes = static_cast<size_t>(dev.get_info<sycl::info::device::AdaptiveCpp_free_memory>());
    }
    catch (const sycl::exception&) {
        // no free-memory query: assume the whole device is free
    }
}

#define SYCLARENA_CHECK(call)                                                          \
    do {                                                                            \
        try { (call); }                                                             \
        catch (const sycl::exception& _e) {                                         \
            syclArena::Logger::error("SYCL error [%s:%d]: %s",                        \
                __FILE__, __LINE__, _e.what());                                     \
            throw syclArena::gpu_error(_e.what());                                    \
        }                                                                           \
    } while (0)

#define SYCLARENA_SYNC(queue)  SYCLARENA_CHECK(syclArena::resolve_queue(queue).wait_and_throw())
#define SYCLARENA_SYNCALL      SYCLARENA_CHECK(syclArena::resolve_queue(nullptr).wait_and_throw())

}
