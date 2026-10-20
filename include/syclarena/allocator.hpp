#pragma once

#include <map>
#include <set>
#include <array>
#include <vector>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <cassert>
#include <stdexcept>
#include "logger.hpp"
#include "sysinfo.hpp"
#include "constants.hpp"
#include "definitions.hpp"

namespace syclArena {

    /**
     * Called by compact_gpu_dynamic for each relocated dynamic allocation.
     * Input: (old_ptr, new_ptr, byte_size).
     * The callback runs after all copies have been queued on the provided
     * queue; the caller must wait on that queue before accessing GPU data
     * through the new pointers.
     */
    using RelocateCallback = std::function<void(addr_t, addr_t, size_t)>;

    struct Pool {
        addr_t mem  = nullptr;
        size_t size = 0;
        size_t cap  = 0;
        size_t off  = 0;
    };

    // Where the free bytes sit. A large request can only come from the gap or a high
    // hole, so the split says whether packing the high region would reach one; live_*
    // is how many base pointers that would move.
    struct FreeHighRegion {
        size_t gap = 0;
        size_t low_free = 0, hi_free = 0;
        size_t low_largest = 0, hi_largest = 0;
        size_t low_live = 0, hi_live = 0;
    };

    struct gpu_memory_error : std::runtime_error {
        explicit gpu_memory_error(const char* msg) : std::runtime_error(msg) {}
    };

    struct cpu_memory_error : std::runtime_error {
        explicit cpu_memory_error(const char* msg) : std::runtime_error(msg) {}
    };

    class DeviceArena {

        static constexpr size_t ALIGNMENT   = 128;
        static constexpr size_t MIN_SPLIT   = ALIGNMENT;
        // Requests at or above LARGE_MIN come from a high sub-region that grows down
        // from the pool end; smaller ones grow up from the bottom. Keeping the large
        // blocks at one end stops small allocations stranding the contiguous space a
        // large request needs. Nothing is relocated, so no caller pointer changes.
        // Override at pool creation with SYCLARENA_LARGE_MIN_MB.
        static constexpr size_t LARGE_MIN   = 16 * MB;
        static constexpr size_t GPU_PENALTY = 256 * MB;
        static constexpr size_t CPU_PENALTY = 512 * MB;

        /**
         * Segregated size-class bins for the GPU dynamic region.
         * Bin k holds free blocks whose size falls in [2^(k+7), 2^(k+8)).
         * Example:
         *   k=0  : [128 B ,  256 B)
         *   k=1  : [256 B ,  512 B)
         *   ...
         *   k=31 : [256 GB, overflow)
         * Each bin stores block base addresses ordered ascending.
         * For bins k+1 and above every block is guaranteed >= size,
         * so the first entry is taken immediately.
         */
        static constexpr int NUM_BINS = 32;

        Pool   _gpool, _cpool;
        size_t _gtot = 0, _gfree_mem = 0, _glimit = 0;
        size_t _ctot = 0, _cfree_mem = 0, _climit = 0;
        size_t _gpu_allocated = 0, _cpu_allocated = 0;
        size_t _gpu_peak = 0;
        sycl::queue* _gqueue = nullptr;
        GPUMemoryType _gtype = GPUMemoryType::Device;
        CPUMemoryType _ctype = CPUMemoryType::Pinned;
        bool          _bypass = false;    // bypass mode: no pools, pass through to SYCL USM

        mutable std::mutex _mutex;

        // Stable region (front of GPU pool). Uses coalescing only.
        size_t _stable_size      = 0;
        size_t _stable_bytes     = 0;
        size_t _stable_off       = 0;
        size_t _stable_cap       = 0;
        size_t _gpu_stable_allocated = 0;

        std::map<addr_t, size_t>           _gpu_stable_free_by_addr;
        std::unordered_map<addr_t, size_t> _gpu_stable_alloc_list;

        // Dynamic sub-region (rest of GPU pool).
        // Uses isolated bins + coalescing + compact().
        std::map<addr_t, size_t>                _gpu_free_by_addr;   // coalescing
        std::array<std::set<addr_t>, NUM_BINS>  _gpu_bins;           // size index
        std::unordered_map<addr_t, size_t>      _gpu_alloc_list;     // dynamic allocs

        // High sub-region. It holds few, large blocks, so an address ordered free map
        // with first fit and coalescing is enough and it needs no size bins.
        size_t                             _large_min = LARGE_MIN;
        size_t                             _gpu_hi = 0;              // its bottom, as an offset
        std::map<addr_t, size_t>           _gpu_hi_free_by_addr;
        std::unordered_map<addr_t, size_t> _gpu_hi_alloc_list;
        size_t                             _gpu_hi_allocated = 0;
        size_t                             _gpu_hi_serves = 0;       // large requests served, ever

        std::map<addr_t, size_t>           _cpu_free_by_addr;
        std::map<size_t, std::set<addr_t>> _cpu_free_by_size;
        std::unordered_map<addr_t, size_t> _cpu_alloc_list;

        int _bin_index(const size_t& size) const noexcept {
            if (size >= (size_t(1) << (NUM_BINS - 1 + 7))) return NUM_BINS - 1;
            int bit = 63 - __builtin_clzll(size);
            int idx = bit - 7;
            return (idx < 0) ? 0 : idx;
        }

        void   _insert_gpu_free    (const addr_t& ptr, const size_t& size);
        void   _insert_cpu_free    (const addr_t& ptr, const size_t& size);
        addr_t _pop_gpu_free       (const size_t& size, size_t& out_size);
        addr_t _pop_cpu_free       (const size_t& size, size_t& out_size);
        addr_t _gpu_alloc          (const size_t& aligned_bytes);
        addr_t _gpu_hi_alloc       (const size_t& aligned_bytes);
        // Plan a region's packing under the lock: `blocks` in the order they must move,
        // `new_ptrs` their destinations, plus the bounce buffer. Rebuilds that region's
        // alloc/free bookkeeping.
        void   _plan_compact_low   (std::vector<std::pair<addr_t, size_t>>& blocks,
                                    std::vector<addr_t>& new_ptrs,
                                    byte_t*& bounce, size_t& bounce_size);
        void   _plan_compact_high  (std::vector<std::pair<addr_t, size_t>>& blocks,
                                    std::vector<addr_t>& new_ptrs,
                                    byte_t*& bounce, size_t& bounce_size);
        // Moves one block with memmove semantics, in either direction.
        void   _move_block         (byte_t* dst, byte_t* src, size_t sz, byte_t* bounce,
                                    size_t bounce_size, sycl::queue* queue);
        void   _insert_gpu_hi_free (const addr_t& ptr, const size_t& size);
        void   _gpu_hi_free_block  (const addr_t& ptr);
        void   _sync_gap           () noexcept { _gpool.cap = _gpu_hi - _gpool.off; }
        addr_t _stable_alloc       (const size_t& aligned_bytes);
        addr_t _cpu_alloc          (const size_t& aligned_bytes);
        void   _gpu_free_block     (const addr_t& ptr);
        void   _stable_free_block  (const addr_t& ptr);
        void   _cpu_free_block     (const addr_t& ptr);

        // Bypass mode: no pools; allocations go straight to SYCL USM / std allocators.
        addr_t _bypass_gpu_alloc  (const size_t& aligned_bytes);
        void   _bypass_gpu_free   (const addr_t& ptr);
        addr_t _bypass_cpu_alloc  (const size_t& aligned_bytes);
        void   _bypass_cpu_free   (const addr_t& ptr);

    public:

        DeviceArena           () = default;
        DeviceArena           (const size_t& cpu_limit,
                               const size_t& gpu_limit,
                               const GPUMemoryType& gtype = GPUMemoryType::Device,
                               const CPUMemoryType& ctype = CPUMemoryType::Pinned,
                               sycl::queue* queue = nullptr,
                               const bool& bypass = false);
        DeviceArena           (const DeviceArena&) = delete;
        DeviceArena& operator=(const DeviceArena&) = delete;
        DeviceArena           (DeviceArena&&)      = delete;
        DeviceArena& operator=(DeviceArena&&)      = delete;

        ~DeviceArena();

        bool create_gpu_pool    (const size_t& limit = 0,
                                 const GPUMemoryType& gtype = GPUMemoryType::Device,
                                 sycl::queue* queue = nullptr,
                                 const size_t& stable_bytes = 0);
        bool create_cpu_pool    (const size_t& limit = 0,
                                 const CPUMemoryType& ctype = CPUMemoryType::Pinned);
        bool destroy_gpu_pool   ();
        bool destroy_cpu_pool   ();
        bool resize_gpu_pool    (const size_t& new_size, sycl::queue* queue = nullptr);
        bool resize_cpu_pool    (const size_t& new_size);
        void reset_gpu_pool     ();
        void reset_cpu_pool     ();

        GPUMemoryType gpu_memory_type() const noexcept { return _gtype; }
        CPUMemoryType cpu_memory_type() const noexcept { return _ctype; }

        constexpr
        size_t        alignment      () const noexcept { return ALIGNMENT; }
        size_t        align_up       (const size_t& n) const noexcept { return (n + ALIGNMENT - 1) & ~(ALIGNMENT - 1); }
        bool          is_aligned     (const size_t& n) const noexcept { return (n & (ALIGNMENT - 1)) == 0; }

        template<class T>
        T* allocate(const size_t& count, Region region = Region::Dynamic) {
            if (!count) return nullptr;
            const size_t aligned = align_up(count * sizeof(T));
            std::lock_guard<std::mutex> lock(_mutex);
            if (_bypass) return static_cast<T*>(_bypass_gpu_alloc(aligned));
            if (region == Region::Stable) {
                if (_stable_size == 0) {
                    Logger::debug("syclArena: pool has no stable region — falling back to dynamic");
                    return static_cast<T*>(_gpu_alloc(aligned));
                }
                return static_cast<T*>(_stable_alloc(aligned));
            }
            return static_cast<T*>(_gpu_alloc(aligned));
        }

        template<class T>
        void deallocate(T* ptr) {
            if (!ptr) return;
            std::lock_guard<std::mutex> lock(_mutex);
            if (_bypass) { _bypass_gpu_free(static_cast<addr_t>(ptr)); return; }
            addr_t p = static_cast<addr_t>(ptr);
            if (_stable_size > 0) {
                byte_t* stable_end = static_cast<byte_t*>(_gpool.mem) + _stable_size;
                if (p >= _gpool.mem && static_cast<byte_t*>(p) < stable_end) {
                    _stable_free_block(p);
                    return;
                }
            }
            if (_gpu_hi_alloc_list.count(p)) _gpu_hi_free_block(p);
            else                             _gpu_free_block(p);
        }

        template<class T>
        void resize(T*& ptr, const size_t& new_count) {
            if (!new_count) {
                deallocate(ptr);
                ptr = nullptr;
                return;
            }
            const size_t aligned = align_up(new_count * sizeof(T));
            std::lock_guard<std::mutex> lock(_mutex);
            if (_bypass) {
                if (ptr) _bypass_gpu_free(static_cast<addr_t>(ptr));
                ptr = static_cast<T*>(_bypass_gpu_alloc(aligned));
                return;
            }
            if (ptr) {
                addr_t p = static_cast<addr_t>(ptr);
                // Resizing a stable allocation is not permitted
                if (_stable_size > 0) {
                    byte_t* stable_end = static_cast<byte_t*>(_gpool.mem) + _stable_size;
                    if (p >= _gpool.mem && static_cast<byte_t*>(p) < stable_end) {
                        Logger::debug("syclArena: resize called on a stable-region pointer %p", p);
                        throw gpu_memory_error("cannot resize a stable region allocation");
                    }
                }
                auto it = _gpu_alloc_list.find(p);
                if (it != _gpu_alloc_list.end()) {
                    if (it->second >= aligned) return;
                    _gpu_free_block(p);
                    ptr = nullptr;
                }
                else {
                    // A previous resize may have crossed LARGE_MIN, moving the
                    // block into the high (large-block) region.
                    auto hi = _gpu_hi_alloc_list.find(p);
                    if (hi == _gpu_hi_alloc_list.end()) {
                        Logger::debug("resize: pointer %p not in GPU alloc list", p);
                        throw gpu_memory_error("invalid pointer");
                    }
                    if (hi->second >= aligned) return;
                    _gpu_hi_free_block(p);
                    ptr = nullptr;
                }
            }
            ptr = static_cast<T*>(_gpu_alloc(aligned));
        }

        template<class T>
        T* allocate_pinned(const size_t& count) {
            if (!count) return nullptr;
            const size_t aligned = align_up(count * sizeof(T));
            std::lock_guard<std::mutex> lock(_mutex);
            if (_bypass) return static_cast<T*>(_bypass_cpu_alloc(aligned));
            return static_cast<T*>(_cpu_alloc(aligned));
        }

        template<class T>
        void deallocate_pinned(T* ptr) {
            if (!ptr) return;
            std::lock_guard<std::mutex> lock(_mutex);
            if (_bypass) { _bypass_cpu_free(static_cast<addr_t>(ptr)); return; }
            _cpu_free_block(static_cast<addr_t>(ptr));
        }

        template<class T>
        void resize_pinned(T*& ptr, const size_t& new_count) {
            if (!new_count) {
                deallocate_pinned(ptr);
                ptr = nullptr;
                return;
            }
            const size_t aligned = align_up(new_count * sizeof(T));
            std::lock_guard<std::mutex> lock(_mutex);
            if (_bypass) {
                if (ptr) _bypass_cpu_free(static_cast<addr_t>(ptr));
                ptr = static_cast<T*>(_bypass_cpu_alloc(aligned));
                return;
            }
            if (ptr) {
                auto it = _cpu_alloc_list.find(static_cast<addr_t>(ptr));
                if (it == _cpu_alloc_list.end()) {
                    Logger::debug("resize pinned: pointer %p not in CPU alloc list", static_cast<addr_t>(ptr));
                    throw cpu_memory_error("invalid pointer");
                }
                if (it->second >= aligned) return;
                _cpu_free_block(static_cast<addr_t>(ptr));
                ptr = nullptr;
            }
            ptr = static_cast<T*>(_cpu_alloc(aligned));
        }

        /**
         * GPU compaction and pointer relocation.
         *
         * Packs one end of the dynamic pool, closing that region's holes: Low (the
         * default) slides the small blocks to the front, High slides the large blocks
         * to the pool end so their holes merge into the shared gap. Returns the number
         * of blocks relocated; wait on `queue` before using the new pointers.
         */
        size_t compact_gpu_dynamic(RelocateCallback callback, sycl::queue* queue = nullptr,
                                   CompactRegion which = CompactRegion::Low);

        addr_t gpu_base             () const noexcept { return _gpool.mem; }
        size_t gpu_capacity         () const noexcept { return _gpool.size; }
        size_t gpu_stable_capacity  () const noexcept { return _stable_size; }
        size_t gpu_dynamic_capacity () const noexcept { return _gpool.size - _stable_size; }
        size_t gpu_stable_used      () const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            return _gpu_stable_allocated;
        }
        size_t gpu_stable_available () const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            size_t freed = 0;
            for (auto& [ptr, sz] : _gpu_stable_free_by_addr)
                freed += sz;
            return _stable_cap + freed;
        }
        // Same as gpu_stable_available() when unfragmented,
        // but much smaller when fragmented.
        size_t gpu_stable_largest_free_block() const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            size_t largest = _stable_cap;
            for (const auto& [ptr, sz] : _gpu_stable_free_by_addr)
                largest = std::max(largest, sz);
            return largest;
        }

        size_t gpu_dynamic_used     () const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            return _gpu_allocated;
        }
        size_t gpu_used             () const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            return _gpu_allocated + _gpu_hi_allocated + _gpu_stable_allocated;
        }

        size_t gpu_peak_used        () const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            return _gpu_peak;
        }

        size_t gpu_available        () const noexcept {   // dynamic region, both ends
            std::lock_guard<std::mutex> lock(_mutex);
            size_t freed = 0;
            for (auto& [ptr, sz] : _gpu_free_by_addr)    freed += sz;
            for (auto& [ptr, sz] : _gpu_hi_free_by_addr) freed += sz;
            return _gpool.cap + freed;                   // cap is the gap between the ends
        }

        // Same as gpu_available() when unfragmented,
        // but much smaller when fragmented.
        size_t gpu_largest_free_block() const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            size_t largest = _gpool.cap;                 // the gap is itself one block
            for (const auto& [ptr, sz] : _gpu_free_by_addr)    largest = std::max(largest, sz);
            for (const auto& [ptr, sz] : _gpu_hi_free_by_addr) largest = std::max(largest, sz);
            return largest;
        }

        // Large requests served, ever. The multiplier on any per-allocation scheme.
        size_t gpu_hi_serves() const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            return _gpu_hi_serves;
        }

        // The per-region breakdown behind gpu_available()/gpu_largest_free_block().
        FreeHighRegion gpu_free_high_region() const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            FreeHighRegion fp;
            fp.gap = _gpool.cap;
            for (const auto& [ptr, sz] : _gpu_free_by_addr) {
                fp.low_free += sz;
                fp.low_largest = std::max(fp.low_largest, sz);
            }
            for (const auto& [ptr, sz] : _gpu_hi_free_by_addr) {
                fp.hi_free += sz;
                fp.hi_largest = std::max(fp.hi_largest, sz);
            }
            fp.low_live = _gpu_alloc_list.size();
            fp.hi_live = _gpu_hi_alloc_list.size();
            return fp;
        }

        size_t cpu_capacity () const noexcept { return _cpool.size; }

        size_t cpu_used     () const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            return _cpu_allocated;
        }

        size_t cpu_available() const noexcept {
            std::lock_guard<std::mutex> lock(_mutex);
            size_t freed = 0;
            for (auto& [sz, addrs] : _cpu_free_by_size)
                freed += sz * addrs.size();
            return _cpool.cap + freed;
        }

    };

}
