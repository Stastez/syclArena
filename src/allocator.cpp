#include <syclarena/allocator.hpp>

namespace syclArena {

    DeviceArena::DeviceArena(const size_t& cpu_limit,
                             const size_t& gpu_limit,
                             const GPUMemoryType& gtype,
                             const CPUMemoryType& ctype,
                             sycl::queue* queue) {
        if (!create_cpu_pool(cpu_limit, ctype)) {
            std::string type = (ctype == CPUMemoryType::Pinned) ? "pinned" : "pageable";
            std::string msg = "failed to create CPU " + type + " pool";
            throw cpu_memory_error(msg.c_str());
        }
        if (!create_gpu_pool(gpu_limit, gtype, queue)) {
            std::string type = (gtype == GPUMemoryType::Device) ? "device" : "managed";
            std::string msg = "failed to create GPU " + type + " pool";
            throw gpu_memory_error(msg.c_str());
        }
    }

    DeviceArena::~DeviceArena() {
        destroy_gpu_pool();
        destroy_cpu_pool();
    }

    bool DeviceArena::create_gpu_pool(const size_t& limit,
                                      const GPUMemoryType& type,
                                      sycl::queue* queue,
                                      const size_t& stable_bytes) {
        if (_gpool.mem) {
            Logger::info("GPU pool already exists (%zu MB)", _gpool.size / MB);
            return true;
        }
        gpu_mem_info(resolve_queue(queue).get_device(), _gfree_mem, _gtot);
        if (_gfree_mem > GPU_PENALTY) _gfree_mem -= GPU_PENALTY;
        _glimit = limit ? align_up(limit) : 0;
        if (_glimit) {
            if (_gfree_mem && _glimit > _gfree_mem) {
                Logger::warn("GPU pool: requested %zu MB, only %zu MB free — capping",
                             _glimit / MB, _gfree_mem / MB);
                _gpool.size = _gfree_mem;
            }
            else {
                _gpool.size = _glimit;
            }
        }
        else {
            if (!_gfree_mem) throw gpu_memory_error("GPU pool: no free GPU memory available");
            _gpool.size = _gfree_mem;
        }

        if (stable_bytes > 0) {
            _stable_size = align_up(stable_bytes);
            if (_stable_size >= _gpool.size) {
                Logger::warn("GPU pool: stable bytes (%zu MB) exceeds pool size (%zu MB) - ignoring stable region",
                             stable_bytes / MB, _gpool.size / MB);
                _stable_size  = 0;
                _stable_bytes = 0;
            }
            else {
                _stable_bytes = _stable_size;
            }
        }
        else {
            _stable_size  = 0;
            _stable_bytes = 0;
        }
        _stable_off = 0;
        _stable_cap = _stable_size;
        // Read once, not per allocation. Changing it mid run would put blocks of one
        // size class in both regions.
        _large_min = LARGE_MIN;
        if (const char* e = std::getenv("SYCLARENA_LARGE_MIN_MB")) {
            const long long v = std::atoll(e);
            if (v >= 0) _large_min = (size_t)v * MB;
        }
        _gpool.off = _stable_size; // Dynamic region starts right after stable region
        _gpu_hi    = _gpool.size;  // large blocks grow down from the pool end
        _gpu_hi_free_by_addr.clear();
        _gpu_hi_alloc_list.clear();
        _gpu_hi_allocated = 0;
        _gpu_hi_serves = 0;
        _gpool.cap = _gpu_hi - _gpool.off;   // the gap between the two ends
        _gtype   = type;
        _gqueue  = queue;

        if (type == GPUMemoryType::Managed) {
            try {
                _gpool.mem = sycl::malloc_shared(_gpool.size, resolve_queue(queue));
            }
            catch (const sycl::exception& e) {
                Logger::error("SYCL error [%s:%d]: %s", __FILE__, __LINE__, e.what());
                _gpool = Pool{};
                return false;
            }
            SYCLARENA_CHECK(resolve_queue(queue).memset(_gpool.mem, 0, _gpool.size).wait_and_throw());
        }
        else {
            try {
                _gpool.mem = sycl::malloc_device(_gpool.size, resolve_queue(queue));
            }
            catch (const sycl::exception& e) {
                Logger::error("SYCL error [%s:%d]: %s", __FILE__, __LINE__, e.what());
                _gpool = Pool{};
                return false;
            }
            SYCLARENA_CHECK(resolve_queue(queue).memset(_gpool.mem, 0, _gpool.size));
        }

        Logger::info("GPU %s pool created: %zu MB (stable: %zu MB, dynamic: %zu MB)",
            _gtype == GPUMemoryType::Device ? "device" : "managed",
            _gpool.size / MB, _stable_size / MB, (_gpool.size - _stable_size) / MB);
        return true;
    }

    bool DeviceArena::destroy_gpu_pool() {
        if (!_gpool.mem) return true;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _gpu_free_by_addr.clear();
            for (auto& bin : _gpu_bins) bin.clear();
            _gpu_alloc_list.clear();
            _gpu_allocated = 0;
            _gpu_hi_free_by_addr.clear();
            _gpu_hi_alloc_list.clear();
            _gpu_hi_allocated = 0;
            _gpu_hi_serves = 0;
            _gpu_stable_free_by_addr.clear();
            _gpu_stable_alloc_list.clear();
            _gpu_stable_allocated = 0;
            _stable_off   = 0;
            _stable_cap   = 0;
            _stable_size  = 0;
            _stable_bytes = 0;
        }
        SYCLARENA_CHECK(sycl::free(_gpool.mem, resolve_queue(_gqueue)));
        SYCLARENA_SYNC(_gqueue);
        _gpool = Pool{};
        Logger::info("GPU %s pool destroyed",
            _gtype == GPUMemoryType::Device ? "device" : "managed");
        return true;
    }

    bool DeviceArena::resize_gpu_pool(const size_t& new_size, sycl::queue* queue) {
        if (!_gpool.mem) {
            Logger::warn("GPU pool: resize called but pool not yet created");
            return false;
        }
        gpu_mem_info(resolve_queue(queue).get_device(), _gfree_mem, _gtot);
        if (new_size > _gfree_mem) {
            Logger::warn("Resizing GPU pool: %zu MB requested but only %zu MB free",
                         new_size / MB, _gfree_mem / MB);
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _gpu_free_by_addr.clear();
            for (auto& bin : _gpu_bins) bin.clear();
            _gpu_alloc_list.clear();
            _gpu_allocated = 0;
            _gpu_hi_free_by_addr.clear();
            _gpu_hi_alloc_list.clear();
            _gpu_hi_allocated = 0;
            _gpu_hi_serves = 0;
            _gpu_stable_free_by_addr.clear();
            _gpu_stable_alloc_list.clear();
            _gpu_stable_allocated = 0;
            SYCLARENA_CHECK(sycl::free(_gpool.mem, resolve_queue(_gqueue)));
            SYCLARENA_SYNC(_gqueue);
            _gpool = Pool{};
        }
        const size_t aligned = align_up(new_size);
        _gpool.size  = aligned;
        // Restore exact stable size; warn and drop it if the new pool is too small
        if (_stable_bytes > 0 && _stable_bytes >= aligned) {
            Logger::warn("GPU pool resize: stable region (%zu MB) >= new pool size (%zu MB) - dropping stable region",
                         _stable_bytes / MB, aligned / MB);
            _stable_size  = 0;
            _stable_bytes = 0;
        }
        else {
            _stable_size = _stable_bytes;
        }
        _stable_off  = 0;
        _stable_cap  = _stable_size;
        _gpool.off   = _stable_size;
        _gpu_hi      = aligned;
        _gpool.cap   = _gpu_hi - _gpool.off;
        _glimit      = aligned;
        _gqueue      = queue;

        if (_gtype == GPUMemoryType::Managed) {
            try {
                _gpool.mem = sycl::malloc_shared(_gpool.size, resolve_queue(queue));
            }
            catch (const sycl::exception& e) {
                Logger::error("SYCL error [%s:%d]: %s", __FILE__, __LINE__, e.what());
                _gpool = Pool{};
                return false;
            }
            SYCLARENA_CHECK(resolve_queue(queue).memset(_gpool.mem, 0, _gpool.size).wait_and_throw());
        }
        else {
            try {
                _gpool.mem = sycl::malloc_device(_gpool.size, resolve_queue(queue));
            }
            catch (const sycl::exception& e) {
                Logger::error("SYCL error [%s:%d]: %s", __FILE__, __LINE__, e.what());
                _gpool = Pool{};
                return false;
            }
            SYCLARENA_CHECK(resolve_queue(queue).memset(_gpool.mem, 0, _gpool.size));
        }
        Logger::info("GPU %s pool resized to %zu MB",
            _gtype == GPUMemoryType::Device ? "device" : "managed",
            _gpool.size / MB);
        return true;
    }

    void DeviceArena::reset_gpu_pool() {
        std::lock_guard<std::mutex> lock(_mutex);
        _gpu_free_by_addr.clear();
        for (auto& bin : _gpu_bins) bin.clear();
        _gpu_alloc_list.clear();
        _gpu_allocated = 0;
        _gpu_hi_free_by_addr.clear();
        _gpu_hi_alloc_list.clear();
        _gpu_hi_allocated = 0;
        _gpu_hi_serves = 0;
        _gpu_stable_free_by_addr.clear();
        _gpu_stable_alloc_list.clear();
        _gpu_stable_allocated = 0;
        _stable_off  = 0;
        _stable_cap  = _stable_size;
        _gpool.off   = _stable_size; // dynamic region starts right after stable region
        _gpu_hi      = _gpool.size;
        _gpool.cap   = _gpu_hi - _gpool.off;
    }

    bool DeviceArena::create_cpu_pool(const size_t& limit, const CPUMemoryType& type) {
        if (_cpool.mem) {
            Logger::info("CPU pool already exists (%zu MB)", _cpool.size / MB);
            return true;
        }
        sys_mem_info(_cfree_mem, _ctot);
        if (_cfree_mem > CPU_PENALTY) _cfree_mem -= CPU_PENALTY;
        _climit = limit ? align_up(limit) : 0;
        if (_climit) {
            if (_cfree_mem && _climit > _cfree_mem) {
                Logger::warn("CPU pool: requested %zu MB, only %zu MB free — capping",
                             _climit / MB, _cfree_mem / MB);
                _cpool.size = _cfree_mem;
            }
            else {
                _cpool.size = _climit;
            }
        }
        else {
            if (!_cfree_mem) _cfree_mem = 4ULL * GB;
            _cpool.size = _cfree_mem;
        }
        _cpool.cap = _cpool.size;
        _cpool.off = 0;
        _ctype = type;
        if (type == CPUMemoryType::Pageable) {
            _cpool.mem = std::malloc(_cpool.size);
            if (!_cpool.mem) { _cpool = Pool{}; return false; }
            std::memset(_cpool.mem, 0, _cpool.size);
        }
        else {
            try {
                _cpool.mem = sycl::malloc_host(_cpool.size, resolve_queue(nullptr));
            }
            catch (const sycl::exception& e) {
                Logger::error("SYCL error [%s:%d]: %s", __FILE__, __LINE__, e.what());
                _cpool = Pool{};
                return false;
            }
        }
        Logger::info("CPU %s pool created: %zu MB",
            _ctype == CPUMemoryType::Pinned ? "pinned" : "pageable",
            _cpool.size / MB);
        return true;
    }

    bool DeviceArena::destroy_cpu_pool() {
        if (!_cpool.mem) return true;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _cpu_free_by_addr.clear();
            _cpu_free_by_size.clear();
            _cpu_alloc_list.clear();
            _cpu_allocated = 0;
        }
        if (_ctype == CPUMemoryType::Pageable)
            std::free(_cpool.mem);
        else
            SYCLARENA_CHECK(sycl::free(_cpool.mem, resolve_queue(nullptr)));
        _cpool = Pool{};
        Logger::info("CPU %s pool destroyed",
            _ctype == CPUMemoryType::Pinned ? "pinned" : "pageable");
        return true;
    }

    bool DeviceArena::resize_cpu_pool(const size_t& new_size) {
        if (!_cpool.mem) {
            Logger::warn("Resizing CPU pool: pool not yet created");
            return false;
        }
        sys_mem_info(_cfree_mem, _ctot);
        if (new_size > _cfree_mem) {
            Logger::warn("Resizing CPU pool: %zu MB requested but only %zu MB free",
                         new_size / MB, _cfree_mem / MB);
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _cpu_free_by_addr.clear();
            _cpu_free_by_size.clear();
            _cpu_alloc_list.clear();
            _cpu_allocated = 0;
            if (_ctype == CPUMemoryType::Pageable)
                std::free(_cpool.mem);
            else
                SYCLARENA_CHECK(sycl::free(_cpool.mem, resolve_queue(nullptr)));
            _cpool = Pool{};
        }
        const size_t aligned = align_up(new_size);
        _cpool.size = _cpool.cap = aligned;
        _climit = aligned;
        if (_ctype == CPUMemoryType::Pageable) {
            _cpool.mem = std::malloc(_cpool.size);
            if (!_cpool.mem) { _cpool = Pool{}; return false; }
            std::memset(_cpool.mem, 0, _cpool.size);
        }
        else {
            try {
                _cpool.mem = sycl::malloc_host(_cpool.size, resolve_queue(nullptr));
            }
            catch (const sycl::exception& e) {
                Logger::error("SYCL error [%s:%d]: %s", __FILE__, __LINE__, e.what());
                _cpool = Pool{};
                return false;
            }
        }
        Logger::info("CPU %s pool resized to %zu MB",
            _ctype == CPUMemoryType::Pinned ? "pinned" : "pageable",
            _cpool.size / MB);
        return true;
    }

    void DeviceArena::reset_cpu_pool() {
        std::lock_guard<std::mutex> lock(_mutex);
        _cpu_free_by_addr.clear();
        _cpu_free_by_size.clear();
        _cpu_alloc_list.clear();
        _cpu_allocated = 0;
        _cpool.off = 0;
        _cpool.cap = _cpool.size;
    }

    void DeviceArena::_insert_gpu_free(const addr_t& ptr, const size_t& size) {
        auto [it, ok] = _gpu_free_by_addr.emplace(ptr, size);
        assert(ok);

        // Coalesce with next block
        auto next = std::next(it);
        if (next != _gpu_free_by_addr.end()) {
            if (static_cast<byte_t*>(it->first) + it->second == static_cast<byte_t*>(next->first)) {
                _gpu_bins[_bin_index(next->second)].erase(next->first);
                it->second += next->second;
                _gpu_free_by_addr.erase(next);
            }
        }

        // Coalesce with previous block
        if (it != _gpu_free_by_addr.begin()) {
            auto prev = std::prev(it);
            if (static_cast<byte_t*>(prev->first) + prev->second == static_cast<byte_t*>(it->first)) {
                _gpu_bins[_bin_index(prev->second)].erase(prev->first);
                prev->second += it->second;
                _gpu_free_by_addr.erase(it);
                it = prev;
            }
        }

        // Insert merged block into the appropriate bin
        _gpu_bins[_bin_index(it->second)].insert(it->first);
    }

    addr_t DeviceArena::_pop_gpu_free(const size_t& size, size_t& out_size) {
        int k = _bin_index(size);

        // search bin k for a block large enough to satisfy the request
        {
            auto& bin = _gpu_bins[k];
            for (auto addr_it = bin.begin(); addr_it != bin.end(); ++addr_it) {
                auto it = _gpu_free_by_addr.find(*addr_it);
                assert(it != _gpu_free_by_addr.end());
                if (it->second >= size) {
                    addr_t block = it->first;
                    const size_t block_size = it->second;
                    bin.erase(addr_it);
                    _gpu_free_by_addr.erase(it);
                    // Split remainder
                    if (block_size >= size + MIN_SPLIT) {
                        addr_t rem = static_cast<byte_t*>(block) + size;
                        const size_t rsz = block_size - size;
                        _gpu_free_by_addr.emplace(rem, rsz);
                        _gpu_bins[_bin_index(rsz)].insert(rem);
                        out_size = size;
                    }
                    else {
                        out_size = block_size;
                    }
                    return block;
                }
            }
        }

        // look in higher bins for a block to split
        for (int j = k + 1; j < NUM_BINS; ++j) {
            auto& bin = _gpu_bins[j];
            if (bin.empty()) continue;
            auto addr_it = bin.begin();
            auto it = _gpu_free_by_addr.find(*addr_it);
            assert(it != _gpu_free_by_addr.end());
            addr_t block = it->first;
            const size_t block_size = it->second;
            bin.erase(addr_it);
            _gpu_free_by_addr.erase(it);
            if (block_size >= size + MIN_SPLIT) {
                addr_t rem = static_cast<byte_t*>(block) + size;
                const size_t rsz = block_size - size;
                _gpu_free_by_addr.emplace(rem, rsz);
                _gpu_bins[_bin_index(rsz)].insert(rem);
                out_size = size;
            }
            else {
                out_size = block_size;
            }
            return block;
        }

        out_size = 0;
        return nullptr;
    }

    addr_t DeviceArena::_gpu_alloc(const size_t& aligned_bytes) {
        assert(is_aligned(aligned_bytes));
        if (_large_min && aligned_bytes >= _large_min) return _gpu_hi_alloc(aligned_bytes);
        size_t out_size = 0;
        addr_t ptr = _pop_gpu_free(aligned_bytes, out_size);
        if (ptr) {
            _gpu_alloc_list.emplace(ptr, out_size);
            _gpu_allocated += out_size;
            _gpu_peak = std::max(_gpu_peak, _gpu_allocated + _gpu_hi_allocated + _gpu_stable_allocated);
            return ptr;
        }
        if (aligned_bytes > _gpool.cap) {
            // Report the request size. A caller that catches this needs to tell a full
            // pool from one merely too fragmented to serve this block.
            size_t largest_free = 0;
            for (const auto& kv : _gpu_free_by_addr) largest_free = std::max(largest_free, kv.second);
            const std::string msg = "GPU pool out of memory (requested " +
                std::to_string(aligned_bytes / MB) + " MB, bump tail " +
                std::to_string(_gpool.cap / MB) + " MB, largest free block " +
                std::to_string(largest_free / MB) + " MB)";
            Logger::debug("syclArena: %s", msg.c_str());
            throw gpu_memory_error(msg.c_str());
        }
        ptr = static_cast<byte_t*>(_gpool.mem) + _gpool.off;
        _gpool.off += aligned_bytes;
        _sync_gap();
        _gpu_alloc_list.emplace(ptr, aligned_bytes);
        _gpu_allocated += aligned_bytes;
        _gpu_peak = std::max(_gpu_peak, _gpu_allocated + _gpu_hi_allocated + _gpu_stable_allocated);
        return ptr;
    }

    // Large blocks. First fit over the high free list, else carve down from its bottom.
    addr_t DeviceArena::_gpu_hi_alloc(const size_t& aligned_bytes) {
        for (auto it = _gpu_hi_free_by_addr.begin(); it != _gpu_hi_free_by_addr.end(); ++it) {
            if (it->second < aligned_bytes) continue;
            addr_t block = it->first;
            const size_t block_size = it->second;
            _gpu_hi_free_by_addr.erase(it);          // it is dead here; the loop returns below
            size_t took = block_size;
            // Split from the top of the hole so the remainder stays at its low end.
            if (block_size >= aligned_bytes + MIN_SPLIT) {
                _gpu_hi_free_by_addr.emplace(block, block_size - aligned_bytes);
                block = static_cast<byte_t*>(block) + (block_size - aligned_bytes);
                took = aligned_bytes;
            }
            _gpu_hi_alloc_list.emplace(block, took);
            _gpu_hi_allocated += took;
            ++_gpu_hi_serves;
            _gpu_peak = std::max(_gpu_peak, _gpu_allocated + _gpu_hi_allocated + _gpu_stable_allocated);
            return block;
        }

        if (_gpu_hi < _gpool.off + aligned_bytes) {
            Logger::debug("syclArena: GPU pool out of memory (high region: gap %zu MB, requested %zu MB)",
                          (_gpu_hi - _gpool.off) / MB, aligned_bytes / MB);
            const std::string msg = "GPU pool out of memory (requested " +
                std::to_string(aligned_bytes / MB) + " MB large-region, shared gap " +
                std::to_string((_gpu_hi - _gpool.off) / MB) + " MB)";
            throw gpu_memory_error(msg.c_str());
        }
        _gpu_hi -= aligned_bytes;
        _sync_gap();
        addr_t ptr = static_cast<byte_t*>(_gpool.mem) + _gpu_hi;
        _gpu_hi_alloc_list.emplace(ptr, aligned_bytes);
        _gpu_hi_allocated += aligned_bytes;
        ++_gpu_hi_serves;
        _gpu_peak = std::max(_gpu_peak, _gpu_allocated + _gpu_hi_allocated + _gpu_stable_allocated);
        return ptr;
    }

    // Coalesce into the high free list, then rewind the region bottom past a free run
    // sitting at it so the gap grows back.
    void DeviceArena::_insert_gpu_hi_free(const addr_t& ptr, const size_t& size) {
        auto [it, ok] = _gpu_hi_free_by_addr.emplace(ptr, size);
        if (!ok) {
            Logger::debug("syclArena: double free in high region %p", ptr);
            throw gpu_memory_error("double free in the large-block region");
        }
        auto next = std::next(it);
        if (next != _gpu_hi_free_by_addr.end() &&
            static_cast<byte_t*>(it->first) + it->second == static_cast<byte_t*>(next->first)) {
            it->second += next->second;
            _gpu_hi_free_by_addr.erase(next);
        }
        if (it != _gpu_hi_free_by_addr.begin()) {
            auto prev = std::prev(it);
            if (static_cast<byte_t*>(prev->first) + prev->second == static_cast<byte_t*>(it->first)) {
                prev->second += it->second;
                _gpu_hi_free_by_addr.erase(it);
                it = prev;
            }
        }
        byte_t* base = static_cast<byte_t*>(_gpool.mem);
        auto bottom = _gpu_hi_free_by_addr.begin();
        if (bottom != _gpu_hi_free_by_addr.end() &&
            (size_t)(static_cast<byte_t*>(bottom->first) - base) == _gpu_hi) {
            _gpu_hi += bottom->second;
            _gpu_hi_free_by_addr.erase(bottom);
            _sync_gap();
        }
    }

    void DeviceArena::_gpu_hi_free_block(const addr_t& ptr) {
        auto it = _gpu_hi_alloc_list.find(ptr);
        if (it == _gpu_hi_alloc_list.end()) {
            Logger::debug("deallocate: pointer %p not in the high alloc list", ptr);
            throw gpu_memory_error("invalid pointer in freeing GPU block");
        }
        const size_t size = it->second;
        _gpu_hi_alloc_list.erase(it);
        _gpu_hi_allocated -= size;
        _insert_gpu_hi_free(ptr, size);
    }

    void DeviceArena::_gpu_free_block(const addr_t& ptr) {
        auto it = _gpu_alloc_list.find(ptr);
        if (it == _gpu_alloc_list.end()) {
            Logger::debug("deallocate: pointer %p not in GPU alloc list", ptr);
            throw gpu_memory_error("invalid pointer in freeing GPU block");
        }
        const size_t size = it->second;
        _gpu_alloc_list.erase(it);
        _gpu_allocated -= size;
        _insert_gpu_free(ptr, size);
    }

    addr_t DeviceArena::_stable_alloc(const size_t& aligned_bytes) {
        assert(is_aligned(aligned_bytes));

        // First-fit from stable free list
        for (auto it = _gpu_stable_free_by_addr.begin();
             it != _gpu_stable_free_by_addr.end(); ++it) {
            if (it->second >= aligned_bytes) {
                addr_t block = it->first;
                size_t block_size   = it->second;
                _gpu_stable_free_by_addr.erase(it);
                // Split remainder back into free list
                if (block_size >= aligned_bytes + MIN_SPLIT) {
                    addr_t rem = static_cast<byte_t*>(block) + aligned_bytes;
                    _gpu_stable_free_by_addr.emplace(rem, block_size - aligned_bytes);
                    block_size = aligned_bytes;
                }
                _gpu_stable_alloc_list.emplace(block, block_size);
                _gpu_stable_allocated += block_size;
                _gpu_peak = std::max(_gpu_peak, _gpu_allocated + _gpu_hi_allocated + _gpu_stable_allocated);
                return block;
            }
        }

        // Stable region allocator
        if (aligned_bytes > _stable_cap) {
            Logger::debug("syclArena: stable GPU region out of memory (%zu MB remaining, requested %zu MB)",
                         _stable_cap / MB, aligned_bytes / MB);
            throw gpu_memory_error("syclArena: stable GPU region out of memory");
        }
        addr_t ptr = static_cast<byte_t*>(_gpool.mem) + _stable_off;
        _stable_off += aligned_bytes;
        _stable_cap -= aligned_bytes;
        _gpu_stable_alloc_list.emplace(ptr, aligned_bytes);
        _gpu_stable_allocated += aligned_bytes;
        _gpu_peak = std::max(_gpu_peak, _gpu_allocated + _gpu_hi_allocated + _gpu_stable_allocated);
        return ptr;
    }

    void DeviceArena::_stable_free_block(const addr_t& ptr) {
        auto it = _gpu_stable_alloc_list.find(ptr);
        if (it == _gpu_stable_alloc_list.end()) {
            Logger::debug("deallocate stable: pointer %p not in stable alloc list", ptr);
            throw gpu_memory_error("invalid pointer in freeing stable GPU block");
        }
        const size_t size = it->second;
        _gpu_stable_alloc_list.erase(it);
        _gpu_stable_allocated -= size;

        // Insert with coalescing
        auto [fit, ok] = _gpu_stable_free_by_addr.emplace(ptr, size);
        assert(ok);

        auto next = std::next(fit);
        if (next != _gpu_stable_free_by_addr.end()) {
            if (static_cast<byte_t*>(fit->first) + fit->second == static_cast<byte_t*>(next->first)) {
                fit->second += next->second;
                _gpu_stable_free_by_addr.erase(next);
            }
        }
        if (fit != _gpu_stable_free_by_addr.begin()) {
            auto prev = std::prev(fit);
            if (static_cast<byte_t*>(prev->first) + prev->second == static_cast<byte_t*>(fit->first)) {
                prev->second += fit->second;
                _gpu_stable_free_by_addr.erase(fit);
                fit = prev;
            }
        }

        if (static_cast<byte_t*>(fit->first) + fit->second ==
            static_cast<byte_t*>(_gpool.mem) + _stable_off) {
            _stable_off -= fit->second;
            _stable_cap += fit->second;
            _gpu_stable_free_by_addr.erase(fit);
        }
    }

    // Move order for the low region.
    static bool by_addr_asc(const std::pair<addr_t, size_t>& a, const std::pair<addr_t, size_t>& b) {
        return a.first < b.first;
    }

    // Move order for the high region.
    static bool by_addr_desc(const std::pair<addr_t, size_t>& a, const std::pair<addr_t, size_t>& b) {
        return a.first > b.first;
    }

    // Pack the small blocks towards the front of the dynamic region.
    void DeviceArena::_plan_compact_low(std::vector<std::pair<addr_t, size_t>>& blocks,
                                        std::vector<addr_t>& new_ptrs,
                                        byte_t*& bounce, size_t& bounce_size) {
        if (_gpu_alloc_list.empty()) {
            // No small allocs: discard the low region's holes outright.
            _gpu_free_by_addr.clear();
            for (auto& bin : _gpu_bins) bin.clear();
            _gpool.off = _stable_size;
            _sync_gap();
            return;
        }

        // The gap above the base pointer is free and belongs to neither region, so it can
        // hold the bounce for an overlapping copy.
        const size_t pre_off = _gpool.off;
        bounce = static_cast<byte_t*>(_gpool.mem) + pre_off;
        bounce_size = (_gpu_hi > pre_off) ? (_gpu_hi - pre_off) : 0;

        blocks.assign(_gpu_alloc_list.begin(), _gpu_alloc_list.end());
        std::sort(blocks.begin(), blocks.end(), by_addr_asc);

        byte_t* dyn_start = static_cast<byte_t*>(_gpool.mem) + _stable_size;
        size_t  new_off   = 0;
        new_ptrs.reserve(blocks.size());
        for (auto& [old_ptr, sz] : blocks) {
            new_ptrs.push_back(dyn_start + new_off);
            new_off += sz;
        }

        _gpu_free_by_addr.clear();
        for (auto& bin : _gpu_bins) bin.clear();
        _gpu_alloc_list.clear();
        _gpu_allocated = 0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            _gpu_alloc_list.emplace(new_ptrs[i], blocks[i].second);
            _gpu_allocated += blocks[i].second;
        }
        _gpool.off = _stable_size + new_off;
        _sync_gap();
    }

    // Pack the large blocks towards the pool end, so their holes end up at the region's
    // bottom and merge into the gap. Packing the other way would eat the gap instead.
    void DeviceArena::_plan_compact_high(std::vector<std::pair<addr_t, size_t>>& blocks,
                                         std::vector<addr_t>& new_ptrs,
                                         byte_t*& bounce, size_t& bounce_size) {
        if (_gpu_hi_alloc_list.empty()) {
            _gpu_hi_free_by_addr.clear();
            _gpu_hi = _gpool.size;
            _sync_gap();
            return;
        }

        // Packing upwards moves every block up, so nothing lands below the region's old
        // bottom and the gap under it is free to bounce an overlapping copy through.
        const size_t pre_hi = _gpu_hi;
        bounce = static_cast<byte_t*>(_gpool.mem) + _gpool.off;
        bounce_size = (pre_hi > _gpool.off) ? (pre_hi - _gpool.off) : 0;

        blocks.assign(_gpu_hi_alloc_list.begin(), _gpu_hi_alloc_list.end());
        std::sort(blocks.begin(), blocks.end(), by_addr_desc);

        size_t packed_bottom = _gpool.size;
        new_ptrs.reserve(blocks.size());
        for (auto& [old_ptr, sz] : blocks) {
            packed_bottom -= sz;
            new_ptrs.push_back(static_cast<byte_t*>(_gpool.mem) + packed_bottom);
        }

        _gpu_hi_free_by_addr.clear();
        _gpu_hi_alloc_list.clear();
        _gpu_hi_allocated = 0;
        for (size_t i = 0; i < blocks.size(); ++i) {
            _gpu_hi_alloc_list.emplace(new_ptrs[i], blocks[i].second);
            _gpu_hi_allocated += blocks[i].second;
        }
        _gpu_hi = packed_bottom;
        _sync_gap();
    }

    // queue::memcpy is undefined where source and destination overlap, which happens
    // when a block moves by less than its own size. Route those through the bounce
    // buffer, or copy in disjoint chunks of delta. Ascending moving down, descending
    // moving up, so each chunk is read before anything overwrites it.
    void DeviceArena::_move_block(byte_t* dst, byte_t* src, size_t sz, byte_t* bounce,
                                  size_t bounce_size, sycl::queue* queue) {
        if (dst == src) return;

        sycl::queue& q = resolve_queue(queue);
        const size_t delta = (dst < src) ? (size_t)(src - dst) : (size_t)(dst - src);
        if (delta >= sz) {
            SYCLARENA_CHECK(q.memcpy(dst, src, sz));
            return;
        }
        if (bounce_size >= sz) {
            SYCLARENA_CHECK(q.memcpy(bounce, src, sz));
            SYCLARENA_CHECK(q.memcpy(dst, bounce, sz));
            return;
        }
        if (dst < src) {
            for (size_t off = 0; off < sz; off += delta) {
                const size_t len = (sz - off < delta) ? (sz - off) : delta;
                SYCLARENA_CHECK(q.memcpy(dst + off, src + off, len));
            }
        }
        else {
            for (size_t off = sz; off > 0; ) {
                const size_t len = (off < delta) ? off : delta;
                off -= len;
                SYCLARENA_CHECK(q.memcpy(dst + off, src + off, len));
            }
        }
    }

    size_t DeviceArena::compact_gpu_dynamic(RelocateCallback callback, sycl::queue* queue,
                                            CompactRegion which) {
        std::vector<std::pair<addr_t, size_t>> active_blocks;  // in move order
        std::vector<addr_t>                    new_ptrs;
        byte_t*                                bounce = nullptr;
        size_t                                 bounce_size = 0;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (which == CompactRegion::High) _plan_compact_high(active_blocks, new_ptrs, bounce, bounce_size);
            else                              _plan_compact_low (active_blocks, new_ptrs, bounce, bounce_size);
        }

        for (size_t i = 0; i < active_blocks.size(); ++i) {
            _move_block(static_cast<byte_t*>(new_ptrs[i]), static_cast<byte_t*>(active_blocks[i].first),
                        active_blocks[i].second, bounce, bounce_size, queue);
        }

        size_t moved = 0;
        for (size_t i = 0; i < active_blocks.size(); ++i) {
            if (new_ptrs[i] == active_blocks[i].first) continue;
            ++moved;
            if (callback) callback(active_blocks[i].first, new_ptrs[i], active_blocks[i].second);
        }
        return moved;
    }

    void DeviceArena::_insert_cpu_free(const addr_t& ptr, const size_t& size) {
        auto [it, ok] = _cpu_free_by_addr.emplace(ptr, size);
        assert(ok);

        auto next = std::next(it);
        if (next != _cpu_free_by_addr.end()) {
            if (static_cast<byte_t*>(it->first) + it->second == static_cast<byte_t*>(next->first)) {
                auto& s = _cpu_free_by_size[next->second];
                s.erase(next->first);
                if (s.empty()) _cpu_free_by_size.erase(next->second);
                it->second += next->second;
                _cpu_free_by_addr.erase(next);
            }
        }

        if (it != _cpu_free_by_addr.begin()) {
            auto prev = std::prev(it);
            if (static_cast<byte_t*>(prev->first) + prev->second == static_cast<byte_t*>(it->first)) {
                auto& s = _cpu_free_by_size[prev->second];
                s.erase(prev->first);
                if (s.empty()) _cpu_free_by_size.erase(prev->second);
                prev->second += it->second;
                _cpu_free_by_addr.erase(it);
                it = prev;
            }
        }

        _cpu_free_by_size[it->second].insert(it->first);
    }

    addr_t DeviceArena::_pop_cpu_free(const size_t& size, size_t& out_size) {
        auto it = _cpu_free_by_size.lower_bound(size);
        if (it == _cpu_free_by_size.end()) { out_size = 0; return nullptr; }

        addr_t block = *it->second.begin();
        const size_t block_size = it->first;

        it->second.erase(it->second.begin());
        if (it->second.empty()) _cpu_free_by_size.erase(it);
        _cpu_free_by_addr.erase(block);

        if (block_size >= size + MIN_SPLIT) {
            addr_t rem = static_cast<byte_t*>(block) + size;
            const size_t rem_size = block_size - size;
            _cpu_free_by_addr.emplace(rem, rem_size);
            _cpu_free_by_size[rem_size].insert(rem);
            out_size = size;
        }
        else {
            out_size = block_size;
        }
        return block;
    }

    addr_t DeviceArena::_cpu_alloc(const size_t& aligned_bytes) {
        assert(is_aligned(aligned_bytes));
        size_t out_size = 0;
        addr_t ptr = _pop_cpu_free(aligned_bytes, out_size);
        if (ptr) {
            _cpu_alloc_list.emplace(ptr, out_size);
            _cpu_allocated += out_size;
            return ptr;
        }
        if (aligned_bytes > _cpool.cap) {
            throw cpu_memory_error("CPU pool out of memory");
        }
        ptr = static_cast<byte_t*>(_cpool.mem) + _cpool.off;
        _cpool.off += aligned_bytes;
        _cpool.cap -= aligned_bytes;
        _cpu_alloc_list.emplace(ptr, aligned_bytes);
        _cpu_allocated += aligned_bytes;
        return ptr;
    }

    void DeviceArena::_cpu_free_block(const addr_t& ptr) {
        auto it = _cpu_alloc_list.find(ptr);
        if (it == _cpu_alloc_list.end()) {
            Logger::debug("deallocate pinned: pointer %p not in CPU alloc list", ptr);
            throw cpu_memory_error("invalid pointer in freeing CPU block");
        }
        const size_t size = it->second;
        _cpu_alloc_list.erase(it);
        _cpu_allocated -= size;
        _insert_cpu_free(ptr, size);
    }

}
