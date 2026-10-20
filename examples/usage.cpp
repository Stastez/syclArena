#include <syclarena/syclarena.hpp>
#include <format>
#include <iostream>
#include <unordered_map>
#include <vector>

/**
 * Persistent buffers: stable region, allocated once.
 * Scratch buffers: dynamic region, frequently created and destroyed.
 */

int main() {
    syclArena::Logger::set_level(1);

    try {
        sycl::queue stream;

        // Pool layout:
        //   total:   4 GB
        //   stable:  512 MB
        //   dynamic: 3584 MB
        syclArena::DeviceArena alloc;
        alloc.create_gpu_pool(4 * syclArena::GB, syclArena::GPUMemoryType::Device, &stream, 512 * syclArena::MB);
        alloc.create_cpu_pool(512 * syclArena::MB, syclArena::CPUMemoryType::Pinned);
        SYCLARENA_SYNC(&stream);

        std::cout << std::format("\n  Pool created with 512 MB stable and {} MB dynamic\n",
            (alloc.gpu_capacity() - 512 * syclArena::MB) / syclArena::MB);

        float* p1 = alloc.allocate<float>(64  * syclArena::MB / sizeof(float), syclArena::Region::Stable);
        float* p2 = alloc.allocate<float>(128 * syclArena::MB / sizeof(float), syclArena::Region::Stable);
        float* p3 = alloc.allocate<float>(256 * syclArena::MB / sizeof(float), syclArena::Region::Stable);
        SYCLARENA_CHECK(stream.memset(p1, 0, 64  * syclArena::MB));
        SYCLARENA_CHECK(stream.memset(p2, 0, 128 * syclArena::MB));
        SYCLARENA_CHECK(stream.memset(p3, 0, 256 * syclArena::MB));
        SYCLARENA_SYNC(&stream);

        std::cout << std::format("  Persistent buffers allocated, stable used: {} MB / 512 MB\n",
            alloc.gpu_stable_used() / syclArena::MB);

        float* h_stage = alloc.allocate_pinned<float>(32 * syclArena::MB / sizeof(float));
        std::cout << std::format("  Pinned mirror: {} MB\n",
            alloc.cpu_used() / syclArena::MB);

        // Fill most of the dynamic pool with equal-sized blocks,
        // then free every other one.
        constexpr size_t BLOCK = 128 * syclArena::MB / sizeof(float);
        constexpr int    N     = 24;

        std::vector<float*> dynamics;
        dynamics.reserve(N);

        std::cout << "\n  Filling dynamic region with " << N << " x 128 MB blocks...\n";
        for (int i = 0; i < N; i++)
            dynamics.push_back(alloc.allocate<float>(BLOCK));
        std::cout << "  Freeing every other block to create fragmentation...\n";
        for (int i = 0; i < N; i += 2) {
            alloc.deallocate(dynamics[i]);
            dynamics[i] = nullptr;
        }

        size_t avail   = alloc.gpu_available();
        size_t used    = alloc.gpu_used() - alloc.gpu_stable_used();
        size_t largest = alloc.gpu_largest_free_block();
        std::cout << std::format("  Before compact:  dynamic used {:4d} MB  available {:4d} MB  largest block {:4d} MB\n",
            used / syclArena::MB, avail / syclArena::MB, largest / syclArena::MB);

        if (largest < avail / 2) {
            std::cout << "  Fragmentation detected, compacting dynamic region...";

            std::unordered_map<syclArena::addr_t, syclArena::addr_t> map;
            alloc.compact_gpu_dynamic(
                [&](syclArena::addr_t old_p, syclArena::addr_t new_p, size_t) {
                    map[old_p] = new_p;
                },
            &stream);
            SYCLARENA_SYNC(&stream);

            for (auto& p : dynamics) {
                if (!p) continue;
                auto it = map.find(p);
                if (it != map.end()) p = static_cast<float*>(it->second);
            }
            std::cout << std::format(" done (largest block: {} MB -> {} MB)\n",
                largest / syclArena::MB, alloc.gpu_largest_free_block() / syclArena::MB);
        }

        for (auto p : dynamics) if (p) alloc.deallocate(p);
        dynamics.clear();

        std::cout << std::format("\n  After all iterations, dynamic used: {} MB\n",
            (alloc.gpu_used() - alloc.gpu_stable_used()) / syclArena::MB);

        alloc.deallocate(p1);
        alloc.deallocate(p2);
        alloc.deallocate(p3);
        alloc.deallocate_pinned(h_stage);

        std::cout << std::format("  Total GPU used after clean-up: {} MB\n",
            alloc.gpu_used() / syclArena::MB);

        alloc.destroy_gpu_pool();
        alloc.destroy_cpu_pool();

    }
    catch (const syclArena::gpu_memory_error& e) {
        std::cerr << std::format("GPU memory error: {}\n", e.what());
        return 1;
    }
    catch (const syclArena::cpu_memory_error& e) {
        std::cerr << std::format("CPU memory error: {}\n", e.what());
        return 1;
    }
    catch (const syclArena::gpu_error& e) {
        std::cerr << std::format("SYCL error: {}\n", e.what());
        return 1;
    }

    return 0;
}
