#pragma once
#include <string>

namespace bmengine {

namespace core {
    
    struct DeviceConfiguration {
        int device_id;
        size_t memory_limit;
    
        DeviceConfiguration(int device_id, size_t memory_limit)
            : device_id(device_id), memory_limit(memory_limit) { }
    };
    
    struct DistConfiguration {
        int tp { -1 };
        std::string dist_init_addr;
        int nnodes { 1 };
        int node_rank { 0 };
    };
    
    struct GPUInfo {
        int real_device_idx;
        int compute_capability;
        size_t total_memory;
        size_t free_memory;
        size_t alloc_memory;
    };

} // namespace core

} // namespace bmengine