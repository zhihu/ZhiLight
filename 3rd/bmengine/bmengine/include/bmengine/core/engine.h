#pragma once
#include "bmengine/core/export.h"
#include "private/engine.h"
#include <memory>
#include <vector>
#include <functional>

namespace bmengine {

namespace core {

class Context;
class MemoryAllocator;

//class EngineImpl;
// Engine can be accessed from multiple threads.
class BMENGINE_EXPORT Engine {
    friend class DistributedTensorImpl;
    std::unique_ptr<EngineImpl> pimpl;

public:
    Engine(const std::vector<DeviceConfiguration>& dev_cfg, const DistConfiguration& dist_cfg);
    Engine(const std::vector<DeviceConfiguration>& dev_cfg): Engine(dev_cfg, {}) {}
    ~Engine();

    Context create_context(const std::vector<int>& devices) const;
    Context create_context() const; // use all devices
    Context create_context_rank(int rank) const;
    int num_gpus() const;
    int world_size() const;
    int local_ranks() const;
    int nnodes() const;
    int node_rank() const;
    template <typename T>
    void broadcast_data(T &data, int nbytes = 0) {
        pimpl->host_comm->broadcast_data(data, nbytes);
    }
    GPUInfo get_gpu_info(int device_idx) const;

    // Disable copy
    Engine(const Engine&) = delete;
    Engine(Engine&&) = delete;

    void device_foreach(std::function<void(int)> fn);
    void print_memory_summary();
    void freeze_model_memory();
    MemoryAllocator* get_allocator(int dev_id);

};

} // namespace core

} // namespace bmengine
