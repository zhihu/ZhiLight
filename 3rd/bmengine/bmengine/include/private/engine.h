#pragma once
//#include "bmengine/core/engine.h"
#include "bmengine/core/thread_pool.h"
#include "bmengine/c10d/host_communicator.h"
#include <mutex>
#include <stack>
#include <nccl.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include "private/allocator.h"
#include "private/stream.h"


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

class DeviceHandles {
public:
    int dev_id;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    ncclComm_t comm;

    // the following four parameters determine the weights range of the model
    int tp_rank;
    int tp_ranks;
    int pp_rank;
    int pp_ranks;

    int compute_capability;
    int mp_count;
    int l2_cache_size;
    int max_shared_memory;

    DeviceHandles(
        int dev_id,
        ncclUniqueId uniqueID,
        int tp_rank = 0,
        int tp_ranks = 1,
        int pp_rank = 0,
        int pp_ranks = 1);
    ~DeviceHandles();
    DeviceHandles(const DeviceHandles&) = delete;
    DeviceHandles& operator=(const DeviceHandles&) = delete;
    DeviceHandles(DeviceHandles&&) = default;
    DeviceHandles& operator=(DeviceHandles&&) = delete;
};

class Tensor;
class Context;
class EngineImpl {
    friend class Engine;
    std::vector<DeviceHandles*> handles;
    std::vector<MemoryAllocator*> allocators;
    std::vector<StreamAllocator*> streams;
    std::vector<std::mutex*> device_lock;
    std::vector<TaskThreadPool*> device_threads;
    // for host
    c10d::HostCommunicator* hc;
    // for nccl
    std::vector<ncclUniqueId> uniqueIDs;
    int world_size_;
    int local_ranks_;

    int debug;
    bool is_mem_frozen { false };

public:
    EngineImpl(const std::vector<DeviceConfiguration>& cfg, const DistConfiguration& dist_cfg);
    ~EngineImpl();
    EngineImpl(const EngineImpl&) = delete;
    EngineImpl& operator=(const EngineImpl&) = delete;
    EngineImpl(EngineImpl&&) = delete;
    EngineImpl& operator=(EngineImpl&&) = delete;

    Context create_context(const std::vector<int>& devices) const;

    /* Thread-safe API */
    DeviceHandles* get_device_handle(int dev_id);
    void alloc_device(int dev_id);
    void release_device(int dev_id);
    cudaStream_t create_stream(int dev_id);
    void destroy_stream(int dev_id, cudaStream_t stream);

    MemoryAllocator* get_allocator(int dev_id) {
        return allocators[dev_id];
    }
    Memory alloc_memory(int dev_id, size_t size, size_t round_up_bytes = 512);
    Tensor alloc_tensor(int dev_id, const std::vector<size_t>& shape, DataType dtype);
    void get_parameter(const std::string& name, Tensor* tensor);
    void init_parameter(const std::string& name, Tensor* tensor);

    GPUInfo get_gpu_info(int dev_id);
    int num_gpus() const;
    int world_size() const { return world_size_; }
    int nnodes() const;
    int node_rank() const;

    void print_memory_summary();
    void freeze_model_memory();

    void device_foreach(std::function<void(int)>& fn);
    std::mutex log_mutex;
};

} // namespace core

} // namespace bmengine
