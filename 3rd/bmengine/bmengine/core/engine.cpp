#include "bmengine/core/engine.h"
#include "bmengine/core/context.h"
#include "bmengine/core/exception.h"
#include "bmengine/logger/std_log_op.hpp"
#include "bmengine/c10d/host_communicator.h"
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <map>
#include <numeric>
#include <stack>
#include <thread>
#include <cublas_v2.h>
#include <curand.h>

#include "private/guard.h"
#include "private/engine.h"
#include "private/context.h"
#include "private/tensor_impl.h"

namespace bmengine {

namespace core {

static inline int get_int_env(const char* name, int def_val = 0) {
    char* env_str = std::getenv(name);
    return env_str != nullptr ? std::atoi(env_str) : def_val;
}

DeviceHandles::DeviceHandles(int dev_id, ncclUniqueId uniqueID, int tp_rank, int tp_ranks, int pp_rank, int pp_ranks)
    : dev_id(dev_id), tp_rank(tp_rank), tp_ranks(tp_ranks), pp_rank(pp_rank), pp_ranks(pp_ranks) {
    DeviceGuard guard(dev_id);
    BM_CUDART_ASSERT(cudaStreamCreate(&stream));
    BM_CUBLAS_ASSERT(cublasCreate(&cublas_handle));
    BM_CUBLAS_ASSERT(cublasSetStream(cublas_handle, stream));
//    ncclConfig_t nccl_config = NCCL_CONFIG_INITIALIZER;
//    nccl_config.blocking = 0;
//    BM_NCCL_ASSERT(ncclCommInitRankConfig(&comm, world_size, uniqueID, rank, &nccl_config));
    if (tp_ranks > 1) {
        BM_NCCL_ASSERT(ncclCommInitRank(&comm, tp_ranks, uniqueID, tp_rank));
        std::cout << "NCCL tp_rank=" << tp_rank << ", pp_rank=" << pp_rank << " connected done!" << std::endl;
    }
    cudaDeviceProp dev_prop{};
    BM_CUDART_ASSERT(cudaGetDeviceProperties(&dev_prop, dev_id));
    int cc_major = dev_prop.major;
    int cc_minor = dev_prop.minor;
    mp_count = dev_prop.multiProcessorCount;
    l2_cache_size = dev_prop.l2CacheSize;
    BM_CUDART_ASSERT(cudaDeviceGetAttribute(
        &max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
    compute_capability = cc_major * 10 + cc_minor;
    if (dev_id == 0)
        std::cout << "CC:" << compute_capability
            << ", mp_count:" << mp_count
            << ", L2 Cache:" << (l2_cache_size / 1024 / 1024) << "MB"
            << ", Max Persistent L2:" << (dev_prop.persistingL2CacheMaxSize / 1024) << "KB"
            << ", max_smem:" << (max_shared_memory / 1024) << "KB\n";
    int max_persist_l2 = get_int_env("MAX_PERSIST_L2", 0);
    if (max_persist_l2)  {
        BM_CUDART_ASSERT(
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, max_persist_l2 * 1024 * 1024));
    }
}
DeviceHandles::~DeviceHandles() {
    DeviceGuard guard(dev_id);
    try {
        BM_CUBLAS_ASSERT(cublasDestroy(cublas_handle));
        BM_CUDART_ASSERT(cudaStreamDestroy(stream));
        if (tp_ranks > 1) {
            BM_NCCL_ASSERT(ncclCommDestroy(comm));
        }
    } catch (const BMEngineException& e) { std::cerr << e.what() << std::endl; }
}

EngineImpl::EngineImpl(const std::vector<DeviceConfiguration>& dev_cfgs, const DistConfiguration& dist_cfg)
    : debug(0) {
    int tp_ranks = dist_cfg.tp;
    int local_dev_count = static_cast<int>(dev_cfgs.size());
    int total_dev_count = local_dev_count * dist_cfg.nnodes;
    if (tp_ranks <= 0) tp_ranks = total_dev_count;
    BM_ASSERT_EQ(total_dev_count % tp_ranks, 0, "dev_count can't mod tp");
    BM_ASSERT(
        std::thread::hardware_concurrency() > dev_cfgs.size(),
        "at least one cpu-core per device required.");
    int pp_ranks = total_dev_count / tp_ranks;
    world_size_ = tp_ranks;
    if (tp_ranks > 1 || pp_ranks > 1) {
        int nccl_version;
        ncclGetVersion(&nccl_version);
        std::cout << "********* world_size=" << world_size_ << ", nccl_version=" << nccl_version << " *********\n";
    }
    
    hc = new c10d::HostCommunicator(dist_cfg.dist_init_addr, dist_cfg.nnodes, dist_cfg.node_rank);

    char* debug_env = std::getenv("BM_DEBUG_LEVEL");
    if (debug_env != nullptr) {
        debug = std::atoi(debug_env);
    }
    char* direct_mem_alloc_str = std::getenv("BM_DIRECT_MEM_ALLOC");
    bool direct_mem_alloc =
        direct_mem_alloc_str != nullptr && strncmp(direct_mem_alloc_str, "1", 1) == 0;

    for (int i = 0; i < local_dev_count; i++) {
        device_threads.push_back(new TaskThreadPool(1, i));
    }

    uniqueIDs.resize(pp_ranks);
    char *data = reinterpret_cast<char *>(uniqueIDs.data());
    int nbytes = sizeof(ncclUniqueId) * pp_ranks;
    // broadcast
    if (dist_cfg.node_rank == 0) {
        for (size_t i = 0; i < uniqueIDs.size(); i++) {
            BM_NCCL_ASSERT(ncclGetUniqueId(&uniqueIDs[i]));
        }
    }
    hc->broadcast(&data, &nbytes);

    ncclGroupStart();
    local_ranks_ = 0;
    int rank_base = dist_cfg.node_rank * local_dev_count;
    for (int i = 0; i < local_dev_count; i++) {
        int tp_rank = (i + rank_base) % tp_ranks;
        int pp_rank = (i + rank_base) / tp_ranks;
        if (pp_rank == 0) {
            local_ranks_++;
        }
        handles.push_back(new DeviceHandles(dev_cfgs[i].device_id, uniqueIDs[pp_rank], tp_rank, tp_ranks, pp_rank, pp_ranks));
        MemoryAllocator* allocator = direct_mem_alloc ?
            new DirectMemoryAllocator(dev_cfgs[i].device_id, i, dev_cfgs[i].memory_limit, handles[i]->stream) :
            new MemoryAllocator(dev_cfgs[i].device_id, i, dev_cfgs[i].memory_limit, handles[i]->stream);
        allocators.push_back(allocator);
        streams.push_back(new StreamAllocator(dev_cfgs[i].device_id));
        device_lock.push_back(new std::mutex());
    }
    ncclGroupEnd();

    if (local_dev_count > 1) {
        int canAccessPeer;
        for (int from = 0; from < local_dev_count; from++) {
            for (int to = 0; to < local_dev_count; to++) {
                BM_CUDART_ASSERT(cudaSetDevice(dev_cfgs[from].device_id));
                if (from != to) {
                    BM_CUDART_ASSERT(cudaDeviceCanAccessPeer(
                        &canAccessPeer, dev_cfgs[from].device_id, dev_cfgs[to].device_id));
                    if (canAccessPeer == 1) {
                        auto result = cudaDeviceEnablePeerAccess(dev_cfgs[to].device_id, 0);
                        // with nccl peer access already enabled;
                        if (result == cudaErrorPeerAccessAlreadyEnabled) {
                            cudaGetLastError();
                        } else
                            BM_CUDART_ASSERT(result);
                    }
                }
            }
        }
        if (canAccessPeer == 1 && debug >= 1) {
            std::cerr << "Done EnablePeerAccess\n";
        }
    }
}
EngineImpl::~EngineImpl() {
    for (int i = 0; i < handles.size(); i++) {
        delete allocators[i];
        delete handles[i];
        delete streams[i];
        delete device_lock[i];
    }
    for (auto th : device_threads) {
        delete th;
    }
    delete hc;
}

DeviceHandles* EngineImpl::get_device_handle(int dev_id) {
    return handles[dev_id];
}
void EngineImpl::alloc_device(int dev_id) {
    device_lock[dev_id]->lock();
    BM_CUDART_ASSERT(cudaSetDevice(handles[dev_id]->dev_id));
}
void EngineImpl::release_device(int dev_id) {
    device_lock[dev_id]->unlock();
}
cudaStream_t EngineImpl::create_stream(int dev_id) {
    return streams[dev_id]->alloc();
}
void EngineImpl::destroy_stream(int dev_id, cudaStream_t stream) {
    streams[dev_id]->free(stream);
}
Memory EngineImpl::alloc_memory(int dev_id, size_t size, size_t round_up_bytes) {
    return allocators[dev_id]->alloc(size, round_up_bytes);
}
Tensor EngineImpl::alloc_tensor(int dev_id, const std::vector<size_t>& shape, DataType dtype) {
    check_no_zero(shape);
    size_t nbytes = get_numel(shape) * get_elem_size(dtype);
    auto mem = alloc_memory(dev_id, nbytes);
    return {std::make_unique<core::TensorImpl>(shape, mem, 0, dtype)};
}
void EngineImpl::init_parameter(const std::string& name, Tensor* tensor) {
    if (tensor->data() == nullptr) {
        *tensor = alloc_tensor(tensor->device(), tensor->size(), tensor->dtype());
    }
}

int EngineImpl::num_gpus() const {
    return handles.size();
}

GPUInfo EngineImpl::get_gpu_info(int dev_id) {
    BM_ASSERT(
        dev_id >= 0 && dev_id < handles.size(), "invalid device id: " + std::to_string(dev_id));
    GPUInfo ret;
    ret.real_device_idx = handles[dev_id]->dev_id;
    ret.compute_capability = handles[dev_id]->compute_capability;

    {
        DeviceGuard guard(ret.real_device_idx);
        BM_CUDART_ASSERT(cudaMemGetInfo(&ret.free_memory, &ret.total_memory));
    }
    ret.alloc_memory = allocators[dev_id]->used_memory();
    return ret;
}

void EngineImpl::print_memory_summary() {
    if (debug >= 1) {
        for (size_t i = 0; i < allocators.size(); ++i) {
            std::cerr << "Dev[" << i << "] mem_blocks=" << allocators[i]->get_block_num()
                      << ", used_memory=" << (allocators[i]->used_memory() / 1000000) << "MB"
                      << ", peak_memory=" << (allocators[i]->peak_memory() / 1000000) << "MB"
                      << std::endl;
        }
    }
}

void EngineImpl::device_foreach(std::function<void(int)>& fn) {
    int local_worker_num = static_cast<int>(device_threads.size());
    for (int i = 0; i < local_worker_num; ++i) {
        device_threads[i]->run(std::bind(fn, i));
    }
    std::exception_ptr e_ptr;
    for (int i = 0; i < local_worker_num; ++i) {
        try {
            device_threads[i]->wait();
        } catch (...) {
            e_ptr = std::current_exception();
        }
    }
    if (e_ptr) {
        std::rethrow_exception(e_ptr);
    }
}

void Engine::print_memory_summary() {
    pimpl->print_memory_summary();
}

void EngineImpl::freeze_model_memory() {
    {
        is_mem_frozen = true;
        for (size_t i = 0; i < allocators.size(); ++i) {
            allocators[i]->freeze_model_memory();
        }
    }
}

void Engine::freeze_model_memory() {
    pimpl->freeze_model_memory();
}

MemoryAllocator* Engine::get_allocator(int dev_id) {
    return pimpl->get_allocator(dev_id);
}

void Engine::device_foreach(std::function<void(int)> fn) {
    pimpl->device_foreach(fn);
}

Engine::Engine(const std::vector<DeviceConfiguration>& cfg, const DistConfiguration& dist_cfg)
    : pimpl(new EngineImpl(cfg, dist_cfg)) {}

Engine::~Engine() { }

Context Engine::create_context(const std::vector<int>& devices) const {
    BM_ASSERT(devices.size() <= (size_t) num_gpus(), "devices.size() too big");
    int rank = pimpl->handles[devices[0]]->tp_rank;
    return Context(std::make_unique<ContextImpl>(this->pimpl.get(), devices, rank));
}
Context Engine::create_context() const {
    std::vector<int> devices(num_gpus()); // use all GPUs
    std::iota(devices.begin(), devices.end(), 0);
    return Context(std::make_unique<ContextImpl>(this->pimpl.get(), devices, 0));
}
Context Engine::create_context_rank(int rank) const {
    std::vector<int> devices;
    int tp_rank = 0;
    for (size_t i = 0; i < pimpl->handles.size(); ++i) {
        if (pimpl->handles[i]->tp_rank % local_ranks() == rank) {
            devices.push_back(int(i));
            tp_rank = pimpl->handles[i]->tp_rank;
        }
    }
    // std::cout << "Rank: " << rank << ", devices: " << devices << "\n";
    BM_ASSERT(!devices.empty(), "Wrong rank " + std::to_string(tp_rank));
    return Context(std::make_unique<ContextImpl>(this->pimpl.get(), devices, tp_rank));
}

int Engine::world_size() const {
    return this->pimpl->world_size_;
}

int Engine::local_ranks() const {
    return this->pimpl->local_ranks_;
}

int Engine::num_gpus() const {
    return this->pimpl->num_gpus();
}

GPUInfo Engine::get_gpu_info(int dev_id) const {
    return pimpl->get_gpu_info(dev_id);
}

} // namespace core
} // namespace bmengine
