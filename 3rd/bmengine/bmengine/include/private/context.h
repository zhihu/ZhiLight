#pragma once
#include "bmengine/core/exception.h"
#include "bmengine/core/stream.h"
#include <atomic>
#include <map>
#include <stack>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "private/allocator.h"
#include "private/guard.h"
#include "private/engine.h"
#include <curand.h>
#include <curand_kernel.h>
#include <nccl.h>
namespace bmengine {
namespace core {

class ContextImpl {
private:
    friend class Context;
    struct EventRecord {
        cudaEvent_t ev;
        std::string name;
        float flops;
    };

    int active_device;
    EngineImpl* engine;
    std::vector<int> devices;
    int rank_;
    size_t used_memory;
    size_t peak_memory;

    // for ScopeDevice
    std::stack<int> old_dev_stack;
    std::stack<int> new_dev_stack;
    std::thread::id thread_id;
    int pid_ { 0 };

    typedef std::unique_ptr<Tensor> TensorPtr;
    std::vector<std::map<long, TensorPtr>> tensor_cache;

    int debug;                   // debug level
    int event_level { -1 };      // event level
    std::atomic<long> tensor_id; // auto increased tensor id generator

    std::vector<EventRecord> events;

    std::vector<DeviceHandles*> dev_handles;
    std::vector<MemoryAllocator*> allocators;
    std::stack<MemoryAllocator*> cache_allocators;
    bool use_cache_alloc_ { false };
    bool aux_;

    void reserve_cache_alloc(size_t s);
    void free_cache_alloc();

    void check_in_same_thread();

    void print_events();
    void print_events_json();

public:
    ContextImpl(EngineImpl* engine, const std::vector<int>& devices, int rank, bool aux=false);
    ~ContextImpl();

    ContextImpl(const ContextImpl&) = delete;
    ContextImpl& operator=(const ContextImpl&) = delete;
    ContextImpl(ContextImpl&&) = delete;
    ContextImpl& operator=(ContextImpl&&) = delete;

    void use_device(int dev_id);
    void release_device();
    DeviceHandles* cur_dev_handle() const;
    MemoryAllocator* get_allocator() const;

    void push_device(int dev_id);
    void pop_device();

    int current_device();
    int rank();
    int world_size();
    int get_compute_capability();
    Stream current_stream();
    void set_current_stream(Stream s);
    ncclComm_t current_comm();
    Stream get_stream();
    cublasHandle_t cublas_handle();
    cublasLtHandle_t current_cublas_handle();
    void run_task(std::function<void()> f) const;

    Memory alloc(size_t nbytes, size_t round_up_bytes = 1024);
    void init_parameter(const std::string& name, Tensor* tensor) const;

    // cache tensor to active_device() if active_device() is different from tensor
    const Tensor* identity(const Tensor* tensor, const std::string& name);
    Tensor copy(const Tensor& tensor);
    void clear_identity_cache();

    void recordEvent(const std::string& name, float flops = 0);
};

} // namespace core
} // namespace bmengine
