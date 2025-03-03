#pragma once
#include "bmengine/core/export.h"
#include "bmengine/core/tensor.h"
#include "bmengine/core/stream.h"
#include "bmengine/core/guard.h"
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <nccl.h>

namespace bmengine {

namespace core {

class ContextImpl;
class DistTensor;
class MemoryAllocator;

// Context can be accessed from one thread only.
class BMENGINE_EXPORT Context {
private:
    friend class ScopeDevice;
    std::unique_ptr<ContextImpl> pimpl;
    int cur_layer_ { -1 };
    int high_precision_ { 0 };
    bool BSHD_ { false };

    // push_device and pop_device should only be called in the same thread with ScopeDevice
    void push_device(int idx);
    void pop_device();

public:
    static const std::string EMPTY_STR;

    explicit Context(std::unique_ptr<ContextImpl>&&);
    virtual ~Context();

    Context(const Context&) = delete;
    Context(Context&&);

    void alloc_device(int idx) const;
    void release_device() const;
    // if idx is different from active_device(), release current device and alloc_device(idx)
    virtual bool switch_to_device(int idx) const;
    const std::vector<int> devices() const;

    int active_device() const;  // device in engine
    int active_device_idx() const;  // index of devices in this context
    int rank() const;
    int world_size() const;
    int get_compute_capability() const;
    int get_mp_count() const;
    int get_L2_cache_size() const;
    int get_max_shared_memory() const;

    virtual void set_current_layer(int i) { cur_layer_ = i; }
    int current_layer() const { return cur_layer_; }
    bool is_layer(int layer, int rank = 0) const {
        return cur_layer_ == layer && this->rank() == rank;
    }

    Stream current_stream() const;
    void set_current_stream(Stream s);
    ncclComm_t current_comm() const;
    Stream get_stream() const;
    cublasHandle_t cublas_handle() const;
    cublasLtHandle_t current_cublas_handle() const;

    Tensor null_tensor() const;
    Tensor tensor(
        const std::vector<size_t>& size,
        DataType dtype,
        const std::string& name = EMPTY_STR,
        size_t round_up_bytes = 1024) const;
    Tensor tensor_s(
        const std::vector<long>& size,
        DataType dtype) const;
    // Define a tensor without allocate memory
    Tensor parameter(const std::vector<size_t>& size, DataType dtype) const;
    Tensor distribute_parameter(const Tensor& param, DistLayout layout) const;
    void load_parameter(
        Tensor* weight,
        const std::string& name,
        const std::map<std::string, const Tensor>& state_dict,
        bool parallel,
        DistLayout layout) const;

    // cache tensor to active_device() if active_device() is different from tensor
    const Tensor* identity(const Tensor* tensor, const std::string& name) const;
    const Tensor copy(const Tensor& tensor) const;

    void clear_identity_cache();

    // copy src to dst if src is a cpu tensor, else just assign
    void assign_or_copy(Tensor* dst, const Tensor* src) const;

    void init_parameter(const std::string& name, Tensor* tensor) const;
    void load_state_dict(
        const std::string& name,
        const std::map<std::string, std::vector<half>>& state_dict,
        Tensor* tensor,
        const bool strict) const;

    size_t used_memory() const;
    size_t peak_memory() const;
    void print_memory_summary() const;
    void print_memory_detail() const;
    MemoryAllocator* get_allocator() const;  // use internally

    WithDevice with_device(int dev_id) const;
    WithDebug with_debug(int debug_level) const;

    // Don't need to exit previous WithDevice or ScopeDevice scope
    ScopeDevice scope_device(int dev_id) const;

    void enable_debug(int level) const;
    int debug() const;
    // Create and record cudaEvent_t to measure time and print timeline
    void recordEvent(const std::string& name, int ev_level = 2, float flops = 0) const;
    void set_event_level(int level) const;
    void print_events();
    std::mutex& log_mutex() const;

    template<typename T, typename DTD = DTypeDeducer<T>>
    Tensor tensor_of(const std::vector<T>& data, const std::vector<size_t>& shape) const {
        if (data.empty()) {
            return std::move(Tensor());
        }
        auto t = tensor(shape, DTD::data_type());
        t.from_buffer(data.data());
        return std::move(t);
    }

    template<typename T>
    Tensor tensor_of(const std::vector<T>& data) const {
        return tensor_of(data, { data.size() });
    }

    virtual Tensor reduce_sum(Tensor& data, DataType out_type) const;

    int high_precision() const { return high_precision_; }
    void set_high_precision(int level) { high_precision_ = level; }

    virtual bool is_BSHD() const { return BSHD_; }
    void set_BSHD(bool b) { BSHD_ = b; }

    Tensor cuda(const Tensor& cpu_tensor) const;

    void reserve_cache_alloc(size_t s);
    void free_cache_alloc();
    void use_cache_alloc(bool b);
    void mem_gc();
};

// Create and record cudaEvent_t to measure time in current scope if debug is enabled
struct EventScope {
    const Context& ctx;
    int debug_level;
    std::string end_name;

    EventScope(const Context& ctx, const std::string& name, int debug_level = 2, float flops = 0)
        : ctx(ctx), debug_level(debug_level) {
        ctx.recordEvent("Start>" + name, debug_level, flops);
        end_name = "End>" + name;
    }
    ~EventScope() { ctx.recordEvent(end_name, debug_level); }
};

class GCStopper {
    const Context& ctx;
public:
    GCStopper(const Context& ctx);
    ~GCStopper();
};

} // namespace core

} // namespace bmengine
