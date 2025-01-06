#include "bmengine/core/context.h"
#include "bmengine/c10d/c10d.h"
#include "bmengine/core/exception.h"
#include "bmengine/functions/typecast.h"
#include "bmengine/logger/std_log_op.hpp"
#include <cstdlib>
#include <map>
#include <mutex>
#include <stack>
#include <vector>
#include <iomanip>
#include <iostream>
#include <curand.h>

#include "private/allocator.h"
#include "private/engine.h"
#include "private/guard.h"
#include "private/context.h"
#include "private/tensor_impl.h"
#include "private/tensor_ops.h"
#include <execinfo.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/syscall.h>

static int _get_tid() {
    return syscall(SYS_gettid);
}
namespace bmengine {
namespace core {

using std::string;

static inline int get_int_env(const char* name, int def_val = 0) {
    char* env_str = std::getenv(name);
    return env_str != nullptr ? std::atoi(env_str) : def_val;
}

ContextImpl::ContextImpl(EngineImpl* engine, const std::vector<int>& devices, int rank, bool aux)
    : engine(engine),
      active_device(-1),
      devices(devices),
      rank_(rank),
      used_memory(0),
      peak_memory(0),
      thread_id(std::this_thread::get_id()),
      debug(0),
      tensor_id(0L),
      aux_(aux) {
    tensor_cache.resize(8);
    debug = get_int_env("BM_DEBUG_LEVEL");

    for (int dev_id : devices) {
        // DeviceHandles* dev_handle = engine->get_device_handle(dev_id, aux);
        DeviceHandles* dev_handle = engine->get_device_handle(dev_id);
        dev_handles.push_back(dev_handle);
        if (aux) {
            BM_ASSERT(false, "");
//            auto org_alloc = engine->get_allocator(dev_id);
//            auto free_mem = org_alloc->get_memory_limit() - org_alloc->used_memory();
//            auto new_alloc = new MemoryAllocator(*org_alloc, free_mem / 2, dev_handle->stream);
//            allocators.push_back(new_alloc);
        } else {
            allocators.push_back(engine->get_allocator(dev_id));
        }
    }
    pid_ = _get_tid();
}

static bool starts_with(const char* str, const char* pre) {
    return strncmp(str, pre, strlen(pre)) == 0;
}

static bool starts_with(const std::string& str, const char* pre) {
    return str.rfind(pre, 0) == 0;
}

// chrome trace event format:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0
void ContextImpl::print_events_json() {
    std::cout << "[\n";
    string main_name = logger::str_cat(events[0].name.c_str() + 6, " (", engine->num_gpus(), " GPUs)");
    float last_ts = 0;
    bool print_connect = get_int_env("TIMELINE_ARROW", 1) > 0;
    for (int i = 0; i < events.size(); ++i) {
        auto& ev = events[i];
        float ts_ms;
        cudaEventElapsedTime(&ts_ms, events[0].ev, ev.ev);
        char buf[1024];
        char ph[2] = {'B', 0};
        char tid[2] = {'0', 0};
        const char* name = ev.name.c_str();
        if (starts_with(ev.name, "End>")) {
            name += 4;
            ph[0] = 'E';
        }
        if (starts_with(ev.name, "Start>")) name += 6;
        if (ev.name.find("ReducePart") != string::npos) tid[0] = '1';
        if (i == 0) name = main_name.c_str();
        if (print_connect && ev.name.find("Reduce") != string::npos && starts_with(ev.name, "Start>")) {
            snprintf(buf, 1024,
                     R"({"id": "Flow%d", "name": "Flow%d", "ph": "s", "ts": %d, "pid": 0, "tid": 0},)",
                     i, i, int(last_ts * 1000));
            std::cout << buf << endl;
            snprintf(buf, 1024,
                     R"({"id": "Flow%d", "name": "Flow%d", "ph": "f", "bp": "e", "ts": %d, "pid": 0, "tid": %s},)",
                     i, i, int(ts_ms * 1000), tid);
            std::cout << buf << endl;
        }
        snprintf(buf, 1024,
                 R"({"name": "%s", "cat": "foo", "ph": "%s", "ts": %d, "pid": 0, "tid": %s},)",
                 name, ph, int(ts_ms * 1000), tid);
        std::cout << buf << endl;
        last_ts = ts_ms;
        if (name == ev.name.c_str()) {
            cudaEventElapsedTime(&ts_ms, events[0].ev, events[i + 1].ev);
            ph[0] = 'E';
            snprintf(buf, 1024,
                     R"({"name": "%s", "cat": "foo", "ph": "%s", "ts": %d, "pid": 0, "tid": %s},)",
                     name, ph, int(ts_ms * 1000), tid);
            std::cout << buf << endl;
        }
    }
    std::cout << R"({"name": "process_name", "ph": "M", "pid": 0, "tid": 0, "args": {"name" : "Cuda Streams"}},)" << endl;
    std::cout << R"({"name": "thread_name", "ph": "M", "pid": 0, "tid": 0, "args": {"name" : "Compute Stream"}},)" << endl;
    std::cout << R"({"name": "thread_name", "ph": "M", "pid": 0, "tid": 1, "args": {"name" : "Reduce Stream"}})" << endl;
    std::cout << "]\n";
    for (auto &ev: events) {
        cudaEventDestroy(ev.ev);
    }
    events.clear();
}

static string format_ev(const string& name, float begin, float taken, float total) {
    char buf[202];
    int buf_len = 200;
    buf[200] = 0;
    buf[201] = 0;
    static float width = get_int_env("TIMELINE_WIDTH", 150);
    int begin_p = int(round(begin / total * width));
    int end_p = int(round((begin + taken) / total * width));
    float percent = std::min(100.f, taken / total * 100);
    for (int i = 0; i < begin_p; ++i) {
        buf[i] = ' ';
    }
    int msg_len = snprintf(buf + begin_p, buf_len - begin_p, "< %s %.1f%% ", name.c_str(), percent);
    if (taken == total)
        msg_len = snprintf(buf + begin_p, buf_len - begin_p,
                           "< %s taken %dus, %.1f%% ", name.c_str(), int(1000 * taken), percent);
    int p = begin_p + msg_len;
    for (; p < end_p; ++p) {
        buf[p] = '-';
    }
    if (buf[p - 2] == '-') buf[p - 2] = ' ';
    buf[p - 1] = '>';
    buf[p] = 0;
    return buf;
}

void ContextImpl::print_events() {
    if (events.size() == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(engine->log_mutex); // lock output.
    try {
        cudaDeviceProp deviceProp;
        int dev_id = engine->get_device_handle(devices[0])->dev_id;
        BM_CUDART_ASSERT(cudaGetDeviceProperties(&deviceProp, dev_id));
        std::cout << "#### " << engine->num_gpus() << " x GPU: " << deviceProp.name
                  << " CC: " << deviceProp.major << "." << deviceProp.minor << " ####" << std::endl;
        BM_CUDART_ASSERT(cudaDeviceSynchronize());

        int timeline_fmt = get_int_env("TIMELINE_FORMAT", 0);
        if (timeline_fmt == 2) {
            print_events_json();
            return;
        }

        std::stack<size_t> end_stack;
        std::string indent;
        std::string pad = "    ";
        float total_ms = 0;
        for (size_t i = 0; i + 1 < events.size(); ++i) {
            if (!end_stack.empty() && i == end_stack.top()) {
                end_stack.pop();
                if (!indent.empty())
                    indent.resize(indent.size() - pad.size());
                continue;
            }
            float begin_ms = 0;
            float elapsed_ms;
            bool has_start = events[i].name.rfind("Start>", 0) == 0;
            // try to find corresponding End event
            bool found_end = false;
            size_t max_end = std::min(i + (has_start ? 10000 : 20), events.size());
            const char* name = has_start ? events[i].name.data() + 6 : events[i].name.data();
            size_t len_name = strlen(name);
            size_t k = i + 1;
            for (; k < max_end; ++k) {
                if (starts_with(events[k].name, "End>")
                    && strncmp(events[k].name.data() + 4, name, len_name) == 0) {
                    // time from start to end
                    cudaEventElapsedTime(&elapsed_ms, events[i].ev, events[k].ev);
                    found_end = true;
                    break;
                }
            }
            BM_ASSERT(!has_start || found_end, events[i].name);
            if (!found_end) {
                // time from start to next
                cudaEventElapsedTime(&elapsed_ms, events[i].ev, events[i + 1].ev);
            }
            if (timeline_fmt && !starts_with(events[i].name, "End")) {
                if (i == 0) {
                    total_ms = elapsed_ms;
                } else {
                    cudaEventElapsedTime(&begin_ms, events[0].ev, events[i].ev);
                }
                std::cout << format_ev(name, begin_ms, elapsed_ms, total_ms) << endl;
            } else if (!starts_with(events[i].name, "End")) {
                std::cout << indent << name << std::fixed << std::setprecision(1);
                if (events[i].flops > 0) {
                    double flops = events[i].flops / elapsed_ms / 1e9;
                    if (flops > 50.)
                        std::cout << ", FLOPS=" << flops << "T,";
                    else
                        std::cout << ", FLOPS=" << (flops * 1000.) << "G,";
                }
                std::cout << std::setprecision(0) << " take " << (1000 * elapsed_ms) << "us\n";
                // std::cout << indent << name << " take " << elapsed_ms << "ms\n";
            }
            if (has_start && found_end) {
                end_stack.push(k);
                indent += pad;
            }
        }
        for (auto &ev: events) {
            cudaEventDestroy(ev.ev);
        }
        events.clear();
    } catch (const std::exception& err) { std::cerr << err.what() << std::endl; }
}

ContextImpl::~ContextImpl() {
    if (aux_) {
        for (auto alloc : allocators) {
            delete alloc;
        }
    }
    if (active_device != -1) {
        std::cerr << "Device " << std::to_string(active_device)
                  << " is not release when context is destroyed" << std::endl;
    }
    if (debug >= 1) {
        engine->print_memory_summary();
        std::cerr << "Context accumulated"
                  << " used_memory " << (used_memory / 1000) << "KBytes"
                  << " peak_memory " << (peak_memory / 1000) << "KBytes" << std::endl;
    }
    if (!events.empty()) {
        print_events();
    }
//    for (auto a: cache_allocators) {
//        delete a;
//    }
    while (!cache_allocators.empty()) {
        auto p = cache_allocators.top();
        delete p;
        cache_allocators.pop();
    }
    // std::cerr << "~Context pid=" << pid_ << "\n";
}

void ContextImpl::check_in_same_thread() {
//    if (thread_id != std::this_thread::get_id()) {
//        std::cerr << "Context expPid=" << pid_ << ", real pid=" << _get_tid() << "\n";
//    }
    BM_ASSERT(thread_id == std::this_thread::get_id(), "Use context in different thread");
}

void ContextImpl::use_device(int dev_id) {
    check_in_same_thread();
    BM_ASSERT(dev_id >= 0 && dev_id < devices.size(), "Invalid dev_id:" + std::to_string(dev_id));
    BM_ASSERT(
        active_device == -1,
        "Previous device " + std::to_string(active_device) + " is not released");
    active_device = dev_id;
    if (!aux_)
        engine->alloc_device(devices[dev_id]);
    else
        BM_CUDART_ASSERT(cudaSetDevice(devices[dev_id]));
}

void ContextImpl::release_device() {
    check_in_same_thread();
    BM_ASSERT(active_device != -1, "No device is active");
    if (!aux_)
        engine->release_device(devices[active_device]);
    active_device = -1;
}

void ContextImpl::push_device(int idx) {
    check_in_same_thread();
    old_dev_stack.push(active_device);
    new_dev_stack.push(idx);
    if (debug >= 2) {
        std::cerr << "Push " << active_device << ", " << idx << ", " << std::this_thread::get_id()
                  << std::endl;
    }
    if (active_device != idx) {
        if (active_device != -1) {
            // BM_CUDART_ASSERT(cudaStreamSynchronize(current_stream()->ptr));
            // BM_CUDART_ASSERT(cudaGetLastError());
            release_device();
        }
        use_device(idx);
    }
}
void ContextImpl::pop_device() {
    check_in_same_thread();
    BM_ASSERT(!old_dev_stack.empty(), "old_dev_stack.empty()");
    int idx = old_dev_stack.top();
    if (debug >= 2) {
        std::cerr << "Pop " << active_device << ", " << idx << std::endl;
    }
    old_dev_stack.pop();
    new_dev_stack.pop();
    if (active_device != idx) {
        if (active_device != -1) {
            // BM_CUDART_ASSERT(cudaStreamSynchronize(current_stream()->ptr));
            // BM_CUDART_ASSERT(cudaGetLastError());
            release_device();
        }
        if (idx != -1) {
            use_device(idx);
        }
    }
}
DeviceHandles* ContextImpl::cur_dev_handle() const {
    BM_ASSERT(active_device != -1, "No device is active");
    return dev_handles[active_device];
}
MemoryAllocator* ContextImpl::get_allocator() const {
    BM_ASSERT(active_device != -1, "No device is active");
    auto allocator = allocators[active_device];
    if (use_cache_alloc_) {
        BM_ASSERT(!cache_allocators.empty(), "");
        allocator = cache_allocators.top();
    }
    return allocator;
}

int ContextImpl::current_device() {
    BM_ASSERT(active_device != -1, "No device is active");
    return devices[active_device];
}
int ContextImpl::rank() {
    return rank_;
}
int ContextImpl::world_size() {
    return engine->world_size();
}
int ContextImpl::get_compute_capability() {
    return cur_dev_handle()->compute_capability;
}
Stream ContextImpl::current_stream() {
    return std::make_shared<Stream_>(
        cur_dev_handle()->stream, [](cudaStream_t) {});
}
void ContextImpl::set_current_stream(Stream s) {
    BM_ASSERT(active_device != -1, "No device is active");
    cur_dev_handle()->stream = s->ptr;
}
ncclComm_t ContextImpl::current_comm() {
    return cur_dev_handle()->comm;
}
Stream ContextImpl::get_stream() {
    BM_ASSERT(active_device != -1, "No device is active");
    int dev_id = devices[active_device];
    cudaStream_t stream = engine->create_stream(dev_id);
    return std::make_shared<Stream_>(
        stream, [this, dev_id](cudaStream_t stream) { engine->destroy_stream(dev_id, stream); });
}

cublasHandle_t ContextImpl::cublas_handle() {
    return cur_dev_handle()->cublas_handle;
}

cublasLtHandle_t ContextImpl::current_cublas_handle() {
    // https://docs.nvidia.com/cuda/cublas/index.html#cublaslthandle-t
    // cuBLAS handle (cublasHandle_t) encapsulates a cuBLASLt handle.
    // Any valid cublasHandle_t can be used in place of cublasLtHandle_t with a simple cast.
    return (cublasLtHandle_t) cur_dev_handle()->cublas_handle;
}

Memory ContextImpl::alloc(size_t nbytes, size_t round_up_bytes) {
    auto allocator = get_allocator();
    auto mem = allocator->alloc(nbytes, round_up_bytes);
    used_memory += nbytes;
    peak_memory = std::max(used_memory, peak_memory);
    mem->add_deleter([this, nbytes](void* ptr) { used_memory -= nbytes; });
    return mem;
}

void ContextImpl::init_parameter(const std::string& name, Tensor* tensor) const {
    return engine->init_parameter(name, tensor);
}

const Tensor* ContextImpl::identity(const Tensor* tensor, const std::string& name) {
    std::map<long, TensorPtr>& map = tensor_cache[active_device];
    auto search = map.find(tensor->id());
    if (search != map.end()) {
        if (debug >= 3) {
            std::cerr << "Use cached tensor " << tensor->info(1) << std::endl;
        }
        return search->second.get();
    }
    size_t nbytes = tensor->nbytes();
    Memory mem = this->alloc(nbytes);
    int srcDevice = tensor->device();
    int dstDevice = mem->dev;
    if (debug >= 3) {
        std::cerr << "Copy tensor " << tensor->info(1) << ", from " << srcDevice << " to "
                  << dstDevice << std::endl;
    }
    if (engine->get_device_handle(srcDevice)->dev_id
        == engine->get_device_handle(dstDevice)->dev_id) {
        BM_CUDART_ASSERT(cudaMemcpyAsync(
            mem->ptr, tensor->data(), nbytes, cudaMemcpyDeviceToDevice, current_stream()->ptr));
    } else {
        BM_CUDART_ASSERT(cudaMemcpyPeer(mem->ptr, dstDevice, tensor->data(), srcDevice, nbytes));
    }
    //    BM_CUDART_ASSERT(cudaDeviceSynchronize());
    //    BM_CUDART_ASSERT(cudaStreamSynchronize(current_stream()->ptr));
    Tensor* new_tensor =
        new Tensor(std::make_unique<TensorImpl>(tensor->size(), mem, 0, tensor->dtype()));
    new_tensor->set_name(name);
    map.emplace(tensor->id(), std::unique_ptr<Tensor>(new_tensor));
    return new_tensor;
}

Tensor ContextImpl::copy(const Tensor& tensor) {
    if (tensor.numel() == 0) {
        return std::move(Tensor());
    }

    size_t nbytes = tensor.nbytes();
    Memory mem = this->alloc(nbytes);
    BM_CUDART_ASSERT(cudaMemcpyAsync(
        mem->ptr, tensor.data(), nbytes, cudaMemcpyDeviceToDevice, current_stream()->ptr));
    return std::move(
        Tensor(std::make_unique<TensorImpl>(tensor.size(), mem, 0, tensor.dtype())));
}

void ContextImpl::clear_identity_cache() {
    for (size_t i = 0; i < tensor_cache.size(); ++i) {
        tensor_cache[i].clear();
    }
}

void Context::assign_or_copy(Tensor* dst, const Tensor* src) const {
    if (src->device() >= 0) {
        *dst = *src;
        return;
    }
    BM_ASSERT_EQ(src->shape(), dst->shape(), "src and dst have different shape");
    // allocate memory
    if (dst->data() == nullptr) {
        init_parameter(dst->name(), dst);
    }
    // copy
    dst->from_buffer(src->data());
}

void ContextImpl::recordEvent(const std::string& name, float flops) {
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, current_stream()->ptr);
    events.push_back({e, name, flops});
}

void Context::recordEvent(const std::string& name, int ev_level, float flops) const {
    if (pimpl->event_level >= ev_level && pimpl->active_device == 0 && pimpl->rank() == 0) {
        pimpl->recordEvent(name, flops);
    }
}

void Context::set_event_level(int level) const {
    pimpl->event_level = level;
}

void Context::print_events() {
    pimpl->print_events();
}
std::mutex& Context::log_mutex() const {
    return pimpl->engine->log_mutex;
}

const std::string Context::EMPTY_STR;

Context::Context(std::unique_ptr<ContextImpl>&& impl) : pimpl(std::move(impl)) {
    char* debug_env = std::getenv("HIGH_PRECISION");
    if (debug_env != nullptr) {
        high_precision_ = std::atol(debug_env);
    }
}
Context::~Context() { }

Context::Context(Context&&) = default;

void Context::alloc_device(int idx) const {
//    std::cout << "thread " << _get_tid() << " use device " << idx << endl;
    pimpl->use_device(idx);
}
void Context::release_device() const {
//    std::cout << "thread " << _get_tid() << " release_device " << endl;
    pimpl->release_device();
}
bool Context::switch_to_device(int idx) const {
    if (pimpl->active_device == idx) {
        return false;
    }
    pimpl->check_in_same_thread();
//    std::cout << "thread " << _get_tid() << " switch device from " << pimpl->active_device << " to " << idx << endl;
    if (pimpl->debug >= 2) {
        std::cerr << "Switch device from " << pimpl->active_device << " to " << idx
                  << std::endl;
    }
    if (pimpl->active_device != -1) {
        std::cout << "WARNING active_device != -1\n";
        BM_CUDART_ASSERT(cudaStreamSynchronize(current_stream()->ptr));
        pimpl->release_device();
    }
    pimpl->use_device(idx);
    return true;
}

const std::vector<int> Context::devices() const {
    return pimpl->devices;
}

void Context::push_device(int idx) {
    pimpl->push_device(idx);
}
void Context::pop_device() {
    pimpl->pop_device();
}

int Context::active_device() const {
    return pimpl->current_device();
}

int Context::active_device_idx() const {
    return pimpl->active_device;
}

int Context::rank() const {
    return pimpl->rank();
}
int Context::world_size() const {
    return pimpl->world_size();
}

int Context::get_compute_capability() const {
    return pimpl->get_compute_capability();
}

int Context::get_mp_count() const {
    return pimpl->cur_dev_handle()->mp_count;
}

int Context::get_L2_cache_size() const {
    return pimpl->cur_dev_handle()->l2_cache_size;
}

int Context::get_max_shared_memory() const {
    return pimpl->cur_dev_handle()->max_shared_memory;
}

Stream Context::current_stream() const {
    return pimpl->current_stream();
}
void Context::set_current_stream(Stream s) {
    pimpl->set_current_stream(s);
}

ncclComm_t Context::current_comm() const {
    return pimpl->current_comm();
}
Stream Context::get_stream() const {
    return pimpl->get_stream();
}
cublasHandle_t Context::cublas_handle() const {
    return pimpl->cublas_handle();
}
cublasLtHandle_t Context::current_cublas_handle() const {
    return pimpl->current_cublas_handle();
}

static inline size_t round_up(size_t num, size_t multiple) {
    return (num + multiple - 1) / multiple * multiple;
//    size_t numel = get_numel(shape);
//    if (shape.size() > 1 && dtype == DataType::kHalf) {
//        numel = round_up(numel, 16 * shape[shape.size() - 1]);
//    }
//    size_t nbytes = numel * get_elem_size(dtype);
}

Tensor Context::tensor(
    const std::vector<size_t>& size,
    DataType dtype,
    const std::string& name,
    size_t round_up_bytes) const {
    check_no_zero(size);
    size_t nbytes = get_numel(size) * get_elem_size(dtype);
    Tensor tensor(
        std::make_unique<TensorImpl>(size, pimpl->alloc(nbytes, round_up_bytes), 0, dtype));
    tensor.set_id(std::atomic_fetch_add(&pimpl->tensor_id, 1L));
    tensor.set_name(name);
    if (pimpl->debug >= 6) {
        std::cerr << "Allocate: " << tensor.info(1) << std::endl;
    }
    return std::move(tensor);
}
Tensor Context::tensor_s(
    const std::vector<long>& size,
    DataType dtype) const {
    std::vector<size_t> shape;
    for (auto d : size) {
        shape.push_back(size_t(d));
    }
    return tensor(shape, dtype);
}

Tensor Context::null_tensor() const {
    return std::move(Tensor());
}

Tensor Context::parameter(const std::vector<size_t>& size, DataType dtype) const {
    check_no_zero(size);
    size_t nbytes = get_numel(size) * get_elem_size(dtype);
    return Tensor(std::make_unique<TensorImpl>(
        size,
        std::make_shared<Memory_>(
            nullptr,
            pimpl->current_device(),
            nbytes,
            [](void* ptr) { BM_ASSERT(ptr == nullptr, "Memory is not nullptr"); }),
        0,
        dtype));
}

Tensor Context::distribute_parameter(const Tensor& param, DistLayout layout) const {
    auto shape = param.shape();
    int shard_dim = shape.size() - (layout == DistLayout::ROW ? 2 : 1);
    BM_ASSERT(shape[shard_dim] % world_size() == 0, "size can't be divided by world_size");
    int shard_len = shape[shard_dim] / world_size();

    Tensor local = param;
    if (rank() != 0) {
        local = tensor(param.shape(), param.dtype());
    }
    BM_NCCL_ASSERT(ncclBroadcast(
        param.data<void*>(),
        local.mutable_data<void*>(),
        local.numel(),
        c10d::dtype2nccl(param.dtype()),
        0,
        current_comm(),
        current_stream()->ptr));
    if (layout == DistLayout::REPLICATED) {
        return local;
    }

    std::vector<int> gather_indices(shard_len);
    int offset = rank() * shard_len;
    for (int i = 0; i < shard_len; ++i) {
        gather_indices[i] = offset + i;
    }
    Tensor d_indices = tensor_of(gather_indices);
    shape[shard_dim] = shard_len;
    Tensor ret = tensor(shape, param.dtype());
    functions::index_select(current_stream()->ptr, local, shard_dim, d_indices, ret);
    return ret;
}

// load distributed parameter w/o NCCL.
// It's faster if there's no NV-Link.
void Context::load_parameter(
    Tensor* weight,
    const std::string& name,
    const std::map<std::string, const Tensor>& state_dict,
    bool parallel,
    DistLayout layout) const {
    auto it = state_dict.find(name);
    BM_ASSERT(it != state_dict.end(), "param " + name + " not found in state_dict");
    auto& param = it->second;
    BM_ASSERT_EQ(weight->numel(), param.numel(), name + " shape mismatch");
    static bool print_param = std::getenv("PRINT_LOAD_PARAM") != nullptr;
    if (print_param) {
        std::cout << "Load " << name << ", shape=" << weight->shape() << ", srcShape=" << param.shape() << endl;
    }
    BM_ASSERT_EQ(weight->shape(), param.shape(), "shape mismatch");
//    if (get_compute_capability() == 80) {
//        if (rank() == 0)
//            assign_or_copy(weight, &param);
//        *weight = distribute_parameter(*weight, layout);
//        return;
//    }
    if (!parallel || world_size() == 1 || layout == DistLayout::REPLICATED) {
        assign_or_copy(weight, &param);
        weight->set_name(name);
        return;
    }
    auto shape = weight->shape();
    size_t shard_dim = shape.size() - (layout == DistLayout::ROW ? 2 : 1);
    BM_ASSERT(shape[shard_dim] % world_size() == 0, "size can't be divided by world_size");
    shape[shard_dim] /= world_size();
    size_t shard_len = shape[shard_dim];
    *weight = tensor(shape, weight->dtype());
    weight->set_name(name);

    // case 1: ROW layout, copy directly
    if (shard_dim == 0) {
        auto part = param.slice_dim0_len(rank() * shard_len, shard_len);
        assign_or_copy(weight, &part);
        return;
    }
    BM_ASSERT_EQ(weight->ndim(), 2, "Unsupported ndim");

    // case 2: COLUMN layout, slice column in CPU, then copy
    thread_local char* buf = nullptr;
    thread_local size_t buf_size = 0;
    bool use_pin_buf = getenv("USE_PIN_BUF");
    if (use_pin_buf) {
        if (buf_size < weight->nbytes())
            BM_CUDART_ASSERT(cudaHostAlloc(&buf, weight->nbytes(), 0));
        buf_size = weight->nbytes();
    } else {
        buf = new char[weight->nbytes()];
    }
    size_t shard_bytes = shard_len * get_elem_size(weight->dtype());
    size_t row_bytes = shard_bytes * world_size();
    size_t col_offset = rank() * shard_bytes;
    size_t num_row = weight->size(0);
    for (int i = 0; i < num_row; ++i) {
        char* dst = buf + i * shard_bytes;
        char* src = param.data<char>() + i * row_bytes;
        memcpy(dst, src + col_offset, shard_bytes);
    }
    weight->from_buffer(buf);
    if (!use_pin_buf)
        delete[] buf;
}

const Tensor* Context::identity(const Tensor* tensor, const std::string& name) const {
    if (tensor == nullptr) {
        return nullptr;
    }
    if (tensor->numel() == 0 || active_device() == tensor->device()) {
        return tensor;
    }
    return pimpl->identity(tensor, name);
}

const Tensor Context::copy(const Tensor& tensor) const {
    return pimpl->copy(tensor);
}

void Context::clear_identity_cache() {
    pimpl->clear_identity_cache();
}

void Context::init_parameter(const std::string& name, Tensor* tensor) const {
    return pimpl->init_parameter(name, tensor);
}

size_t Context::used_memory() const {
    return pimpl->used_memory;
}

size_t Context::peak_memory() const {
    return pimpl->peak_memory;
}

void Context::print_memory_summary() const {
    return pimpl->engine->print_memory_summary();
}
void Context::print_memory_detail() const {
    // return pimpl->engine->print_mem_info();
}
MemoryAllocator* Context::get_allocator() const {
    return pimpl->get_allocator();
}

WithDevice Context::with_device(int dev_id) const {
    return WithDevice(*this, dev_id);
}

WithDebug Context::with_debug(int debug_level) const {
    return WithDebug(*this, debug_level);
}

ScopeDevice Context::scope_device(int dev_id) const {
    return ScopeDevice(*this, dev_id);
}
void Context::enable_debug(int level) const {
    pimpl->debug = level;
}
int Context::debug() const {
    return pimpl->debug;
}

Tensor Context::reduce_sum(Tensor& data, DataType out_type) const {
    BM_CUDART_ASSERT(cudaDeviceSynchronize());
    c10d::NCCLAllReduce(*this, data, data, ncclSum);  // reduce in-place
    BM_CUDART_ASSERT(cudaDeviceSynchronize());
    return functions::typecast(*this, data, out_type);
}

Tensor Context::cuda(const Tensor& cpu_tensor) const {
    BM_ASSERT_EQ(cpu_tensor.device(), -1, "Not a cpu tensor");
    Tensor out = this->tensor(cpu_tensor.shape(), cpu_tensor.dtype());
    out.set_name(cpu_tensor.name());
    out.from_buffer(cpu_tensor.data());
    return out;
}

void ContextImpl::reserve_cache_alloc(size_t s) {
    BM_ASSERT_EQ(1, devices.size() == 1, "TP only");
    auto alloc =  allocators[0];
    auto stream = engine->get_device_handle(devices[0])->stream;
    cache_allocators.push(new MemoryAllocator(*alloc, s, stream));

//    BM_ASSERT(cache_allocators.empty(), "reserve_cache_alloc called twice");
//    for (size_t i = 0; i < devices.size(); ++i) {
//        auto alloc =  allocators[i];
//        auto stream = engine->get_device_handle(devices[i])->stream;
//        cache_allocators.push(new MemoryAllocator(*alloc, s, stream));
//    }
}
void ContextImpl::free_cache_alloc() {
    BM_ASSERT_EQ(1, devices.size() == 1, "TP only");
    BM_ASSERT(!cache_allocators.empty(), "");
    delete cache_allocators.top();
    cache_allocators.pop();
}
void Context::reserve_cache_alloc(size_t s) {
    pimpl->reserve_cache_alloc(s);
}
void Context::free_cache_alloc() {
    pimpl->free_cache_alloc();
}
void Context::use_cache_alloc(bool b) {
    BM_ASSERT(!b || !pimpl->cache_allocators.empty(), "No cache allocators");
    pimpl->use_cache_alloc_ = b;
}

void Context::mem_gc() {
    BM_ASSERT_EQ(1, pimpl->devices.size() == 1, "TP only");
    pimpl->allocators[0]->defragmentation();
}

GCStopper::GCStopper(const Context& ctx) : ctx(ctx) {
    ctx.get_allocator()->set_allow_gc(false);
}
GCStopper::~GCStopper() {
    ctx.get_allocator()->set_allow_gc(true);
}

} // namespace core
} // namespace bmengine
