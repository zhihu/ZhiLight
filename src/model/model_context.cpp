#include "model/model_context.h"
#include "model/allocate_util.hpp"
#include "model/dyn_batch_context.h"
#include "model/rag_buffer_context.h"
#include "model/model.h"
#include "nn/quant/int8/quant_kernel.h"
#include "utils/env.h"
#include <bmengine/c10d/c10d.h>
#include <bmengine/functions/all.h>
#include <bmengine/functions/element.h>
#include <bmengine/functions/tensor_ops.h>
#include <bmengine/functions/typecast.h>
#include <bmengine/logger/std_log_op.hpp>
#include "bmengine/logger/kernel_time_trace.hpp"
#include <memory>
#include <numeric>
#include <utility>

namespace model {

using bmengine::core::Context;
using bmengine::functions::BinaryElementwiseOp;
using kvcache::KVCacheConfig;
using std::string;

WithBuffer::WithBuffer(ModelContext& context, std::shared_ptr<BufferContext> buf_ctx)
    : ctx(&context), buf_ctx_(context.buffer_context()) {
    ctx->switch_buffer(std::move(buf_ctx));
}
WithBuffer::~WithBuffer() {
    if (buf_ctx_ != nullptr)
        ctx->switch_buffer(buf_ctx_);
}

WithBuffer::WithBuffer(WithBuffer&& other)
    : ctx(other.ctx), buf_ctx_(std::move(other.buf_ctx_)) {
    other.ctx = nullptr;
    other.buf_ctx_ = nullptr;
}

ModelContext::ModelContext(
    Context&& ctx, const ModelBase& m, int batch_size, bool parallel, bool BSHD)
    : Context(std::move(ctx)), model_(m), cfg(m.cfg), parallel_(parallel) {
    set_BSHD(BSHD);
    layer_devices = model::partition_layer_devices(*this, m.num_layers);
    buf_ctx_ =
        std::make_shared<TransformerBufferContext>(m, batch_size, parallel, world_size(), BSHD);
    buf_ctx_->set_layer_devices(layer_devices);
    latent_cache_ = cfg.kv_lora_rank > 0 && utils::get_int_env("LATENT_CACHE", 0) == 1;
}

ModelContext::~ModelContext() {
    if (debug() >= 1) {
        print_memory_summary();
        std::cerr << "ModelContext accumulated"
                  << " used_memory " << (used_memory() / 1000) << "KBytes"
                  << " peak_memory " << (peak_memory() / 1000) << "KBytes" << std::endl;
    }
};

KVCacheConfig ModelContext::get_kv_cache_config() {
    int num_kv_heads = parallel_ ? cfg.num_kv_heads / world_size() : cfg.num_kv_heads;
    int dim_head = cfg.dim_head;
    DataType dtype = cfg.dtype;
    std::shared_ptr<core::DataType> scale_dtype;
    char* env = std::getenv("KV_CACHE_DTYPE");
    if (env && *env) {
        if (string("int8") == env) {
            dtype = DataType::kInt8;
            scale_dtype = std::make_shared<DataType>(DataType::kFloat); // TODO
        } else {
            throw std::runtime_error("Unsupported dtype: " + string(env));
        }
    }
    KVCacheConfig cache_config =
        {cfg.num_layers, num_kv_heads, dim_head, dtype, is_BSHD(), scale_dtype};
    cache_config.layer_devices = layer_devices;
    return cache_config;
}

void ModelContext::resize_task_buf(int b, size_t new_length) {
    rag_buffer()->resize_task_buf(*this, b, new_length);
}

void ModelContext::free_task_buf(int b) {
    rag_buffer()->free_task_buf(b);
}

// for batch_generator
ModelContext ModelContext::create(
    Engine& engine, const ModelBase& md, const DynBatchConfig& batch_config, int dev, bool parallel
) {
    std::vector<int> devices(dev == -1 ? engine.num_gpus() : 1);
    if (dev == -1) {
        std::iota(devices.begin(), devices.end(), 0);
    } else {
        devices[0] = dev;
    }
    // std::cout << "devices: " << devices << "\n";

    Context ctx = parallel ? engine.create_context_rank(dev) : engine.create_context(devices);
    ModelContext model_ctx(std::move(ctx), md, batch_config.max_batch, parallel);
    model_ctx.set_BSHD(batch_config.flash_attention);
    model_ctx.dyn_batch_ = std::make_shared<DynBatchContext>();
    if (batch_config.rag_buffer) {
        auto k_cfg = model_ctx.get_kv_cache_config();
        if (model_ctx.cfg.qk_rope_head_dim > 0) {
            k_cfg.dim_head = model_ctx.cfg.qk_rope_head_dim + model_ctx.cfg.qk_nope_head_dim;
        }
        auto v_cfg = k_cfg;
        if (model_ctx.latent_cache_ && model_ctx.cfg.kv_lora_rank > 0) {
            k_cfg.num_heads = 1;
            v_cfg.num_heads = 1;
            k_cfg.dim_head = model_ctx.cfg.kv_lora_rank + model_ctx.cfg.qk_rope_head_dim;
            v_cfg.dim_head = 0;
        }
        model_ctx.set_rag_buffer(std::make_shared<RagBufferContext>(k_cfg, v_cfg));
    }
    model_ctx.reducer_ = std::make_shared<ReduceContext>();
    return model_ctx;
}

Tensor ModelContext::copy_peer(const Tensor& src) {
    Tensor dst;
    if (src.numel()) {
        dst = this->tensor(src.shape(), src.dtype());
        BM_ASSERT(src.device() != dst.device(), "Copy tensor to same device.");
        BM_CUDART_ASSERT(cudaMemcpyPeer(dst.data(), dst.device(), src.data(), src.device(), src.nbytes()));
    }
    return dst;
}

void ModelContext::copy2(const Tensor& src, Tensor* dst) {
    BM_CUDART_ASSERT(cudaMemcpyAsync(
        dst->data(), src.data(), src.nbytes(), cudaMemcpyDeviceToDevice, current_stream()->ptr));
}

using namespace std::chrono_literals;

std::pair<ReduceContext*, Tensor> ReduceContext::pop_buffer() {
    bool timeout = false;
    {
        std::unique_lock<std::mutex> lock(buf_mutex_);
        if (peer_buffers_.empty()) {
            if (buf_cond_.wait_for(lock, 2000ms) == std::cv_status::timeout) {
                std::cerr << "buf_cond_.wait timeout\n";
                timeout = true;
            }
        }
        if (!peer_buffers_.empty()) {
            auto buf = peer_buffers_.front();
            peer_buffers_.pop();
            return buf;
        }
    }
    if (timeout) {
        throw std::runtime_error("Reduce Recv timeout");
    }
    throw std::runtime_error("Unknown error");
}

void ReduceContext::push_buffer(std::pair<ReduceContext*, Tensor>& buf) {
    std::unique_lock<std::mutex> lock(buf_mutex_);
    peer_buffers_.push(buf);
    buf_cond_.notify_one();
}

void ReduceContext::begin() {
    std::unique_lock<std::mutex> lock(done_mutex_);
    count_ = peers_.size();
}

void ReduceContext::end() {
    std::unique_lock<std::mutex> lock(done_mutex_);
    if (--count_ == 0) {
        done_cond_.notify_all();
    }
}

void ReduceContext::wait_peer() {
    std::unique_lock<std::mutex> lock(done_mutex_);
    if (count_ > 0) {
        if (done_cond_.wait_for(lock, 2000ms) == std::cv_status::timeout) {
            std::cerr << "done_cond_.wait timeout\n";
            throw std::runtime_error("Reduce wait timeout");
        }
    }
}

Tensor ModelContext::all_gather(const Tensor& data) const {
    auto shape = data.shape();
    shape[0] *= world_size();
    Tensor out = tensor(shape, data.dtype());
    c10d::NCCLAllGather(*this, data, out);
    return out;
}

void ModelContext::set_host_reducer(std::shared_ptr<HostAllReducer> reducer) {
    host_reducer_ = std::move(reducer);
    cudaStream_t red_stream;
    BM_CUDART_ASSERT(cudaStreamCreateWithPriority(&red_stream, cudaStreamNonBlocking, -1));
    reducer_stream_ = std::make_shared<core::Stream_>(red_stream, [&](cudaStream_t s) { cudaStreamDestroy(s); });
    reducer_thread_ = std::make_shared<core::TaskThreadPool>(1, 10 + rank());
    int dev_id;
    BM_CUDART_ASSERT(cudaGetDevice(&dev_id));
    reducer_thread_->run([=]() { BM_CUDART_ASSERT(cudaSetDevice(dev_id)); });
}

#pragma GCC push_options
#pragma GCC optimize ("O0")
Tensor ModelContext::reduce_sum(Tensor& data, DataType out_type) const {
    core::EventScope ev(*this, "AllReduce", 1, data.nbytes());
    Tensor output = tensor(data.shape(), data.dtype());
    static int host_reduce_thres = utils::get_int_env("HOST_REDUCE_THRES", 128);
    if (host_reducer_ && data.size(-2) >= host_reduce_thres && data.dtype() == DataType::kHalf) {
//        host_reducer_->reduce_sum(rank(), current_layer(), data);
        host_reducer_->reduce_sum_async(rank(), current_layer(), data, output, current_stream()->ptr, reducer_stream_->ptr);
        BM_CUDART_ASSERT(cudaStreamSynchronize(reducer_stream_->ptr));
        return functions::typecast(*this, output, out_type);
    }
    reduce_sum2(data, &output, out_type);
    return output;
    // return functions::typecast(*this, output, out_type);
}

void ModelContext::reduce_sum2(const Tensor& data, Tensor* out, DataType out_type, bool quant) const {
    static int int8_thres = utils::get_int_env("REDUCE_TP_INT8_THRES", INT_MAX);
    static int direct_alloc = utils::get_int_env("BM_DIRECT_MEM_ALLOC", 0);
    if (quant && data.size(0) > int8_thres && world_size() > 1 && get_compute_capability() > 80) {
        reduce_tp_int8(data, out_type, out);
        return;
    }

    int syn_level = utils::get_int_env("CPM_SYNC_NCCL", 0);
    if (data.size(0) > 100 || syn_level > 0) {
        // Synchronize is need for large input, because MemoryAllocator may trigger GC
        BM_CUDART_ASSERT(cudaStreamSynchronize(current_stream()->ptr));
    }
    if (out->numel() == 0)
        *out = tensor(data.shape(), data.dtype());
    c10d::NCCLAllReduce(*this, data, *out, ncclSum);
    if (syn_level > 1 || direct_alloc > 0) { //  || data.size(0) >= 32
        BM_CUDART_ASSERT(cudaStreamSynchronize(current_stream()->ptr));
    }
    // TODO: remove reduce by float feature
    // return functions::typecast(*this, data, out_type);
}

void ModelContext::reduce_tp_int8(const Tensor& data, DataType out_type, Tensor* output) const {
    int event_level = current_layer() == 300 && rank() == 0 ? 0 : 3;
//    core::EventScope ev(*this, "Reduce_tp_int8", event_level);
    size_t WS = world_size();
    size_t GROUP_SIZE = 32;
    size_t M = data.numel() / WS / GROUP_SIZE;
    BM_ASSERT(world_size() > 1, "");
    BM_ASSERT_EQ(data.numel() % (WS * GROUP_SIZE), 0, "");
    BM_ASSERT_LE(WS * M, INT_MAX, "");

    Tensor g_data = data.view({WS, M, GROUP_SIZE});
    // step0: quant
    recordEvent("0.quant", event_level);
    auto [q_send, scale_send] = int8_op::quant_group_32(*this, g_data);
//    if (rank() == 0 && current_layer() == 0) {
//        std::cout << "g_data: " << g_data << endl;
//        std::cout << "q_send: " << q_send << endl;
//        std::cout << "scale_send: " << scale_send << endl;
//        auto x = nn::quant_calc_scale(*this, g_data);
////        std::cout << "BASE q: " << x << endl;
////        std::cout << "BASE q_scale: " << *x.quant_scale << endl;
//        auto deq = int8_op::dequant_group_32(*this, q_send, scale_send);
//        std::cout << "dequant: " << deq << endl;
//    }
    Tensor q_recv = tensor({WS - 1, M, GROUP_SIZE}, q_send.dtype());
    Tensor scale_recv = tensor({WS - 1, M}, scale_send.dtype());
    auto q_send_buf = q_send.chunk();
    auto scale_send_buf = scale_send.chunk();
    auto q_recv_buf = q_recv.chunk();
    auto scale_recv_buf = scale_recv.chunk();

    // step1: scatter
    recordEvent("1.scatter", event_level);
    c10d::NCCLGroupStart();
    for (int i = 0; i < WS - 1; ++i) {
        size_t rank_distance = i + 1;
        size_t src_rank = (rank() + WS + rank_distance) % WS;
        size_t dst_rank = (rank() + WS - rank_distance) % WS;
        // size_t src_p = (rank() + i + WS - 1) % WS;
        c10d::NCCLSend(*this, q_send_buf[dst_rank], dst_rank);
        c10d::NCCLRecv(*this, q_recv_buf[i], src_rank);

        c10d::NCCLSend(*this, scale_send_buf[dst_rank], dst_rank);
        c10d::NCCLRecv(*this, scale_recv_buf[i], src_rank);
    }
    c10d::NCCLGroupEnd();

    // step2: reduce
    recordEvent("2.reduce", event_level);
    auto my = g_data.slice_dim0_len(rank(), 1).view({M, GROUP_SIZE});
    Tensor q_sum = tensor({WS, M, GROUP_SIZE}, q_send.dtype());
    Tensor scale_sum = tensor({WS, M}, scale_send.dtype());
    auto q_sum_buf = q_sum.chunk();
    auto scale_sum_buf = scale_sum.chunk();
    int8_op::dequant_sum_quant_g32(*this, my, q_recv, scale_recv, &q_sum_buf[rank()], &scale_sum_buf[rank()]);

    // step3: gather
    recordEvent("3.gather", event_level);
    c10d::NCCLGroupStart();
    for (int i = 0; i < WS - 1; ++i) {
        size_t rank_distance = i + 1;
        size_t dst_rank = (rank() + WS + rank_distance) % WS;
        size_t src_rank = (rank() + WS - rank_distance) % WS;
        // size_t src_p = (rank() + i + WS - 1) % WS;
        c10d::NCCLSend(*this, q_sum_buf[rank()], dst_rank);
        c10d::NCCLRecv(*this, q_sum_buf[src_rank], src_rank);
        c10d::NCCLSend(*this, scale_sum_buf[rank()], dst_rank);
        c10d::NCCLRecv(*this, scale_sum_buf[src_rank], src_rank);
    }
    c10d::NCCLGroupEnd();
//    c10d::NCCLGroupEndCheck(current_comm());

    static int fuse_dequant = utils::get_int_env("REDUCE_TP_INT8_FUSE_DEQUANT", 1);
    // step4: dequant
    if (fuse_dequant && current_layer() >= 0 && !dual_stream()) {
        int8_op::set_quant_scale(q_sum, scale_sum);
        *output = q_sum;
        return;
    }
    recordEvent("4.dequant", event_level);
    int8_op::dequant_group_32(*this, q_sum, scale_sum, output);
    BM_ASSERT_EQ(output->dtype(), out_type, "Output type mismatch");
}

void ModelContext::update_act_scale(const std::string& name, const Tensor& act) {
    Tensor x = functions::reduce_abs_max(*this, act);
    if (act_scale_map_.count(name) > 0) {
        x = functions::BinaryElementwiseOp(*this, functions::BinaryElementwiseOp::Max)
            .forward(*this, act_scale_map_[name], x);
    }
    act_scale_map_[name] = x;
}

void ModelContext::check_numeric(const core::Tensor& tensor) const {
    if (checking_numerics()) {
        functions::check_numeric(*this, tensor);
    }
}

#pragma GCC pop_options

}
