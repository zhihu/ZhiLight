#pragma once

#include "model/model_config.hpp"
#include <bmengine/core/context.h>
#include <bmengine/core/engine.h>
#include <bmengine/core/thread_pool.h>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <future>
#include <queue>
#include <thread>
#include "model/buffer_context.h"
#include "kvcache/kvcache.h"

namespace model {

using bmengine::core::DataType;
using bmengine::core::Engine;
using bmengine::core::Tensor;
using bmengine::core::WithDevice;
using kvcache::KVCache;
using kvcache::KVCacheConfig;

class ModelBase;
struct DynBatchConfig;
struct DynBatchContext;
class ModelContext;
class RagBufferContext;

class HostAllReducer {
public:
    HostAllReducer() = default;
    virtual ~HostAllReducer() = default;

    virtual Tensor reduce_sum(int rank, int layer, Tensor& data) = 0;
    virtual Tensor reduce_sum_async(int rank, int layer, Tensor& data, Tensor& out, cudaStream_t is, cudaStream_t os, bool copy_only=false) = 0;
};

struct ReduceContext {
    std::vector<ModelContext*> peers_;
    std::queue<std::pair<ReduceContext*, Tensor>> peer_buffers_;
    std::mutex buf_mutex_;
    std::condition_variable buf_cond_;

    volatile int count_;
    std::mutex done_mutex_;
    std::condition_variable done_cond_;

    void push_buffer(std::pair<ReduceContext*, Tensor>& buf);
    std::pair<ReduceContext*, Tensor> pop_buffer();

    void begin();
    void end();
    void wait_peer();
};

class WithBuffer {
private:
    ModelContext* ctx;
    std::shared_ptr<BufferContext> buf_ctx_;

public:
    WithBuffer(ModelContext& ctx, std::shared_ptr<BufferContext> gen_ctx);
    ~WithBuffer();
    WithBuffer(const WithBuffer&) = delete;
    WithBuffer& operator=(const WithBuffer&) = delete;
    WithBuffer(WithBuffer&&);
    WithBuffer& operator=(WithBuffer&&) = delete;
};

/**
 * Extend Context to hold more info for LLM model inference
 */
class ModelContext : public bmengine::core::Context {
public:
    const ModelConfig cfg;
private:
    const ModelBase& model_;
    bool parallel_;

    std::vector<int> layer_devices;
    std::shared_ptr<BufferContext> buf_ctx_;

    std::shared_ptr<DynBatchContext> dyn_batch_;
    std::shared_ptr<RagBufferContext> rag_buffer_;

    std::shared_ptr<ReduceContext> reducer_;
    std::shared_ptr<HostAllReducer> host_reducer_;
    std::shared_ptr<bmengine::core::TaskThreadPool> reducer_thread_;
    core::Stream reducer_stream_;

    float smooth_quant_alpha_ { -1 };
    float smooth_quant_min_scale_ { 1e-5 };
    float smooth_quant_max_scale_ { 1e5 };
    bool calc_act_scales_ { false };
    std::map<std::string, Tensor> act_scale_map_;

    std::map<std::string, Tensor> layer_cache_;
    bool dual_stream_ { false };
    bool latent_cache_ { false };

public:
    ModelContext(
        bmengine::core::Context&& ctx,
        const ModelBase& md,
        int batch_size = 1,
        bool parallel = false,
        bool BSHD = false);
    ~ModelContext() override;
    ModelContext(ModelContext&&) = default;

    static ModelContext* cast(const core::Context& ctx) {
        return dynamic_cast<ModelContext*>(const_cast<core::Context*>(&ctx));
    }

    bool is_parallel() { return parallel_; }

    std::shared_ptr<BufferContext> buffer_context() const { return buf_ctx_; }
    void switch_buffer(std::shared_ptr<BufferContext> gen_ctx) { buf_ctx_ = gen_ctx; }

    WithBuffer with_buffer(std::shared_ptr<BufferContext> buf_ctx) {
        return std::move(WithBuffer(*this, buf_ctx));
    };

    KVCache* buf_k() { return buf_ctx_->buf_k(); }
    KVCache* buf_v() { return buf_ctx_->buf_v(); }
    Tensor* buf_k(size_t layer) { return buf_ctx_ == nullptr ? nullptr : buf_ctx_->buf_k(layer); };
    Tensor* buf_v(size_t layer) { return buf_ctx_ == nullptr ? nullptr : buf_ctx_->buf_v(layer); };

    size_t add_queries(std::vector<std::vector<int32_t>> query_tokens) {
        return buf_ctx_ == nullptr ? 0 : buf_ctx_->add_queries(*this, query_tokens);
    }
    size_t add_sequence(const std::vector<int32_t>& prompt_tokens) {
        return buf_ctx_ == nullptr ? 0 : buf_ctx_->add_sequence(*this, prompt_tokens);
    }
    const Tensor* block_table(size_t layer) {
        return buf_ctx_ == nullptr ? nullptr : buf_ctx_->block_table(layer);
    };

    size_t kvcache_len() { return buf_ctx_->kvcache_len(); }
    void resize_transformer_buf(size_t new_length) {
        return buf_ctx_->resize_transformer_buf(*this, new_length);
    }

    // For ragged buffer: each task has a different buffer length.
    void resize_task_buf(int b, size_t new_length);
    void free_task_buf(int b);

    static ModelContext create(
        Engine& engine,
        const ModelBase& md,
        const DynBatchConfig& batch_config,
        int dev,
        bool parallel);

    std::shared_ptr<DynBatchContext> dyn_batch() const { return dyn_batch_; }
    void set_dyn_batch(const std::shared_ptr<DynBatchContext>& ptr) { dyn_batch_ = ptr; }

    std::shared_ptr<RagBufferContext> rag_buffer() { return rag_buffer_; }
    void set_rag_buffer(const std::shared_ptr<RagBufferContext>& buffer) { rag_buffer_ = buffer; }

    ReduceContext* reducer() { return reducer_.get(); }

    Tensor copy_peer(const Tensor& src);
    void copy2(const Tensor& src, Tensor* dst);

    Tensor reduce_sum(Tensor& data, DataType out_type) const;
    void reduce_sum2(const Tensor& data, Tensor* out, DataType out_type, bool quant = true) const;
    Tensor all_gather(const Tensor& data) const;

    HostAllReducer* create_host_reducer();
    void set_host_reducer(std::shared_ptr<HostAllReducer> reducer);
    std::shared_ptr<HostAllReducer> get_host_reducer() {
        return host_reducer_;
    }
    std::shared_ptr<bmengine::core::TaskThreadPool> get_reducer_thread() {
        return reducer_thread_;
    }
    core::Stream get_reducer_stream() { return reducer_stream_; }

    float smooth_quant_alpha() const { return smooth_quant_alpha_; }
    float smooth_quant_min_scale() const { return smooth_quant_min_scale_; }
    float smooth_quant_max_scale() const { return smooth_quant_max_scale_; }

    void set_smooth_quant(
        const std::map<std::string, Tensor>& act_scale_map,
        float alpha,
        float min_scale,
        float max_scale) {
        act_scale_map_ = act_scale_map;
        smooth_quant_alpha_ = alpha;
        smooth_quant_min_scale_ = min_scale;
        smooth_quant_max_scale_ = max_scale;
    }

    bool is_calc_act_scales() const { return calc_act_scales_; }

    void set_calc_act_scales(bool calc_act_scales) { calc_act_scales_ = calc_act_scales; }

    void update_act_scale(const std::string& name, const Tensor& act);

    const std::map<std::string, Tensor>& get_act_scale_map() const { return act_scale_map_; }

    void set_current_layer(int i) { Context::set_current_layer(i); layer_cache_.clear(); }
    std::map<std::string, Tensor>& layer_cache() { return layer_cache_; }

    bool dual_stream() const { return dual_stream_; }
    void set_dual_stream(bool b) { dual_stream_ = b; }

    bool latent_cache() const { return latent_cache_; }

    int num_layers() const { return layer_devices.size(); }

    void check_numeric(const core::Tensor& tensor) const;

private:
    void reduce_tp_int8(const Tensor& data, DataType out_type, Tensor* output) const;
    KVCacheConfig get_kv_cache_config();
};
}
