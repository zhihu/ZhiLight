#include "nn/block/block.h"
#include "nn/block/block_kernel.h"
#include "nn/attention/attention.h"
#include "nn/embedding/embedding.h"
#include "nn/feedforward/feedforward.h"
#include "nn/functions/functions.h"
#include "nn/quant/int8/quant_kernel.h"
#include "nn/layernorm/layernorm.h"
#include "nn/linear/linear.h"
#include "model/model_context.h"
#include "model/dyn_batch_context.h"
#include "utils/env.h"

#include <bmengine/core/core.h>
#include <bmengine/core/utils.h>
#include <bmengine/functions/element.h>
#include <bmengine/functions/arthmetic.h>
#include <bmengine/functions/tensor_ops.h>
#include <bmengine/functions/typecast.h>
#include <bmengine/logger/std_log_op.hpp>
#include "private/allocator.h"
#include <limits>

namespace nn {

using namespace bmengine;
using bmengine::core::DataType;
using bmengine::core::ScopeDevice;
using bmengine::core::Tensor;
using model::ModelContext;

class EncoderLayer::impl {
public:
    class CohereImpl;
    LayerNorm ln_attn, ln_ff;
    Attention attn;
    FeedForward ff;
    float scale;
    bool scale_residual;
    std::vector<bool> mask_modules;
    std::string prefix;
    int dev;
    bool parallel;
    core::DataType dtype;

    impl(
        const core::Context& ctx,
        model::ModelConfig cfg,
        model::QuantConfig quant_config,
        bool parallel)
        : ln_attn(ctx, cfg.dim_model, quant_config.fuse_ln_attn(), cfg.eps, 1.0, cfg.dtype),
          ln_ff(ctx, cfg.dim_model, quant_config.fuse_ln_ff(), cfg.eps, 1.0, cfg.dtype),
          attn(ctx, cfg, quant_config, parallel),
          ff(ctx, cfg, quant_config, parallel),
          scale(
              cfg.model_type == "cpm_dragonfly" ? sqrtf(float(cfg.num_layers)) / cfg.scale_depth
                                                : 1.0),
          scale_residual(cfg.model_type == "cpm_dragonfly" ? false : true),
          mask_modules(cfg.mask_modules[ctx.current_layer()]),
          parallel(parallel),
          dtype(cfg.dtype),
          dev(ctx.active_device_idx()) {}
    virtual ~impl() = default;

    core::Tensor slice_for_mla(ModelContext& ctx, const core::Tensor& input) {
        static int reduce_scatter_thres = utils::get_int_env("PROJ_A_DP_THRES", -1);
        static int data_parallel = utils::get_int_env("ATTN_DATA_PARALLEL", 0);
        size_t len_q = input.size(0);
        if (reduce_scatter_thres > 0 && len_q > reduce_scatter_thres) {
            BM_ASSERT_EQ(input.ndim(), 2, "slice_for_mla: input is not 2D");
            BM_ASSERT(ctx.dyn_batch(), "slice_for_mla: No dyn batch");
            ctx.dyn_batch()->input_len_before_dp = len_q;
            size_t num_s = ctx.dyn_batch()->get_num_search();
            size_t part_len = ceil_div(len_q, ctx.world_size());
            size_t begin = ctx.rank() * part_len;
            size_t end = std::min(len_q, begin + part_len);
            if (data_parallel == 2 && num_s > 0) {
                Tensor input_s = input.slice_dim0_len(input.size(0) - num_s, num_s);
                ctx.dyn_batch()->s_mla_input = ln_attn.forward(ctx, input_s);
            }
            return input.slice_dim0(begin, end);
        }
        return input;
    }

    virtual core::Tensor forward(
        const core::Context& ctx,
        const core::Tensor& inp,           // (batch, len_q, dim_model)
        const core::Tensor& mask,          // (batch, len_q, len_buf)
        const core::Tensor& position_bias, // if relative (batch, num_head, len_q, len_buf) else if
                                           // rotary (batch, len_q)
        const core::Tensor& seqlens_q,     // (batch)
        const core::Tensor& seqlens_kv,    // (batch)
        const core::Tensor* past_k,        // (batch, num_head, len_buf, dim_head)
        const core::Tensor* past_v,        // (batch, num_head, len_buf, dim_head)
        const core::Tensor* block_table,   // (batch, blocks_per_seq)
        const core::Tensor* placement      // (batch, len_q,)    int32
    ) {
        ModelContext* m_ctx = ModelContext::cast(ctx);
        core::Tensor ret;

        if (!mask_modules[0]) {
            Tensor inp1 = m_ctx ? slice_for_mla(*m_ctx, inp) : inp;
            auto ln_out = ln_attn.forward(ctx, inp1);
            if (m_ctx && m_ctx->is_calc_act_scales()) {
                m_ctx->update_act_scale(ln_attn.prefix + ".max_out", ln_out);
            }

            ret = attn(
                ctx,
                ln_out,
                mask,
                position_bias,
                seqlens_q,
                seqlens_kv,
                past_k,
                past_v,
                block_table,
                placement,
                nullptr);
            m_ctx->check_numeric(ret);
            BM_ASSERT_EQ(ret.dtype(), dtype, "dtype mismatch");
            if (parallel)
                ret = ctx.reduce_sum(ret, dtype);
            element_add_scale_out(ctx, inp, ret, ret, 1 / scale, scale_residual); // residual first
        } else {
            ret = inp;
        }

        if (!mask_modules[1]) {
            auto ln_out = ln_ff(ctx, ret);
            BM_ASSERT_EQ(ln_out.dtype(), dtype, "dtype mismatch");
            if (m_ctx && m_ctx->is_calc_act_scales()) {
                m_ctx->update_act_scale(ln_ff.prefix + ".max_out", ln_out);
            }
            Tensor ff_out = ff.forward(ctx, ln_out);
            m_ctx->check_numeric(ff_out);
            if (parallel)
                ff_out = ctx.reduce_sum(ff_out, dtype);
            element_add_scale_out(ctx, ret, ff_out, ret, 1 / scale, scale_residual); // residual first
        }
        return ret;
    }

    static Tensor single_stream_encode(
        model::ModelContext& ctx,
        const std::vector<EncoderLayer::impl*>& encoders,
        const core::Tensor& input,       // (grouped_len_q, dim_model)
        const core::Tensor& position     // (grouped_len_q)
    );
    static Tensor dual_stream_encode(
        model::ModelContext& ctx,
        const std::vector<EncoderLayer::impl*>& encoders,
        core::Tensor& input_g,             // (grouped_len_q, dim_model)
        const core::Tensor& position_g     // (grouped_len_q)
    );

    // Fuse c = a + b with layer norm
    Tensor add_fuse_ln(model::ModelContext& ctx, const Tensor& a, const Tensor& b, Tensor& c, LayerNorm& ln) {
        if (scale == 1.f && b.dtype() != core::DataType::kInt8) {
            return ln.fuse_add(ctx, a, b, c);
        } else {
            element_add_scale_out(ctx, a, b, c, 1 / scale, scale_residual);
            return ln.forward(ctx, c);
        }
    }
};

Tensor EncoderLayer::impl::single_stream_encode(
    model::ModelContext& ctx,
    const std::vector<EncoderLayer::impl*>& encoders,
    const core::Tensor& input,       // (grouped_len_q, dim_model)
    const core::Tensor& position     // (grouped_len_q)
) {
    int debug_layer = utils::get_int_env("CPM_DEBUG_LAYER", -1);
    int debug_layer_level = utils::get_int_env("CPM_DEBUG_LAYER_LEVEL", 2);

    Tensor hidden = input;
    Tensor ret;
    auto dtype = encoders[0]->dtype;
    for (int i = 0; i < encoders.size(); i++) {
        ctx.set_current_layer(i);
        int org_debug_level = ctx.debug();
        if (i == debug_layer)
            ctx.enable_debug(debug_layer_level);

        Tensor ln_out = encoders[i]->ln_attn(ctx, hidden);
        ret = encoders[i]->attn.dyn_rag_forward(ctx, ln_out, position);
        ret = ctx.reduce_sum(ret, dtype);
        element_add_scale_out(ctx, hidden, ret, ret, 1 / encoders[i]->scale, encoders[i]->scale_residual);

        ln_out = encoders[i]->ln_ff(ctx, ret);
        Tensor ff_out = encoders[i]->ff.forward(ctx, ln_out);
        ff_out = ctx.reduce_sum(ff_out, dtype);
        element_add_scale_out(ctx, ret, ff_out, ret, 1 / encoders[i]->scale, encoders[i]->scale_residual);

        hidden = ret;
        ctx.enable_debug(org_debug_level);
    }
    return ret;
}

using std::shared_ptr;
using std::vector;
Tensor EncoderLayer::impl::dual_stream_encode(
    model::ModelContext& ctx,
    const std::vector<EncoderLayer::impl*>& encoders,
    core::Tensor& input_g,             // (grouped_len_q, dim_model)
    const core::Tensor& position_g     // (grouped_len_q)
) {
    bool use_host_reducer = ctx.get_host_reducer().get();
    bool copy_only = utils::get_int_env("HOST_REDUCE_COPY_ONLY", 0) > 0 && ctx.world_size() == 2;
    static bool logged = false;
    if (!logged) {
        logged = true;
        std::cout << ">>> HOST_REDUCE: " << use_host_reducer << ", copy_only: " << copy_only << endl;
    }

    // Two streams: one compute; one reduce.
    auto main_stream = ctx.current_stream();
    cudaStream_t red_stream;
    int stream_priority = utils::get_int_env("DUAL_STREAM_PRIORITY", -1);
    BM_CUDART_ASSERT(cudaStreamCreateWithPriority(&red_stream, cudaStreamNonBlocking, stream_priority));
    auto reduce_stream = std::make_shared<core::Stream_>(red_stream, [&](cudaStream_t s) { cudaStreamDestroy(s); });

    BM_ASSERT(ctx.dyn_batch(), "");
    BM_ASSERT_EQ(input_g.ndim(), 2, "");
    ctx.set_dual_stream(true);
    auto org_dyn_batch = ctx.dyn_batch();

    int num_split = utils::get_int_env("DUAL_STREAM_NUM_SPLIT", 2);
    int split_round_up = utils::get_int_env("DUAL_STREAM_SPLIT_ROUND_UP", 16);
    int debug_layer = utils::get_int_env("CPM_DEBUG_LAYER", -1);
    int debug_layer_level = utils::get_int_env("CPM_DEBUG_LAYER_LEVEL", 2);
    int event_level = utils::get_int_env("CPM_DEBUG_LAYER_EV_LEVEL", debug_layer_level);
    auto dtype = encoders[0]->dtype;
    static int quant_reduce_thres = utils::get_int_env("REDUCE_TP_INT8_THRES", INT_MAX);
    bool enabled_chunk_prefill = utils::get_int_env("CHUNKED_PREFILL", 0) > 0;

    // Split dyn context and input to num_split parts.
    int part_size = round_up(int(input_g.size(0) + num_split - 1) / num_split, split_round_up);
    auto dyn_ctx = org_dyn_batch->split_encode(num_split, part_size);
    vector<Tensor> hidden = org_dyn_batch->split_tensor(input_g, num_split, part_size);
    vector<Tensor> position = org_dyn_batch->split_tensor(position_g, num_split, part_size);
    // global tensor for reduce, address should not change during allocator GC.
    core::DataType reduce_dtype = use_host_reducer && !copy_only ? core::DataType::kHalf : input_g.dtype();
    Tensor reduce_in_all = ctx.tensor(input_g.shape(), reduce_dtype);
    Tensor reduce_out_all = ctx.tensor(input_g.shape(), reduce_dtype);
    vector<Tensor> reduce_in = org_dyn_batch->split_tensor(reduce_in_all, num_split, part_size);
    vector<Tensor> reduce_out = org_dyn_batch->split_tensor(reduce_out_all, num_split, part_size);
    vector<Tensor> ret(num_split);
    ctx.mem_gc();

    bool reduce_need_mem = hidden[0].size(0) > quant_reduce_thres;
    if (reduce_need_mem) {
        size_t reduce_mem = round_up(hidden[0].nbytes(), 1024) * 5 + 64 * 1024 * 1024;
        ctx.reserve_cache_alloc(reduce_mem);
    }

    std::vector<cudaEvent_t> main_events(num_split);
    std::vector<cudaEvent_t> events(num_split);
    for (auto& ev: main_events) {
        BM_CUDART_ASSERT(cudaEventCreateWithFlags(&ev, cudaEventDefault));
    }
    for (auto& ev: events) {
        BM_CUDART_ASSERT(cudaEventCreateWithFlags(&ev, cudaEventDefault));
    }

    std::function<void(int, cudaEvent_t, bool)> reduce_fn = [&](int k, cudaEvent_t e, bool quant) {
        // BM_CUDART_ASSERT(cudaStreamSynchronize(ctx.current_stream()->ptr));
        BM_CUDART_ASSERT(cudaEventRecord(main_events[k], main_stream->ptr));

         // switch stream and allocator
        ctx.set_current_stream(reduce_stream);
        if (reduce_need_mem) ctx.use_cache_alloc(true);

        BM_CUDART_ASSERT(cudaStreamWaitEvent(reduce_stream->ptr, main_events[k]));
        {
            std::string ev_name = quant && reduce_need_mem ? "Int8ReducePart" : "AllReducePart";
            core::EventScope ev(ctx, ev_name + std::to_string(k), 1, reduce_out[k].nbytes());
            ctx.reduce_sum2(reduce_in[k], &reduce_out[k], dtype, quant);
        }
        BM_CUDART_ASSERT(cudaEventRecord(e, ctx.current_stream()->ptr));

        // switch back stream and allocator
        if (reduce_need_mem) ctx.use_cache_alloc(false);
        ctx.set_current_stream(main_stream);
    };
    // Host reduce
    core::Stream reduce_stream2 = ctx.get_reducer_stream();
    shared_ptr<model::HostAllReducer> host_reducer = ctx.get_host_reducer();
    auto reducer_thread = ctx.get_reducer_thread();
    std::promise<void> reduce_promise[10];
    for (int k = 0; k < num_split; ++k) {
        reduce_promise[k].set_value();
    }
    volatile bool ready[10];
    auto host_reduce_fn = [&](int k, cudaEvent_t e, bool quant) {
        reduce_promise[k] = std::promise<void>(); // reset promise k
        ready[k] = false;
        BM_CUDART_ASSERT(cudaEventRecord(main_events[k], main_stream->ptr));
        BM_CUDART_ASSERT(cudaStreamWaitEvent(reduce_stream->ptr, main_events[k]));
        auto async_fn = [&, k, e]() {
            try {
                host_reducer->reduce_sum_async(
                    ctx.rank(), ctx.current_layer(),
                    reduce_in[k], reduce_out[k],
                    reduce_stream->ptr, reduce_stream2->ptr,
                    copy_only); // syn internal if copy_only
                if (!copy_only)
                    BM_CUDART_ASSERT(cudaStreamSynchronize(reduce_stream2->ptr));
                // BM_CUDART_ASSERT(cudaEventRecord(e, reduce_stream2->ptr));
                reduce_promise[k].set_value();
                ready[k] = true;
            } catch(...) {
                reduce_promise[k].set_exception(std::current_exception());
            }
        };
        reducer_thread->run(async_fn);
    };
    if (use_host_reducer) {
        reduce_fn = std::move(host_reduce_fn);
    }
    auto wait_reduce_fn = [&](int k) {
        if (use_host_reducer) {
            reduce_promise[k].get_future().get();
        } else {
            BM_CUDART_ASSERT(cudaEventSynchronize(events[k]));
        }
    };
    auto post_reduce_fn = [&](int k) -> Tensor {
        if (use_host_reducer) {
            if (!copy_only) {
                return functions::typecast(ctx, reduce_out[k], dtype);
            } else {
                functions::BinaryElementwiseOp add_op(ctx, functions::BinaryElementwiseOp::Add);
                return add_op.forward(ctx, reduce_in[k], reduce_out[k]);
            }
        } else {
            return reduce_out[k];
        }
    };
    auto is_ready_fn = [&](int k) -> bool {
        if (use_host_reducer)
            return ready[k];
        return cudaSuccess == cudaEventQuery(events[k]);
    };

    Tensor ret_g = ctx.tensor(input_g.shape(), input_g.dtype());
    ret = org_dyn_batch->split_tensor(ret_g, num_split, part_size);

    // for quantized kv cache attention
    shared_ptr<Tensor> unquant_key_buf;
    shared_ptr<Tensor> unquant_val_buf;

    for (size_t i = 0; i < encoders.size(); i++) {
        ctx.set_current_layer(i);
        int org_debug_level = ctx.debug();
        if (ctx.is_layer(debug_layer, 0)) {
            ctx.enable_debug(debug_layer_level);
            ctx.set_event_level(event_level);
        }
        size_t M = input_g.numel() / input_g.size(-1);
        auto event_scope = std::make_unique<core::EventScope>(ctx, logger::str_cat("EncoderLayer[M=", M, "]"), 1);
        if (!enabled_chunk_prefill) {
            unquant_key_buf = std::make_shared<Tensor>();
            unquant_val_buf = std::make_shared<Tensor>();
        }
        for (int k = 0; k < num_split; ++k) {
            Tensor ln_attn_out;
            if (i > 0) {
                wait_reduce_fn(k);
                Tensor a = post_reduce_fn(k); // reduce_out[k], reduced sum of previous layer
                ln_attn_out = encoders[i]->add_fuse_ln(ctx, ret[k], a, ret[k], encoders[i]->ln_attn);
                hidden[k] = ret[k];
            } else {
                ln_attn_out = encoders[i]->ln_attn.forward(ctx, hidden[k]);
            }
            dyn_ctx[k]->unquant_key_buf = unquant_key_buf;
            dyn_ctx[k]->unquant_val_buf = unquant_val_buf;
            ctx.set_dyn_batch(dyn_ctx[k]);
            Tensor attn_out = encoders[i]->attn.dyn_rag_forward(ctx, ln_attn_out, position[k]);
            if (use_host_reducer && !copy_only && attn_out.dtype() != DataType::kHalf)
                attn_out = functions::typecast(ctx, attn_out, DataType::kHalf);
            ctx.copy2(attn_out, &reduce_in[k]);
            reduce_fn(k, events[k], ctx.world_size() == 8 || k == 0);
        }
        ctx.layer_cache().clear();
        {
            static bool dequant_cache_weight = utils::get_int_env("NEED_DEQUANT_WEIGHT", 0) > 0;
            // BM_CUDART_ASSERT(cudaStreamSynchronize(ctx.current_stream()->ptr));
            if (dequant_cache_weight && !is_ready_fn(0)) {
                core::EventScope ev_dequant(ctx, "dequant_cache_weight", 3);
                encoders[i]->ff.dequant_cache_weight(ctx, hidden[0]);
            }
        }

        for (int k = 0; k < num_split; ++k) {
            wait_reduce_fn(k);
            Tensor a = post_reduce_fn(k); // reduce_out[k]
            ctx.set_dyn_batch(dyn_ctx[k]);
            // ret[k] = ctx.tensor(hidden[k].shape(), hidden[k].dtype());
            Tensor ln_ff_out = encoders[i]->add_fuse_ln(ctx, hidden[k], a, ret[k], encoders[i]->ln_ff);
            hidden[k] = Tensor();
            Tensor ff_out = encoders[i]->ff.forward(ctx, ln_ff_out);
            if (use_host_reducer && !copy_only && ff_out.dtype() != DataType::kHalf)
                ff_out = functions::typecast(ctx, ff_out, DataType::kHalf);
            ctx.copy2(ff_out, &reduce_in[k]);
            reduce_fn(k, events[k], ctx.world_size() == 8 || k == num_split - 1);
        }
        ctx.layer_cache().clear();

        event_scope.reset();
        ctx.enable_debug(org_debug_level);
        ctx.set_event_level(-1);
    }

    size_t last_layer = encoders.size() - 1;
    for (int k = 0; k < num_split; ++k) {
        wait_reduce_fn(k);
        Tensor a = post_reduce_fn(k);
        element_add_scale_out(ctx, ret[k], a, ret[k], 1 / encoders[last_layer]->scale,
                              encoders[last_layer]->scale_residual);
        hidden[k] = ret[k];
    }

    // ctx.set_dyn_aux(std::shared_ptr<model::DynBatchContext>()); // reset
    if (reduce_need_mem) {
        ctx.free_cache_alloc();
    }
    for (auto& ev: main_events) {
        cudaEventDestroy(ev);
    }
    for (auto& ev: events) {
        cudaEventDestroy(ev);
    }
    ctx.set_dual_stream(false);
    ctx.set_dyn_batch(org_dyn_batch);
    return ret_g;
    // return *hidden.rbegin();
}

class EncoderLayer::impl::CohereImpl : public EncoderLayer::impl {
public:
    CohereImpl(
        const core::Context& ctx,
        model::ModelConfig config,
        model::QuantConfig quant_config,
        bool parallel)
        : EncoderLayer::impl(
            ctx,
            config,
            quant_config,
            parallel) {
        ln_attn.set_rms(false);
    }

    core::Tensor dyn_forward(
        const core::Context& ctx,
        const core::Tensor& input,           // (batch, len_q, dim_model)
        const core::Tensor& position)  {
        ModelContext* m_ctx = ModelContext::cast(ctx);

        const Tensor& residual = input;
        Tensor hidden_states = ln_attn(ctx, input);
//        std::cout << "input: " << input << endl;
//        std::cout << "hidden_states: " << hidden_states << endl;
        Tensor attn_out = attn.dyn_rag_forward(*m_ctx, hidden_states, position);
        Tensor mlp_out = ff.forward(ctx, hidden_states);

        // Add everything together
        functions::BinaryElementwiseOp add_op(ctx, functions::BinaryElementwiseOp::Add);
        Tensor ret = add_op.forward(ctx, attn_out, mlp_out);
        if (parallel)
            ret = ctx.reduce_sum(ret, dtype);
        add_op.inplace(ctx, ret, residual);
        return ret;
    }
};

EncoderLayer::EncoderLayer(
    const core::Context& ctx,
    model::ModelConfig config,
    model::QuantConfig quant_config,
    bool parallel) : core::Layer(), parallel(parallel) {
    bool is_cohere = config.model_type == "cohere";
    if (is_cohere) {
        pimpl.reset(new impl::CohereImpl(
            ctx,
            config,
            quant_config,
            parallel));
    } else {
        pimpl.reset(new impl(
            ctx,
            config,
            quant_config,
            parallel));
    }
    this->layer_id = ctx.current_layer();
    this->dev = ctx.active_device_idx();
    this->output_dev = dev;

    auto mask_modules = config.mask_modules[this->layer_id];
    if (!mask_modules[0]) {
        add_submodule("ln_attn", pimpl->ln_attn);
        add_submodule("attn", pimpl->attn);
    }
    if (!mask_modules[1]) {
        if (!is_cohere)
            add_submodule("ln_ff", pimpl->ln_ff);
        add_submodule("ff", pimpl->ff);
    }
}

void EncoderLayer::set_mask_modules(const std::vector<bool>& mask_modules) {
    pimpl->mask_modules = mask_modules;
}

core::Tensor EncoderLayer::forward(
    const core::Context& ctx,
    const core::Tensor& inp,           // (batch, len_q, dim_model)
    const core::Tensor& mask,          // (batch, len_q, len_buf)
    const core::Tensor& position_bias, // if relative (batch, num_head, len_q, len_buf) else if
                                       // rotary (batch, len_q)
    const core::Tensor& seqlens_q,     // (batch)
    const core::Tensor& seqlens_kv,    // (batch)
    const core::Tensor* past_k,        // (batch, num_head, len_buf, dim_head)
    const core::Tensor* past_v,        // (batch, num_head, len_buf, dim_head)
    const core::Tensor* block_table,   // (batch, blocks_per_seq)
    const core::Tensor* placement      // (batch, len_q,)    int32
) {
    size_t M = inp.numel() / inp.size(-1);
    core::EventScope event_scope(ctx, logger::str_cat("EncoderLayer[M=", M, "]"), 1);
    {
        bool switched = ctx.switch_to_device(dev);
        if (switched && ctx.debug() >= 2) {
            std::cerr << "EncoderLayer[" << layer_id << "]::forward() switch to device " << dev
                      << std::endl;
        }
    }
    // copy and cache to current layer's device, if necessary
    const core::Tensor* p_input = ctx.identity(&inp, "EncoderInput");
    const core::Tensor* p_mask = ctx.identity(&mask, "EncoderMask");
    const core::Tensor* p_pos_bias = ctx.identity(&position_bias, "EncoderPosBias");
    const core::Tensor* p_placement = ctx.identity(placement, "EncoderPosBias");
    const core::Tensor* p_seqlens_q = ctx.identity(&seqlens_q, "EncoderQSeqLens");
    const core::Tensor* p_seqlens_kv = ctx.identity(&seqlens_kv, "EncoderKVSeqLens");

    impl::CohereImpl* cohere = dynamic_cast<impl::CohereImpl*>(pimpl.get());
    if (cohere) {
        return cohere->dyn_forward(ctx, *p_input, *p_pos_bias);
    }
    core::Tensor tensor = pimpl->forward(
        ctx,
        *p_input,
        *p_mask,
        *p_pos_bias,
        *p_seqlens_q,
        *p_seqlens_kv,
        past_k,
        past_v,
        block_table,
        p_placement);

    if (output_dev != dev) {
        // at last layer, switch back to device 0, and copy output
        ctx.switch_to_device(output_dev);
        if (ctx.debug() >= 2) {
            std::cerr << "EncoderLayer[" << layer_id
                      << "]::forward(), last layer, switch back device to " << output_dev
                      << std::endl;
        }
        tensor = *ctx.identity(&tensor, "EncoderFinalOutput");
    }
    return std::move(tensor);
}

EncoderLayer::~EncoderLayer() = default;

void EncoderLayer::load_state_dict(
    const core::Context& ctx,
    const std::map<std::string, const core::Tensor>& state_dict,
    const std::string& prefix,
    bool allow_missing) {
    ScopeDevice scope_device(ctx, dev);

    using BinaryOp = bmengine::functions::BinaryElementwiseOp;
    BinaryOp mul_op(ctx, BinaryOp::Mul);
    BinaryOp div_op(ctx, BinaryOp::Div);

    ModelContext* m_ctx = dynamic_cast<ModelContext*>(const_cast<core::Context*>(&ctx));

    pimpl->prefix = prefix;
    if (m_ctx && m_ctx->smooth_quant_alpha() > 0) {
        BM_ASSERT(m_ctx != nullptr, "Not ModelContext.");
        float alpha = m_ctx->smooth_quant_alpha();
        BM_ASSERT(1 > alpha, "wrong alpha");
        std::map<std::string, core::Tensor> state_dict_tmp =
            reinterpret_cast<const std::map<std::string, core::Tensor>&>(state_dict);
        {
            // attention
            ScopeDevice scope_dev(ctx, pimpl->dev);
            auto name_ln = prefix + ".ln_attn.weight";
            auto name_q = prefix + ".attn.project_q.weight";
            auto name_k = prefix + ".attn.project_k.weight";
            auto name_v = prefix + ".attn.project_v.weight";
            Tensor w_ln = ctx.cuda(state_dict.at(name_ln));  // (dim_model)
            Tensor w_q = ctx.cuda(state_dict.at(name_q));  // (dim_out, dim_model)
            Tensor w_k = ctx.cuda(state_dict.at(name_k));
            Tensor w_v = ctx.cuda(state_dict.at(name_v));

            Tensor scale_ln = m_ctx->get_act_scale_map().at(prefix + ".ln_attn.max_out");
            scale_ln = ctx.cuda(scale_ln);

            Tensor scale_q = functions::reduce_abs_max(ctx, w_q);
            Tensor scale_k = functions::reduce_abs_max(ctx, w_k);
            Tensor scale_v = functions::reduce_abs_max(ctx, w_v);
            Tensor qkv = functions::stack_tensor(ctx, {scale_q, scale_k, scale_v});
            Tensor scale_qkv = functions::reduce_abs_max(ctx, qkv);

            scale_qkv = functions::clamp(ctx, scale_qkv, 1e-5, std::numeric_limits<float>::max());

            Tensor scale_ln_pow = functions::pow(ctx, scale_ln, alpha);
            Tensor scale_qkv_pow = functions::pow(ctx, scale_qkv, (1 - alpha));
            Tensor scale_s = div_op.forward(ctx, scale_ln_pow, scale_qkv_pow);
            scale_s = functions::clamp(ctx, scale_s, m_ctx->smooth_quant_min_scale(), m_ctx->smooth_quant_max_scale());

            w_ln = div_op.forward(ctx, w_ln, scale_s);
            w_q = mul_op.broadcast_y(ctx, w_q, scale_s);
            w_k = mul_op.broadcast_y(ctx, w_k, scale_s);
            w_v = mul_op.broadcast_y(ctx, w_v, scale_s);

            state_dict_tmp[name_ln] = w_ln;
            state_dict_tmp[name_q] = w_q;
            state_dict_tmp[name_k] = w_k;
            state_dict_tmp[name_v] = w_v;
        }
        {
            // ff
            ScopeDevice scope_dev(ctx, pimpl->dev);
            auto name_ln = prefix + ".ln_ff.weight";
            auto name_in1 = prefix + ".ff.w_in.weight";
            auto name_in2 = prefix + ".ff.w_gated.weight";
            Tensor w_ln = ctx.cuda(state_dict.at(name_ln));  // (dim_model)
            Tensor w_in1 = ctx.cuda(state_dict.at(name_in1));  // (dim_out, dim_model)
            Tensor w_in2 = ctx.cuda(state_dict.at(name_in2));  // (dim_out, dim_model)

            Tensor scale_ln = m_ctx->get_act_scale_map().at(prefix + ".ln_ff.max_out");
            scale_ln = ctx.cuda(scale_ln);

            Tensor scale_in1 = functions::reduce_abs_max(ctx, w_in1);
            Tensor scale_in2 = functions::reduce_abs_max(ctx, w_in2);
            Tensor stacked = functions::stack_tensor(ctx, {scale_in1, scale_in2});
            Tensor scale_in = functions::reduce_abs_max(ctx, stacked);

            scale_in = functions::clamp(ctx, scale_in, 1e-5, std::numeric_limits<float>::max());

            Tensor scale_ln_pow = functions::pow(ctx, scale_ln, alpha);
            Tensor scale_in_pow = functions::pow(ctx, scale_in, (1 - alpha));
            Tensor scale_s = div_op.forward(ctx, scale_ln_pow, scale_in_pow);
            scale_s = functions::clamp(ctx, scale_s, m_ctx->smooth_quant_min_scale(), m_ctx->smooth_quant_max_scale());

            w_ln = div_op.forward(ctx, w_ln, scale_s);
            w_in1 = mul_op.broadcast_y(ctx, w_in1, scale_s);
            w_in2 = mul_op.broadcast_y(ctx, w_in2, scale_s);

            state_dict_tmp[name_ln] = w_ln;
            state_dict_tmp[name_in1] = w_in1;
            state_dict_tmp[name_in2] = w_in2;
        }
        auto& dict1 = reinterpret_cast<std::map<std::string, const core::Tensor>&>(state_dict_tmp);
        core::Layer::load_state_dict(ctx, dict1, prefix, allow_missing);
    } else {
        core::Layer::load_state_dict(ctx, state_dict, prefix, allow_missing);
        int freeze_mem = utils::get_int_env("FREEZE_MEM_EACH_LAYER", 0);
        if (freeze_mem) {
            ctx.get_allocator()->freeze_model_memory();
        }
    }
}

core::Tensor EncoderLayer::dual_stream_encode(
    model::ModelContext& ctx,
    functions::ModuleList<EncoderLayer>& encoders,
    core::Tensor& input,           // (grouped_len_q, dim_model)
    const core::Tensor& position   // (grouped_len_q)
) {
    std::vector<EncoderLayer::impl*> impls;
    for (size_t i = 0; i < encoders.size(); i++) {
        impls.push_back(encoders[i].pimpl.get());
    }
    return EncoderLayer::impl::dual_stream_encode(ctx, impls, input, position);
}

}
