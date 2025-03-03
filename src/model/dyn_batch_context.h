#pragma once
#include "model/model_context.h"
#include "bmengine/core/tensor.h"
#include <memory>
#include <numeric>
#include <vector>
#include "kvcache/transformer_buffer.h"
#include "utils/ts_queue.hpp"

namespace model {

using bmengine::core::Tensor;
using kvcache::TransformerBuffer;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

static inline bool is_power_of_2(int x) {
    return (x > 0) && !(x & (x - 1));
}

#define CHECK_IS_POWER_OF_2(x)                                                                   \
    do {                                                                                         \
        if (!is_power_of_2(x)) {                                                                 \
            throw std::invalid_argument(#x "=" + std::to_string(x) + " is not power of 2");      \
        }                                                                                        \
    } while (0)

struct DynBatchConfig {
    int max_batch { 20 };
    int max_beam_size { 1 * 8 }; // set to n * 8 for best performance
    int task_queue_size { 8 };
    int max_total_token { 4096 };  // input and output tokens
    int chunk_size { 256 };
    int max_chunk { 20 };
    int seed { 0 };
    int eos_id { 0 };
    int bos_id { 6 };
    int unk_id { 1 };
    int first_batch { 1 };
    int nccl { -1 };
    bool rag_buffer { false };
    bool ignore_eos { false };
    bool keep_eos { false };
    int reserved_work_mem_mb { 1024 };
    int high_precision { 0 };
    bool flash_attention { false };
    bool enable_prompt_caching { false };
};

struct AuxInfo {
    cudaEvent_t e;
};

typedef utils::TSQueue<AuxInfo> AuxInfoQueue;
class RagBufferContext;

struct RopeCache {
    Tensor cos;
    Tensor sin;
    void clear() {
        cos = Tensor();
        sin = Tensor();
    }
};

struct DynBatchContext {
    RopeCache rope_cache;
    // search
    Tensor s_token;
    Tensor s_sub;
    Tensor s_placement;
    Tensor s_position;
    Tensor s_mask;
    vector<Tensor> s_position_buckets; // for CPMBee ragged buffer
    vector<Tensor> s_position_biases;  // for CPMBee ragged buffer
    vector<int> sv_len_buf;  // for ragged buffer
    Tensor s_len_buf;  // for ragged buffer
    // encode
    Tensor e_token;
    Tensor e_sub;
    Tensor e_placement;
    Tensor e_position;
    Tensor e_mask;
    vector<Tensor> e_position_buckets; // for CPMBee
    vector<Tensor> e_position_biases;  // for CPMBee

    vector<int> ev_batch;
    Tensor e_batch;
    vector<int> ev_input_len;
    vector<int> full_input_len;
    Tensor e_input_len;
    vector<int> ev_len_buf;

    int debug_batch { -1 };

    std::vector<void *> host_position_bias_addresses;
    Tensor position_bias_addresses;

    Tensor cu_q_seqlens; // for FlashDecoding
    Tensor cu_k_seqlens; // for FlashDecoding
    size_t total_k {0};
    int max_q_seqlen { 0 };
    int max_k_seqlen { 0 };

    shared_ptr<Tensor> unquant_key_buf;
    shared_ptr<Tensor> unquant_val_buf;
    int input_len_no_split;

    Tensor get_position_bias_addresses(ModelContext &ctx) {
        std::vector<void *> pb_addrs;
        for (size_t i = 0; i < s_position_biases.size(); ++i) {
            pb_addrs.push_back(s_position_biases[i].data());
        }
        if (host_position_bias_addresses != pb_addrs) {
            host_position_bias_addresses = pb_addrs;
            position_bias_addresses = ctx.tensor({pb_addrs.size()}, bmengine::core::DataType::kDouble);
            position_bias_addresses.from_buffer(pb_addrs.data());
        }
        return position_bias_addresses;
    }

    void copy_from(ModelContext &ctx, const DynBatchContext& other) {
        s_token = ctx.copy_peer(other.s_token);
        s_sub = ctx.copy_peer(other.s_sub);
        s_placement = ctx.copy_peer(other.s_placement);
        s_position = ctx.copy_peer(other.s_position);
        s_mask = ctx.copy_peer(other.s_mask);
        sv_len_buf = other.sv_len_buf;
        s_len_buf = ctx.copy_peer(other.s_len_buf);

        e_token = ctx.copy_peer(other.e_token);
        e_sub = ctx.copy_peer(other.e_sub);
        e_placement = ctx.copy_peer(other.e_placement);
        e_position = ctx.copy_peer(other.e_position);
        e_mask = ctx.copy_peer(other.e_mask);
        e_batch = ctx.copy_peer(other.e_batch);

        ev_batch = other.ev_batch;
        ev_input_len = other.ev_input_len;
        ev_len_buf = other.ev_len_buf;

        e_position_buckets.resize(other.e_position_buckets.size());
        for (int i = 0; i < e_position_buckets.size(); ++i) {
            e_position_buckets[i] = ctx.copy_peer(other.e_position_buckets[i]);
        }
        s_position_buckets.resize(other.s_position_buckets.size());
        for (int i = 0; i < s_position_buckets.size(); ++i) {
            s_position_buckets[i] = ctx.copy_peer(other.s_position_buckets[i]);
        }

        cu_q_seqlens = ctx.copy_peer(other.cu_q_seqlens);
        cu_k_seqlens = ctx.copy_peer(other.cu_k_seqlens);
    }

    Tensor encode_mask(ModelContext &ctx, int b) {
        int offset = 0;
        for (int i = 0; i < b; ++i) {
            offset += ev_input_len[i] * ev_len_buf[i];
        }
        return ctx.identity(&e_mask, "e_mask")
            ->slice_dim0(offset, offset + ev_input_len[b] * ev_len_buf[b])
            .view({ size_t(ev_input_len[b]), size_t(ev_len_buf[b]) });
    }
    // for ragged buffer
    Tensor search_mask(ModelContext &ctx, int b, size_t len_q) {
        int offset = 0;
        for (int i = 0; i < b; ++i) {
            offset += len_q * sv_len_buf[i];
        }
        return ctx.identity(&s_mask, "s_mask")
            ->slice_dim0(offset, offset + len_q * sv_len_buf[b])
            .view({ len_q, size_t(sv_len_buf[b]) });
    }

    Tensor e_position_bias(ModelContext &ctx, int b) {
        return *ctx.identity(&e_position_biases[b], "e_position_bias");
    }
    Tensor s_position_bias(ModelContext &ctx, int b) {
//        size_t num_heads = pos_bias.size(0);
//        int offset = 0;
//        for (int i = 0; i < b; ++i) {
//            offset += len_q * sv_len_buf[i];
//        }
//        return pos_bias.view({pos_bias.numel()}).slice_dim0_len(num_heads * offset, num_heads * len_q * sv_len_buf[b])
//            .view({ num_heads, len_q, size_t(sv_len_buf[b]) });
        return *ctx.identity(&s_position_biases[b], "s_position_bias");
    }

    void set_search(
        const Tensor& token_ids,
        const Tensor& token_sub,
        const Tensor& placement,
        const Tensor& position,
        const Tensor& mask) {
        s_token = token_ids;
        s_sub = token_sub;
        s_placement = placement;
        s_position = position;
        s_mask = mask;
    }

    void set_encode(
        const Tensor& token_ids,
        const Tensor& token_sub,
        const Tensor& placement,
        const Tensor& position,
        const Tensor& mask) {
        e_token = token_ids;
        e_sub = token_sub;
        e_placement = placement;
        e_position = position;
        e_mask = mask;
    }

    void set_encode_batch(
        const vector<int>& v_batch,
        const Tensor& batch) {
        this->ev_batch = v_batch;
        this->e_batch = batch;
    }

    void set_encode_len(
        const vector<int>& v_input_len,
        const vector<int>& full_input_lens,
        const Tensor& input_len) {
        this->ev_input_len = v_input_len;
        this->full_input_len = full_input_lens;
        this->e_input_len = input_len;
    }

    void clear_encode() {
        set_encode(Tensor(), Tensor(), Tensor(), Tensor(), Tensor());
        set_encode_batch(vector<int>(), Tensor());
        set_encode_len({}, {}, Tensor());
        ev_len_buf.clear();
    }

    vector<Tensor> split_tensor(const Tensor& tensor, int num_split, int p_size0) {
        vector<Tensor> parts;
        int total_size = int(tensor.size(0));
        for (int i = 0; i < num_split; ++i) {
            int from = i * p_size0;
            int to = std::min(from + p_size0, total_size);
            parts.push_back(tensor.slice_dim0(from, to));
        }
        return parts;
    }

    vector<std::shared_ptr<DynBatchContext>> split_encode(int num_split, int p_size0) {
        vector<std::shared_ptr<DynBatchContext>> parts;
        BM_ASSERT_EQ(full_input_len.size(), 1, "");
        int full_len = full_input_len[0];
        int total_size = int(e_token.size(0));
        BM_ASSERT_LE(total_size, full_len, "");
        int full_offset = full_len - total_size;
        for (int i = 0; i < num_split; ++i) {
            std::shared_ptr<DynBatchContext> part = std::make_shared<DynBatchContext>();
            int from = i * p_size0;
            int to = std::min(from + p_size0, total_size);
            int p_size = to - from;
            part->set_encode(
                e_token.slice_dim0(from, to),
                Tensor(),
                e_placement.slice_dim0(from, to),
                e_position.slice_dim0(from, to),
                e_mask.slice_dim0(from * ev_len_buf[0], to * ev_len_buf[0])
                );
            part->set_encode_batch(ev_batch, e_batch);
            part->set_encode_len({p_size}, {full_offset + to}, Tensor());
            part->ev_len_buf = ev_len_buf;

            if (i == num_split - 1) {
                part->set_search(s_token, s_sub, s_placement, s_position, s_mask);
                part->sv_len_buf = sv_len_buf;
                part->s_len_buf = s_len_buf;
            }
            part->input_len_no_split = full_len;
            parts.push_back(part);
        }
        // split rope_cache
        if (rope_cache.cos.numel() > 0) {
            auto cos = split_tensor(rope_cache.cos, num_split, p_size0);
            auto sin = split_tensor(rope_cache.sin, num_split, p_size0);
            for (int i = 0; i < num_split; ++i) {
                parts[i]->rope_cache.cos = cos[i];
                parts[i]->rope_cache.sin = sin[i];
            }
        }

        return parts;
    }

    size_t get_max_len_buf() const {
        int max_len_buf = 0;
        for (auto n: sv_len_buf) {
            max_len_buf = std::max(max_len_buf, n);
        }
        return max_len_buf;
    }
    size_t sum_len_buf() const {
        return std::accumulate(sv_len_buf.begin(), sv_len_buf.end(), 0);
    }
};

}  // namespace model