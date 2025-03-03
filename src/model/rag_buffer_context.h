#pragma once
#include "model/model_context.h"
#include "bmengine/core/tensor.h"
#include <bmengine/logger/std_log_op.hpp>
#include <memory>
#include <vector>
#include "kvcache/transformer_buffer.h"

namespace model {

using kvcache::KVCacheConfig;
struct RagBufferContext {
    typedef std::vector<void *> AddressVector;

    kvcache::KVCacheConfig config_k_;
    kvcache::KVCacheConfig config_v_;
    size_t num_layers;
    vector<unique_ptr<TransformerBuffer>> buf_k_; // buffer per task
    vector<unique_ptr<TransformerBuffer>> buf_v_; // buffer per task

    AddressVector h_buf_k_addresses;
    AddressVector h_buf_v_addresses;
    AddressVector h_scale_k_addresses;
    AddressVector h_scale_v_addresses;
    Tensor buf_k_addresses; // (num_layers, batch)
    Tensor buf_v_addresses; // (num_layers, batch)
    Tensor scale_k_address; // (num_layers, batch)
    Tensor scale_v_address; // (num_layers, batch)

    bool skip_last { false };

private:
    typedef const Tensor& (TransformerBuffer::*BufPtr)(int i) const;
    static constexpr bool BUF_K = true;
    static constexpr bool BUF_V = false;
    size_t active_batch() const {
        return buf_k_.size() - size_t(skip_last);
    }

    AddressVector get_buf_addresses(bool is_k, BufPtr buf_ptr) {
        auto& buffers = is_k ? buf_k_ : buf_v_;
        AddressVector addresses; // (num_layers, batch)
        size_t batch = active_batch();
        addresses.reserve(num_layers * batch);
        for (int x = 0; x < num_layers; ++x) {
            for (size_t i = 0; i < batch; ++i) {
                void* ptr = (buffers[i].get()->*buf_ptr)(x).data();
                addresses.push_back(ptr);
            }
        }
        return addresses;
    }
    AddressVector get_buf_addresses(bool is_k, BufPtr buf_ptr, int layer) {
        auto& buffers = is_k ? buf_k_ : buf_v_;
        AddressVector addresses; // (num_layers, batch)
        size_t batch = active_batch();
        addresses.reserve(buffers.size());
        {
            for (size_t i = 0; i < batch; ++i) {
                void* ptr = (buffers[i].get()->*buf_ptr)(layer).data();
                addresses.push_back(ptr);
            }
        }
        return addresses;
    }

    AddressVector slice_address(const AddressVector& addresses, int layer) {
        size_t batch = active_batch();
        BM_ASSERT_EQ(num_layers * batch, addresses.size(), "");
        BM_ASSERT_LT(layer, num_layers, "");
        size_t offset = layer * batch;
        return AddressVector(addresses.begin() + offset, addresses.begin() + offset + batch);
    }

    bool is_address_changed(const AddressVector& addresses, bool is_k, BufPtr buf_ptr, int layer) {
        auto old_address = slice_address(addresses, layer);
        auto new_address = get_buf_addresses(is_k, buf_ptr, layer);
        if (old_address != new_address) {
            std::cout << "Address changed\n";
            return true;
        }
        return false;
    }

    bool has_v() const {
        return config_v_.dim_head > 0;
    }

public:
    RagBufferContext(KVCacheConfig config_k, KVCacheConfig config_v)
        : config_k_(config_k), config_v_(config_v) {
        BM_ASSERT_EQ(config_k.num_layers, config_v.num_layers, "num_layers mismatch");
        this->num_layers = config_k.num_layers;
    }

    bool is_cache_quant() const { return config_k_.is_quant(); }

    void check_task_index(int i_task) {
        if (i_task >= buf_k_.size())
            throw std::out_of_range("Invalid task index");
    }
    TransformerBuffer& buf_k(int b) {
        check_task_index(b);
        return (*buf_k_[b]);
    }
    TransformerBuffer& buf_v(int b) {
        check_task_index(b);
        return (*buf_v_[b]);
    }
    const Tensor& buf_k(int b, int layer) {
        check_task_index(b);
        return (*buf_k_[b])[layer];
    }
    const Tensor& buf_v(int b, int layer) {
        check_task_index(b);
        return (*buf_v_[b])[layer];
    }

    void resize_task_buf(const core::Context& ctx, int b, size_t new_length) {
        if (buf_k_.size() < b + 1) {
            buf_k_.resize(b + 1);
            buf_v_.resize(b + 1);
        }
        if (!buf_k_[b]) {
            buf_k_[b] = std::make_unique<TransformerBuffer>(config_k_);
            if (has_v()) {
                buf_v_[b] = std::make_unique<TransformerBuffer>(config_v_);
            }
        }
        buf_k_[b]->resize(ctx, new_length);
        if (has_v())
            buf_v_[b]->resize(ctx, new_length);
    }

    void free_task_buf(int b) {
        check_task_index(b);
        buf_k_.erase(buf_k_.begin() + b);
        buf_v_.erase(buf_v_.begin() + b);
    }

    void set_buffer_addr(ModelContext &ctx) {
        if (active_batch() > 0) {
            h_buf_k_addresses = get_buf_addresses(BUF_K, &TransformerBuffer::get_layer);
            buf_k_addresses = ctx.tensor_of(h_buf_k_addresses, {num_layers, active_batch()});
            if (config_k_.is_quant()) {
                h_scale_k_addresses = get_buf_addresses(BUF_K, &TransformerBuffer::get_scale);
                scale_k_address = ctx.tensor_of(h_scale_k_addresses, {num_layers, active_batch()});
            }
            // std::cout << "set_buffer_addr::buf_k_addresses:" << buf_k_addresses.shape() << "\n";
        }
        if (active_batch() > 0 && has_v()) {
            h_buf_v_addresses = get_buf_addresses(BUF_V, &TransformerBuffer::get_layer);
            buf_v_addresses = ctx.tensor_of(h_buf_v_addresses, {num_layers, active_batch()});
            if (config_v_.is_quant()) {
                h_scale_v_addresses = get_buf_addresses(BUF_V, &TransformerBuffer::get_scale);
                scale_v_address = ctx.tensor_of(h_scale_v_addresses, {num_layers, active_batch()});
            }
        }
    }

    Tensor buf_k_addr(ModelContext &ctx, int layer) {
        if (is_address_changed(h_buf_k_addresses, BUF_K, &TransformerBuffer::get_layer, layer)) {
            set_buffer_addr(ctx);
        }
        return ctx.identity(&buf_k_addresses, "buf_k_addresses")->index_dim0(layer);
    }
    Tensor scale_k_addr(ModelContext &ctx, int layer) {
        if (!config_k_.is_quant())
            return Tensor();
        if (is_address_changed(h_scale_k_addresses, BUF_K, &TransformerBuffer::get_scale, layer)) {
            set_buffer_addr(ctx);
        }
        return ctx.identity(&scale_k_address, "scale_k_addr")->index_dim0(layer);
    }

    Tensor buf_v_addr(ModelContext &ctx, int layer) {
        if (is_address_changed(h_buf_v_addresses, BUF_V, &TransformerBuffer::get_layer, layer)) {
            set_buffer_addr(ctx);
        }
        return ctx.identity(&buf_v_addresses, "buf_v_addresses")->index_dim0(layer);
    }
    Tensor scale_v_addr(ModelContext &ctx, int layer) {
        if (!config_v_.is_quant())
            return Tensor();
        if (is_address_changed(h_scale_v_addresses, BUF_V, &TransformerBuffer::get_scale, layer)) {
            set_buffer_addr(ctx);
        }
        return ctx.identity(&scale_v_address, "scale_v_addr")->index_dim0(layer);
    }

    size_t get_buf_len(size_t b) {
        if (b < buf_k_.size()) {
            auto& t = (*buf_k_[b])[0];
            return t.numel() ? t.size(buf_k_[b]->is_BSHD() ? -3 : -2) : 0;
        }
        return 0;
    }

    void dump_task_buf(int b, int num_layers, char* ptr, size_t len) {
        BM_ASSERT_LE(size_t(b + 1), buf_k_.size(), "out of range");
        auto buf_k = *buf_k_[b];
        auto buf_v = *buf_v_[b];
        BM_ASSERT_EQ(len, 2 * num_layers * buf_k[0].nbytes(), "buffer len mismatch");
        for (int i = 0; i < num_layers; ++i) {
            auto& t = buf_k[i];
            t.to_buffer(ptr);
            ptr += t.nbytes();
        }
        for (int i = 0; i < num_layers; ++i) {
            auto& t = buf_v[i];
            t.to_buffer(ptr);
            ptr += t.nbytes();
        }
        BM_CUDART_ASSERT(cudaDeviceSynchronize());
    }

    void load_task_buf(int b, int num_layers, char* ptr, size_t len) {
        BM_ASSERT_LE(size_t(b + 1), buf_k_.size(), "out of range");
        auto buf_k = *buf_k_[b];
        auto buf_v = *buf_v_[b];
        BM_ASSERT_EQ(len, 2 * num_layers * buf_k[0].nbytes(), "buffer len mismatch");
        BM_CUDART_ASSERT(cudaDeviceSynchronize());
        for (int i = 0; i < num_layers; ++i) {
            auto& t = buf_k[i];
            t.from_buffer(ptr);
            ptr += t.nbytes();
        }
        for (int i = 0; i < num_layers; ++i) {
            auto& t = buf_v[i];
            t.from_buffer(ptr);
            ptr += t.nbytes();
        }
    }
}; // class RagBufferContext

} // namespace model
