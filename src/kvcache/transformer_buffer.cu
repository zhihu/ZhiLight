#include "kvcache/transformer_buffer.h"

#include <iostream>
#include <limits>
#include <memory>
#include <assert.h>

#include "bmengine/functions/init.h"
#include "nn/quant/int8/quant_kernel.h"
#include "utils/env.h"
#include "private/allocator.h"

namespace kvcache {

using bmengine::core::Tensor;

// clang-format off
// gridDim (batch, len_kv, num_heads),  blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(copy_to_buffer)(
    int len_buf,
    int num_heads,
    int dim_head,
    size_t src_stride,
    size_t dst_stride,
    int place_stride,
    const int32_t* __restrict__ placement, // (batch, len_kv)
    const T* __restrict__ src,             // (batch, len_kv, num_heads, dim_head)
    T* __restrict__ dst,                   // (batch, (num_heads, len_buf|len_buf, num_heads), dim_head)
    bool BSHD
) {
    int batch_id = blockIdx.x;
    int pos_buf =
        (placement == nullptr) ? blockIdx.y : placement[batch_id * place_stride + blockIdx.y];
    // padded query when pos == -1, just ignore.
    if (pos_buf < 0)
        return;
    assert(pos_buf < len_buf);
    size_t offset_src = batch_id * src_stride + (blockIdx.y * num_heads + blockIdx.z) * dim_head;
    size_t offset_dst;
    if (BSHD)
        offset_dst = batch_id * dst_stride + (blockIdx.z + num_heads * pos_buf) * dim_head;
    else
        offset_dst = batch_id * dst_stride + (blockIdx.z * len_buf + pos_buf) * dim_head;
    for (int i = threadIdx.x; i < dim_head; i += blockDim.x) {
        dst[offset_dst + i] = src[offset_src + i];
    }
}

void copy_to_buffer(
    int num_heads,
    int len_kv,
    int len_buf,
    int dim_head,
    const core::Tensor* placement,  // (batch, len_q)
    const core::Tensor& src, // (batch, len_q, num_heads, dim_head)
    const core::Tensor& dst, // (batch, num_heads, len_buf, dim_head)
    cudaStream_t stream,
    bool BSHD) {
    int batch = (src.ndim() == 3) ? 1 : src.size(0);
    size_t src_stride = (src.ndim() == 3) ? 0 : src.stride(0);
    size_t dst_stride = (dst.ndim() == 3) ? 0 : dst.stride(0);
    int place_stride = (placement == nullptr || placement->ndim() == 1) ? 0 : placement->stride(0);
    dim3 gridDim(batch, len_kv, num_heads);
    dim3 blockDim(min(1024, round_up(dim_head, 32)), 1, 1);
    auto dtype = src.dtype();

    BM_ASSERT(
        (src.ndim() == 3 && dst.ndim() == 3) || (src.ndim() == 4 && dst.ndim() == 4),
        "src and dst must be 3/4-dimensional");
    BM_ASSERT_EQ(dst.dtype(), dtype, "dst.dtype() != src.dtype()");

    BM_ASSERT_EQ(src.size(-1), dim_head, "dim mismatch");
    BM_ASSERT_EQ(src.size(-2), num_heads, "dim mismatch");
    BM_ASSERT_EQ(src.size(-3), len_kv, "dim mismatch");
    BM_ASSERT_EQ(dst.size(-1), dim_head, "dim mismatch");

    if (BSHD) {
        BM_ASSERT_EQ(dst.size(-2), num_heads, "dim mismatch");
        if (batch > 1) {
            BM_ASSERT_EQ(dst.size(-3), len_buf, "len_buf mismatch");
        } else {
            BM_ASSERT_LE(len_buf, dst.size(-3), "len_buf mismatch");
        }
    } else {
        BM_ASSERT_EQ(dst.size(-3), num_heads, "dim mismatch");
        BM_ASSERT_EQ(dst.size(-2), len_buf, "dim mismatch");
    }

    BM_DTYPE_DISPATCH(dtype, {
        BM_KERNEL(copy_to_buffer)<<<gridDim, blockDim, 0, stream>>>(
            len_buf,
            num_heads,
            dim_head,
            src_stride,
            dst_stride,
            place_stride,
            (placement == nullptr ? nullptr : placement->data<int32_t>()),
            src.data<scalar_t>(),
            dst.data<scalar_t>(),
            BSHD);
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

core::Tensor TransformerBuffer::copy(
    const core::Context& ctx,
    int layer,
    const core::Tensor& src,
    const core::Tensor& placement,
    int start,
    bool need_dequant) {
    BM_ASSERT_LT(layer, buffer.size(), "Out of range");
    Tensor dst = buffer[layer];
    BM_ASSERT(BSHD, "flash attention only");
    BM_ASSERT_EQ(src.ndim(), 3, "Wrong ndim, input should be SHD");
    BM_ASSERT_EQ(dst.ndim(), 3, "Wrong ndim, buffer should be SHD");

    cudaStream_t stream = ctx.current_stream()->ptr;

    int len_kv = src.size(0);
    int len_buf = dst.size(0);
    if (!is_quantized()) {
        copy_to_buffer(num_heads, len_kv, len_buf, dim_head, &placement, src, dst, stream, BSHD);
        return dst;
    }

    // quantized cache
    BM_ASSERT_LT(layer, scales_.size(), "Out of range");
    Tensor scale = scales_[layer];
    // TODO: check placement
    Tensor dst_new = dst.slice_dim0_len(start, len_kv);
    Tensor scale_new = scale.slice_dim0_len(start, len_kv);
    int8_op::quant_calc_scale(ctx, src, &dst_new, &scale_new, 127, 128);  // +128 => unsigned
    if (need_dequant && start > 0) {
        // BM_ASSERT(false, "start > 0");
        // need de-quantize old cache
        auto out_shape = src.shape();
        out_shape[0] = len_buf;
        Tensor out = ctx.tensor(out_shape, src.dtype());
        // dequant old part
        Tensor out_old = out.slice_dim0_len(0, start);
        Tensor q_old = dst.slice_dim0_len(0, start);
        Tensor scale_old = scale.slice_dim0_len(0, start);
        int8_op::dequant_group(ctx, q_old, scale_old, &out_old, 128);
        // copy new part
        Tensor out_new = out.slice_dim0_len(start, len_kv);
        BM_CUDART_ASSERT(cudaMemcpyAsync(
            out_new.mutable_data(), src.data(), src.nbytes(), cudaMemcpyDeviceToDevice, stream));
        return out;
    }
    return src;
}

// gridDim (n / 1024, 1, 1),    blockDim (1024, 1, 1)
template<typename T>
__global__ void BM_KERNEL(resize_buffer)(
    size_t n, size_t old_stride, size_t new_stride, const T* old_buffer, T* new_buffer) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int in_pos = i % old_stride;
        if (in_pos < new_stride) {
            new_buffer[in_pos + (i / old_stride * new_stride)] = old_buffer[i];
        }
    }
}

__host__ void launch_resize_buffer(
    size_t old_stride,
    size_t new_stride,
    const core::Tensor& src,
    const core::Tensor& dst,
    cudaStream_t stream,
    size_t numel = 0) {
    BM_ASSERT(src.device() == dst.device(), "src and dst must be on the same device");
    size_t n = numel == 0 ? src.numel() : numel;
    int threads = round_up(min(n, (size_t) 1024), 32);
    int blocks = round_up(n, threads) / threads;

    dim3 blockDim(threads, 1, 1);
    dim3 gridDim(blocks, 1, 1);

    auto dtype = src.dtype();
    BM_DTYPE_DISPATCH(src.dtype(), {
        BM_KERNEL(resize_buffer)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
            n, old_stride, new_stride, src.data<scalar_t>(), dst.data<scalar_t>());
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

core::Tensor resize_buffer(
    const core::Context& ctx, const core::Tensor& buffer, int dim, size_t new_length) {
    auto shape = buffer.size();
    int normalized_dim = dim < 0 ? (shape.size() + dim) : dim;
    BM_ASSERT(
        (normalized_dim >= 0) && (normalized_dim < shape.size()),
        "Invalid dimension: dim must in [0, " + std::to_string(shape.size()) + "), but got "
            + std::to_string(dim));
    BM_ASSERT_EQ(ctx.active_device(), buffer.device(), "Invalid device");

    size_t stride_base = 1;
    for (int i = normalized_dim + 1; i < shape.size(); i++) {
        stride_base *= shape[i];
    }
    size_t old_stride = shape[normalized_dim] * stride_base;
    size_t new_stride = new_length * stride_base;

    auto new_shape = shape;
    new_shape[normalized_dim] = new_length;
    auto new_buffer = ctx.tensor(new_shape, buffer.dtype());
    BM_CUDART_ASSERT(
        cudaMemsetAsync(new_buffer.data(), 0, new_buffer.nbytes(), ctx.current_stream()->ptr));

    launch_resize_buffer(old_stride, new_stride, buffer, new_buffer, ctx.current_stream()->ptr);

    return new_buffer;
}

TransformerBuffer::TransformerBuffer(
    int batch_size,
    int num_layers,
    int num_heads,
    int dim_head,
    core::DataType dtype,
    bool parallel,
    bool BSHD)
    : KVCache(batch_size, num_layers, num_heads, dim_head, dtype, parallel, BSHD) {
    BM_ASSERT(is_dyn_batch() || batch_size > 0, "batch_size must be greater than 0");
    BM_ASSERT(num_layers > 0, "num_layers must be greater than 0");
    BM_ASSERT(num_heads > 0, "num_heads must be greater than 0");
    BM_ASSERT(dim_head > 0, "dim_head must be greater than 0");
    buffer.resize(num_layers);
}
TransformerBuffer::TransformerBuffer(
    const KVCacheConfig& c)
    : TransformerBuffer(-1, c.num_layers, c.num_heads, c.dim_head, c.dtype, false, c.BSHD) {
    scale_dtype_ = c.scale_dtype;
    layer_devices = c.layer_devices;
//    std::cout << "batch_size=" << batch_size << ", num_layers:" << this->num_layers << ", num_heads=" << this->num_heads
//        << ", BSHD=" << this->BSHD
//        << std::endl;
}
TransformerBuffer::~TransformerBuffer() { }

void TransformerBuffer::check_layer(int i) const {
    BM_ASSERT(
        i >= 0 && i < num_layers,
        "Invalid layer index: i must in [0, " + std::to_string(num_layers) + "), but got "
        + std::to_string(i));
}
const core::Tensor& TransformerBuffer::operator[](int i) const {
    check_layer(i);
    return buffer[i];
}

core::Tensor& TransformerBuffer::operator[](int i) {
    check_layer(i);
    return buffer[i];
}

const core::Tensor& TransformerBuffer::get_scale(int i) const {
    BM_ASSERT_EQ(num_layers, scales_.size(), "Wrong scales_ size");
    check_layer(i);
    return scales_[i];
}

void TransformerBuffer::resize(const core::Context& ctx, size_t new_length) {
    BM_ASSERT_EQ(layer_devices.size(), (size_t) (num_layers), "Invalid layer_devices");
    // std::cout << "TransformerBuffer::resize: new_length=" << new_length << "\n";
    if (is_dyn_batch()) {
        resize_dyn_batch(ctx, new_length);
        return;
    }
    for (int i = 0; i < num_layers; i++) {
        ctx.switch_to_device(layer_devices[i]);
        std::vector<size_t> shape;
        if (buffer[i].numel() == 0) {
            if (BSHD)
                shape = { new_length, num_heads, dim_head };
            else
                shape = { num_heads, new_length, dim_head };
            if (batch_size > 0) {
                shape.insert(shape.begin(), size_t(batch_size));
            }
            buffer[i] = ctx.tensor(shape, dtype);
            functions::zeros_(ctx, buffer[i]);
        } else {
            buffer[i] = resize_buffer(ctx, buffer[i], BSHD ? -3 : -2, new_length);
        }
    }
    ctx.switch_to_device(layer_devices[0]);
}

const static size_t max_block_size = 128 * 1024 * 1024; // 128M

void TransformerBuffer::resize_dyn_batch(const core::Context& ctx, size_t new_length) {
    for (int i = 0; i < num_layers;) {
        int dev = layer_devices[i];
        int j = i + 1;
        while (j < num_layers && layer_devices[j] == dev)
            j++;
        // std::cout << "dev, i, j " << dev << ", " << i << ", " << j << "\n";
        // resize MULTIPLE layer buffers on SAME device.
        ctx.switch_to_device(layer_devices[i]);
        resize_multi_layer(ctx, new_length, i, j, BSHD ? -3 : -2);
        if (scale_dtype_.get()) {
            BM_ASSERT_EQ(j, num_layers, "Only support TP mode");
            resize_scale(ctx, new_length, i, j, BSHD ? -2 : -1);
        }
        i = j;
    }
    ctx.switch_to_device(layer_devices[0]);
}

// try to allocate multiple layers together if each one is small.
void TransformerBuffer::resize_multi_layer(
    const core::Context& ctx, size_t new_length, int begin, int end, int dim) {
    size_t nbytes = new_length * num_heads * dim_head * core::get_elem_size(dtype);
    // std::cout << "resize_multi_layer, new_len=" << new_length << ", nbytes=" << nbytes << "\n";
    size_t alloc_layers = std::max(size_t(1), max_block_size / nbytes);
    if (continuous)
        alloc_layers = end - begin;
    bool first = buffer[begin].numel() == 0;
    for (size_t i = begin; i < end; i += alloc_layers) {
        size_t align_bytes = 1024;
        // Allow kv cache be used in paged attention kernels.
        static int align_page = utils::get_int_env("KV_CACHE_ALIGN_PAGE", 0);
        if (num_heads == 1 && align_page > 0) {
            // for MLA to use paged table
            align_bytes = align_page * dim_head * core::get_elem_size(dtype);
            BM_ASSERT_EQ(new_length % align_page, 0, "new_length can't mod page size.");
        }
        alloc_layers = std::min(end - i, alloc_layers);
        // std::cout << "resize_multi_layer,  i=" << i << ", alloc_layers=" << alloc_layers << "\n";
        core::Tensor tensor =
            dim == -2 ? ctx.tensor({ alloc_layers, num_heads, new_length, dim_head }, dtype, "1", align_bytes)
                      : ctx.tensor({ alloc_layers, new_length, num_heads, dim_head }, dtype, "1", align_bytes);
        if (num_heads == 1 && align_page > 0) {
            char* base_ptr = ctx.get_allocator()->get_base_ptr();
            size_t offset = tensor.data<char>() - base_ptr;
            BM_ASSERT_EQ(offset % align_bytes, 0, "data_ptr is not aligned as expected.");
        }
        auto chunk = tensor.chunk();
        std::vector<core::Tensor> old_chunk(alloc_layers);
        if (false && num_heads == 1) {  // MLA
            functions::fill(ctx, tensor, -std::numeric_limits<float>::infinity());
        } else {
            functions::zeros_(ctx, tensor);
        }
        for (int j = 0; j < alloc_layers; j++) {
            old_chunk[j] = buffer[i + j];
            buffer[i + j] = chunk[j]; // .view({1, num_heads, new_length, dim_head});
        }
        if (first) {
            continue;
        }
        size_t stride_base = old_chunk[0].stride(dim);
        size_t old_stride = old_chunk[0].size(dim) * stride_base;
        size_t new_stride = new_length * stride_base;
        size_t nbytes = old_chunk[0].nbytes();
        auto stream = ctx.current_stream()->ptr;
        for (int j = 0; j < alloc_layers;) {
            // copy (k - j) continuous chunks
            int k = j + 1;
            while (k < alloc_layers
                   && old_chunk[k - 1].data<char>() + nbytes == old_chunk[k].data<char>()) {
                k++;
            }
            // std::cout << "multiCopy i=" << i << ", j=" << j << ", k=" << k << "\n";
            size_t numel = old_chunk[j].numel() * (k - j);
            launch_resize_buffer(old_stride, new_stride, old_chunk[j], chunk[j], stream, numel);

            j = k;
        }
    }
}

void TransformerBuffer::resize_scale(
    const core::Context& ctx, size_t new_length, int begin, int end, int dim) {
    bool first = all_scale_.numel() == 0;
    if (first) {
        all_scale_ = BSHD
            ? ctx.tensor({ num_layers, new_length, num_heads }, *scale_dtype_)
            : ctx.tensor({ num_layers, num_heads, new_length }, *scale_dtype_);
        functions::zeros_(ctx, all_scale_);
    } else {
        all_scale_ = resize_buffer(ctx, all_scale_, dim, new_length);
    }
    scales_ = all_scale_.chunk();
}

// (num_layers * num_heads, copy_len, 1) (dim_head)
template<typename T>
__global__ void  KERNEL_copy_slice(
    const T* src, // (num_layers * num_heads, src_len, dim_head)
    T* dst,       // (num_layers * num_heads, dst_len, dim_head)
    unsigned int src_offset,
    unsigned int src_len,
    unsigned int dst_offset,
    unsigned int dst_len) {
    dst[((blockIdx.x * dst_len) + dst_offset + blockIdx.y) * blockDim.x + threadIdx.x] = \
        src[((blockIdx.x * src_len) + src_offset + blockIdx.y) * blockDim.x + threadIdx.x];
}

// (num_layers, copy_len, num_heads) (dim_head)
template<typename T>
__global__ void  KERNEL_copy_slice_BSHD(
    const T* src, // (num_layers, src_len, num_heads, dim_head)
    T* dst,       // (num_layers, dst_len, num_heads, dim_head)
    unsigned int src_offset,
    unsigned int src_len,
    unsigned int dst_offset,
    unsigned int dst_len) {
    unsigned int R = gridDim.z * blockDim.x;
    unsigned int r = blockIdx.z * blockDim.x + threadIdx.x;
    dst[((blockIdx.x * dst_len) + dst_offset + blockIdx.y) * R + r] = \
        src[((blockIdx.x * src_len) + src_offset + blockIdx.y) * R + r];
}

// std::shared_ptr<TransformerBuffer>
core::Tensor TransformerBuffer::dump_slice(
    core::Context& ctx, size_t start, size_t len, core::Tensor* out) {
    int dim = BSHD ? -3 : -2;
    int dim_h = !BSHD ? -3 : -2;
    BM_ASSERT_EQ(batch_size, -1, "Not a dynamic batch buffer");
    BM_ASSERT(buffer[0].numel(), "buffer no data");
    BM_ASSERT_EQ(dim_head % 32, 0, "Wrong dim_head");
    BM_ASSERT_LE(start + len, buffer[0].size(dim), "out of range");
    if (out) {
        BM_ASSERT_EQ(out->size(0), num_layers, "Wrong num_layers");
        BM_ASSERT_EQ(out->size(-1), dim_head, "Wrong dim_head");
        BM_ASSERT_EQ(out->size(dim), len, "Wrong len");
        BM_ASSERT_EQ(out->size(dim_h), num_heads, "Wrong num_heads");
    }

    // std::shared_ptr<TransformerBuffer> ret =
    //    std::make_shared(num_layers, num_heads, dim_head, dtype, {}, parallel, BSHD);
    std::vector<size_t> shape =
        dim == -2 ? std::vector<size_t>({ num_layers, num_heads, len, dim_head })
                  : std::vector<size_t>({ num_layers, len, num_heads, dim_head });
    core::Tensor ret = out ? out->view(shape) : ctx.tensor(shape, dtype);
//        dim == -2 ? ctx.tensor({ num_layers, num_heads, len, dim_head }, dtype)
//                  : ctx.tensor({ num_layers, len, num_heads, dim_head }, dtype);

    size_t nbytes = buffer[0].nbytes();
    auto stream = ctx.current_stream()->ptr;
    for (size_t j = 0; j < num_layers;) {
        // copy [j, k] continuous chunks
        size_t k = j + 1;
        while (k < num_layers && buffer[k - 1].data<char>() + nbytes == buffer[k].data<char>()) {
            k++;
        }
        auto chunk = ret.slice_dim0(j, k);
        // std::cout << "multiCopy start=" << start << ", j=" << j << ", k=" << k << "\n";
        if (BSHD) {
            dim3 gridDim(k - j, len, num_heads);
            BM_DTYPE_DISPATCH_FLOAT(dtype, {
                KERNEL_copy_slice_BSHD<scalar_t><<<gridDim, dim_head, 0, stream>>>(
                    buffer[j].data<scalar_t>(),
                    chunk.mutable_data<scalar_t>(),
                    start,
                    buffer[j].size(dim),
                    0,
                    len);
            });
        } else {
            dim3 gridDim((k - j) * num_heads, len, 1);
            BM_DTYPE_DISPATCH_FLOAT(dtype, {
                KERNEL_copy_slice<scalar_t><<<gridDim, dim_head, 0, stream>>>(
                    buffer[j].data<scalar_t>(),
                    chunk.mutable_data<scalar_t>(),
                    start,
                    buffer[j].size(dim),
                    0,
                    len);
            });
        }
        BM_CUDART_ASSERT(cudaGetLastError());

        j = k;
    }
    return ret;
}

void TransformerBuffer::load_slice(core::Context& ctx, size_t start, size_t len, const core::Tensor& input) {
    int dim = BSHD ? -3 : -2;
    int dim_h = !BSHD ? -3 : -2;
    BM_ASSERT_EQ(batch_size, -1, "Not a dynamic batch buffer");
    BM_ASSERT(buffer[0].numel(), "buffer must be resized first");
    BM_ASSERT_EQ(dim_head % 32, 0, "Wrong dim_head");
    BM_ASSERT_LE(start + len, buffer[0].size(dim), "out of range");
    BM_ASSERT_EQ(input.size(0), num_layers, "Wrong num_layers");
    BM_ASSERT_EQ(input.size(-1), dim_head, "Wrong dim_head");
    BM_ASSERT_EQ(input.size(dim), len, "Wrong len");
    BM_ASSERT_EQ(input.size(dim_h), num_heads, "Wrong num_heads");

    size_t nbytes = buffer[0].nbytes();
    auto stream = ctx.current_stream()->ptr;
    for (size_t j = 0; j < num_layers;) {
        // copy [j, k] continuous chunks
        size_t k = j + 1;
        while (k < num_layers && buffer[k - 1].data<char>() + nbytes == buffer[k].data<char>()) {
            k++;
        }
        auto chunk = input.slice_dim0(j, k);
        // std::cout << "multiCopy i=" << i << ", j=" << j << ", k=" << k << "\n";
        if (BSHD) {
            dim3 gridDim(k - j, len, num_heads);
            BM_DTYPE_DISPATCH_FLOAT(dtype, {
                KERNEL_copy_slice_BSHD<scalar_t><<<gridDim, dim_head, 0, stream>>>(
                    chunk.data<scalar_t>(),
                    buffer[j].mutable_data<scalar_t>(),
                    0,
                    len,
                    start,
                    buffer[j].size(dim));
            });
        } else {
            dim3 gridDim((k - j) * num_heads, len, 1);
            BM_DTYPE_DISPATCH_FLOAT(dtype, {
                KERNEL_copy_slice<scalar_t><<<gridDim, dim_head, 0, stream>>>(
                    chunk.data<scalar_t>(),
                    buffer[j].mutable_data<scalar_t>(),
                    0,
                    len,
                    start,
                    buffer[j].size(dim));
            });
        }
        BM_CUDART_ASSERT(cudaGetLastError());

        j = k;
    }
}

} // namespace kvcache
