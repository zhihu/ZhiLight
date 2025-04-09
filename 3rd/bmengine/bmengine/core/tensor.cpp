#include "bmengine/core/tensor.h"
#include "bmengine/core/context.h"
#include "bmengine/core/exception.h"
#include <cuda_runtime.h>
#include <memory>

#include "private/tensor_impl.h"

namespace bmengine {
namespace core {

DistLayout transpose_layout(DistLayout dist_layout) {
    switch (dist_layout) {
        case DistLayout::COLUMNAR: return DistLayout::ROW;
        case DistLayout::ROW: return DistLayout::COLUMNAR;
        default: return dist_layout;
    }
};

Tensor::Tensor() : pimpl_(new TensorImpl({}, nullptr, 0, DataType::kFloat)) { }
Tensor::Tensor(std::unique_ptr<TensorImpl>&& impl) : pimpl_(std::move(impl)) { }
Tensor::Tensor(const Tensor& other)
    : pimpl_(std::make_unique<TensorImpl>(*other.pimpl_)), quant_scale(other.quant_scale) { }
Tensor::Tensor(Tensor&& other)
    : pimpl_(std::move(other.pimpl_)), quant_scale(std::move(other.quant_scale)) { }

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        pimpl_ = std::make_unique<TensorImpl>(*other.pimpl_);
        quant_scale = other.quant_scale;
    }
    return *this;
}
Tensor& Tensor::operator=(Tensor&& other) {
    pimpl_ = std::move(other.pimpl_);
    quant_scale = std::move(other.quant_scale);
    return *this;
}

TensorImpl* Tensor::pimpl() const {
    BM_ASSERT(pimpl_.get(), "Tensor is empty");
    return pimpl_.get();
}

Tensor Tensor::view(const std::vector<size_t>& size) const {
    return Tensor(pimpl_->view_type(size, pimpl()->dtype()));
}

Tensor Tensor::view_unchecked(const std::vector<size_t>& size, DataType dtype) const {
    return Tensor(pimpl_->view_unchecked(size, dtype));
}

Tensor Tensor::view_type(const std::vector<size_t>& size, DataType dtype) const {
    BM_ASSERT(pimpl() != nullptr, "Tensor is empty");
    return Tensor(pimpl_->view_type(size, dtype));
}

void Tensor::from_buffer(const void* ptr, bool async, cudaStream_t stream) {
    BM_CUDART_ASSERT(cudaMemcpyAsync(mutable_data(), ptr, nbytes(), cudaMemcpyHostToDevice, stream));
    if (!async) {
        BM_CUDART_ASSERT(cudaStreamSynchronize(stream));
    }
    BM_CUDART_ASSERT(cudaGetLastError());
}

Tensor Tensor::to_device(int dev_id) const {
    // TODO combine ctx.identity() with this.
    BM_ASSERT(false, "not implemented.");
    return Tensor();
}

Tensor::~Tensor() {};


long Tensor::id() const {
    return pimpl()->id();
}
void Tensor::set_id(long id) const {
    pimpl()->set_id(id);
}
const std::string& Tensor::name() const {
    return pimpl()->name();
}
void Tensor::set_name(const std::string& name) const {
    pimpl()->set_name(name);
}

std::string Tensor::info(int level) const {
    return pimpl()->info(level);
}

DataType Tensor::dtype() const {
    return pimpl()->dtype();
}
int Tensor::ndim() const {
    return pimpl()->ndim();
}
size_t Tensor::nbytes() const {
    return pimpl()->nbytes();
}
size_t Tensor::numel() const {
    return pimpl()->numel();
}
int Tensor::normalize_dim(int dim) const {
    return pimpl()->normalize_dim(dim);
}
size_t Tensor::size(int dim) const {
    return pimpl()->size(dim);
}
const std::vector<size_t>& Tensor::size() const {
    return pimpl()->size();
}
size_t Tensor::stride(int dim) const {
    return pimpl()->stride(dim);
}
size_t Tensor::stride_bytes(int dim) const {
    return pimpl()->stride_bytes(dim);
}

void* Tensor::data() const {
    return pimpl()->data();
}
void* Tensor::nullable_data() const {
    return pimpl()->nullable_data();
}
void* Tensor::mutable_data() {
    return pimpl()->mutable_data();
}

size_t Tensor::mem_bytes() const {
    return pimpl()->mem_bytes();
}

int Tensor::device() const {
    return pimpl()->device();
}

void Tensor::to_buffer(void* ptr, cudaStream_t stream) const {
    BM_CUDART_ASSERT(cudaMemcpyAsync(ptr, data(), nbytes(), cudaMemcpyDeviceToHost, stream));
    BM_CUDART_ASSERT(cudaStreamSynchronize(stream));
}

std::vector<Tensor> Tensor::chunk() const {
    std::vector<Tensor> chunks;
    BM_ASSERT(pimpl() != nullptr, "Tensor is empty");
    auto chunk_impls = pimpl()->chunk();
    for (int i = 0; i < chunk_impls.size(); ++i) {
        chunks.emplace_back(Tensor(std::move(chunk_impls[i])));
    }
    return chunks;
}

Tensor Tensor::index_dim0(size_t i) const {
    BM_ASSERT_LT(i, size(0), "Out of range");
    return Tensor(pimpl()->index_dim0(i));
}

Tensor Tensor::slice_dim0(size_t from, size_t to) const {
    BM_ASSERT(from < to, "Wrong range");
    BM_ASSERT_LE(to, size(0), "Wrong range");
    return Tensor(pimpl()->slice_dim0(from, to));
}

Tensor Tensor::virtual_slice(size_t from, size_t len, int dim) const {
    return Tensor(dynamic_cast<TensorImpl*>(pimpl())->virtual_slice(from, len, dim));
}

Tensor Tensor::squeeze() const {
    auto s = shape();
    for (auto it = s.begin(); it != s.end();) {
        if (*it == 1)
            it = s.erase(it);
        else
            it++;
    }
    return view(s);
}

Tensor Tensor::from_external(
    const std::vector<size_t>& shape, DataType dtype, void* ptr, size_t nbytes, int device, bool own_ptr) {
    // Don't take over ownership of ptr, deleter do nothing
    Memory mem = std::make_shared<Memory_>(
        ptr, device, nbytes, [own_ptr](void* ptr) { if (own_ptr) {delete[] (char*)ptr;} });
    return Tensor(std::make_unique<TensorImpl>(shape, mem, 0, dtype));
};

} // namespace core
} // namespace bmengine
