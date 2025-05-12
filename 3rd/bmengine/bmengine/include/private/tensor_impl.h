#pragma once
#include "bmengine/core/tensor.h"
#include "private/memory.h"
#include <string>
#include <vector>

namespace bmengine {
namespace core {

class TensorImpl {
private:
    long id_ { 0 };
    std::string name_;
    DataType dtype_;

    std::vector<size_t> shape_;
    std::vector<size_t> strides;
    const size_t offset;
    size_t numel_;
    size_t nbytes_;

    Memory mem;
    void check_mem() const;

public:
    TensorImpl(const std::vector<size_t>& shape, Memory mem, size_t offset, DataType dtype);
    TensorImpl(const TensorImpl&) = default;
    TensorImpl(TensorImpl&&) = default;
    ~TensorImpl() = default;

    long id() const { return id_; };
    void set_id(long id) { id_ = id; };
    const std::string& name() const;
    void set_name(const std::string& name);

    DataType dtype() const;

    int ndim() const { return shape_.size(); }
    size_t nbytes() const { return nbytes_; }
    size_t numel() const { return numel_; }
    const std::vector<size_t>& size() const { return shape_; }

    int normalize_dim(int dim) const;
    size_t size(int dim) const;
    size_t stride(int dim) const;
    size_t stride_bytes(int dim) const;

    void* data() const; // TODO: should return const void *
    void* nullable_data() const;
    void* mutable_data();
    size_t mem_bytes() const;
    int device() const;

    // void from_buffer(const void* ptr, bool async);

    std::unique_ptr<TensorImpl> view(const std::vector<size_t>& size) const;
    std::unique_ptr<TensorImpl> view_unchecked(const std::vector<size_t>& size, DataType dtype) const;
    std::unique_ptr<TensorImpl> view_type(const std::vector<size_t>& size, DataType dtype, bool check_size = true) const;
    std::unique_ptr<TensorImpl> view_uncontinuous(const std::vector<size_t>& size, DataType dtype) const;
    std::vector<std::unique_ptr<TensorImpl>> chunk() const;
    std::unique_ptr<TensorImpl> index_dim0(size_t i) const;
    std::unique_ptr<TensorImpl> slice_dim0(size_t from, size_t to) const;
    std::unique_ptr<TensorImpl> virtual_slice(size_t from, size_t to, int dim) const;
    bool is_continuous() const;

    std::string info(int level = 0) const;
    long get_mem_use_count() const { return mem.use_count(); }
};

} // namespace core
} // namespace bmengine
