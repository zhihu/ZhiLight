#pragma once
#include "bmengine/core/export.h"
#include "bmengine/core/dtype.h"
#include <memory>
#include <initializer_list>
#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

namespace bmengine {
namespace core {

enum class DistLayout { COLUMNAR, ROW, REPLICATED };

BMENGINE_EXPORT DistLayout transpose_layout(DistLayout dist_layout);
BMENGINE_EXPORT const char* get_dist_layout_name(DistLayout dist_layout);

BMENGINE_EXPORT size_t get_elem_size(DataType dtype);
BMENGINE_EXPORT size_t get_numel(std::initializer_list<size_t> size);
BMENGINE_EXPORT size_t get_numel(const std::vector<size_t>& size);
BMENGINE_EXPORT void check_no_zero(const std::vector<size_t>& size);

class TensorImpl;
class Tensor;
class BMENGINE_EXPORT Tensor {
public:
    std::shared_ptr<Tensor> quant_scale;

protected:
    friend class Context;
    friend class ContextImpl;

    std::unique_ptr<TensorImpl> pimpl_;
    TensorImpl* pimpl() const;

public:
    Tensor();
    Tensor(std::unique_ptr<TensorImpl>&&);
    Tensor(const Tensor&);
    Tensor(Tensor&&);
    Tensor& operator=(const Tensor&);
    Tensor& operator=(Tensor&&);
    ~Tensor();

    long id() const;
    void set_id(long it) const;
    const std::string& name() const;
    void set_name(const std::string& name) const; // const for easily set name to debug

    DataType dtype() const;
    int ndim() const;
    size_t nbytes() const;
    size_t numel() const;
    bool empty() const { return numel() == 0; }
    const std::vector<size_t>& size() const;
    const std::vector<size_t>& shape() const { return size(); }

    int normalize_dim(int dim) const;
    size_t size(int dim) const;
    size_t stride(int dim) const;
    size_t stride_bytes(int dim) const;

    /**************************** Data functions ****************************/
    template<typename T>
    T* data() const {
        return reinterpret_cast<T*>(data());
    }
    void* data() const; // TODO: should return const void *
    void* nullable_data() const;
    template<typename T>
    void* nullable_data() const{
        return reinterpret_cast<T*>(nullable_data());
    }
    template<typename T>
    T* mutable_data() {
        return reinterpret_cast<T*>(mutable_data());
    }
    void* mutable_data();
    size_t mem_bytes() const;
    int device() const;


    /**************************** Transform functions ****************************/
    Tensor view(const std::vector<size_t>& size) const;
    Tensor view_unchecked(const std::vector<size_t>& size, DataType dtype) const;
    Tensor view_type(const std::vector<size_t>& size, DataType dtype) const;
    Tensor view_uncontinuous(const std::vector<size_t>& size) const;

    Tensor index_dim0(size_t i) const;
    Tensor slice_dim0(size_t from, size_t to) const;
    Tensor slice_dim0_len(size_t from, size_t len) const {
        return this->slice_dim0(from, from + len);
    }
    // Note: After slice, the storage is no longer continuous!!!
    Tensor virtual_slice(size_t from, size_t len, int dim = -1) const;
    Tensor virtual_transpose(int dim0, int dim1) const;
    bool is_continuous() const;
    std::vector<Tensor> chunk() const;
    Tensor squeeze() const;


    /**************************** Load/Save functions ****************************/
    void from_buffer(const void* ptr, bool async = false, cudaStream_t stream=0);

    // Create a tensor from external memory
    // device=-1 means CPU tensor
    static Tensor from_external(
        const std::vector<size_t>& shape,
        DataType dtype,
        void* ptr,
        size_t nbytes,
        int device = -1,
        bool own_ptr=false);

    void to_buffer(void* ptr, cudaStream_t stream=0) const;
    Tensor to_device(int dev_id = -1) const;

    template<typename T>
    std::vector<T> to_vector() const {
        std::vector<T> t(nbytes() / sizeof(T));
        to_buffer(t.data());
        return std::move(t);
    }


    /**************************** IO functions ****************************/
    BMENGINE_EXPORT friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    std::string info(int level = 0) const; // info exclude data

};

} // namespace core
} // namespace bmengine

