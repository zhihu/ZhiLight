
#include "bmengine/c10d/c10d.h"
#include "bmengine/core/tensor.h"
#include "bmengine/core/exception.h"

namespace bmengine {
namespace c10d {

#ifdef ENABLE_NCCL_TP
ncclDataType_t dtype2nccl(core::DataType dtype) {
    switch (dtype) {
        case core::DataType::kInt8: return ncclInt8;
        case core::DataType::kDouble: return ncclDouble;
        case core::DataType::kFloat: return ncclFloat;
        case core::DataType::kHalf: return ncclHalf;
        case core::DataType::kBFloat16: return ncclBfloat16;
        case core::DataType::kInt32: return ncclInt32;
        default:
            BM_ASSERT(false, "Unsupport dtype " + std::string(get_data_type_name(dtype)));
            return ncclNumTypes;
    }
}

void NCCLAllGather(const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff) {
    BM_NCCL_ASSERT(ncclAllGather(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLCheckAsync(ncclComm_t comm) {
    auto state = ncclInProgress;
    do {
        ncclCommGetAsyncError(comm, &state);
    } while (state == ncclInProgress);
    BM_NCCL_ASSERT(state);
}

void NCCLAllReduce(
    const core::Context& ctx,
    const core::Tensor& sendbuff,
    core::Tensor& recvbuff,
    ncclRedOp_t op) {
    BM_NCCL_ASSERT(ncclAllReduce(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        op,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
//    NCCLCheckAsync(ctx.current_comm());
}

void NCCLBroadcast(
    const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff, int root) {
    BM_NCCL_ASSERT(ncclBroadcast(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        root,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLReduce(
    const core::Context& ctx,
    const core::Tensor& sendbuff,
    core::Tensor& recvbuff,
    ncclRedOp_t op,
    int root) {
    BM_NCCL_ASSERT(ncclReduce(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        op,
        root,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLReduceScatter(
    const core::Context& ctx,
    const core::Tensor& sendbuff,
    core::Tensor& recvbuff,
    ncclRedOp_t op) {
    BM_NCCL_ASSERT(ncclReduceScatter(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        recvbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        op,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLSend(const core::Context& ctx, const core::Tensor& sendbuff, int peer) {
    BM_NCCL_ASSERT(ncclSend(
        sendbuff.data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        peer,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLRecv(const core::Context& ctx, core::Tensor& recvbuff, int peer) {
    BM_NCCL_ASSERT(ncclRecv(
        recvbuff.mutable_data<void*>(),
        recvbuff.numel(),
        dtype2nccl(recvbuff.dtype()),
        peer,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLGroupStart() {
    BM_NCCL_ASSERT(ncclGroupStart());
}

void NCCLGroupEnd() {
    BM_NCCL_ASSERT(ncclGroupEnd());
}
void NCCLGroupEndCheck(ncclComm_t comm) {
    ncclResult_t ret = ncclGroupEnd();
    if (ret == ncclInProgress) {
        NCCLCheckAsync(comm);
    } else {
        BM_NCCL_ASSERT(ret);
    }
}
int NCCLCommCount(const core::Context& ctx) {
    int res;
    BM_NCCL_ASSERT(ncclCommCount(ctx.current_comm(), &res));
    return res;
}
int NCCLCommUserRank(const core::Context& ctx) {
    int rank;
    BM_NCCL_ASSERT(ncclCommUserRank(ctx.current_comm(), &rank));
    return rank;
}

#else
void NCCLAllGather(const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}
void NCCLAllReduce(
    const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff, ncclRedOp_t op) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}

void NCCLBroadcast(
    const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff, int root) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}

void NCCLReduce(
    const core::Context& ctx,
    const core::Tensor& sendbuff,
    core::Tensor& recvbuff,
    ncclRedOp_t op,
    int root) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}

void NCCLReduceScatter(const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff, ncclRedOp_t op) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}

void NCCLSend(const core::Context& ctx, const core::Tensor& sendbuff, int peer) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}

void NCCLRecv(const core::Context& ctx, core::Tensor& recvbuff, int peer) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}

void NCCLGroupStart() {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}
void NCCLGroupEnd() {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}
void NCCLGroupEndCheck(ncclComm_t comm) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}
int NCCLCommCount(const core::Context& ctx) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}
int NCCLCommUserRank(const core::Context& ctx) {
    throw std::runtime_error("ENABLE_NCCL_TP is off");
}

#endif
} // namespace c10d
} // namespace bmengine