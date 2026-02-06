#pragma once
#ifdef ENABLE_NCCL_TP
#include <nccl.h>
#endif
#include "bmengine/core/tensor.h"
#include "bmengine/core/context.h"

namespace bmengine {

namespace c10d {

#ifdef ENABLE_NCCL_TP
ncclDataType_t dtype2nccl(core::DataType dtype);
#endif

void NCCLAllGather(const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff);
void NCCLAllReduce(
    const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff, ncclRedOp_t op);

void NCCLBroadcast(
    const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff, int root);

void NCCLReduce(
    const core::Context& ctx,
    const core::Tensor& sendbuff,
    core::Tensor& recvbuff,
    ncclRedOp_t op,
    int root);

void NCCLReduceScatter(const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff, ncclRedOp_t op);

void NCCLSend(const core::Context& ctx, const core::Tensor& sendbuff, int peer);

void NCCLRecv(const core::Context& ctx, core::Tensor& recvbuff, int peer);

void NCCLGroupStart();
void NCCLGroupEnd();
void NCCLGroupEndCheck(ncclComm_t comm);
int NCCLCommCount(const core::Context& ctx);
int NCCLCommUserRank(const core::Context& ctx);
} // namespace c10d
} // namespace bmengine