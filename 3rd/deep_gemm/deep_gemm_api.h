#pragma once

extern "C"
int deep_gemm_fp8_block_h20_group(
    void* __raw_lhs,
    void* __raw_lhs_scales,
    void* __raw_rhs,
    void* __raw_rhs_scales,
    void* __raw_out,
    void* __raw_grouped_layout,
    void* __raw_stream,
    int m,
    int n,
    int k,
    int block_m,
    int num_groups // =1 for normal Gemm
);