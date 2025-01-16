#pragma once

static __device__ __forceinline__ int get_half_dim_index(
    int col, int half_dim, bool neox_style
) {
    if (neox_style) {
        return (col < half_dim) ? col : (col - half_dim); // = col % half_dim
    } else {
        return col / 2;
    }
}

template<typename T>
static __device__ __forceinline__ float rope_one_value(
    const T *__restrict__ in,
    int offset,
    float cos_freq,
    float sin_freq,
    int col, int half_dim, bool neox_style
) {
    if (neox_style) {
        if (col < half_dim) {
            return float(in[offset]) * cos_freq - float(in[offset + half_dim]) * sin_freq;
        } else {
            return float(in[offset]) * cos_freq + float(in[offset - half_dim]) * sin_freq;
        }
    } else {
        if ((col % 2) == 0) {
            return float(in[offset]) * cos_freq - float(in[offset + 1]) * sin_freq;
        } else {
            return float(in[offset]) * cos_freq + float(in[offset - 1]) * sin_freq;
        }
    }
}