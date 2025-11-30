#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "llaisys.h"
#include "rope_nvidia.cuh"

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// RoPE Kernel : Basic version - one thread per (seq, head, dim_pair)
// ============================================================================
template <typename T>
__global__ void rope_kernel(
    T *__restrict__ out,
    const T *__restrict__ in,
    const int64_t *__restrict__ pos_ids,
    const float theta,
    const int seqlen,
    const int nhead,
    const int d) {
    const int half_d = d / 2;

    // Global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = seqlen * nhead * half_d;

    if (idx >= total_elements) {
        return;
    }

    // Decompose index into (i, j, k) - sequence, head, dimension
    const int k = idx % half_d;
    const int j = (idx / half_d) % nhead;
    const int i = idx / (half_d * nhead);

    // Compute angle
    const float expo = (2.0f * k) / static_cast<float>(d);
    const float inv_theta = 1.0f / powf(theta, expo);
    const float angle = static_cast<float>(pos_ids[i]) * inv_theta;

    float cos_val, sin_val;
    sincosf(angle, &sin_val, &cos_val);

    // Input/output offset
    const int base_offset = (i * nhead + j) * d;

    // Load input values
    const float a = to_float(in[base_offset + k]);
    const float b = to_float(in[base_offset + k + half_d]);

    // Apply rotation
    out[base_offset + k] = from_float<T>(a * cos_val - b * sin_val);
    out[base_offset + k + half_d] = from_float<T>(b * cos_val + a * sin_val);
}

namespace llaisys::ops::nvidia {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {
    const size_t numel = seqlen * nhead * d / 2;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(div_ceil(numel, BLOCK_SIZE));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_kernel<<<gridDim, blockDim>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_kernel<<<gridDim, blockDim>>>(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_kernel<<<gridDim, blockDim>>>(reinterpret_cast<cuda_bfloat16 *>(out), reinterpret_cast<const cuda_bfloat16 *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
