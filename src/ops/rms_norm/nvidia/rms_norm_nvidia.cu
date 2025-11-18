#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "rms_norm_nvidia.cuh"

#include <cuda_runtime.h>
#include <type_traits>

namespace llaisys::ops::nvidia {

constexpr size_t WARP_SIZE = 32;

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FLOAT4_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])

// One block processes one row.
// Shared memory size must be blockDim.x * sizeof(float).
template <typename T>
__global__ void rmsnorm_kernel_smem(T *output, const T *input, const T *weight,
                                    const float eps, size_t len) {
    extern __shared__ float smem[]; // shared buffer for reductions
    const int tid = threadIdx.x;
    const T *row_in = input + blockIdx.x * len;
    T *row_out = output + blockIdx.x * len;

    // ---- Pass 1: compute squre sum ----
    float sqsum = 0.0f;
    for (size_t i = tid; i < len; i += blockDim.x) {
        float d = 0.0f;
        if constexpr (std::is_same_v<T, float>) {
            d = row_in[i];
        } else if constexpr (std::is_same_v<T, half>) {
            d = __half2float(row_in[i]);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            d = __bfloat162float(row_in[i]);
        }
        sqsum += d * d;
    }

    smem[tid] = sqsum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float var = smem[0] / (float)len;
    float inv_std = rsqrtf(var + eps);

    // ---- Normalize ----
    for (size_t i = tid; i < len; i += blockDim.x) {
        if constexpr (std::is_same_v<T, float>) {
            row_out[i] = row_in[i] * weight[i] * inv_std;
        } else if constexpr (std::is_same_v<T, half>) {
            row_out[i] = __float2half(__half2float(row_in[i]) * __half2float(weight[i]) * inv_std);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            row_out[i] = __float2bfloat16(__bfloat162float(row_in[i]) * __bfloat162float(weight[i]) * inv_std);
        }
    }
}

// One block processes one row.
// Shared memory size must be blockDim.x /  WARP_SIZE * sizeof(float).
template <typename T>
__global__ void rmsnorm_kernel_warp(T *output, const T *input, const T *weight,
                                    const float eps, size_t len) {
    extern __shared__ float smem[]; // shared buffer for reductions
    const int tid = threadIdx.x;
    const int lane_id = tid & 0x1f;
    const int warp_num = blockDim.x / WARP_SIZE;
    const T *row_in = input + blockIdx.x * len;
    T *row_out = output + blockIdx.x * len;

    // ---- Pass 1: compute square sum ----
    float sqsum = 0.0f;
    for (size_t i = tid; i < len; i += blockDim.x) {
        float d = 0.0f;
        if constexpr (std::is_same_v<T, float>) {
            d = row_in[i];
        } else if constexpr (std::is_same_v<T, half>) {
            d = __half2float(row_in[i]);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            d = __bfloat162float(row_in[i]);
        }
        sqsum += d * d;
    }

    for (size_t stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
        sqsum += __shfl_down_sync(0xffffffffu, sqsum, stride);
    }
    if (lane_id == 0) {
        smem[tid / WARP_SIZE] = sqsum;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        float block_sqsum = (tid < warp_num) ? smem[tid] : 0.0f;
        for (size_t offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
            block_sqsum += __shfl_down_sync(0xffffffffu, block_sqsum, offset);
        }
        if (tid == 0) {
            smem[tid] = block_sqsum;
        }
    }
    __syncthreads();

    float var = smem[0] / (float)len;
    float inv_std = rsqrtf(var + eps);

    // ---- Normalize ----
    for (size_t i = tid; i < len; i += blockDim.x) {
        if constexpr (std::is_same_v<T, float>) {
            row_out[i] = row_in[i] * weight[i] * inv_std;
        } else if constexpr (std::is_same_v<T, half>) {
            row_out[i] = __float2half(__half2float(row_in[i]) * __half2float(weight[i]) * inv_std);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            row_out[i] = __float2bfloat16(__bfloat162float(row_in[i]) * __bfloat162float(weight[i]) * inv_std);
        }
    }
}

// One block processes one row.
// Shared memory size must be blockDim.x * sizeof(float).
__global__ void rmsnorm_kernel_smem_vec(float *output, const float *input,
                                        const float *weight, const float eps,
                                        size_t len) {
    extern __shared__ float smem[]; // shared buffer for reductions
    const int tid = threadIdx.x;
    const float *row_in = input + blockIdx.x * len;
    float *row_out = output + blockIdx.x * len;
    size_t vec_len = len / 4;

    // ---- Pass 1: compute square sum ----
    float sqsum = 0.0f;
    for (size_t i = tid; i < vec_len; i += blockDim.x) {
        float d = 0.0f;
        const float4 v = FLOAT4_CONST(row_in[i * 4]);
        d = v.x;
        sqsum += d * d;
        d = v.y;
        sqsum += d * d;
        d = v.z;
        sqsum += d * d;
        d = v.w;
        sqsum += d * d;
    }

    smem[tid] = sqsum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float var = smem[0] / (float)len;
    float inv_std = rsqrtf(var + eps);

    // ---- Normalize ----
    for (size_t i = tid; i < vec_len; i += blockDim.x) {
        float4 res;
        const float4 v = FLOAT4_CONST(row_in[i * 4]);
        const float4 w = FLOAT4_CONST(weight[i * 4]);
        res.x = v.x * w.x * inv_std;
        res.y = v.y * w.y * inv_std;
        res.z = v.z * w.z * inv_std;
        res.w = v.w * w.w * inv_std;
        FLOAT4(row_out[i * 4]) = res;
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t nrow, size_t ncol) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(nrow);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rmsnorm_kernel_warp<<<grid_dim, block_dim>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, ncol);
    case LLAISYS_DTYPE_BF16:
        return rmsnorm_kernel_warp<<<grid_dim, block_dim>>>(reinterpret_cast<cuda_bfloat16 *>(out), reinterpret_cast<const cuda_bfloat16 *>(in),
                                                            reinterpret_cast<const cuda_bfloat16 *>(weight), eps, ncol);
    case LLAISYS_DTYPE_F16:
        return rmsnorm_kernel_warp<<<grid_dim, block_dim>>>(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
                                                            reinterpret_cast<const half *>(weight), eps, ncol);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
