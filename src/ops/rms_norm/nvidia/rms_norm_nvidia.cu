#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "rms_norm_nvidia.cuh"

#include <cuda_runtime.h>
#include <type_traits>

namespace llaisys::ops::nvidia {

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
    __shared__ float smem[NUM_WARPS]; // shared buffer for reductions
    const int tid = threadIdx.x;
    const int lane_id = tid & 0x1f;
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
        float block_sqsum = (tid < NUM_WARPS) ? smem[tid] : 0.0f;
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
// Shared memory size must be blockDim.x /  WARP_SIZE * sizeof(float).
template <typename T>
__global__ void rmsnorm_kernel_vec(T *output, const T *input,
                                   const T *weight, const float eps,
                                   size_t len) {
    __shared__ float smem[NUM_WARPS]; // shared buffer for reductions
    const int tid = threadIdx.x;
    const int lane_id = tid & 0x1f;

    const T *row_in = input + blockIdx.x * len;
    T *row_out = output + blockIdx.x * len;

    constexpr size_t VEC_SIZE = PackedUtils<T>::pack_size;
    size_t vec_len = len / VEC_SIZE;

    // ---- Pass 1: compute square sum ----
    float sqsum = 0.0f;
    for (size_t i = tid; i < vec_len; i += blockDim.x) {
        sqsum += vec128_dot<T>(row_in + i * VEC_SIZE, row_in + i * VEC_SIZE);
    }

    size_t offset = vec_len * VEC_SIZE;
    for (size_t i = offset + tid; i < len; i += blockDim.x) {
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
        float block_sqsum = (tid < NUM_WARPS) ? smem[tid] : 0.0f;
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
    for (size_t i = tid; i < vec_len; i += blockDim.x) {
        float4 res;
        if constexpr (std::is_same_v<T, float>) {
            const float4 v = FLOAT4_CONST(row_in[i * VEC_SIZE]);
            const float4 w = FLOAT4_CONST(weight[i * VEC_SIZE]);
            res.x = v.x * w.x * inv_std;
            res.y = v.y * w.y * inv_std;
            res.z = v.z * w.z * inv_std;
            res.w = v.w * w.w * inv_std;
        } else if constexpr (std::is_same_v<T, half>) {
            const float4 v = FLOAT4_CONST(row_in[i * VEC_SIZE]);
            const float4 w = FLOAT4_CONST(weight[i * VEC_SIZE]);
            const half *v_half = reinterpret_cast<const half *>(&v);
            const half *w_half = reinterpret_cast<const half *>(&w);
            half *res_half = reinterpret_cast<half *>(&res);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float res_f = to_float(v_half[i]) * to_float(w_half[i]) * inv_std;
                res_half[i] = from_float<half>(res_f);
            }
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            const float4 v = FLOAT4_CONST(row_in[i * VEC_SIZE]);
            const float4 w = FLOAT4_CONST(weight[i * VEC_SIZE]);
            const cuda_bfloat16 *v_bf = reinterpret_cast<const cuda_bfloat16 *>(&v);
            const cuda_bfloat16 *w_bf = reinterpret_cast<const cuda_bfloat16 *>(&w);
            cuda_bfloat16 *res_bf = reinterpret_cast<cuda_bfloat16 *>(&res);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float res_f = to_float(v_bf[i]) * to_float(w_bf[i]) * inv_std;
                res_bf[i] = from_float<cuda_bfloat16>(res_f);
            }
        }
        FLOAT4(row_out[i * VEC_SIZE]) = res;
    }

    for (size_t i = offset + tid; i < len; i += blockDim.x) {
        if constexpr (std::is_same_v<T, float>) {
            row_out[i] = row_in[i] * weight[i] * inv_std;
        } else if constexpr (std::is_same_v<T, half>) {
            row_out[i] = __float2half(__half2float(row_in[i]) * __half2float(weight[i]) * inv_std);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            row_out[i] = __float2bfloat16(__bfloat162float(row_in[i]) * __bfloat162float(weight[i]) * inv_std);
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t nrow, size_t ncol) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(nrow);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rmsnorm_kernel_vec<<<grid_dim, block_dim>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, ncol);
    case LLAISYS_DTYPE_BF16:
        return rmsnorm_kernel_vec<<<grid_dim, block_dim>>>(reinterpret_cast<cuda_bfloat16 *>(out), reinterpret_cast<const cuda_bfloat16 *>(in),
                                                            reinterpret_cast<const cuda_bfloat16 *>(weight), eps, ncol);
    case LLAISYS_DTYPE_F16:
        return rmsnorm_kernel_vec<<<grid_dim, block_dim>>>(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
                                                            reinterpret_cast<const half *>(weight), eps, ncol);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
