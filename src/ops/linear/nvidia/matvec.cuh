#pragma once

#include "../../../device/nvidia/nvidia_common.cuh"

#define FLOAT4_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])

__global__ void matvec_kernel_warp(float *c, const float *a, const float *B, const float *bias, const size_t N, const size_t K) {
    size_t bid = blockIdx.x;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];
    const int warp_num = blockDim.x / WARP_SIZE;
    for (size_t i = bid; i < N; i += gridDim.x) {
        const float *B_ptr = B + i * K;
        size_t tid = threadIdx.x;
        size_t lane_id = tid % 32;

        float sum = 0.0f;
        for (size_t j = tid; j < K; j += blockDim.x) {
            sum += a[j] * B_ptr[j];
        }
        for (size_t stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, stride);
        }
        if (lane_id == 0) {
            smem[tid / 32] = sum;
        }
        __syncthreads();

        if (tid < WARP_SIZE) {
            float block_sum = (tid < warp_num) ? smem[tid] : 0.0f;
            for (size_t stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
                block_sum += __shfl_down_sync(0xffffffffu, block_sum, stride);
            }
            if (tid == 0) {
                smem[tid] = block_sum;
                c[i] = smem[0] + bias[i];
            }
        }
        __syncthreads();
    }
}

// Load 128-bit data as float4 and compute dot product
template <typename T>
__device__ __forceinline__ float vec128_dot(const T *a, const T *b);

// Specialization for float: 4 elements per float4
template <>
__device__ __forceinline__ float vec128_dot<float>(const float *a, const float *b) {
    float4 a_vec = *reinterpret_cast<const float4 *>(a);
    float4 b_vec = *reinterpret_cast<const float4 *>(b);
    return a_vec.x * b_vec.x + a_vec.y * b_vec.y + a_vec.z * b_vec.z + a_vec.w * b_vec.w;
}

// Specialization for half: 8 elements per float4
template <>
__device__ __forceinline__ float vec128_dot<half>(const half *a, const half *b) {
    float4 a_vec = *reinterpret_cast<const float4 *>(a);
    float4 b_vec = *reinterpret_cast<const float4 *>(b);
    const half *a_h = reinterpret_cast<const half *>(&a_vec);
    const half *b_h = reinterpret_cast<const half *>(&b_vec);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum += __half2float(a_h[i]) * __half2float(b_h[i]);
    }
    return sum;
}

// Specialization for bfloat16: 8 elements per float4
template <>
__device__ __forceinline__ float vec128_dot<cuda_bfloat16>(const cuda_bfloat16 *a, const cuda_bfloat16 *b) {
    float4 a_vec = *reinterpret_cast<const float4 *>(a);
    float4 b_vec = *reinterpret_cast<const float4 *>(b);
    const cuda_bfloat16 *a_bf = reinterpret_cast<const cuda_bfloat16 *>(&a_vec);
    const cuda_bfloat16 *b_bf = reinterpret_cast<const cuda_bfloat16 *>(&b_vec);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum += __bfloat162float(a_bf[i]) * __bfloat162float(b_bf[i]);
    }
    return sum;
}

// Warp-based matvec kernel with vectorized loads
template <typename T>
__global__ void matvec_kernel_warp_vec(T *c, const T *a, const T *B, const T *bias,
                                       const size_t N, const size_t K) {
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];
    const size_t bid = blockIdx.x;
    const int warp_num = blockDim.x / WARP_SIZE;
    const T *B_ptr = B + bid * K;
    const size_t tid = threadIdx.x;
    const size_t lane_id = tid % WARP_SIZE;

    constexpr int vec_size = sizeof(float4) / sizeof(T); // 4 for float, 8 for half/bf16
    const size_t vec_len = K / vec_size;
    const size_t remainder = K % vec_size;

    float sum = 0.0f;

    // Vectorized accumulation
    for (size_t j = tid; j < vec_len; j += blockDim.x) {
        sum += vec128_dot<T>(B_ptr + j * vec_size, a + j * vec_size);
    }

    // Handle remaining elements
    if (remainder > 0) {
        const size_t base = vec_len * vec_size;
        for (size_t j = tid; j < remainder; j += blockDim.x) {
            sum += float(B_ptr[base + j]) * float(a[base + j]);
        }
    }

    // Warp-level reduction
    for (int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, stride);
    }

    if (lane_id == 0) {
        smem[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    // Block-level reduction
    if (tid < WARP_SIZE) {
        float block_sum = (tid < warp_num) ? smem[tid] : 0.0f;
        for (int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, stride);
        }
        if (tid == 0) {
            float result = block_sum + float(bias[bid]);
            if constexpr (std::is_same_v<T, float>) {
                c[bid] = result;
            } else if constexpr (std::is_same_v<T, half>) {
                c[bid] = __float2half_rn(result);
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                c[bid] = __float2bfloat16_rn(result);
            }
        }
    }
}
