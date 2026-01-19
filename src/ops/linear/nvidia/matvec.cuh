#pragma once

#include "../../../device/nvidia/nvidia_common.cuh"

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

template <typename T, typename WType>
__global__ void matvec_kernel_warp_vec(T *c, const T *a, const WType *B, const T *bias,
                                               const size_t N, const size_t K, 
                                               const T *scale_ptr) { // 新增 Scale 参数
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];
    const size_t bid = blockIdx.x;
    const int warp_num = blockDim.x / WARP_SIZE;
    
    const WType *B_ptr = B + bid * K;
    
    float scale_val = 1.0f;
    if constexpr (std::is_same_v<WType, int8_t>) {
        if (scale_ptr) scale_val = to_float(scale_ptr[bid]);
    }

    const size_t tid = threadIdx.x;
    const size_t lane_id = tid % WARP_SIZE;

    float bias_val = bias ? to_float(bias[bid]) : 0.0f;

    constexpr int PackedSize = sizeof(float4) / sizeof(T);
    const size_t num_packs = K / PackedSize;
    const size_t remainder = K % PackedSize;

    float sum = 0.0f;

    for (size_t j = tid; j < num_packs; j += blockDim.x) {
        sum += dot_packed_128b<T, WType>(
            B_ptr + j * PackedSize, 
            a + j * PackedSize, 
            scale_val
        );
    }


    if (remainder > 0) {
        const size_t base = num_packs * PackedSize;
        for (size_t j = tid; j < remainder; j += blockDim.x) {
            float w_val;
            if constexpr (std::is_same_v<WType, int8_t>) {
                w_val = float(B_ptr[base + j]) * scale_val;
            } else {
                w_val = to_float(B_ptr[base + j]);
            }
            sum += w_val * to_float(a[base + j]);
        }
    }

    for (int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, stride);
    }

    if (lane_id == 0) {
        smem[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        float block_sum = (tid < warp_num) ? smem[tid] : 0.0f;
        for (int stride = WARP_SIZE >> 1; stride > 0; stride >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, stride);
        }
        if (tid == 0) {
            float result = block_sum + bias_val;
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
