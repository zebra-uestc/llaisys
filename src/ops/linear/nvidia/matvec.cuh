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

__global__ void matvec_kernel_warp_vec(float *c, const float *a, const float *B, const float *bias, const size_t N, const size_t K) {
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];
    size_t bid = blockIdx.x;
    const int warp_num = blockDim.x / WARP_SIZE;

    const float *B_ptr = B + bid * K;
    size_t tid = threadIdx.x;
    size_t lane_id = tid % 32;

    size_t vec_len = K / 4;
    float sum = 0.0f;
    for (size_t j = tid; j < vec_len; j += blockDim.x) {
        const float4 matval = FLOAT4_CONST(B_ptr[j * 4]);
        const float4 vecval = FLOAT4_CONST(a[j * 4]);
        sum += (matval.x * vecval.x + matval.y * vecval.y + matval.z * vecval.z + matval.w * vecval.w);
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
            c[bid] = smem[0] + bias[bid];
        }
    }
    __syncthreads();
}