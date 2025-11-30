#include "../../../device/nvidia/nvidia_common.cuh"
#include "llaisys.h"
#include "swiglu_nvidia.cuh"

#include "../../../utils.hpp"
#include <type_traits>

namespace llaisys::ops::nvidia {

/**
 * SiLU: x / (1 + exp(-x))
 */
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

template <typename T>
__global__ void swiglu_kernel_vec(T *__restrict__ out,
                                  const T *__restrict__ gate,
                                  const T *__restrict__ up,
                                  size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        // ---------------------------------------------------
        // Float: Process 4 elements per thread (128-bit)
        // ---------------------------------------------------
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

        if (idx + 3 < n) {
            // Vectorized Load
            float4 g_vec = FLOAT4_CONST(gate[idx]);
            float4 u_vec = FLOAT4_CONST(up[idx]);
            float4 out_vec;

            // Vectorized Math: Compute components manually
            out_vec.x = u_vec.x * silu(g_vec.x);
            out_vec.y = u_vec.y * silu(g_vec.y);
            out_vec.z = u_vec.z * silu(g_vec.z);
            out_vec.w = u_vec.w * silu(g_vec.w);

            // Vectorized Store
            FLOAT4(out[idx]) = out_vec;
        } else if (idx < n) {
            // Tail Handling
            for (size_t i = idx; i < n; i++) {
                out[i] = up[i] * silu(gate[i]);
            }
        }

    } else if constexpr (std::is_same_v<T, half>) {
        // ---------------------------------------------------
        // Half: Process 8 elements per thread (8 * 16-bit = 128-bit)
        // ---------------------------------------------------
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

        if (idx + 7 < n) {
            // Use float4 as a 128-bit container
            float4 g_vec = FLOAT4_CONST(gate[idx]);
            float4 u_vec = FLOAT4_CONST(up[idx]);
            float4 out_vec;

            // Reinterpret cast: Treat 128-bit data as array of half2
            const half2 *g_h2 = reinterpret_cast<const half2 *>(&g_vec);
            const half2 *u_h2 = reinterpret_cast<const half2 *>(&u_vec);
            half2 *out_h2 = reinterpret_cast<half2 *>(&out_vec);

// SIMD Math loop (4 pairs of half2 = 8 elements)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                // 由于标准库没有直接的 __hsilu2，通常先转为 float 计算以保证精度
                float2 g_f2 = __half22float2(g_h2[k]);
                float2 u_f2 = __half22float2(u_h2[k]);

                float2 res_f2;
                res_f2.x = u_f2.x * silu(g_f2.x);
                res_f2.y = u_f2.y * silu(g_f2.y);

                out_h2[k] = __float22half2_rn(res_f2);
            }

            FLOAT4(out[idx]) = out_vec;
        } else if (idx < n) {
            // Tail Handling
            for (size_t i = idx; i < n; i++) {
                float g_val = __half2float(gate[i]);
                float u_val = __half2float(up[i]);
                out[i] = __float2half(u_val * silu(g_val));
            }
        }

    } else if constexpr (std::is_same_v<T, nv_bfloat16>) { // 注意: CUDA中通常用 nv_bfloat16
        // ---------------------------------------------------
        // Bfloat16: Process 8 elements per thread
        // ---------------------------------------------------
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

        if (idx + 7 < n) {
            float4 g_vec = FLOAT4_CONST(gate[idx]);
            float4 u_vec = FLOAT4_CONST(up[idx]);
            float4 out_vec;

            // Reinterpret cast to bf16 vector type
            const nv_bfloat162 *g_bf2 = reinterpret_cast<const nv_bfloat162 *>(&g_vec);
            const nv_bfloat162 *u_bf2 = reinterpret_cast<const nv_bfloat162 *>(&u_vec);
            nv_bfloat162 *out_bf2 = reinterpret_cast<nv_bfloat162 *>(&out_vec);

// SIMD Math
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                // Convert to float for SiLU calculation
                float2 g_f2 = __bfloat1622float2(g_bf2[k]);
                float2 u_f2 = __bfloat1622float2(u_bf2[k]);

                float2 res_f2;
                res_f2.x = u_f2.x * silu(g_f2.x);
                res_f2.y = u_f2.y * silu(g_f2.y);

                out_bf2[k] = __float22bfloat162_rn(res_f2);
            }

            FLOAT4(out[idx]) = out_vec;
        } else if (idx < n) {
            // Tail Handling
            for (size_t i = idx; i < n; i++) {
                float g_val = __bfloat162float(gate[i]);
                float u_val = __bfloat162float(up[i]);
                out[i] = __float2bfloat16(u_val * silu(g_val));
            }
        }
    }
}

template <typename T>
__global__ void swiglu_kernel_naive(T *out, const T *gate, const T *up, const size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if constexpr (std::is_same_v<T, float>) {
            const float gate_v = gate[idx];
            const float up_v = up[idx];
            out[idx] = up_v * silu(gate_v);
        } else if constexpr (std::is_same_v<T, half>) {
            const float gate_v = to_float(gate[idx]);
            const float up_v = to_float(up[idx]);
            out[idx] = from_float<half>(up_v * silu(gate_v));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            const float gate_v = to_float(gate[idx]);
            const float up_v = to_float(up[idx]);
            out[idx] = from_float<cuda_bfloat16>(up_v * silu(gate_v));
        }
    }
}

template <typename T>
void launch_swiglu_kernel(T *out, const T *gate, const T *up, const size_t N) {
    // Determine packing size (4 for float, 8 for half/bf16)
    constexpr size_t VEC_SIZE = PackedUtils<T>::pack_size;
    dim3 block_dim(BLOCK_SIZE);

    // Adjust grid size to account for vectorized processing
    dim3 grid_dim(div_ceil(N, BLOCK_SIZE * VEC_SIZE));

    swiglu_kernel_vec<<<grid_dim, block_dim>>>(out, gate, up, N);
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    // dim3 blockDim(BLOCK_SIZE);
    // dim3 gridDim(div_ceil(numel, BLOCK_SIZE));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_swiglu_kernel(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return launch_swiglu_kernel(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(gate), reinterpret_cast<const half *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return launch_swiglu_kernel(reinterpret_cast<cuda_bfloat16 *>(out), reinterpret_cast<const cuda_bfloat16 *>(gate), reinterpret_cast<const cuda_bfloat16 *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
