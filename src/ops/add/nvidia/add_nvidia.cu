#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "add_nvidia.cuh"

#include <cuda_runtime.h>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compile-time dispatch for scalar operations based on type T
        if constexpr (std::is_same_v<T, float>) {
            c[idx] = a[idx] + b[idx];
        } else if constexpr (std::is_same_v<T, half>) {
            c[idx] = __hadd(a[idx], b[idx]); // Half-precision intrinsic
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            c[idx] = __hadd(a[idx], b[idx]); // BF16 intrinsic
        }
    }
}

template <typename T>
__global__ void add_kernel_vec(T *c, const T *a, const T *b, size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        // Float: Process 4 elements per thread (128-bit load/store)
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

        if (idx + 3 < n) {
            // Vectorized Load: Fetch 128-bit data into registers
            float4 a_vec = FLOAT4_CONST(a[idx]);
            float4 b_vec = FLOAT4_CONST(b[idx]);
            float4 c_vec;

            // Vectorized Math: Compute components manually
            c_vec.x = a_vec.x + b_vec.x;
            c_vec.y = a_vec.y + b_vec.y;
            c_vec.z = a_vec.z + b_vec.z;
            c_vec.w = a_vec.w + b_vec.w;

            // Vectorized Store
            FLOAT4(c[idx]) = c_vec;
        } else if (idx < n) {
            // Tail Handling: Process remaining elements scalar-wise
            for (size_t i = idx; i < n; i++) {
                c[i] = a[i] + b[i];
            }
        }

    } else if constexpr (std::is_same_v<T, half>) {
        // Half: Process 8 elements per thread (8 * 16-bit = 128-bit)
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

        if (idx + 7 < n) {
            // Use float4 as a 128-bit container for memory access
            float4 a_vec = FLOAT4_CONST(a[idx]);
            float4 b_vec = FLOAT4_CONST(b[idx]);
            float4 c_vec;

            // Reinterpret cast: Treat 128-bit data as array of half2
            half2 *a_h2 = reinterpret_cast<half2 *>(&a_vec);
            half2 *b_h2 = reinterpret_cast<half2 *>(&b_vec);
            half2 *c_h2 = reinterpret_cast<half2 *>(&c_vec);

            // SIMD Math: Perform 4 x half2 additions (8 elements total)
            c_h2[0] = __hadd2(a_h2[0], b_h2[0]);
            c_h2[1] = __hadd2(a_h2[1], b_h2[1]);
            c_h2[2] = __hadd2(a_h2[2], b_h2[2]);
            c_h2[3] = __hadd2(a_h2[3], b_h2[3]);

            FLOAT4(c[idx]) = c_vec;
        } else if (idx < n) {
            // Tail Handling
            for (size_t i = idx; i < n; i++) {
                c[i] = __hadd(a[i], b[i]);
            }
        }

    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        // Bfloat16: Process 8 elements per thread (same logic as half)
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

        if (idx + 7 < n) {
            float4 a_vec = FLOAT4_CONST(a[idx]);
            float4 b_vec = FLOAT4_CONST(b[idx]);
            float4 c_vec;

            // Reinterpret cast to bf16 vector type
            cuda_bfloat162 *a_bf2 = reinterpret_cast<cuda_bfloat162 *>(&a_vec);
            cuda_bfloat162 *b_bf2 = reinterpret_cast<cuda_bfloat162 *>(&b_vec);
            cuda_bfloat162 *c_bf2 = reinterpret_cast<cuda_bfloat162 *>(&c_vec);

            // SIMD Math for bf16 pairs
            c_bf2[0] = __hadd2(a_bf2[0], b_bf2[0]);
            c_bf2[1] = __hadd2(a_bf2[1], b_bf2[1]);
            c_bf2[2] = __hadd2(a_bf2[2], b_bf2[2]);
            c_bf2[3] = __hadd2(a_bf2[3], b_bf2[3]);

            FLOAT4(c[idx]) = c_vec;
        } else if (idx < n) {
            // Tail Handling
            for (size_t i = idx; i < n; i++) {
                c[i] = __hadd(a[i], b[i]);
            }
        }
    } else if constexpr (std::is_same_v<T, int8_t>) {
        // Int8: Process 16 elements per thread (16 * 8-bit = 128-bit)
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;

        if (idx + 15 < n) {
            // Use int4 as a 128-bit container for memory access
            int4 a_vec = INT4_CONST(a[idx]);
            int4 b_vec = INT4_CONST(b[idx]);
            int4 c_vec;

            // Reinterpret cast: Treat 128-bit data as array of int8_t[16]
            int8_t *a_i8 = reinterpret_cast<int8_t *>(&a_vec);
            int8_t *b_i8 = reinterpret_cast<int8_t *>(&b_vec);
            int8_t *c_i8 = reinterpret_cast<int8_t *>(&c_vec);

            // SIMD-like Math: Perform element-wise addition
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                c_i8[i] = a_i8[i] + b_i8[i];
            }

            INT4(c[idx]) = c_vec;
        } else if (idx < n) {
            // Tail Handling
            for (size_t i = idx; i < n; i++) {
                c[i] = a[i] + b[i];
            }
        }
    }
}

template <typename T>
void launch_add_kernel(T *c, const T *a, const T *b, size_t n) {
    // Determine packing size (4 for float, 8 for half/bf16)
    constexpr size_t VEC_SIZE = PackedUtils<T>::pack_size;
    dim3 block_dim(BLOCK_SIZE);

    // Adjust grid size to account for vectorized processing
    dim3 grid_dim(div_ceil(n, BLOCK_SIZE * VEC_SIZE));

    add_kernel_vec<<<grid_dim, block_dim>>>(c, a, b, n);
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_add_kernel(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return launch_add_kernel(reinterpret_cast<cuda_bfloat16 *>(c), reinterpret_cast<const cuda_bfloat16 *>(a),
                                 reinterpret_cast<const cuda_bfloat16 *>(b), numel);
    case LLAISYS_DTYPE_F16:
        return launch_add_kernel(reinterpret_cast<half *>(c), reinterpret_cast<const half *>(a),
                                 reinterpret_cast<const half *>(b), numel);
    case LLAISYS_DTYPE_I8:
        return launch_add_kernel(reinterpret_cast<int8_t *>(c), reinterpret_cast<const int8_t *>(a),
                                 reinterpret_cast<const int8_t *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia