#include "../../../device/nvidia/nvidia_common.cuh"
#include "llaisys.h"
#include "swiglu_nvidia.cuh"

#include "../../../utils.hpp"
#include <type_traits>

namespace llaisys::ops::nvidia {

/**
 * Packed type traits
 */
template <typename T>
struct PackedUtils;

// FP32: 1 float4 = 4 floats
template <>
struct PackedUtils<float> {
    using PackedType = float4;
    static constexpr int pack_size = 4;
};

// FP16: 1 float4 = 8 halfs
template <>
struct PackedUtils<__half> {
    using PackedType = float4;
    static constexpr int pack_size = 8;
};

// BF16: 1 float4 = 8 bf16s
template <>
struct PackedUtils<__nv_bfloat16> {
    using PackedType = float4;
    static constexpr int pack_size = 8;
};

/**
 * Type conversion helpers
 */
template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float(float x) { return x; }

template <>
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float(float x) { return x; }

template <>
__device__ __forceinline__ __half from_float(float x) { return __float2half(x); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }

/**
 * SiLU: x / (1 + exp(-x))
 */
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

/**
 * SwiGLU kernel (vectorized)
 */
template <typename T>
__global__ void swiglu_kernel_vec(T *__restrict__ out,
                                  const T *__restrict__ gate,
                                  const T *__restrict__ up,
                                  const size_t N) {
    using Packed = typename PackedUtils<T>::PackedType;
    constexpr int pack_size = PackedUtils<T>::pack_size;

    // Number of full vector packs
    const size_t num_packs = N / pack_size;

    // Vector views
    Packed *out_packed = reinterpret_cast<Packed *>(out);
    const Packed *gate_packed = reinterpret_cast<const Packed *>(gate);
    const Packed *up_packed = reinterpret_cast<const Packed *>(up);

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    // -------------------------
    // Vectorized main loop
    // -------------------------
    for (size_t i = idx; i < num_packs; i += stride) {
        Packed g_vec = gate_packed[i];
        Packed u_vec = up_packed[i];
        Packed r_vec;

        // Treat packed data as scalar arrays
        T *g_arr = reinterpret_cast<T *>(&g_vec);
        T *u_arr = reinterpret_cast<T *>(&u_vec);
        T *r_arr = reinterpret_cast<T *>(&r_vec);

#pragma unroll
        for (int k = 0; k < pack_size; ++k) {
            float g_val = to_float(g_arr[k]);
            float u_val = to_float(u_arr[k]);
            float val = u_val * silu(g_val);
            r_arr[k] = from_float<T>(val);
        }

        out_packed[i] = r_vec;
    }

    // -------------------------
    // Scalar tail loop
    // -------------------------
    const size_t processed = num_packs * pack_size;

    for (size_t i = processed + idx; i < N; i += stride) {
        float g_val = to_float(gate[i]);
        float u_val = to_float(up[i]);
        float val = u_val * silu(g_val);
        out[i] = from_float<T>(val);
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
            const float gate_v = __half2float(gate[idx]);
            const float up_v = __half2float(up[idx]);
            out[idx] = __float2half(up_v * silu(gate_v));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            const float gate_v = __bfloat162float(gate[idx]);
            const float up_v = __bfloat162float(up[idx]);
            out[idx] = __float2bfloat16(up_v * silu(gate_v));
        }
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil_div(numel, BLOCK_SIZE));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_kernel_naive<<<gridDim, blockDim>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_kernel_naive<<<gridDim, blockDim>>>(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(gate), reinterpret_cast<const half *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_kernel_naive<<<gridDim, blockDim>>>(reinterpret_cast<cuda_bfloat16 *>(out), reinterpret_cast<const cuda_bfloat16 *>(gate), reinterpret_cast<const cuda_bfloat16 *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
