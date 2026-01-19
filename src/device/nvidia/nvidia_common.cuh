#pragma once

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h> // for cudaError_t, cudaSuccess, cudaGetErrorString
#include <stdexcept>      // IWYU pragma: keep
#include <stdio.h>        // IWYU pragma: keep  for fprintf, stderr
#include <stdlib.h>       // for exit()

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t error = call;                                       \
        if (error != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,            \
                    cudaGetErrorString(error));                         \
            exit(1);                                                    \
        }                                                               \
    } while (0)

#define EXCEPTION_LOCATION_MSG \
    " from " << __func__ << " at " << __FILE__ << ":" << __LINE__ << "."

#define ASSERT(condition, message)                            \
    do {                                                      \
        if (!(condition)) {                                   \
            std::cerr << "[ERROR] " << message << std::endl   \
                      << "Assertion failed: " << #condition   \
                      << EXCEPTION_LOCATION_MSG << std::endl; \
            throw std::runtime_error("Assertion failed");     \
        }                                                     \
    } while (0)

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS 8

// Helper macros
#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FLOAT4_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LD128BITS_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define INT4_CONST(value) (reinterpret_cast<const int4 *>(&(value))[0])

using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;

// Ceiling division
__host__ __device__ constexpr size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

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
struct PackedUtils<half> {
    using PackedType = float4;
    static constexpr int pack_size = 8;
};

// BF16: 1 float4 = 8 bf16s
template <>
struct PackedUtils<cuda_bfloat16> {
    using PackedType = float4;
    static constexpr int pack_size = 8;
};

// INT8: 1 int4 = 16 int8s
template <>
struct PackedUtils<int8_t> {
    using PackedType = int4;
    static constexpr int pack_size = 16;
};

// ============================================================================
// Type conversion utilities
// ============================================================================
template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float(float x) { return x; }

template <>
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <>
__device__ __forceinline__ float to_float(int8_t x) { return static_cast<float>(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float(float x) { return x; }

template <>
__device__ __forceinline__ __half from_float(float x) { return __float2half(x); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }

template <>
__device__ __forceinline__ int8_t from_float(float x) { 
    int val = __float2int_rn(x); 
    return static_cast<int8_t>(max(-128, min(127, val)));
 }

// Load 128-bit data as float4 and compute dot product
template <typename T>
__device__ __forceinline__ float vec128_dot(const T *a, const T *b);

// Specialization for float: 4 elements per float4
template <>
__device__ __forceinline__ float vec128_dot<float>(const float *a, const float *b) {
    float4 a_vec = FLOAT4_CONST(a[0]);
    float4 b_vec = FLOAT4_CONST(b[0]);
    return a_vec.x * b_vec.x + a_vec.y * b_vec.y + a_vec.z * b_vec.z + a_vec.w * b_vec.w;
}

// Specialization for half: 8 elements per float4
template <>
__device__ __forceinline__ float vec128_dot<half>(const half *a, const half *b) {
    float4 a_vec = FLOAT4_CONST(a[0]);
    float4 b_vec = FLOAT4_CONST(b[0]);
    const half *a_h = reinterpret_cast<const half *>(&a_vec);
    const half *b_h = reinterpret_cast<const half *>(&b_vec);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum += to_float(a_h[i]) * to_float(b_h[i]);
    }
    return sum;
}

// Specialization for bfloat16: 8 elements per float4
template <>
__device__ __forceinline__ float vec128_dot<cuda_bfloat16>(const cuda_bfloat16 *a, const cuda_bfloat16 *b) {
    float4 a_vec = FLOAT4_CONST(a[0]);
    float4 b_vec = FLOAT4_CONST(b[0]);
    const cuda_bfloat16 *a_bf = reinterpret_cast<const cuda_bfloat16 *>(&a_vec);
    const cuda_bfloat16 *b_bf = reinterpret_cast<const cuda_bfloat16 *>(&b_vec);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum += to_float(a_bf[i]) * to_float(b_bf[i]);
    }
    return sum;
}

template <typename T, typename WType>
__forceinline__ __device__ float dot_packed_128b(const WType* w_ptr, const T* a_ptr, float scale);

template <>
__forceinline__ __device__ float dot_packed_128b<cuda_bfloat16, int8_t>(const int8_t* w_ptr, const cuda_bfloat16* a_ptr, float scale) {
    // 1. Load 16 bytes (128-bit) of data
    //    - Int8: load 16 weights (int4 × 4 or char16)
    //    - BF16: load 8 inputs (float4)
    // Note: stride sizes do not match here!
    // Standard packed_traits typically align to 128 bits.
    // For BF16, 128 bits hold 8 elements.
    // For Int8, 128 bits hold 16 elements.
    
    // To reuse the existing loop structure (iterate by input PackedSize), process 8 elements per step.
    // Input (BF16): 8 elements = 16 bytes (float4)
    // Weight (Int8): 8 elements = 8 bytes (int2)
    
    // Load input (8 × BF16)
    float4 a_vec = *reinterpret_cast<const float4*>(a_ptr); // 128-bit load
    // Load weights (8 × Int8)
    int2 w_vec = *reinterpret_cast<const int2*>(w_ptr);     // 64-bit load

    // Unpack and compute
    // Treat int2 as 8 int8 values
    const int8_t* w_i8 = reinterpret_cast<const int8_t*>(&w_vec);
    
    // Treat float4 as 8 BF16 values
    const cuda_bfloat16* a_bf16 = reinterpret_cast<const cuda_bfloat16*>(&a_vec);

    float partial_sum = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float w = static_cast<float>(w_i8[i]) * scale; // Fuse dequantization + scaling
        float a = __bfloat162float(a_bf16[i]);
        partial_sum += w * a;
    }
    return partial_sum;
}
