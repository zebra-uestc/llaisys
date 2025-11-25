#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h> // for cudaError_t, cudaSuccess, cudaGetErrorString
#include <stdexcept>      // IWYU pragma: keep
#include <stdio.h>        // IWYU pragma: keep  for fprintf, stderr
#include <stdlib.h>       // for exit()

#define BLOCK_SIZE 256
constexpr inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;

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

// ============================================================================
// Type conversion utilities
// ============================================================================
template <typename T>
__device__ __forceinline__ float to_float(T val) { return static_cast<float>(val); }

template <>
__device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) { return __bfloat162float(val); }

template <typename T>
__device__ __forceinline__ T from_float(float val) { return static_cast<T>(val); }

template <>
__device__ __forceinline__ half from_float<half>(float val) { return __float2half(val); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) { return __float2bfloat16(val); }