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
