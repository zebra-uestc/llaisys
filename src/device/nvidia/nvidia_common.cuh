#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 256
constexpr inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;