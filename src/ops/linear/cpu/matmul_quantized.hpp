#pragma once
#include <cstdint>

void matmul_quantized(const float *A, const int8_t *B, float *C, int M, int N, int K, const float *scales);