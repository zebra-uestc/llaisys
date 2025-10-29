#include "vecmul.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

void vecmul(const float *a, const float *B, float *c, int N, int K) {
    // Fall back to scalar implementation for small problems
    if (K < 64 || N < 4) {
        for (int i = 0; i < N; i++) {
            float sum = 0.0f;
            const float *B_row = B + i * K;
            for (int k = 0; k < K; k++) {
                sum += a[k] * B_row[k];
            }
            c[i] += sum;
        }
        return;
    }

// Parallel processing over output rows
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            const float *B_row = B + (long long)i * K;

            // Use 8 accumulators to reduce dependency chain latency
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();
            __m512 sum4 = _mm512_setzero_ps();
            __m512 sum5 = _mm512_setzero_ps();
            __m512 sum6 = _mm512_setzero_ps();
            __m512 sum7 = _mm512_setzero_ps();

            int k = 0;

            // Main loop: process 128 floats per iteration (8x unroll, 16 floats each)
            for (; k + 127 < K; k += 128) {
                __m512 a0 = _mm512_loadu_ps(a + k);
                __m512 a1 = _mm512_loadu_ps(a + k + 16);
                __m512 a2 = _mm512_loadu_ps(a + k + 32);
                __m512 a3 = _mm512_loadu_ps(a + k + 48);
                __m512 a4 = _mm512_loadu_ps(a + k + 64);
                __m512 a5 = _mm512_loadu_ps(a + k + 80);
                __m512 a6 = _mm512_loadu_ps(a + k + 96);
                __m512 a7 = _mm512_loadu_ps(a + k + 112);

                __m512 b0 = _mm512_loadu_ps(B_row + k);
                __m512 b1 = _mm512_loadu_ps(B_row + k + 16);
                __m512 b2 = _mm512_loadu_ps(B_row + k + 32);
                __m512 b3 = _mm512_loadu_ps(B_row + k + 48);
                __m512 b4 = _mm512_loadu_ps(B_row + k + 64);
                __m512 b5 = _mm512_loadu_ps(B_row + k + 80);
                __m512 b6 = _mm512_loadu_ps(B_row + k + 96);
                __m512 b7 = _mm512_loadu_ps(B_row + k + 112);

                // Fused multiply-add
                sum0 = _mm512_fmadd_ps(a0, b0, sum0);
                sum1 = _mm512_fmadd_ps(a1, b1, sum1);
                sum2 = _mm512_fmadd_ps(a2, b2, sum2);
                sum3 = _mm512_fmadd_ps(a3, b3, sum3);
                sum4 = _mm512_fmadd_ps(a4, b4, sum4);
                sum5 = _mm512_fmadd_ps(a5, b5, sum5);
                sum6 = _mm512_fmadd_ps(a6, b6, sum6);
                sum7 = _mm512_fmadd_ps(a7, b7, sum7);
            }

            // Handle remaining 64-127 elements
            for (; k + 63 < K; k += 64) {
                __m512 a0 = _mm512_loadu_ps(a + k);
                __m512 a1 = _mm512_loadu_ps(a + k + 16);
                __m512 a2 = _mm512_loadu_ps(a + k + 32);
                __m512 a3 = _mm512_loadu_ps(a + k + 48);

                __m512 b0 = _mm512_loadu_ps(B_row + k);
                __m512 b1 = _mm512_loadu_ps(B_row + k + 16);
                __m512 b2 = _mm512_loadu_ps(B_row + k + 32);
                __m512 b3 = _mm512_loadu_ps(B_row + k + 48);

                sum0 = _mm512_fmadd_ps(a0, b0, sum0);
                sum1 = _mm512_fmadd_ps(a1, b1, sum1);
                sum2 = _mm512_fmadd_ps(a2, b2, sum2);
                sum3 = _mm512_fmadd_ps(a3, b3, sum3);
            }

            // Handle remaining 32-63 elements
            for (; k + 31 < K; k += 32) {
                __m512 a0 = _mm512_loadu_ps(a + k);
                __m512 a1 = _mm512_loadu_ps(a + k + 16);

                __m512 b0 = _mm512_loadu_ps(B_row + k);
                __m512 b1 = _mm512_loadu_ps(B_row + k + 16);

                sum0 = _mm512_fmadd_ps(a0, b0, sum0);
                sum1 = _mm512_fmadd_ps(a1, b1, sum1);
            }

            // Handle remaining 16-31 elements
            for (; k + 15 < K; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(a + k);
                __m512 b_vec = _mm512_loadu_ps(B_row + k);
                sum0 = _mm512_fmadd_ps(a_vec, b_vec, sum0);
            }

            // Reduce 8 accumulators into one
            sum0 = _mm512_add_ps(sum0, sum1);
            sum2 = _mm512_add_ps(sum2, sum3);
            sum4 = _mm512_add_ps(sum4, sum5);
            sum6 = _mm512_add_ps(sum6, sum7);

            sum0 = _mm512_add_ps(sum0, sum2);
            sum4 = _mm512_add_ps(sum4, sum6);
            sum0 = _mm512_add_ps(sum0, sum4);

            // Horizontal reduction using AVX-512 intrinsic
            float sum = _mm512_reduce_add_ps(sum0);

            // Scalar tail processing for remaining elements (<16)
            for (; k < K; k++) {
                sum += a[k] * B_row[k];
            }

            c[i] += sum;
        }
    }
}