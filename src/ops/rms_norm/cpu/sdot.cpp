#include <cstddef>
#include <cstdio>
#include <immintrin.h>

namespace llaisys::ops::cpu {
float sdot(const float *x, const float *y, const size_t N) {
    size_t mod = N % 16;
    size_t align = N - mod;
    __m512 accum = _mm512_setzero_ps();

    for (int i = 0; i <= (int)align - 16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        accum = _mm512_fmadd_ps(vx, vy, accum);
    }

    float sum = 0.0f;
    sum = _mm512_reduce_add_ps(accum);
    for (size_t i = align; i < N; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}
} // namespace llaisys::ops::cpu