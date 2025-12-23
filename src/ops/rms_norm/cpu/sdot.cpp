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

int32_t sdot_int8(const int8_t *x, const int8_t *y, const size_t N) {
    int32_t final_sum = 0;
    size_t i = 0;

#if defined(__AVX512VNNI__)
    size_t vec_size = 64;
    size_t mod = N % vec_size;
    size_t align = N - mod;

    __m512i acc_dot = _mm512_setzero_si512();
    __m512i acc_sum_y = _mm512_setzero_si512();

    const __m512i v_flip = _mm512_set1_epi8((int8_t)0x80);
    const __m512i v_ones = _mm512_set1_epi8(1);

    for (; i < align; i += vec_size) {
        __m512i vx = _mm512_loadu_si512((const __m512i*)(x + i));
        __m512i vy = _mm512_loadu_si512((const __m512i*)(y + i));

        __m512i vx_unsigned = _mm512_xor_si512(vx, v_flip);

        acc_dot = _mm512_dpbusd_epi32(acc_dot, vx_unsigned, vy);

        acc_sum_y = _mm512_dpbusd_epi32(acc_sum_y, v_ones, vy);
    }

    int32_t dot_val = _mm512_reduce_add_epi32(acc_dot);
    int32_t sum_y_val = _mm512_reduce_add_epi32(acc_sum_y);

    final_sum = dot_val - (sum_y_val << 7);

#else
    size_t vec_size = 64;
    size_t mod = N % vec_size;
    size_t align = N - mod;

    __m512i accum = _mm512_setzero_si512();

    for (; i < align; i += 64) {
        __m512i vx_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)(x + i)));
        __m512i vy_lo = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)(y + i)));
        accum = _mm512_add_epi32(accum, _mm512_madd_epi16(vx_lo, vy_lo));

        __m512i vx_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)(x + i + 32)));
        __m512i vy_hi = _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i*)(y + i + 32)));
        accum = _mm512_add_epi32(accum, _mm512_madd_epi16(vx_hi, vy_hi));
    }
    final_sum = _mm512_reduce_add_epi32(accum);
#endif

    for (; i < N; ++i) {
        final_sum += static_cast<int32_t>(x[i]) * static_cast<int32_t>(y[i]);
    }

    return final_sum;
}
} // namespace llaisys::ops::cpu