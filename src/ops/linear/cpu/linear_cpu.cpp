#include "linear_cpu.hpp"
#include "matmul.hpp"
#include "matmul_quantized.hpp"
#include "vecmul.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <type_traits>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t nrow, size_t ncol_out, size_t ncol_in) {

    const size_t M = nrow;
    const size_t N = ncol_out;
    const size_t K = ncol_in;

    if constexpr (std::is_same_v<T, float>) {
        if (bias) {
            for (size_t i = 0; i < M; ++i) {
                std::memcpy(out + i * N, bias, N * sizeof(T));
            }
        } else {
            memset(out, 0, M * N * sizeof(T));
        }
        if (M == 1) {
            vecmul(in, weight, out, N, K);
        } else {
            matmul(in, weight, out, M, N, K);
        }

    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        std::vector<float> in_fp32(M * K);
        std::vector<float> weight_fp32(N * K);
        std::vector<float> out_fp32(M * N, {});

        llaisys::utils::fp16_to_fp32_batch_f16c(in_fp32.data(), in, M * K);
        llaisys::utils::fp16_to_fp32_batch_f16c(weight_fp32.data(), weight, N * K);

        if (bias) {
            std::vector<float> bias_fp32(N);
            llaisys::utils::fp16_to_fp32_batch_f16c(bias_fp32.data(), bias, N);

            for (size_t i = 0; i < M; ++i) {
                std::memcpy(out_fp32.data() + i * N, bias_fp32.data(), N * sizeof(float));
            }
        }
        
        if (M == 1) {
            vecmul(in_fp32.data(), weight_fp32.data(), out_fp32.data(), N, K);
        } else {
            matmul(in_fp32.data(), weight_fp32.data(), out_fp32.data(), M, N, K);
        }

        llaisys::utils::fp32_to_fp16_batch_f16c(out, out_fp32.data(), M * N);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        std::vector<float> in_fp32(M * K);
        std::vector<float> weight_fp32(N * K);
        std::vector<float> out_fp32(M * N, {});

        llaisys::utils::bf16_to_fp32_batch(in_fp32.data(), in, M * K);
        llaisys::utils::bf16_to_fp32_batch(weight_fp32.data(), weight, N * K);

        if (bias) {
            std::vector<float> bias_fp32(N);
            llaisys::utils::bf16_to_fp32_batch(bias_fp32.data(), bias, N);

            for (size_t i = 0; i < M; ++i) {
                std::memcpy(out_fp32.data() + i * N, bias_fp32.data(), N * sizeof(float));
            }
        }

        if (M == 1) {
            vecmul(in_fp32.data(), weight_fp32.data(), out_fp32.data(), N, K);
        } else {
            matmul(in_fp32.data(), weight_fp32.data(), out_fp32.data(), M, N, K);
        }

        llaisys::utils::fp32_to_bf16_batch(out, out_fp32.data(), M * N);
    }
}

static inline float hsum_avx256(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void linear_decode_simd(float *output, const float *in, const int8_t *weight, 
                        size_t N, size_t K, const float *scale, const float *bias) {

    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        
        const int8_t* w_ptr = weight + n * K;
        const float* in_ptr = in;
        
        float final_sum = 0.0f;
        size_t k = 0;

#if defined(__AVX512F__)
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();

        for (; k + 63 < K; k += 64) {
            __m128i w_raw0 = _mm_loadu_si128((__m128i const*)(w_ptr + k));
            __m128i w_raw1 = _mm_loadu_si128((__m128i const*)(w_ptr + k + 16));
            __m128i w_raw2 = _mm_loadu_si128((__m128i const*)(w_ptr + k + 32));
            __m128i w_raw3 = _mm_loadu_si128((__m128i const*)(w_ptr + k + 48));

            __m512 w_f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w_raw0));
            __m512 w_f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w_raw1));
            __m512 w_f2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w_raw2));
            __m512 w_f3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(w_raw3));

            __m512 in0 = _mm512_loadu_ps(in_ptr + k);
            __m512 in1 = _mm512_loadu_ps(in_ptr + k + 16);
            __m512 in2 = _mm512_loadu_ps(in_ptr + k + 32);
            __m512 in3 = _mm512_loadu_ps(in_ptr + k + 48);

            sum0 = _mm512_fmadd_ps(w_f0, in0, sum0);
            sum1 = _mm512_fmadd_ps(w_f1, in1, sum1);
            sum2 = _mm512_fmadd_ps(w_f2, in2, sum2);
            sum3 = _mm512_fmadd_ps(w_f3, in3, sum3);
        }
        
        __m512 sum_total = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        final_sum = _mm512_reduce_add_ps(sum_total);

#elif defined(__AVX2__)
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        for (; k + 31 < K; k += 32) {
            __m128i w_raw0 = _mm_loadl_epi64((__m128i const*)(w_ptr + k));
            __m256 w_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_raw0));

            __m128i w_raw1 = _mm_loadl_epi64((__m128i const*)(w_ptr + k + 8));
            __m256 w_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_raw1));

            __m128i w_raw2 = _mm_loadl_epi64((__m128i const*)(w_ptr + k + 16));
            __m256 w_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_raw2));

            __m128i w_raw3 = _mm_loadl_epi64((__m128i const*)(w_ptr + k + 24));
            __m256 w_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_raw3));

            __m256 in0 = _mm256_loadu_ps(in_ptr + k);
            __m256 in1 = _mm256_loadu_ps(in_ptr + k + 8);
            __m256 in2 = _mm256_loadu_ps(in_ptr + k + 16);
            __m256 in3 = _mm256_loadu_ps(in_ptr + k + 24);

            sum0 = _mm256_fmadd_ps(w_f0, in0, sum0);
            sum1 = _mm256_fmadd_ps(w_f1, in1, sum1);
            sum2 = _mm256_fmadd_ps(w_f2, in2, sum2);
            sum3 = _mm256_fmadd_ps(w_f3, in3, sum3);
        }

        __m256 sum_total = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        final_sum = hsum_avx256(sum_total);
#endif

        for (; k < K; ++k) {
            final_sum += in_ptr[k] * static_cast<float>(w_ptr[k]);
        }

        if (scale) final_sum *= scale[n];
        if (bias) final_sum += bias[n];
        
        output[n] = final_sum;
    }
}

void linear_(float *output, const float *in, const int8_t *weight, const float *bias, 
             size_t nrow, size_t ncol_out, size_t ncol_in, const float *scale) {
    
    const size_t M = nrow;
    const size_t N = ncol_out;
    const size_t K = ncol_in;
    if (M == 1) {
        linear_decode_simd(output, in, weight, ncol_out, ncol_in, scale, bias);
    } else {
        if (bias) {
            for (size_t i = 0; i < M; ++i) {
                std::memcpy(output + i * N, bias, N * sizeof(float));
            }
        } else {
            memset(output, 0, M * N * sizeof(float));
        }

        matmul_quantized(in, weight, output, M, N, K, scale);
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t nrow, size_t ncol_out, size_t ncol_in, const std::byte *scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), nrow, ncol_out, ncol_in);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), nrow, ncol_out, ncol_in);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), nrow, ncol_out, ncol_in);
    case LLAISYS_DTYPE_I8:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const int8_t *>(weight), reinterpret_cast<const float *>(bias), nrow, ncol_out, ncol_in, reinterpret_cast<const float *>(scale));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
