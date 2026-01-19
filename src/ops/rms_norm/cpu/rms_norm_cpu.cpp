#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"
#include "sdot.hpp"

#include <cmath>
#include <cstddef>
#include <omp.h>
#include <type_traits>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t nrow, size_t ncol) {
#pragma omp parallel for
    for (size_t i = 0; i < nrow; ++i) {
        T *out_t = out + i * ncol;
        const T *in_t = in + i * ncol;

        if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
            float rms{}, square_sum{}, avg_square_sum{};
            std::vector<float> in_f32(ncol);
            std::vector<float> weight_f32(ncol);
            std::vector<float> out_f32(ncol);

            llaisys::utils::bf16_to_fp32_batch(in_f32.data(), in_t, ncol);
            llaisys::utils::bf16_to_fp32_batch(weight_f32.data(), weight, ncol);

            square_sum = llaisys::ops::cpu::sdot(in_f32.data(), in_f32.data(), ncol);
            avg_square_sum = square_sum / static_cast<float>(ncol);
            rms = std::sqrt(avg_square_sum + eps);

            for (size_t j = 0; j < ncol; ++j) {
                out_f32[j] = weight_f32[j] * in_f32[j] / rms;
            }
            llaisys::utils::fp32_to_bf16_batch(out_t, out_f32.data(), ncol);

        } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
            float rms{}, square_sum{}, avg_square_sum{};
            std::vector<float> in_f32(ncol);
            std::vector<float> weight_f32(ncol);
            std::vector<float> out_f32(ncol);

            llaisys::utils::fp16_to_fp32_batch_f16c(in_f32.data(), in_t, ncol);
            llaisys::utils::fp16_to_fp32_batch_f16c(weight_f32.data(), weight, ncol);

            square_sum = llaisys::ops::cpu::sdot(in_f32.data(), in_f32.data(), ncol);
            avg_square_sum = square_sum / static_cast<float>(ncol);
            rms = std::sqrt(avg_square_sum + eps);

            for (size_t j = 0; j < ncol; ++j) {
                out_f32[j] = weight_f32[j] * in_f32[j] / rms;
            }
            llaisys::utils::fp32_to_fp16_batch_f16c(out_t, out_f32.data(), ncol);

        } else if constexpr (std::is_same_v<T, int8_t>) {
            int32_t square_sum = llaisys::ops::cpu::sdot_int8(in_t, in_t, ncol);
            float mean = static_cast<float>(square_sum) / static_cast<float>(ncol);
            float rms = std::sqrt(mean + eps);
            float scale = 1.0f / rms;

            size_t j = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
            __m512 v_scale = _mm512_set1_ps(scale);
            __m512i v_max = _mm512_set1_epi32(127);
            __m512i v_min = _mm512_set1_epi32(-128);

            for (; j + 15 < ncol; j += 16) {
                __m128i v_in_128 = _mm_loadu_si128((const __m128i*)(in_t + j));
                __m128i v_w_128  = _mm_loadu_si128((const __m128i*)(weight + j));

                __m512i v_in_i32 = _mm512_cvtepi8_epi32(v_in_128);
                __m512i v_w_i32  = _mm512_cvtepi8_epi32(v_w_128);
                
                __m512 v_in_f32 = _mm512_cvtepi32_ps(v_in_i32);
                __m512 v_w_f32  = _mm512_cvtepi32_ps(v_w_i32);

                __m512 v_res = _mm512_mul_ps(v_in_f32, v_w_f32);
                v_res = _mm512_mul_ps(v_res, v_scale);

                __m512i v_res_i32 = _mm512_cvtps_epi32(v_res);

                v_res_i32 = _mm512_min_epi32(v_res_i32, v_max);
                v_res_i32 = _mm512_max_epi32(v_res_i32, v_min);

                __m128i v_out_128 = _mm512_cvtepi32_epi8(v_res_i32);
                _mm_storeu_si128((__m128i*)(out_t + j), v_out_128);
            }
#endif

            for (; j < ncol; ++j) {
                float w_val = static_cast<float>(weight[j]);
                float x_val = static_cast<float>(in_t[j]);
                float res = x_val * w_val * scale;

                out_t[j] = llaisys::utils::cast<int8_t>(res);
            }
        } else {
            T rms{}, square_sum{}, avg_square_sum{};
            square_sum = llaisys::ops::cpu::sdot(in_t, in_t, ncol);
            avg_square_sum = square_sum / static_cast<T>(ncol);
            rms = std::sqrt(avg_square_sum + eps);

            for (size_t j = 0; j < ncol; ++j) {
                out_t[j] = weight[j] * in_t[j] / rms;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t nrow, size_t ncol) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, nrow, ncol);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), eps, nrow, ncol);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), eps, nrow, ncol);
    case LLAISYS_DTYPE_I8:
        return rms_norm_(reinterpret_cast<int8_t *>(out), reinterpret_cast<const int8_t *>(in), reinterpret_cast<const int8_t *>(weight), eps, nrow, ncol);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
