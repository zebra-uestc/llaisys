#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seqlen, size_t nhead, size_t d, const std::vector<float> &inv_theta) {
    size_t half_d = d / 2;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < seqlen; ++i) {
        std::vector<float> cos_cache(half_d);
        std::vector<float> sin_cache(half_d);
        int64_t p_i = pos_ids[i];

#pragma omp simd
        for (size_t k = 0; k < half_d; ++k) {
            float angle = static_cast<float>(p_i * inv_theta[k]);
#ifdef __linux__
            sincosf(angle, &sin_cache[k], &cos_cache[k]);
#else
            cos_cache[k] = std::cos(angle);
            sin_cache[k] = std::sin(angle);
#endif
        }

        for (size_t j = 0; j < nhead; ++j) {
            T *out_t = out + (i * nhead + j) * d;
            const T *in_t = in + (i * nhead + j) * d;

#pragma omp simd
            for (size_t k = 0; k < half_d; ++k) {
                float cos_val = cos_cache[k];
                float sin_val = sin_cache[k];
                float a = llaisys::utils::cast<float>(in_t[k]);
                float b = llaisys::utils::cast<float>(in_t[k + half_d]);
                out_t[k] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                out_t[k + half_d] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {
    size_t half_d = d / 2;
    std::vector<float> inv_theta(half_d);
    for (size_t k = 0; k < half_d; ++k) {
        float expo = static_cast<float>(2 * k) / static_cast<float>(d);
        inv_theta[k] = 1.0 / std::pow(theta, expo);
    }

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d, inv_theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d, inv_theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d, inv_theta);
    case LLAISYS_DTYPE_I8:
        return rope_(reinterpret_cast<int8_t *>(out), reinterpret_cast<const int8_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d, inv_theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
