#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seqlen, size_t nhead, size_t d) {
    T *out_t{};
    const T *in_t{};
    for (size_t i = 0; i < seqlen; ++i) {
        out_t = out + (i * nhead * d);
        in_t = in + (i * nhead * d);
        int64_t p_i = pos_ids[i];
        out_t -= d;
        in_t -= d;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            for (size_t j = 0; j < nhead; ++j) {
                out_t += d;
                in_t += d;
                for (size_t k = 0; k < d / 2; ++k) {
                    float expo = static_cast<float>(2 * k) / static_cast<float>(d);
                    float angle = p_i / std::pow(theta, expo);
                    float sin_val = std::sin(angle);
                    float cos_val = std::cos(angle);

                    float a_ik = llaisys::utils::cast<float>(in_t[k]);
                    float b_ik = llaisys::utils::cast<float>(in_t[k + d / 2]);
                    out_t[k] = llaisys::utils::cast<T>(a_ik * cos_val - b_ik * sin_val);
                    out_t[k + d / 2] = llaisys::utils::cast<T>(b_ik * cos_val + a_ik * sin_val);
                }
            }
        } else {
            for (size_t j = 0; j < nhead; ++j) {
                out_t += d;
                in_t += d;
                for (size_t k = 0; k < d / 2; ++k) {
                    float expo = static_cast<float>(2 * k) / static_cast<float>(d);
                    T angle = p_i / std::pow(theta, expo);
                    T sin_val = std::sin(angle);
                    T cos_val = std::cos(angle);

                    T a_ik = in_t[k];
                    T b_ik = in_t[k + d / 2];
                    out_t[k] = a_ik * cos_val - b_ik * sin_val;
                    out_t[k + d / 2] = b_ik * cos_val + a_ik * sin_val;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
