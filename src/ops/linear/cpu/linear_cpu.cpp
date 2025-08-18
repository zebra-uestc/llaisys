#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t nrow, size_t ncol_out, size_t ncol_in) {
    T *out_t{};
    const T *in_t{}, *weight_t{};

    for (size_t i = 0; i < nrow; ++i) {
        out_t = out + i * ncol_out;
        in_t = in + i * ncol_in;
        for (size_t j = 0; j < ncol_out; ++j) {
            weight_t = weight + j * ncol_in;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float ans{0};
                for (size_t k = 0; k < ncol_in; ++k) {
                    ans += llaisys::utils::cast<float>(in_t[k]) * llaisys::utils::cast<float>(weight_t[k]);
                }
                if (bias) {
                    ans += llaisys::utils::cast<float>(bias[j]);
                }
                out_t[j] = llaisys::utils::cast<T>(ans);
            } else {
                T ans{0};
                for (size_t k = 0; k < ncol_in; ++k) {
                    ans += in_t[k] * weight_t[k];
                }
                if (bias) {
                    ans += bias[j];
                }
                out_t[j] = ans;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t nrow, size_t ncol_out, size_t ncol_in) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), nrow, ncol_out, ncol_in);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), nrow, ncol_out, ncol_in);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), nrow, ncol_out, ncol_in);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
