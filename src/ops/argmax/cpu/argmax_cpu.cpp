#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    int64_t ans_idx{};
    T ans_val{};
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float f_tmp = llaisys::utils::cast<float>(vals[i]);
            float f_ans_val = llaisys::utils::cast<float>(ans_val);
            if (f_tmp > f_ans_val) {
                ans_val = llaisys::utils::cast<T>(f_tmp);
                ans_idx = i;
            }
        } else {
            T tmp = vals[i];
            if (tmp > ans_val) {
                ans_val = tmp;
                ans_idx = i;
            }
        }
    }
    *max_idx = ans_idx;
    *max_val = ans_val;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
