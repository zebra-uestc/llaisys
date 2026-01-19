#include "argmax_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <omp.h>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        using MaxPair = std::pair<float, int64_t>;

#pragma omp declare reduction(max_pair:MaxPair : \
    omp_out = (omp_in.first >  omp_out.first ||  \
              (omp_in.first == omp_out.first && omp_in.second < omp_out.second)) \
              ? omp_in : omp_out) \
    initializer(omp_priv = MaxPair{std::numeric_limits<float>::lowest(), INT64_MAX})

        MaxPair result{std::numeric_limits<float>::lowest(), 0};

#pragma omp parallel for reduction(max_pair : result)
        for (size_t i = 0; i < numel; ++i) {
            float f_val = llaisys::utils::cast<float>(vals[i]);
            if (f_val > result.first) {
                result = {f_val, i};
            }
        }

        *max_val = llaisys::utils::cast<T>(result.first);
        *max_idx = result.second;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        using MaxPair = std::pair<int, int64_t>;

        #pragma omp declare reduction(max_pair_int : MaxPair : \
            omp_out = (omp_in.first > omp_out.first || \
                      (omp_in.first == omp_out.first && omp_in.second < omp_out.second)) \
                      ? omp_in : omp_out) \
            initializer(omp_priv = MaxPair{std::numeric_limits<int>::lowest(), INT64_MAX})

        MaxPair result{std::numeric_limits<int>::lowest(), INT64_MAX};

        #pragma omp parallel for reduction(max_pair_int : result)
        for (size_t i = 0; i < numel; ++i) {
            int val = static_cast<int>(vals[i]);
            if (val > result.first) {
                result = {val, i};
            }
        }

        *max_val = static_cast<T>(result.first);
        *max_idx = result.second;
    } else {
        using MaxPair = std::pair<T, int64_t>;

#pragma omp declare reduction(max_pair:MaxPair : \
    omp_out = (omp_in.first >  omp_out.first ||  \
              (omp_in.first == omp_out.first && omp_in.second < omp_out.second)) \
              ? omp_in : omp_out) \
    initializer(omp_priv = MaxPair{std::numeric_limits<T>::lowest(), INT64_MAX})

        MaxPair result{std::numeric_limits<T>::lowest(), 0};

#pragma omp parallel for reduction(max_pair : result)
        for (size_t i = 0; i < numel; ++i) {
            T f_val = vals[i];
            if (f_val > result.first) {
                result = {f_val, i};
            }
        }

        *max_val = result.first;
        *max_idx = result.second;
    }
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
    case LLAISYS_DTYPE_I8:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<int8_t *>(max_val), reinterpret_cast<const int8_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
