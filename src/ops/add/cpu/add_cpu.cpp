#include "add_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <omp.h>

template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numel; ++i) {
            float f_a = llaisys::utils::cast<float>(a[i]);
            float f_b = llaisys::utils::cast<float>(b[i]);
            c[i] = llaisys::utils::cast<T>(f_a + f_b);
        }
    } else {
#pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
    }
}

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return add_(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                    reinterpret_cast<const llaisys::bf16_t *>(b), numel);
    case LLAISYS_DTYPE_F16:
        return add_(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                    reinterpret_cast<const llaisys::fp16_t *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
