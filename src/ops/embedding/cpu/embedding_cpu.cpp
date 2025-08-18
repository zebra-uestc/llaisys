#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t numel, size_t len) {
    size_t nlen = numel / len;
    for (size_t i = 0; i < nlen; i++) {
        int64_t idx = index[i];
        std::memcpy(out + (i * len), weight + (idx * len), len * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t numel, size_t len) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), numel, len);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), numel, len);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), numel, len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
