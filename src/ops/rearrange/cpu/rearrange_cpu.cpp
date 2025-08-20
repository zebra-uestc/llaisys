#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>

template <typename T>
void rearrange_(T *out, const T *in, size_t numel, const std::vector<size_t> &out_shape, const std::vector<ptrdiff_t> &out_strides, const std::vector<size_t> &in_shape, const std::vector<ptrdiff_t> &in_strides) {
    size_t ndim = out_shape.size();
    for (size_t i = 0; i < numel; ++i) {
        size_t out_idx = llaisys::ops::indexToOffset(i, ndim, out_shape, out_strides);
        size_t in_idx = llaisys::ops::indexToOffset(i, ndim, in_shape, in_strides);
        out[out_idx] = in[in_idx];
    }
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, size_t numel, const std::vector<size_t> &out_shape, const std::vector<ptrdiff_t> &out_strides, const std::vector<size_t> &in_shape, const std::vector<ptrdiff_t> &in_strides) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), numel, out_shape, out_strides, in_shape, in_strides);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                          numel, out_shape, out_strides, in_shape, in_strides);
    case LLAISYS_DTYPE_F16:
        return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                          numel, out_shape, out_strides, in_shape, in_strides);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
