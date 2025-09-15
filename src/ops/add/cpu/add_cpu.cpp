#include "add_cpu.hpp"
#include "add_opt.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_f32(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return add_bf16(reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
                        reinterpret_cast<const llaisys::bf16_t *>(b), numel);
    case LLAISYS_DTYPE_F16:
        return add_f16(reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
                       reinterpret_cast<const llaisys::fp16_t *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
