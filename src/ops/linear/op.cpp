#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#include "nvidia/linear_nvidia.cuh"
#include <cstddef>

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, tensor_t scale) {
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        // Only support contiguous inputs with same shape for now.
        if (weight->dtype() == LLAISYS_DTYPE_I8) {
            CHECK_SAME_DTYPE(out->dtype(), in->dtype(), bias->dtype(), scale->dtype());
        } else {
            CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        }
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(), "Linear: all tensors must be contiguous.");
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        // Only support contiguous inputs with same shape for now.
        if (weight->dtype() == LLAISYS_DTYPE_I8) {
            CHECK_SAME_DTYPE(out->dtype(), in->dtype(), scale->dtype());
        } else {
            CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
        }
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");
    }

    std::byte *bias_data = nullptr;
    if (bias) {
        bias_data = bias->data();
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, weight->dtype(), out->dim(0), out->dim(1), in->dim(1), scale ? scale->data() : nullptr);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, weight->dtype(), out->dim(0), out->dim(1), in->dim(1), scale ? scale->data() : nullptr);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias_data, weight->dtype(), out->dim(0), out->dim(1), in->dim(1), scale ? scale->data() : nullptr);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
