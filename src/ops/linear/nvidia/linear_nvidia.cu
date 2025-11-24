#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "128x128x8.cuh"
#include "linear_nvidia.cuh"
#include "matvec.cuh"

#include <cuda_runtime.h>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t nrow, size_t ncol_out, size_t ncol_in) {
    const size_t M = nrow;
    const size_t N = ncol_out;
    const size_t K = ncol_in;

    dim3 blockDim(BLOCK_SIZE);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        if (likely(M == 1)) {
            dim3 gridDim(N);
            return matvec_kernel_warp_vec<<<gridDim, blockDim>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), N, K);
        } else {
            dim3 gridDim;
            gridDim.y = ceil_div(M, 128);
            gridDim.x = ceil_div(N, 128);
            return linear_128x128x8_kernel<<<gridDim, blockDim>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), M, N, K);
        }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia