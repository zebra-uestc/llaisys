#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "linear_bf16_kernel.cuh"
#include "linear_fp16_kernel.cuh"
#include "linear_fp32_kernel.cuh"
#include "linear_nvidia.cuh"
#include "llaisys.h"
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
            return linear_fp32_kernel<<<gridDim, blockDim>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), M, N, K);
        }
        break;
    case LLAISYS_DTYPE_F16:
        if (likely(M == 1)) {
            dim3 gridDim(N);
            return matvec_kernel_warp_vec<<<gridDim, blockDim>>>(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in), reinterpret_cast<const half *>(weight), reinterpret_cast<const half *>(bias), N, K);
        } else {
            constexpr size_t BM = 128;
            constexpr size_t BN = 256;
            constexpr size_t BK = 32;
            constexpr size_t PAD = 8;
            constexpr size_t smem_size = 2 * (BM + BN) * (BK + PAD) * sizeof(half) + 8 * 16 * 16 * sizeof(float) + 8 * 16 * 16 * sizeof(half);
            cudaFuncSetAttribute(linear_fp16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            dim3 gridDim;
            gridDim.y = ceil_div(M, BM);
            gridDim.x = ceil_div(N, BN);
            return linear_fp16_kernel<<<gridDim, blockDim, smem_size>>>(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in), reinterpret_cast<const half *>(weight), reinterpret_cast<const half *>(bias), M, N, K);
        }
        break;
    case LLAISYS_DTYPE_BF16:
        if (likely(M == 1)) {
            dim3 gridDim(N);
            return matvec_kernel_warp_vec<<<gridDim, blockDim>>>(reinterpret_cast<cuda_bfloat16 *>(out), reinterpret_cast<const cuda_bfloat16 *>(in), reinterpret_cast<const cuda_bfloat16 *>(weight), reinterpret_cast<const cuda_bfloat16 *>(bias), N, K);
        } else {
            constexpr size_t BM = 128;
            constexpr size_t BN = 256;
            constexpr size_t BK = 32;
            constexpr size_t PAD = 8;
            constexpr size_t smem_size = 2 * (BM + BN) * (BK + PAD) * sizeof(cuda_bfloat16) + 8 * 16 * 16 * sizeof(float) + 8 * 16 * 16 * sizeof(cuda_bfloat16);
            cudaFuncSetAttribute(linear_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            dim3 gridDim;
            gridDim.y = ceil_div(M, 128);
            gridDim.x = ceil_div(N, 256);
            return linear_bf16_kernel<<<gridDim, blockDim, smem_size>>>(reinterpret_cast<cuda_bfloat16 *>(out), reinterpret_cast<const cuda_bfloat16 *>(in), reinterpret_cast<const cuda_bfloat16 *>(weight), reinterpret_cast<const cuda_bfloat16 *>(bias), M, N, K);
        }
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia