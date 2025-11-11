#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "add_nvidia.cuh"

#include <cuda_runtime.h>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if constexpr (std::is_same_v<T, half2>) {
            c[idx] = __hadd2(a[idx], b[idx]);
        } else if constexpr (std::is_same_v<T, half>) {
            c[idx] = __hadd(a[idx], b[idx]);
        } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
            c[idx] = __hadd(a[idx], b[idx]);
        } else if constexpr (std::is_same_v<T, float>) {
            c[idx] = __fadd_rn(a[idx], b[idx]);
        }
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(ceil_div(numel, BLOCK_SIZE));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return add_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<cuda_bfloat16 *>(c), reinterpret_cast<const cuda_bfloat16 *>(a),
                                                   reinterpret_cast<const cuda_bfloat16 *>(b), numel);
    case LLAISYS_DTYPE_F16:
        return add_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<half *>(c), reinterpret_cast<const half *>(a),
                                                   reinterpret_cast<const half *>(b), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
