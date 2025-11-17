#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "embedding_nvidia.cuh"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(T *output, const T *weight, const int64_t *index,
                                 size_t len) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int offset = index[bx] * len;
    for (int i = tx; i < len; i += blockDim.x) {
        output[bx * len + i] = weight[offset + i];
    }
}

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS_CONST(value) (reinterpret_cast<const float4 *>(&(value))[0])

template <typename T>
struct VecSize {
    static constexpr int value = 16 / sizeof(T); // 128-bit vectorization
};

template <typename T>
__global__ void embedding_kernel_vec(T *output, const T *weight,
                                     const int64_t *index, size_t len) {
    constexpr int VEC_SIZE = VecSize<T>::value;

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int idx = index[bx];

    size_t w_offset = static_cast<size_t>(idx) * len;
    size_t o_offset = static_cast<size_t>(bx) * len;

    size_t vec_len = len / VEC_SIZE;

    // Vectorized copy
    for (size_t i = tx; i < vec_len; i += blockDim.x) {
        LDST128BITS(output[o_offset + i * VEC_SIZE]) = LDST128BITS_CONST(weight[w_offset + i * VEC_SIZE]);
    }

    // Handle remaining elements
    size_t remain_start = vec_len * VEC_SIZE;
    for (size_t i = remain_start + tx; i < len; i += blockDim.x) {
        output[o_offset + i] = weight[w_offset + i];
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t size, size_t len) {
    const size_t nlen = size / len;
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(nlen);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(weight), reinterpret_cast<const int64_t *>(index), len);
    case LLAISYS_DTYPE_BF16:
        return embedding_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<cuda_bfloat16 *>(out), reinterpret_cast<const cuda_bfloat16 *>(weight), reinterpret_cast<const int64_t *>(index), len);
    case LLAISYS_DTYPE_F16:
        return embedding_kernel<<<grid_dim, block_dim>>>(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(weight), reinterpret_cast<const int64_t *>(index), len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
