#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "argmax_nvidia.cuh"

#include <cstring>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__device__ __forceinline__ T get_lowest_value() {
    if constexpr (std::is_same_v<T, float>) {
        return -CUDART_INF_F;
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(-CUDART_INF_F);
    } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        return __float2bfloat16(-CUDART_INF_F);
    }
}

__host__ __device__ inline unsigned long long pack(float val, uint32_t idx) {
#ifdef __CUDA_ARCH__
    unsigned int vb = __float_as_uint(val);
#else
    unsigned int vb;
    std::memcpy(&vb, &val, sizeof(vb));
#endif
    return ((unsigned long long)vb << 32) | (unsigned long long)idx;
}

__host__ __device__ inline float unpack_val(unsigned long long packed) {
    unsigned int vb = (unsigned int)(packed >> 32);
#ifdef __CUDA_ARCH__
    return __uint_as_float(vb);
#else
    float f;
    std::memcpy(&f, &vb, sizeof(f));
    return f;
#endif
}

__host__ __device__ inline uint32_t unpack_idx(unsigned long long packed) {
    return (uint32_t)(packed & 0xffffffffu);
}

// Warp-only kernel
template <typename T>
__global__ void reduce_argmax_kernel_warp(unsigned long long *global_packed,
                                          const T *input, size_t N,
                                          std::byte *d_max_idx, // device pointer
                                          std::byte *d_max_val, // device pointer
                                          llaisysDataType_t type) {
    const unsigned mask = 0xffffffffu;
    size_t tid = threadIdx.x;
    size_t lane_id = tid & 31;
    size_t gidx = blockIdx.x * blockDim.x + tid;

    // local reduction
    T thread_max_val_t = get_lowest_value<T>();
    uint32_t thread_max_idx = UINT32_MAX; // invalid initially

    for (size_t j = gidx; j < N; j += blockDim.x * gridDim.x) {
        T v = input[j];
        if (v > thread_max_val_t) {
            thread_max_val_t = v;
            thread_max_idx = static_cast<uint32_t>(j);
        }
    }

    // convert to float for shuffles/comparisons
    float v = to_float<T>(thread_max_val_t);
    uint32_t idx = thread_max_idx;

    // warp-level reduce: shuffle float values and indices
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_v = __shfl_down_sync(mask, v, offset);
        uint32_t other_idx = __shfl_down_sync(mask, idx, offset);
        // other_idx may be UINT32_MAX if that lane had no element
        if (other_v > v || (other_v == v && other_idx != UINT32_MAX && other_idx < idx)) {
            v = other_v;
            idx = other_idx;
        }
    }

    if (lane_id == 0 && idx != UINT32_MAX) {
        // attempt atomic CAS with packed (float value + idx)
        unsigned long long my_packed = pack(v, idx);
        unsigned long long old = *global_packed;
        while (true) {
            float old_val = unpack_val(old);
            uint32_t old_idx = unpack_idx(old);
            if (v < old_val || (v == old_val && idx >= old_idx)) {
                break;
            }
            unsigned long long prev = atomicCAS(global_packed, old, my_packed);
            if (prev == old) {
                // SUCCESS: write outputs into provided device buffers
                // write index (int64_t *)
                int64_t *out_idx = reinterpret_cast<int64_t *>(d_max_idx);
                *out_idx = static_cast<int64_t>(idx);

                // write value according to type
                switch (type) {
                case LLAISYS_DTYPE_F32: {
                    float *out_val = reinterpret_cast<float *>(d_max_val);
                    *out_val = v;
                    break;
                }
                case LLAISYS_DTYPE_F16: {
                    half *out_val = reinterpret_cast<half *>(d_max_val);
                    *out_val = __float2half(v);
                    break;
                }
                case LLAISYS_DTYPE_BF16: {
                    cuda_bfloat16 *out_val = reinterpret_cast<cuda_bfloat16 *>(d_max_val);
                    *out_val = __float2bfloat16(v);
                    break;
                }
                default:
                    break;
                }
                break; // done
            }
            old = prev;
        }
    }
}

// smem per-block kernel
template <typename T>
__global__ void
reduce_argmax_kernel_warp_smem(unsigned long long *global_packed,
                               const T *input, size_t N) {
    const unsigned mask = 0xffffffffu;
    // Use fixed 32 entries (max 32 warps per block for 1024 threads)
    __shared__ float s_val[32];
    __shared__ uint32_t s_idx[32];

    size_t tid = threadIdx.x;
    size_t lane_id = tid & 31;
    size_t gidx = blockIdx.x * blockDim.x + tid;

    // local reduction
    T thread_max_val_t = get_lowest_value<T>();
    uint32_t thread_max_idx = UINT32_MAX;

    for (size_t j = gidx; j < N; j += blockDim.x * gridDim.x) {
        T v = input[j];
        if (v > thread_max_val_t) {
            thread_max_val_t = v;
            thread_max_idx = static_cast<uint32_t>(j);
        }
    }

    // warp-level reduce into (v, idx) using float
    float v = to_float<T>(thread_max_val_t);
    uint32_t idx = thread_max_idx;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_v = __shfl_down_sync(mask, v, offset);
        uint32_t other_idx = __shfl_down_sync(mask, idx, offset);
        if (other_v > v || (other_v == v && other_idx != UINT32_MAX && other_idx < idx)) {
            v = other_v;
            idx = other_idx;
        }
    }

    // warp leader writes to shared memory
    unsigned warp_id = tid >> 5; // tid/32
    if (lane_id == 0) {
        // if this warp had no valid idx, write lowest sentinel
        if (idx == UINT32_MAX) {
            s_val[warp_id] = to_float<T>(get_lowest_value<T>());
            s_idx[warp_id] = UINT32_MAX;
        } else {
            s_val[warp_id] = v;
            s_idx[warp_id] = idx;
        }
    }
    __syncthreads();

    // first warp reduces across warp results (only first 32 threads used)
    if (tid < 32) {
        unsigned num_warps = (blockDim.x + 31) / 32;
        float block_v = to_float<T>(get_lowest_value<T>());
        uint32_t block_idx = UINT32_MAX;
        if (tid < num_warps) {
            block_v = s_val[tid];
            block_idx = s_idx[tid];
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_v = __shfl_down_sync(mask, block_v, offset);
            uint32_t other_idx = __shfl_down_sync(mask, block_idx, offset);
            if (other_v > block_v || (other_v == block_v && other_idx != UINT32_MAX && other_idx < block_idx)) {
                block_v = other_v;
                block_idx = other_idx;
            }
        }

        if (tid == 0 && block_idx != UINT32_MAX) {
            unsigned long long my_packed = pack(block_v, block_idx);
            unsigned long long old = *global_packed;
            while (true) {
                float old_val = unpack_val(old);
                uint32_t old_idx = unpack_idx(old);
                if (block_v < old_val || (block_v == old_val && block_idx >= old_idx)) {
                    break;
                }
                unsigned long long prev = atomicCAS(global_packed, old, my_packed);
                if (prev == old) {
                    break;
                }
                old = prev;
            }
        }
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(BLOCK_SIZE);

    unsigned long long h_packed_res = pack(-std::numeric_limits<float>::infinity(), UINT32_MAX);
    unsigned long long *d_packed_res;
    CUDA_CHECK(cudaMalloc(&d_packed_res, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_packed_res, &h_packed_res, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        reduce_argmax_kernel_warp<<<grid_dim, block_dim>>>(d_packed_res, reinterpret_cast<const float *>(vals), size, max_idx, max_val, type);
        break;
    case LLAISYS_DTYPE_BF16:
        reduce_argmax_kernel_warp<<<grid_dim, block_dim>>>(d_packed_res, reinterpret_cast<const cuda_bfloat16 *>(vals), size, max_idx, max_val, type);
        break;
    case LLAISYS_DTYPE_F16:
        reduce_argmax_kernel_warp<<<grid_dim, block_dim>>>(d_packed_res, reinterpret_cast<const half *>(vals), size, max_idx, max_val, type);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
