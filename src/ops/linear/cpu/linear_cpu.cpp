#include "linear_cpu.hpp"
#include "matmul.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <type_traits>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t nrow, size_t ncol_out, size_t ncol_in) {

    const size_t M = nrow;
    const size_t N = ncol_out;
    const size_t K = ncol_in;

    if constexpr (std::is_same_v<T, float>) {
        if (bias) {
            for (size_t i = 0; i < M; ++i) {
                std::memcpy(out + i * N, bias, N * sizeof(T));
            }
            matmul(in, weight, out, M, N, K);
        } else {
            memset(out, 0, M * N * sizeof(T));
            matmul(in, weight, out, M, N, K);
        }

    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        std::vector<float> in_fp32(M * K);
        std::vector<float> weight_fp32(N * K);
        std::vector<float> out_fp32(M * N, {});

        // for (size_t i = 0; i < M * K; ++i) {
        //     in_fp32[i] = llaisys::utils::cast<float>(in[i]);
        // }
        // for (size_t i = 0; i < N * K; ++i) {
        //     weight_fp32[i] = llaisys::utils::cast<float>(weight[i]);
        // }
        llaisys::utils::fp16_to_fp32_batch_f16c(in_fp32.data(), in, M * K);
        llaisys::utils::fp16_to_fp32_batch_f16c(weight_fp32.data(), weight, N * K);

        if (bias) {
            std::vector<float> bias_fp32(N);
            // for (size_t j = 0; j < N; ++j) {
            //     bias_fp32[j] = llaisys::utils::cast<float>(bias[j]);
            // }
            llaisys::utils::fp16_to_fp32_batch_f16c(bias_fp32.data(), bias, N);

            for (size_t i = 0; i < M; ++i) {
                std::memcpy(out_fp32.data() + i * N, bias_fp32.data(), N * sizeof(float));
            }

            matmul(in_fp32.data(), weight_fp32.data(), out_fp32.data(), M, N, K);
        } else {
            matmul(in_fp32.data(), weight_fp32.data(), out_fp32.data(), M, N, K);
        }

        // for (size_t i = 0; i < M * N; ++i) {
        //     out[i] = llaisys::utils::cast<T>(out_fp32[i]);
        // }
        llaisys::utils::fp32_to_fp16_batch_f16c(out, out_fp32.data(), M * N);
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        std::vector<float> in_fp32(M * K);
        std::vector<float> weight_fp32(N * K);
        std::vector<float> out_fp32(M * N, {});

        // for (size_t i = 0; i < M * K; ++i) {
        //     in_fp32[i] = llaisys::utils::cast<float>(in[i]);
        // }
        // for (size_t i = 0; i < N * K; ++i) {
        //     weight_fp32[i] = llaisys::utils::cast<float>(weight[i]);
        // }
        llaisys::utils::bf16_to_fp32_batch(in_fp32.data(), in, M * K);
        llaisys::utils::bf16_to_fp32_batch(weight_fp32.data(), weight, N * K);

        if (bias) {
            std::vector<float> bias_fp32(N);
            // for (size_t j = 0; j < N; ++j) {
            //     bias_fp32[j] = llaisys::utils::cast<float>(bias[j]);
            // }
            llaisys::utils::bf16_to_fp32_batch(bias_fp32.data(), bias, N);

            for (size_t i = 0; i < M; ++i) {
                std::memcpy(out_fp32.data() + i * N, bias_fp32.data(), N * sizeof(float));
            }

            matmul(in_fp32.data(), weight_fp32.data(), out_fp32.data(), M, N, K);
        } else {
            matmul(in_fp32.data(), weight_fp32.data(), out_fp32.data(), M, N, K);
        }

        // for (size_t i = 0; i < M * N; ++i) {
        //     out[i] = llaisys::utils::cast<T>(out_fp32[i]);
        // }
        llaisys::utils::fp32_to_bf16_batch(out, out_fp32.data(), M * N);
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t nrow, size_t ncol_out, size_t ncol_in) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), nrow, ncol_out, ncol_in);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), nrow, ncol_out, ncol_in);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), nrow, ncol_out, ncol_in);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
