#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cblas.h>
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
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, in, K, weight, K, 1.0f, out, N);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, in, K, weight, K, 0.0f, out, N);
        }

    } else if constexpr (std::is_same_v<T, double>) {
        if (bias) {
            for (size_t i = 0; i < M; ++i) {
                std::memcpy(out + i * N, bias, N * sizeof(T));
            }
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0, in, K, weight, K, 1.0, out, N);
        } else {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0, in, K, weight, K, 0.0, out, N);
        }
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
        std::vector<float> in_fp32(M * K);
        std::vector<float> weight_fp32(N * K);
        std::vector<float> out_fp32(M * N);

        for (size_t i = 0; i < M * K; ++i) {
            in_fp32[i] = llaisys::utils::cast<float>(in[i]);
        }
        for (size_t i = 0; i < N * K; ++i) {
            weight_fp32[i] = llaisys::utils::cast<float>(weight[i]);
        }

        if (bias) {
            std::vector<float> bias_fp32(N);
            for (size_t j = 0; j < N; ++j) {
                bias_fp32[j] = llaisys::utils::cast<float>(bias[j]);
            }

            for (size_t i = 0; i < M; ++i) {
                std::memcpy(out_fp32.data() + i * N, bias_fp32.data(), N * sizeof(float));
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, in_fp32.data(), K,
                        weight_fp32.data(), K, 1.0f, out_fp32.data(), N);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, in_fp32.data(), K,
                        weight_fp32.data(), K, 0.0f, out_fp32.data(), N);
        }

        for (size_t i = 0; i < M * N; ++i) {
            out[i] = llaisys::utils::cast<T>(out_fp32[i]);
        }
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
