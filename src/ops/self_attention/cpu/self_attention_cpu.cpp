#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <omp.h>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     float scale, size_t seqlen, size_t nhead, size_t dv,
                     size_t total_len, size_t nkvhead, size_t d) {

    const size_t group_size = nhead / nkvhead;

    // 1. Sanity check
    if (nhead % nkvhead != 0) {
        throw std::invalid_argument("nhead must be a multiple of nkvhead");
    }

    // 2. Clear Output
    // Safety precaution. For long sequences, memset overhead is negligible.
    std::memset(attn_val, 0, seqlen * nhead * dv * sizeof(T));

    // Block size for Query tokens, optimized for L1/L2 cache residency.
    constexpr size_t TILE_Q = 16;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i_base = 0; i_base < seqlen; i_base += TILE_Q) {
        for (size_t h = 0; h < nhead; ++h) {

            const size_t kvh = h / group_size;
            // Handle boundary conditions for the last tile
            const size_t current_tile_q = std::min(TILE_Q, seqlen - i_base);

            // Local accumulation buffer (fp32)
            // Shape: [current_tile_q, dv]
            std::vector<float> local_out(current_tile_q * dv, 0.0f);

            // Online Softmax statistics: max_score and exponential sum
            std::vector<float> local_max(current_tile_q, -std::numeric_limits<float>::infinity());
            std::vector<float> local_sum(current_tile_q, 0.0f);

            // ---------- Loop over KV Cache (Keys & Values) ----------
            // Stream through K/V. Total_len is usually large, so we scan it once.
            for (size_t j = 0; j < total_len; ++j) {

                // Precompute KV offsets
                const size_t k_offset = j * nkvhead * d + kvh * d;
                const size_t v_offset = j * nkvhead * dv + kvh * dv;

                // Iterate over queries in the current tile
                for (size_t ii = 0; ii < current_tile_q; ++ii) {
                    size_t i = i_base + ii;

                    // Apply causal masking
                    if (j > (total_len - seqlen) + i) {
                        continue;
                    }

                    // --- 1. Compute Dot Product (Q * K) ---
                    float score = 0.0f;
                    const size_t q_offset = i * nhead * d + h * d;

// SIMD-friendly reduction for contiguous memory access
#pragma omp simd reduction(+ : score)
                    for (size_t dim = 0; dim < d; ++dim) {
                        float q_val = llaisys::utils::cast<float>(q[q_offset + dim]);
                        float k_val = llaisys::utils::cast<float>(k[k_offset + dim]);
                        score += q_val * k_val;
                    }
                    score *= scale;

                    // --- 2. Online Softmax Update ---
                    // Math: m_new = max(m_old, score)
                    //       d_new = d_old * exp(m_old - m_new) + exp(score - m_new)
                    //       o_new = o_old * exp(m_old - m_new) + v * exp(score - m_new)

                    float prev_max = local_max[ii];
                    float curr_max = std::max(prev_max, score);

                    // Update if max changes or first valid score
                    if (curr_max > prev_max || std::isinf(prev_max)) {
                        float exp_diff = std::exp(prev_max - curr_max); // 0 if prev_max is -inf
                        float exp_score = std::exp(score - curr_max);

                        // Update accumulator
                        local_sum[ii] = local_sum[ii] * exp_diff + exp_score;
                        local_max[ii] = curr_max;

// --- 3. Accumulate V (Fused) ---
// Key Optimization: The inner loop iterates over 'dv', ensuring
// contiguous memory access for 'v', drastically reducing cache misses.
#pragma omp simd
                        for (size_t dim_v = 0; dim_v < dv; ++dim_v) {
                            float v_val = llaisys::utils::cast<float>(v[v_offset + dim_v]);
                            local_out[ii * dv + dim_v] = local_out[ii * dv + dim_v] * exp_diff + v_val * exp_score;
                        }
                    } else {
                        // If score is small, max doesn't change.
                        // Just accumulate the exponentiated term.
                        float exp_score = std::exp(score - curr_max);
                        local_sum[ii] += exp_score;

#pragma omp simd
                        for (size_t dim_v = 0; dim_v < dv; ++dim_v) {
                            float v_val = llaisys::utils::cast<float>(v[v_offset + dim_v]);
                            local_out[ii * dv + dim_v] += v_val * exp_score;
                        }
                    }
                }
            }

            // --- 4. Finalize and Store ---
            for (size_t ii = 0; ii < current_tile_q; ++ii) {
                size_t i = i_base + ii;
                const size_t out_offset = i * nhead * dv + h * dv;
                float inv_sum = 1.0f / (local_sum[ii] + 1e-6f); // Avoid division by zero

#pragma omp simd
                for (size_t dim_v = 0; dim_v < dv; ++dim_v) {
                    float val = local_out[ii * dv + dim_v] * inv_sum;

                    // Cast back to destination type
                    if constexpr (std::is_same_v<T, float>) {
                        attn_val[out_offset + dim_v] = val;
                    } else {
                        attn_val[out_offset + dim_v] = llaisys::utils::cast<T>(val);
                    }
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t dv, size_t total_len, size_t nkvhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale, seqlen, nhead, dv, total_len, nkvhead, d);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), scale, seqlen, nhead, dv, total_len, nkvhead, d);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), scale, seqlen, nhead, dv, total_len, nkvhead, d);
    case LLAISYS_DTYPE_I8:
        return self_attention_(reinterpret_cast<int8_t *>(attn_val), reinterpret_cast<const int8_t *>(q),
                               reinterpret_cast<const int8_t *>(k), reinterpret_cast<const int8_t *>(v), scale, seqlen, nhead, dv, total_len, nkvhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
