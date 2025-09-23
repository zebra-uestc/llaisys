#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <omp.h>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t seqlen, size_t nhead, size_t dv, size_t total_len, size_t nkvhead, size_t d) {
    // ========== Group-Query Attention (GQA) with causal mask ==========

    const size_t group_size = nhead / nkvhead;

    /* ---------- 1. zero the output buffer ---------- */
    const size_t attn_val_size = seqlen * nhead * dv;
    std::memset(attn_val, 0, attn_val_size * sizeof(T));

    /* ---------- 2. sanity check ---------- */
    if (nhead % nkvhead != 0) {
        throw std::invalid_argument("nhead must be a multiple of nkvhead");
    }

/* ---------- 3. main compute loop ---------- */
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seqlen; ++i) {    // current token position
        for (size_t h = 0; h < nhead; ++h) { // query-head index

            /* --- map q-head -> kv-head (GQA) --- */
            const size_t kvh = h / group_size; // KV-head for this query-head

            /* ---------- step-1: compute attention scores ---------- */
            std::vector<float> scores(total_len,
                                      -std::numeric_limits<float>::infinity());
            float max_score = -std::numeric_limits<float>::infinity();

            const size_t causal_end = (total_len - seqlen) + i + 1; // causal boundary
            for (size_t j = 0; j < causal_end; ++j) {               // position in KV-cache
                float dot = 0.0f;

                const size_t q_base = i * nhead * d + h * d;
                const size_t k_base = j * nkvhead * d + kvh * d;

                #pragma omp simd reduction(+ : dot)
                for (size_t dim = 0; dim < d; ++dim) {
                    float q_val = llaisys::utils::cast<float>(q[q_base + dim]);
                    float k_val = llaisys::utils::cast<float>(k[k_base + dim]);
                    dot += q_val * k_val;
                }
                scores[j] = dot * scale; // scale factor
                max_score = std::max(max_score, scores[j]);
            }

            /* ---------- step-2: causal softmax ---------- */
            float exp_sum = 0.0f;
            
            #pragma omp simd reduction(+ : exp_sum)
            for (size_t j = 0; j < causal_end; ++j) {
                scores[j] = std::exp(scores[j] - max_score); // numerically stable
                exp_sum += scores[j];
            }

            /* ---------- step-3: aggregate V with attention weights ---------- */
            for (size_t dv_dim = 0; dv_dim < dv; ++dv_dim) {
                float out_val = 0.0f;

                #pragma omp simd reduction(+ : out_val)
                for (size_t j = 0; j < causal_end; ++j) {
                    const size_t v_idx = j * nkvhead * dv + kvh * dv + dv_dim;
                    const float v_val = llaisys::utils::cast<float>(v[v_idx]);
                    out_val += scores[j] * v_val;
                }

                const size_t attn_idx = i * nhead * dv + h * dv + dv_dim;
                out_val /= exp_sum; // finalize softmax

                /* down-cast if necessary */
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[attn_idx] = llaisys::utils::cast<T>(out_val);
                } else {
                    attn_val[attn_idx] = out_val;
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
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
