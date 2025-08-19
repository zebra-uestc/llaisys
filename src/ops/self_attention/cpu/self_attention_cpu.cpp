#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t seqlen, size_t nhead, size_t dv, size_t total_len, size_t nkvhead, size_t d) {

    const size_t group_size = nhead / nkvhead; // GQA分组大小

    // 1. 初始化输出为0
    const size_t attn_val_size = seqlen * nhead * dv;
    std::memset(attn_val, 0, attn_val_size * sizeof(T));

    // 2. 检查维度约束
    if (nhead % nkvhead != 0) {
        throw std::invalid_argument("nhead must be multiple of nkvhead");
    }

// 3. 主计算循环
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seqlen; ++i) {    // 当前序列位置
        for (size_t h = 0; h < nhead; ++h) { // 查询头索引

            // 确定对应的键值头分组 (GQA)
            const size_t kvh = h / group_size;

            // --- 步骤1：计算注意力分数 ---
            // 临时存储分数和softmax变量
            std::vector<float> scores(total_len, -std::numeric_limits<float>::infinity());
            float max_score = -std::numeric_limits<float>::infinity();

            // 仅计算因果位置 (j <= past_len + i)
            const size_t causal_end = (total_len - seqlen) + i + 1;
            for (size_t j = 0; j < causal_end; ++j) {
                float dot = 0.0f;

                // Q[i,h]·K[j,kvh] 点积
                for (size_t dim = 0; dim < d; ++dim) {
                    const size_t q_idx = i * nhead * d + h * d + dim;
                    const size_t k_idx = j * nkvhead * d + kvh * d + dim;
                    // 使用统一的类型转换
                    const float q_val = llaisys::utils::cast<float>(q[q_idx]);
                    const float k_val = llaisys::utils::cast<float>(k[k_idx]);
                    dot += q_val * k_val;
                }

                scores[j] = dot * scale;
                if (scores[j] > max_score) {
                    max_score = scores[j];
                }
            }

            // --- 步骤2：因果softmax归一化 ---
            float exp_sum = 0.0f;
            for (size_t j = 0; j < causal_end; ++j) {
                float exp_val = std::exp(scores[j] - max_score);
                scores[j] = exp_val;
                exp_sum += exp_val;
            }

            // --- 步骤3：加权聚合V矩阵 ---
            for (size_t dv_dim = 0; dv_dim < dv; ++dv_dim) {
                float out_val = 0.0f;

                for (size_t j = 0; j < causal_end; ++j) {
                    const size_t v_idx = j * nkvhead * dv + kvh * dv + dv_dim;
                    const float v_val = llaisys::utils::cast<float>(v[v_idx]);
                    out_val += scores[j] * v_val;
                }

                const size_t attn_idx = i * nhead * dv + h * dv + dv_dim;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[attn_idx] = llaisys::utils::cast<T>(out_val / exp_sum); // softmax归一化
                } else {
                    attn_val[attn_idx] = out_val / exp_sum; // softmax归一化
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
