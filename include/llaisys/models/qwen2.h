#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"
#include "llaisys.h"
#include <cstddef>

__LLAISYS__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2KVCache {
        llaisysTensor_t *kcache;
        llaisysTensor_t *vcache;
    };

    struct LlaisysQwen2Model {
        struct LlaisysQwen2Meta meta;
        struct LlaisysQwen2Weights weights;
        struct LlaisysQwen2KVCache kvcache;
        llaisysDeviceType_t device;
        int ndevice;
        int *device_ids;
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int ndevice, int *device_ids);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2KVCache *llaisysQwen2KVCacheCreate(struct LlaisysQwen2Model * meta, size_t max_len);

    __export void llaisysQwen2KVCacheDestroy(struct LlaisysQwen2KVCache * kvcache, size_t nlayer);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken, struct LlaisysQwen2KVCache *kvcache, size_t past_len);
}
#endif // LLAISYS_MODELS_QWEN2_H
