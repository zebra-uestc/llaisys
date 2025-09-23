#include "llaisys/models/qwen2.h"
#include "../../utils.hpp"
#include "llaisys.h"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <iostream>
#include <omp.h>
#include <vector>

// #define DEBUG

__LLAISYS__C {
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int ndevice, int *device_ids) {
        LOG("⚙️  llaisysQwen2ModelCreate: begin\n");
        struct LlaisysQwen2Model *model = (struct LlaisysQwen2Model *)malloc(sizeof(struct LlaisysQwen2Model));
        model->meta = *meta;
        model->device = device;
        model->ndevice = ndevice;

        model->device_ids = (int *)malloc(sizeof(int) * ndevice);
        memcpy(model->device_ids, device_ids, sizeof(int) * ndevice);

        size_t nlayer = model->meta.nlayer;

        model->weights.attn_norm_w = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.attn_q_w = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.attn_q_b = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.attn_k_w = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.attn_k_b = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.attn_v_w = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.attn_v_b = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.attn_o_w = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.mlp_norm_w = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.mlp_gate_w = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.mlp_up_w = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        model->weights.mlp_down_w = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);

        size_t voc = model->meta.voc;
        size_t di = model->meta.di;
        size_t hs = model->meta.hs;
        size_t dh = model->meta.dh;
        size_t nh = model->meta.nh;
        size_t nkvh = model->meta.nkvh;
        size_t dqh = hs;
        size_t dkvh = model->meta.nkvh * dh;

        LOG("📋 Qwen2 Model Configs:")
        print_config("num_hidden_layers:", nlayer, "📊");
        print_config("vocab_size:", voc, "🗄️");
        print_config("intermediate_size:", di, "🔧");
        print_config("hidden_size:", hs, "🔒");
        print_config("head_size:", dh, "🧠");
        print_config("num_query_heads:", nh, "🔄");
        print_config("q_head_dim:", dqh, "🧩");
        print_config("num_key_value_heads:", nkvh, "🔁");
        print_config("k_v_head_dim:", dkvh, "🔤");
        std::cout << std::endl;

        // in_embed (151936, 1536)
        std::vector<size_t> shape = {voc, hs};
        model->weights.in_embed = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
        tensorLoad(model->weights.in_embed, weights->in_embed);

        // out_embed (151936, 1536)
        shape = {voc, hs};
        model->weights.out_embed = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
        tensorLoad(model->weights.out_embed, weights->out_embed);

        // out_norm (1536)
        shape = {hs};
        model->weights.out_norm_w = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
        tensorLoad(model->weights.out_norm_w, weights->out_norm_w);

        for (size_t i = 0; i < nlayer; ++i) {
            // attn_norm
            shape = {hs};
            model->weights.attn_norm_w[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.attn_norm_w[i], weights->attn_norm_w[i]);

            // attn_q
            shape = {hs, hs};
            model->weights.attn_q_w[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.attn_q_w[i], weights->attn_q_w[i]);

            // attn_q_b
            shape = {hs};
            model->weights.attn_q_b[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.attn_q_b[i], weights->attn_q_b[i]);

            // attn_k
            shape = {dkvh, hs};
            model->weights.attn_k_w[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.attn_k_w[i], weights->attn_k_w[i]);

            // attn_k_b
            shape = {dkvh};
            model->weights.attn_k_b[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.attn_k_b[i], weights->attn_k_b[i]);

            // attn_v
            shape = {dkvh, hs};
            model->weights.attn_v_w[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.attn_v_w[i], weights->attn_v_w[i]);

            // attn_v_b
            shape = {dkvh};
            model->weights.attn_v_b[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.attn_v_b[i], weights->attn_v_b[i]);

            // attn_o
            shape = {hs, hs};
            model->weights.attn_o_w[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.attn_o_w[i], weights->attn_o_w[i]);

            // mlp_norm
            shape = {hs};
            model->weights.mlp_norm_w[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.mlp_norm_w[i], weights->mlp_norm_w[i]);

            // mlp_gate
            shape = {di, hs};
            model->weights.mlp_gate_w[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.mlp_gate_w[i], weights->mlp_gate_w[i]);

            // mlp_up
            shape = {di, hs};
            model->weights.mlp_up_w[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.mlp_up_w[i], weights->mlp_up_w[i]);

            // mlp_down
            shape = {hs, di};
            model->weights.mlp_down_w[i] = tensorCreate(shape.data(), shape.size(), model->meta.dtype, device, device_ids[0]);
            tensorLoad(model->weights.mlp_down_w[i], weights->mlp_down_w[i]);
        }

        LOG("🎉 llaisysQwen2ModelCreate: end\n\n");
        return model;
    }

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (model == NULL) {
            return;
        }
        LOG("🗑️  llaisysQwen2ModelDestroy: begin")

        // 1. 释放设备ID数组
        if (model->device_ids != NULL) {
            free(model->device_ids);
            model->device_ids = NULL;
        }

        // 2. 释放非数组的独立张量
        if (model->weights.in_embed != NULL) {
            tensorDestroy(model->weights.in_embed);
            model->weights.in_embed = NULL;
        }
        if (model->weights.out_embed != NULL) {
            tensorDestroy(model->weights.out_embed);
            model->weights.out_embed = NULL;
        }
        if (model->weights.out_norm_w != NULL) {
            tensorDestroy(model->weights.out_norm_w);
            model->weights.out_norm_w = NULL;
        }

        // 3. 释放 weights 中的指针数组
        size_t nlayer = model->meta.nlayer;
        llaisysTensor_t *ptrs[] = {
            model->weights.attn_norm_w,
            model->weights.attn_q_w,
            model->weights.attn_q_b,
            model->weights.attn_k_w,
            model->weights.attn_k_b,
            model->weights.attn_v_w,
            model->weights.attn_v_b,
            model->weights.attn_o_w,
            model->weights.mlp_norm_w,
            model->weights.mlp_gate_w,
            model->weights.mlp_up_w,
            model->weights.mlp_down_w};

        // 遍历二级指针数组逐个释放
        for (size_t p = 0; p < sizeof(ptrs) / sizeof(ptrs[0]); ++p) {
            if (ptrs[p] != NULL) {
                // 释放每个数组元素指向的张量
                for (size_t i = 0; i < nlayer; ++i) {
                    if (ptrs[p][i] != NULL) {
                        tensorDestroy(ptrs[p][i]); // 释放张量内部数据
                        ptrs[p][i] = NULL;
                    }
                }
                // 释放指针数组本身
                free(ptrs[p]);
                ptrs[p] = NULL;
            }
        }

        // 4. 最后释放顶层结构体
        free(model);

        LOG("✅  llaisysQwen2ModelDestroy: end\n\n");
        return;
    }

    __export struct LlaisysQwen2KVCache *llaisysQwen2KVCacheCreate(struct LlaisysQwen2Model * model, size_t max_len) {
        struct LlaisysQwen2KVCache *kvcache = (struct LlaisysQwen2KVCache *)malloc(sizeof(struct LlaisysQwen2KVCache));
        LOG("🧠 llaisysQwen2KVCacheCreate: begin\n");

        size_t nlayer = model->meta.nlayer;
        kvcache->kcache = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);
        kvcache->vcache = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * nlayer);

        size_t nkvh = model->meta.nkvh;
        size_t dh = model->meta.dh;
        llaisysDataType_t dtype = model->meta.dtype;
        llaisysDeviceType_t device = model->device;
        int device_id = model->device_ids[0];
        std::vector<size_t> shape{max_len, nkvh, dh};

        LOG("📊 KVCache Configs:")
        LOG("   💾 kcache: nlayer × (max_len, nkvh, d)")
        LOG("   💾 vcache: nlayer × (max_len, nkvh, dv)")
        print_config("nlayer", nlayer, "📦");
        print_config("nkvh", nkvh, "🔢");
        print_config("d", dh, "🧱");
        print_config("dv", dh, "🧱");
        std::cout << std::endl;

        for (size_t i = 0; i < nlayer; ++i) {
            kvcache->kcache[i] = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            kvcache->vcache[i] = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
        }

        LOG("🎉 llaisysQwen2KVCacheCreate: end\n\n")

        return kvcache;
    }

    __export void llaisysQwen2KVCacheDestroy(struct LlaisysQwen2KVCache * kvcache, size_t nlayer) {
        if (kvcache == NULL) {
            return;
        }
        LOG("🗑️  llaisysQwen2KVCacheDestroy: begin");

        LOG("   🧹 freeing kvcache memory...")
        for (size_t i = 0; i < nlayer; ++i) {
            if (kvcache->kcache[i] != NULL) {
                tensorDestroy(kvcache->kcache[i]);
                kvcache->kcache[i] = NULL;
            }

            if (kvcache->vcache[i] != NULL) {
                tensorDestroy(kvcache->vcache[i]);
                kvcache->vcache[i] = NULL;
            }
        }

        free(kvcache->kcache);
        free(kvcache->vcache);
        kvcache->kcache = NULL;
        kvcache->vcache = NULL;

        free(kvcache);
        LOG("   ✅ kvcache destroyed successfully")

        LOG("🎉 llaisysQwen2KVCacheDestroy: end\n\n");

        return;
    }

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken, struct LlaisysQwen2KVCache *kvcache, size_t past_len) {
        ASSERT(model != NULL, "model is NULL");
        ASSERT(token_ids != NULL, "token_ids is NULL");
        ASSERT(ntoken != 0, "ntoken is 0");
        ASSERT(kvcache != NULL, "kvcache is NULL");

        omp_set_num_threads(24);

        // std::cout << "llaisysQwen2ModelInfer==> ntoken: " << ntoken << ", pastlen: " << past_len << std::endl;

        size_t nlayer = model->meta.nlayer;
        size_t voc = model->meta.voc;
        size_t di = model->meta.di;
        size_t hs = model->meta.hs;
        size_t dh = model->meta.dh;
        size_t nh = model->meta.nh;
        size_t nkvh = model->meta.nkvh;
        size_t dkvh = model->meta.nkvh * dh;
        float eps = model->meta.epsilon;
        float theta = model->meta.theta;
        float scale = 1.0f / std::sqrt(static_cast<float>(dh));

        size_t seqlen = ntoken;
        size_t total_len = seqlen + past_len;

        llaisysDataType_t dtype = model->meta.dtype;
        llaisysDeviceType_t device = model->device;
        int device_id = model->device_ids[0];

        // used for create tensor
        std::vector<size_t> shape;

        // create pos_ids for RoPE
        shape = {seqlen};
        llaisysTensor_t pos_ids_tensor = tensorCreate(shape.data(), shape.size(), LLAISYS_DTYPE_I64, device, device_id);
        int64_t *pos_ids = (int64_t *)tensorGetData(pos_ids_tensor);
        for (size_t i = 0; i < seqlen; ++i) {
            pos_ids[i] = past_len + i;
        }

        // 1. input embedding
        shape = {seqlen};
        llaisysTensor_t input_token_tensor = tensorCreate(shape.data(), shape.size(), LLAISYS_DTYPE_I64, device, device_id);
        tensorLoad(input_token_tensor, token_ids);
#ifdef DEBUG
        std::cout << "input_token_tensor: " << std::endl;
        tensorDebug(input_token_tensor);
#endif

        shape = {seqlen, hs};
        llaisysTensor_t input_embed_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
        llaisysEmbedding(input_embed_tensor, input_token_tensor, model->weights.in_embed);

        llaisysTensor_t attn_layer_tensor = input_embed_tensor;
#ifdef DEBUG
        std::cout << "attn_layer_tensor: " << std::endl;
        tensorDebug(tensorSlice(attn_layer_tensor, 0, seqlen - 1, seqlen));
#endif

        // 2. tranformers
        for (size_t i = 0; i < nlayer; ++i) {
            // 2.1 RMS Norm
            shape = {seqlen, hs};
            llaisysTensor_t attn_layernorm_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysRmsNorm(attn_layernorm_tensor, attn_layer_tensor, model->weights.attn_norm_w[i], eps);
#ifdef DEBUG
            std::cout << "attn_layernorm_tensor: " << std::endl;
            tensorDebug(tensorSlice(attn_layernorm_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.2 Attention
            // 2.2.1 q_proj
            shape = {seqlen, hs};
            llaisysTensor_t q_proj_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysLinear(q_proj_tensor, attn_layernorm_tensor, model->weights.attn_q_w[i], model->weights.attn_q_b[i]);
#ifdef DEBUG
            std::cout << "q_proj_tensor: " << std::endl;
            tensorDebug(tensorSlice(q_proj_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.2.2 q_rope
            shape = {seqlen, nh, dh};
            q_proj_tensor = tensorView(q_proj_tensor, shape.data(), shape.size());
            llaisysTensor_t q_rope_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysROPE(q_rope_tensor, q_proj_tensor, pos_ids_tensor, theta);
#ifdef DEBUG
            std::cout << "q_rope_tensor: " << std::endl;
            tensorDebug(tensorSlice(q_rope_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.2.3 k_proj
            shape = {seqlen, dkvh};
            llaisysTensor_t k_proj_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysLinear(k_proj_tensor, attn_layernorm_tensor, model->weights.attn_k_w[i], model->weights.attn_k_b[i]);
#ifdef DEBUG
            std::cout << "k_proj_tensor: " << std::endl;
            tensorDebug(tensorSlice(k_proj_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.2.4 k_rope
            shape = {seqlen, nkvh, dh};
            k_proj_tensor = tensorView(k_proj_tensor, shape.data(), shape.size());
            llaisysTensor_t k_rope_tensor = tensorSlice(kvcache->kcache[i], 0, past_len, total_len);
            llaisysROPE(k_rope_tensor, k_proj_tensor, pos_ids_tensor, theta);
#ifdef DEBUG
            std::cout << "k_rope_tensor: " << std::endl;
            tensorDebug(tensorSlice(k_rope_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.2.5 v_proj
            shape = {seqlen, dkvh};
            llaisysTensor_t v_proj_tensor = tensorView(tensorSlice(kvcache->vcache[i], 0, past_len, total_len), shape.data(), shape.size());
            llaisysLinear(v_proj_tensor, attn_layernorm_tensor, model->weights.attn_v_w[i], model->weights.attn_v_b[i]);
#ifdef DEBUG
            std::cout << "v_proj_tensor: " << std::endl;
            tensorDebug(tensorSlice(v_proj_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.2.6 self-attention
            shape = {seqlen, nh, dh};
            llaisysTensor_t attn_val_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysTensor_t attn_q_tensor = q_rope_tensor;
            llaisysTensor_t attn_k_tensor = tensorSlice(kvcache->kcache[i], 0, 0, total_len);
            llaisysTensor_t attn_v_tensor = tensorSlice(kvcache->vcache[i], 0, 0, total_len);
            llaisysSelfAttention(attn_val_tensor, attn_q_tensor, attn_k_tensor, attn_v_tensor, scale);
#ifdef DEBUG
            std::cout << "attn_val_tensor: " << std::endl;
            tensorDebug(tensorSlice(attn_val_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.2.7 o_proj
            shape = {seqlen, hs};
            attn_val_tensor = tensorView(attn_val_tensor, shape.data(), shape.size());
            llaisysTensor_t o_proj_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysLinear(o_proj_tensor, attn_val_tensor, model->weights.attn_o_w[i], nullptr);
#ifdef DEBUG
            std::cout << "o_proj_tensor: " << std::endl;
            tensorDebug(tensorSlice(o_proj_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.3 RMS Norm
            shape = {seqlen, hs};
            llaisysTensor_t mlp_layer_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysAdd(mlp_layer_tensor, o_proj_tensor, attn_layer_tensor);
#ifdef DEBUG
            std::cout << "mlp_layer_tensor: " << std::endl;
            tensorDebug(tensorSlice(mlp_layer_tensor, 0, seqlen - 1, seqlen));
#endif

            shape = {seqlen, hs};
            llaisysTensor_t mlp_layernorm_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysRmsNorm(mlp_layernorm_tensor, mlp_layer_tensor, model->weights.mlp_norm_w[i], eps);
#ifdef DEBUG
            std::cout << "mlp_layernorm_tensor: " << std::endl;
            tensorDebug(tensorSlice(mlp_layernorm_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.4 MLP
            // 2.4.1 gate
            shape = {seqlen, di};
            llaisysTensor_t mlp_gate_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysLinear(mlp_gate_tensor, mlp_layernorm_tensor, model->weights.mlp_gate_w[i], nullptr);
#ifdef DEBUG
            std::cout << "mlp_gate_tensor: " << std::endl;
            tensorDebug(tensorSlice(mlp_gate_tensor, 0, seqlen - 1, seqlen));
#endif
            // 2.4.2 up
            shape = {seqlen, di};
            llaisysTensor_t mlp_up_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysLinear(mlp_up_tensor, mlp_layernorm_tensor, model->weights.mlp_up_w[i], nullptr);
#ifdef DEBUG
            std::cout << "mlp_up_tensor: " << std::endl;
            tensorDebug(tensorSlice(mlp_up_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.4.3 swiglu
            shape = {seqlen, di};
            llaisysTensor_t mlp_swiglu_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysSwiGLU(mlp_swiglu_tensor, mlp_gate_tensor, mlp_up_tensor);
#ifdef DEBUG
            std::cout << "mlp_swiglu_tensor: " << std::endl;
            tensorDebug(tensorSlice(mlp_swiglu_tensor, 0, seqlen - 1, seqlen));
#endif

            // 2.4.5 down
            shape = {seqlen, hs};
            llaisysTensor_t mlp_down_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
            llaisysLinear(mlp_down_tensor, mlp_swiglu_tensor, model->weights.mlp_down_w[i], nullptr);
#ifdef DEBUG
            std::cout << "mlp_down_tensor: " << std::endl;
            tensorDebug(tensorSlice(mlp_down_tensor, 0, seqlen - 1, seqlen));
#endif

            llaisysAdd(attn_layer_tensor, mlp_down_tensor, mlp_layer_tensor);
#ifdef DEBUG
            std::cout << "attn_layer_tensor: " << std::endl;
            tensorDebug(tensorSlice(attn_layer_tensor, 0, seqlen - 1, seqlen));
#endif
            // 2.5 destroy
            // release inner intermediate tensors
            tensorDestroy(attn_layernorm_tensor);
            tensorDestroy(q_proj_tensor);
            tensorDestroy(q_rope_tensor);
            tensorDestroy(k_proj_tensor);
            tensorDestroy(attn_val_tensor);
            tensorDestroy(o_proj_tensor);
            tensorDestroy(mlp_layer_tensor);
            tensorDestroy(mlp_layernorm_tensor);
            tensorDestroy(mlp_gate_tensor);
            tensorDestroy(mlp_up_tensor);
            tensorDestroy(mlp_swiglu_tensor);
            tensorDestroy(mlp_down_tensor);
        }

        // 3. Output
        // 3.1 RMS Norm
        shape = {seqlen, hs};
        llaisysTensor_t output_layernorm_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
        llaisysRmsNorm(output_layernorm_tensor, attn_layer_tensor, model->weights.out_norm_w, eps);
#ifdef DEBUG
        std::cout << "output_layernorm_tensor: " << std::endl;
        tensorDebug(tensorSlice(output_layernorm_tensor, 0, seqlen - 1, seqlen));
#endif

        // 3.2 output embedding
        shape = {seqlen, voc};
        llaisysTensor_t output_embed_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
        llaisysLinear(output_embed_tensor, output_layernorm_tensor, model->weights.out_embed, nullptr);
#ifdef DEBUG
        std::cout << "output_embed_tensor: " << std::endl;
        tensorDebug(tensorSlice(output_embed_tensor, 0, seqlen - 1, seqlen));
#endif

        // 3.3 Argmax
        llaisysTensor_t last_output_embed_tensor = tensorSlice(output_embed_tensor, 0, seqlen - 1, seqlen);
#ifdef DEBUG
        std::cout << "last_output_embed_tensor: " << std::endl;
        tensorDebug(last_output_embed_tensor);
#endif
        shape = {1};
        llaisysTensor_t max_idx_tensor = tensorCreate(shape.data(), shape.size(), LLAISYS_DTYPE_I64, device, device_id);
        llaisysTensor_t max_vals_tensor = tensorCreate(shape.data(), shape.size(), dtype, device, device_id);
        llaisysArgmax(max_idx_tensor, max_vals_tensor, last_output_embed_tensor);
#ifdef DEBUG
        std::cout << "max_idx_tensor: " << std::endl;
        tensorDebug(max_idx_tensor);
#endif

        int64_t token_id = *(int64_t *)tensorGetData(max_idx_tensor);

        // 4. release outer intermediate tensors
        tensorDestroy(pos_ids_tensor);
        tensorDestroy(input_token_tensor);
        tensorDestroy(attn_layer_tensor);
        tensorDestroy(output_layernorm_tensor);
        tensorDestroy(output_embed_tensor);
        tensorDestroy(max_idx_tensor);
        tensorDestroy(max_vals_tensor);

        return token_id;
    }
}