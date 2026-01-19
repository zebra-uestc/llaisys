#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <omp.h>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
#pragma omp parallel for simd
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float gate_i = llaisys::utils::cast<float>(gate[i]);
            float up_i = llaisys::utils::cast<float>(up[i]);
            float denom = 1.0f + std::exp(-gate_i);
            float ans = up_i * gate_i / denom;
            out[i] = llaisys::utils::cast<T>(ans);
        } else {
            T denom = T{1} + std::exp(-gate[i]);
            out[i] = up[i] * gate[i] / denom;
        }
    }
}

template <typename T>
void swiglu_(T *output, const T *gate, const T *up, size_t numel, 
             float gate_scale, float up_scale, float output_scale) {
    if constexpr (std::is_same_v<T, int8_t>) {
        alignas(64) float swish_lut[256]; 
    
        for (int i = 0; i < 256; i++) {
            int8_t q_val = static_cast<int8_t>(i - 128); 
            
            float g_val = q_val * gate_scale;
            
            float swish_val = g_val / (1.0f + std::exp(-g_val));
            
            swish_lut[i] = swish_val;
        }

        float effective_scale = up_scale / output_scale;

        #pragma omp parallel for
        for (size_t i = 0; i < numel; i++) {
            int idx = static_cast<int>(gate[i]) + 128; 
            float val_swish = swish_lut[idx];

            float val_up = static_cast<float>(up[i]);

            float res_f = val_swish * val_up * effective_scale;

            output[i] = llaisys::utils::cast<int8_t>(res_f);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel, 
            float gate_scale, float up_scale, float output_scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    case LLAISYS_DTYPE_I8:
        return swiglu_(reinterpret_cast<int8_t *>(out), reinterpret_cast<const int8_t *>(gate),
                       reinterpret_cast<const int8_t *>(up), numel, gate_scale, up_scale, output_scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
