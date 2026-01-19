#include "types.hpp"

#include <cstring>
#include <emmintrin.h>

namespace llaisys::utils {
float _f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;
    if (exponent == 31) {
        if (mantissa != 0) {
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
            f32 = sign | 0x7F800000;
        }
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

fp16_t _f32_to_f16(float val) {
    uint32_t f32;
    memcpy(&f32, &val, sizeof(f32));               // Read the bits of the float32
    uint16_t sign = (f32 >> 16) & 0x8000;          // Extract the sign bit
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127; // Extract and de-bias the exponent
    uint32_t mantissa = f32 & 0x7FFFFF;            // Extract the mantissa (fraction part)

    if (exponent >= 16) { // Special cases for Inf and NaN
        // NaN
        if (exponent == 128 && mantissa != 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7E00)};
        }
        // Infinity
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00)};
    } else if (exponent >= -14) { // Normalized case
        return fp16_t{(uint16_t)(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    } else if (exponent >= -24) {
        mantissa |= 0x800000; // Add implicit leading 1
        mantissa >>= (-14 - exponent);
        return fp16_t{(uint16_t)(sign | (mantissa >> 13))};
    } else {
        // Too small for subnormal: return signed zero
        return fp16_t{(uint16_t)sign};
    }
}

float _bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;

    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const uint32_t rounding_bias = 0x00007FFF + // 0111 1111 1111 1111
                                   ((bits32 >> 16) & 1);

    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_t{bf16_bits};
}

float _fp8_e5m2_to_fp32(f8a_t x) {
    return FP8_E5M2_LUT[x._v];
}

f8a_t _fp32_to_fp8_e5m2(float x) {
    if (x <= 0.0f) {
        return f8a_t{0};
    }

    uint8_t best_index = 0;
    float smallest_diff = std::fabs(x - FP8_E5M2_LUT[0]);

    for (uint16_t i = 1; i < 256; ++i) {
        float diff = std::fabs(x - FP8_E5M2_LUT[i]);
        if (diff < smallest_diff) {
            smallest_diff = diff;
            best_index = static_cast<uint8_t>(i);
        }
    }

    return f8a_t{best_index};
}

float _fp8_e4m3_to_fp32(f8b_t x) {
    return FP8_E4M3_LUT[x._v];
}

f8b_t _fp32_to_fp8_e4m3(float x) {
    if (x <= 0.0f) {
        return f8b_t{0};
    }

    uint8_t best_index = 0;
    float smallest_diff = std::fabs(x - FP8_E4M3_LUT[0]);

    for (uint16_t i = 1; i < 256; ++i) {
        float diff = std::fabs(x - FP8_E4M3_LUT[i]);
        if (diff < smallest_diff) {
            smallest_diff = diff;
            best_index = static_cast<uint8_t>(i);
        }
    }

    return f8b_t{best_index};
}

#ifdef __F16C__
// Convert a single FP16 value to FP32 using F16C instructions
// This function leverages hardware acceleration for optimal performance
float fp16_to_fp32_f16c(fp16_t x) {
    // Load the 16-bit FP16 value into the low 16 bits of a 128-bit register
    __m128i vec = _mm_cvtsi32_si128(static_cast<int>(x._v));

    // Convert the FP16 value to FP32 using F16C instruction
    __m128 vec_f32 = _mm_cvtph_ps(vec);

    // Extract the converted FP32 value from the SIMD register
    return _mm_cvtss_f32(vec_f32);
}

// Convert a single FP32 value to FP16 using F16C instructions
// This function leverages hardware acceleration for optimal performance
fp16_t fp32_to_fp16_f16c(float x) {
    // Load the FP32 value into the low 32 bits of a 128-bit register
    __m128 vec = _mm_set_ss(x);

    // Convert the FP32 value to FP16 using F16C instruction with nearest-even rounding
    __m128i vec_f16 = _mm_cvtps_ph(vec, _MM_FROUND_TO_NEAREST_INT);

    // Extract the converted FP16 value from the SIMD register
    return fp16_t{static_cast<uint16_t>(_mm_extract_epi16(vec_f16, 0))};
}

// Convert a batch of FP16 values to FP32 using F16C instructions
// Processes elements in groups of 8 for optimal performance
void fp16_to_fp32_batch_f16c(float *dst, const fp16_t *src, size_t count) {
    size_t i = 0;

    // Process elements in groups of 8 using SIMD instructions
    for (; i + 7 < count; i += 8) {
        // Load 8 FP16 values (16 bytes) into a 128-bit register
        __m128i vec_f16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));

        // Convert 8 FP16 values to 8 FP32 values using F16C instruction
        __m256 vec_f32 = _mm256_cvtph_ps(vec_f16);

        // Store the 8 converted FP32 values to the destination array
        _mm256_storeu_ps(dst + i, vec_f32);
    }

    // Process any remaining elements (less than 8) using single-element conversion
    for (; i < count; ++i) {
        dst[i] = fp16_to_fp32_f16c(src[i]);
    }
}

// Convert a batch of FP32 values to FP16 using F16C instructions
// Processes elements in groups of 8 for optimal performance
void fp32_to_fp16_batch_f16c(fp16_t *dst, const float *src, size_t count) {
    size_t i = 0;

    // Process elements in groups of 8 using SIMD instructions
    for (; i + 7 < count; i += 8) {
        // Load 8 FP32 values into a 256-bit register
        __m256 vec_f32 = _mm256_loadu_ps(src + i);

        // Convert 8 FP32 values to 8 FP16 values using F16C instruction with nearest-even rounding
        __m128i vec_f16 = _mm256_cvtps_ph(vec_f32, _MM_FROUND_TO_NEAREST_INT);

        // Store the 8 converted FP16 values (16 bytes) to the destination array
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), vec_f16);
    }

    // Process any remaining elements (less than 8) using single-element conversion
    for (; i < count; ++i) {
        dst[i] = fp32_to_fp16_f16c(src[i]);
    }
}
#endif

// AVX-512 optimized versions (if available)
#ifdef __AVX512F__
// Convert a batch of BF16 values to FP32 using AVX-512 instructions
// Processes 16 elements at a time for optimal performance
void bf16_to_fp32_batch_avx512(float *dst, const bf16_t *src, size_t count) {
    size_t i = 0;

    // Process elements in groups of 16 using AVX-512
    for (; i + 15 < count; i += 16) {
        // Load 16 BF16 values into a 256-bit register
        __m256i bf16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));

        // Convert BF16 to FP32
        __m512i bf16_extended = _mm512_cvtepu16_epi32(bf16_vec);
        __m512i bf16_shifted = _mm512_slli_epi32(bf16_extended, 16);
        __m512 fp32_vec = _mm512_castsi512_ps(bf16_shifted);

        // Store 16 FP32 results to destination array
        _mm512_storeu_ps(dst + i, fp32_vec);
    }

    // Handle remaining elements
    for (; i < count; ++i) {
        // Single element conversion
        dst[i] = _bf16_to_f32(src[i]);
    }
}

// Convert a batch of FP32 values to BF16 using AVX-512 instructions
// Processes 16 elements at a time for optimal performance
void fp32_to_bf16_batch_avx512(bf16_t *dst, const float *src, size_t count) {
    size_t i = 0;

    // Process elements in groups of 16 using AVX-512
    for (; i + 15 < count; i += 16) {
        // Load 16 FP32 values into a 512-bit register
        __m512 fp32_vec = _mm512_loadu_ps(src + i);

        // Convert FP32 to BF16
        __m512i fp32_int = _mm512_castps_si512(fp32_vec);
        __m512i bf16_shifted = _mm512_srli_epi32(fp32_int, 16);
        __m256i bf16_vec = _mm512_cvtepi32_epi16(bf16_shifted);

        // Store 16 BF16 results to destination array
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), bf16_vec);
    }

    // Handle remaining elements
    for (; i < count; ++i) {
        // Single element conversion
        dst[i] = _f32_to_bf16(src[i]);
    }
} // __AVX512F__
#elif defined(__AVX2__)
// Convert a batch of BF16 values to FP32 using AVX2 instructions
// Processes 8 elements at a time for optimal performance
void bf16_to_fp32_batch_avx2(float *dst, const bf16_t *src, size_t count) {
    size_t i = 0;

    // Process elements in groups of 8 using AVX2
    for (; i + 7 < count; i += 8) {
        // Load 8 BF16 values (16 bytes) into a 128-bit register
        __m128i bf16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));

        // Convert BF16 to FP32:
        // 1. Zero-extend 16-bit BF16 values to 32-bit integers
        __m256i bf16_extended = _mm256_cvtepu16_epi32(bf16_vec);

        // 2. Shift left by 16 bits to place BF16 bit pattern in upper 16 bits of FP32
        __m256i bf16_shifted = _mm256_slli_epi32(bf16_extended, 16);

        // 3. Reinterpret the integer bits as floating-point values
        __m256 fp32_vec = _mm256_castsi256_ps(bf16_shifted);

        // Store 8 FP32 results to destination array
        _mm256_storeu_ps(dst + i, fp32_vec);
    }

    // Handle remaining elements (less than 8)
    for (; i < count; ++i) {
        // Single element conversion
        dst[i] = _bf16_to_f32(src[i]);
    }
}

// Convert a batch of FP32 values to BF16 using AVX2 instructions
// Processes 8 elements at a time for optimal performance
void fp32_to_bf16_batch_avx2(bf16_t *dst, const float *src, size_t count) {
    size_t i = 0;

    // Process elements in groups of 8 using AVX2
    for (; i + 7 < count; i += 8) {
        // Load 8 FP32 values into a 256-bit register
        __m256 fp32_vec = _mm256_loadu_ps(src + i);

        // Convert FP32 to BF16:
        // 1. Reinterpret floating-point bits as integers
        __m256i fp32_int = _mm256_castps_si256(fp32_vec);

        // 2. Shift right by 16 bits to extract upper 16 bits as BF16
        __m256i bf16_shifted = _mm256_srli_epi32(fp32_int, 16);

        // 3. Pack 32-bit integers into 16-bit values
        __m128i bf16_vec = _mm256_cvtepi32_epi16(bf16_shifted);

        // Store 8 BF16 results to destination array
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), bf16_vec);
    }

    // Handle remaining elements (less than 8)
    for (; i < count; ++i) {
        // Single element conversion
        dst[i] = _f32_to_bf16(src[i]);
    }
}
#endif // __AVX2__

// Wrapper functions that automatically select the best available implementation
// Convert a batch of BF16 values to FP32 using the best available instruction set
void bf16_to_fp32_batch(float *dst, const bf16_t *src, size_t count) {
#if defined(__AVX512F__)
    bf16_to_fp32_batch_avx512(dst, src, count);
#elif defined(__AVX2__)
    bf16_to_fp32_batch_avx2(dst, src, count);
#else
    // Fallback to scalar implementation if no SIMD support
    for (size_t i = 0; i < count; ++i) {
        dst[i] = _bf16_to_f32(src[i]);
    }
#endif
}

// Convert a batch of FP32 values to BF16 using the best available instruction set
void fp32_to_bf16_batch(bf16_t *dst, const float *src, size_t count) {
#if defined(__AVX512F__)
    fp32_to_bf16_batch_avx512(dst, src, count);
#elif defined(__AVX2__)
    fp32_to_bf16_batch_avx2(dst, src, count);
#else
    // Fallback to scalar implementation if no SIMD support
    for (size_t i = 0; i < count; ++i) {
        dst[i] = _f32_to_bf16(src[i]);
    }
#endif
}

void int8_to_fp32_batch(float* dst, const int8_t* src, size_t size) {
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (; i + 16 <= size; i += 16) {
        __m128i v8 = _mm_loadu_si128((const __m128i*)(src + i));
        __m512i v32 = _mm512_cvtepi8_epi32(v8);
        __m512 f32 = _mm512_cvtepi32_ps(v32);
        _mm512_storeu_ps(dst + i, f32);
    }
#elif defined(__AVX2__)
    for (; i + 8 <= size; i += 8) {
        __m128i v8 = _mm_loadu_si128((const __m128i*)(src + i));
        __m256i v32 = _mm256_cvtepi8_epi32(v8);
        __m256 f32 = _mm256_cvtepi32_ps(v32);
        _mm256_storeu_ps(dst + i, f32);
    }
#endif
    for (; i < size; ++i) dst[i] = static_cast<float>(src[i]);
}

void fp32_to_int8_batch(int8_t* dst, const float* src, size_t size) {
    size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (; i + 16 <= size; i += 16) {
        __m512 f32 = _mm512_loadu_ps(src + i);
        __m512i v32 = _mm512_cvtps_epi32(f32);
        _mm_storeu_si128((__m128i*)(dst + i), _mm512_cvtsepi32_epi8(v32));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= size; i += 8) {
        __m256 f32 = _mm256_loadu_ps(src + i);
        __m256i v32 = _mm256_cvtps_epi32(f32);
        for(size_t j=0; j<8; ++j) {
            int val = _mm256_extract_epi32(v32, j);
            dst[i+j] = llaisys::utils::cast<int8_t>(val);
        }
    }
#endif
    for (; i < size; ++i) {
        dst[i] = llaisys::utils::cast<int8_t>(src[i]);
    }
}

void f8a_to_fp32_batch(float *dst, const f8a_t *src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = _fp8_e5m2_to_fp32(src[i]);
    }
}

void fp32_to_f8a_batch(f8a_t *dst, const float *src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = _fp32_to_fp8_e5m2(src[i]);
    }
}

void f8b_to_fp32_batch(float *dst, const f8b_t *src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = _fp8_e4m3_to_fp32(src[i]);
    }
}

void fp32_to_f8b_batch(f8b_t *dst, const float *src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = _fp32_to_fp8_e4m3(src[i]);
    }
}

} // namespace llaisys::utils
