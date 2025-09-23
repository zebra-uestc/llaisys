#include "add_opt.hpp"
#include <immintrin.h>
#include <omp.h>

void add_bf16(llaisys::bf16_t *c, const llaisys::bf16_t *a, const llaisys::bf16_t *b, size_t numel) {
#if defined(__AVX512F__)
    // AVX512 implementation: process 16 bf16 elements at once (converted to fp32 for addition)
    size_t mod = numel % 16;
    size_t align = numel - mod;

#pragma omp parallel for
    for (int i = 0; i <= (int)align - 16; i += 16) {
        // Load 16 bf16 values (each 2 bytes) as 256-bit integer vector
        __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
        __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));

        // Convert 16x bf16 (uint16_t) to 16x int32_t (zero-extended)
        __m512i a_int = _mm512_cvtepu16_epi32(a_vec);
        __m512i b_int = _mm512_cvtepu16_epi32(b_vec);

        // Shift left by 16 bits to convert bf16 to fp32 format (implicit mantissa alignment)
        __m512i a_shifted = _mm512_slli_epi32(a_int, 16);
        __m512i b_shifted = _mm512_slli_epi32(b_int, 16);

        // Reinterpret as float vectors for IEEE 754 arithmetic
        __m512 a_f32 = _mm512_castsi512_ps(a_shifted);
        __m512 b_f32 = _mm512_castsi512_ps(b_shifted);

        // Perform floating-point addition
        __m512 c_f32 = _mm512_add_ps(a_f32, b_f32);

        // Convert result back to bf16: cast to int, right-shift 16 bits, then truncate to uint16_t
        __m512i c_int = _mm512_castps_si512(c_f32);
        __m512i c_shifted = _mm512_srli_epi32(c_int, 16);
        __m256i c_vec = _mm512_cvtepi32_epi16(c_shifted);

        // Store 16 bf16 results back to memory
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(c + i), c_vec);
    }

    // Handle remaining elements (< 16) using scalar conversion
    for (size_t i = align; i < numel; ++i) {
        float f_a = llaisys::utils::cast<float>(a[i]);
        float f_b = llaisys::utils::cast<float>(b[i]);
        c[i] = llaisys::utils::cast<llaisys::bf16_t>(f_a + f_b);
    }
#else
    // Fallback: scalar implementation with OpenMP parallelization for general CPUs
#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        float f_a = llaisys::utils::cast<float>(a[i]);
        float f_b = llaisys::utils::cast<float>(b[i]);
        c[i] = llaisys::utils::cast<llaisys::bf16_t>(f_a + f_b);
    }
#endif
}

void add_f16(llaisys::fp16_t *c, const llaisys::fp16_t *a, const llaisys::fp16_t *b, size_t numel) {
#if defined(__AVX512F__) && defined(__F16C__)
    // AVX512 + F16C implementation: Process 16 fp16 elements in parallel using native conversion
    size_t mod = numel % 16;
    size_t align = numel - mod;

#pragma omp parallel for
    for (int i = 0; i <= (int)align - 16; i += 16) {
        // Load 16 fp16 values (32 bytes) as 256-bit integer vector
        __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
        __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));

        // Convert 16x fp16 to 16x fp32 using F16C intrinsic (hardware-accelerated)
        __m512 a_f32 = _mm512_cvtph_ps(a_vec);
        __m512 b_f32 = _mm512_cvtph_ps(b_vec);

        // Perform floating-point addition
        __m512 c_f32 = _mm512_add_ps(a_f32, b_f32);

        // Convert result back to fp16 with round-to-nearest-even (IEEE 754 compliant)
        __m256i c_vec = _mm512_cvtps_ph(c_f32, _MM_FROUND_TO_NEAREST_INT);

        // Store 16 fp16 results back to memory
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(c + i), c_vec);
    }

    // Handle remaining elements (< 16) with scalar fallback
    for (size_t i = align; i < numel; ++i) {
        float f_a = llaisys::utils::cast<float>(a[i]);
        float f_b = llaisys::utils::cast<float>(b[i]);
        c[i] = llaisys::utils::cast<llaisys::fp16_t>(f_a + f_b);
    }
#else
    // Fallback: Scalar implementation with OpenMP parallelization for platforms without F16C support
#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        float f_a = llaisys::utils::cast<float>(a[i]);
        float f_b = llaisys::utils::cast<float>(b[i]);
        c[i] = llaisys::utils::cast<llaisys::fp16_t>(f_a + f_b);
    }
#endif
}

void add_f32(float *c, const float *a, const float *b, size_t numel) {
#if defined(__AVX512F__)
    // AVX512 implementation: Process 16 single-precision floats per iteration
    size_t mod = numel % 16;
    size_t align = numel - mod;

    // Use a single OpenMP parallel region for the entire function to avoid nested parallelism
#pragma omp parallel for
    for (int i = 0; i <= (int)align - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i); // Load 16 unaligned floats
        __m512 vb = _mm512_loadu_ps(b + i); // Load 16 unaligned floats
        __m512 vc = _mm512_add_ps(va, vb);  // Vectorized addition
        _mm512_storeu_ps(c + i, vc);        // Store result back
    }

    // Handle remaining elements (< 16) with scalar fallback
    for (size_t i = align; i < numel; ++i) {
        c[i] = a[i] + b[i];
    }
#else
    // Fallback: Scalar loop with OpenMP SIMD directive for automatic vectorization
    // Compiler will auto-vectorize if target supports SSE/AVX and optimizations enabled
#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        c[i] = a[i] + b[i];
    }
#endif
}