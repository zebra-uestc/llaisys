#include "add_opt.hpp"
#include <immintrin.h>
#include <omp.h>

// Performance tuning thresholds - adjust based on actual hardware benchmarking
#ifndef SIMD_THRESHOLD
#define SIMD_THRESHOLD 64 // Below this: use pure scalar (fastest for tiny data)
#endif

#ifndef PARALLEL_THRESHOLD
#define PARALLEL_THRESHOLD 16384 // Below this: use single-threaded SIMD
#endif

// Force inline to eliminate function call overhead
#define FORCE_INLINE __attribute__((always_inline)) inline

// Fast path for tiny data - optimized for shapes like (2,3)
template <typename T>
FORCE_INLINE void scalar_add_tiny(T *c, const T *a, const T *b, size_t numel) {
    // Simplest loop - let compiler optimize
    // For <64 elements, this is faster than any SIMD approach
    for (size_t i = 0; i < numel; ++i) {
        if constexpr (std::is_arithmetic_v<T>) {
            c[i] = a[i] + b[i];
        } else {
            float f_a = llaisys::utils::cast<float>(a[i]);
            float f_b = llaisys::utils::cast<float>(b[i]);
            c[i] = llaisys::utils::cast<T>(f_a + f_b);
        }
    }
}

void add_i8(int8_t *c, const int8_t *a, const int8_t *b, size_t numel) {
    // Fast path: tiny data - avoid all vectorization overhead
    if (numel <= SIMD_THRESHOLD) {
        scalar_add_tiny(c, a, b, numel);
        return;
    }

#if defined(__AVX512BW__)
    const size_t vec_size = 64; // 512 bits / 8 bits = 64 elements
    size_t align = numel & ~(vec_size - 1); // Bit-wise modulo, faster than % operator

    // Medium scale: single-threaded SIMD (avoid OpenMP thread creation overhead)
    if (numel < PARALLEL_THRESHOLD) {
        size_t i = 0;
        // Main loop: process 64 int8 elements at once
        for (; i < align; i += vec_size) {
            // Load 64 int8 values (512 bits) directly from memory
            __m512i a_vec = _mm512_loadu_si512((const void*)(a + i));
            __m512i b_vec = _mm512_loadu_si512((const void*)(b + i));

            // Perform integer addition
            // Note: _mm512_add_epi8 performs standard wrap-around addition.
            // Use _mm512_adds_epi8 for saturation (e.g., 120 + 10 = 127), common in quantization.
            __m512i c_vec = _mm512_add_epi8(a_vec, b_vec);

            // Store 64 int8 results back to memory
            _mm512_storeu_si512((void*)(c + i), c_vec);
        }

        // Handle remaining elements (< 64) using scalar addition
        for (; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
        return;
    }

// Large scale: multi-threaded SIMD
// Manual thread partitioning for better cache locality (static scheduling)
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Manual chunking: ensure each thread processes multiples of vec_size
        size_t chunk_size = align / nthreads;
        chunk_size = (chunk_size / vec_size) * vec_size;

        size_t start = tid * chunk_size;
        size_t end = (tid == nthreads - 1) ? align : start + chunk_size;

        for (size_t i = start; i < end; i += vec_size) {
            // Load 64 int8 values
            __m512i a_vec = _mm512_loadu_si512((const void*)(a + i));
            __m512i b_vec = _mm512_loadu_si512((const void*)(b + i));

            // Perform addition (wrap-around)
            __m512i c_vec = _mm512_add_epi8(a_vec, b_vec); 

            // Store results
            _mm512_storeu_si512((void*)(c + i), c_vec);
        }
    }

    // Single-threaded tail processing for remaining elements
    for (size_t i = align; i < numel; ++i) {
        c[i] = a[i] + b[i];
    }

#else
    // Fallback: scalar implementation with OpenMP parallelization for general CPUs
    // (Note: An AVX2 branch could be added here using _mm256_add_epi8)
    if (numel < PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < numel; ++i) c[i] = a[i] + b[i];
    } else {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numel; ++i) c[i] = a[i] + b[i];
    }
#endif
}

void add_bf16(llaisys::bf16_t *c, const llaisys::bf16_t *a, const llaisys::bf16_t *b, size_t numel) {
    // Fast path: tiny data - avoid all vectorization overhead
    if (numel <= SIMD_THRESHOLD) {
        scalar_add_tiny(c, a, b, numel);
        return;
    }

#if defined(__AVX512F__)
    const size_t vec_size = 16;
    size_t align = numel & ~(vec_size - 1); // Bit-wise modulo, faster than % operator

    // Medium scale: single-threaded SIMD (avoid OpenMP thread creation overhead)
    if (numel < PARALLEL_THRESHOLD) {
        size_t i = 0;

        // Main loop: process 16 bf16 elements at once (converted to fp32 for addition)
        for (; i < align; i += vec_size) {
            // Load 16 bf16 values (each 2 bytes) as 256-bit integer vector
            __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
            __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));

            // Convert 16x bf16 (uint16_t) to 16x int32_t (zero-extended)
            __m512i a_int = _mm512_cvtepu16_epi32(a_vec);
            __m512i b_int = _mm512_cvtepu16_epi32(b_vec);

            // Shift left by 16 bits to convert bf16 to fp32 format (implicit mantissa alignment)
            // Reinterpret as float vectors for IEEE 754 arithmetic
            __m512 a_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(a_int, 16));
            __m512 b_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(b_int, 16));

            // Perform floating-point addition
            __m512 c_f32 = _mm512_add_ps(a_f32, b_f32);

            // Convert result back to bf16: cast to int, right-shift 16 bits, then truncate to uint16_t
            __m256i c_vec = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(c_f32), 16));

            // Store 16 bf16 results back to memory
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(c + i), c_vec);
        }

        // Handle remaining elements (< 16) using scalar conversion
        for (; i < numel; ++i) {
            float f_a = llaisys::utils::cast<float>(a[i]);
            float f_b = llaisys::utils::cast<float>(b[i]);
            c[i] = llaisys::utils::cast<llaisys::bf16_t>(f_a + f_b);
        }
        return;
    }

// Large scale: multi-threaded SIMD
// Manual thread partitioning for better cache locality (static scheduling)
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Manual chunking: ensure each thread processes multiples of vec_size
        size_t chunk_size = align / nthreads;
        chunk_size = (chunk_size / vec_size) * vec_size;

        size_t start = tid * chunk_size;
        size_t end = (tid == nthreads - 1) ? align : start + chunk_size;

        for (size_t i = start; i < end; i += vec_size) {
            // Load 16 bf16 values (each 2 bytes) as 256-bit integer vector
            __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
            __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));

            // Convert 16x bf16 (uint16_t) to 16x int32_t (zero-extended)
            __m512i a_int = _mm512_cvtepu16_epi32(a_vec);
            __m512i b_int = _mm512_cvtepu16_epi32(b_vec);

            // Shift left by 16 bits to convert bf16 to fp32 format
            // Reinterpret as float vectors for IEEE 754 arithmetic
            __m512 a_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(a_int, 16));
            __m512 b_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(b_int, 16));

            // Perform floating-point addition
            __m512 c_f32 = _mm512_add_ps(a_f32, b_f32);

            // Convert result back to bf16: cast to int, right-shift 16 bits, then truncate
            __m256i c_vec = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(c_f32), 16));

            // Store 16 bf16 results back to memory
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(c + i), c_vec);
        }
    }

    // Single-threaded tail processing for remaining elements
    for (size_t i = align; i < numel; ++i) {
        float f_a = llaisys::utils::cast<float>(a[i]);
        float f_b = llaisys::utils::cast<float>(b[i]);
        c[i] = llaisys::utils::cast<llaisys::bf16_t>(f_a + f_b);
    }

#else
    // Fallback: scalar implementation with OpenMP parallelization for general CPUs
    if (numel < PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < numel; ++i) {
            float f_a = llaisys::utils::cast<float>(a[i]);
            float f_b = llaisys::utils::cast<float>(b[i]);
            c[i] = llaisys::utils::cast<llaisys::bf16_t>(f_a + f_b);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numel; ++i) {
            float f_a = llaisys::utils::cast<float>(a[i]);
            float f_b = llaisys::utils::cast<float>(b[i]);
            c[i] = llaisys::utils::cast<llaisys::bf16_t>(f_a + f_b);
        }
    }
#endif
}

void add_f16(llaisys::fp16_t *c, const llaisys::fp16_t *a, const llaisys::fp16_t *b, size_t numel) {
    // Fast path: tiny data
    if (numel <= SIMD_THRESHOLD) {
        scalar_add_tiny(c, a, b, numel);
        return;
    }

#if defined(__AVX512F__) && defined(__F16C__)
    const size_t vec_size = 16;
    size_t align = numel & ~(vec_size - 1); // Bit-wise modulo

    // Medium scale: single-threaded SIMD
    if (numel < PARALLEL_THRESHOLD) {
        size_t i = 0;

        // Process 16 fp16 elements in parallel using native conversion
        for (; i < align; i += vec_size) {
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
        for (; i < numel; ++i) {
            float f_a = llaisys::utils::cast<float>(a[i]);
            float f_b = llaisys::utils::cast<float>(b[i]);
            c[i] = llaisys::utils::cast<llaisys::fp16_t>(f_a + f_b);
        }
        return;
    }

// Large scale: multi-threaded SIMD with manual partitioning
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Manual chunking for cache-friendly access patterns
        size_t chunk_size = align / nthreads;
        chunk_size = (chunk_size / vec_size) * vec_size;

        size_t start = tid * chunk_size;
        size_t end = (tid == nthreads - 1) ? align : start + chunk_size;

        for (size_t i = start; i < end; i += vec_size) {
            // Load 16 fp16 values (32 bytes) as 256-bit integer vector
            __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
            __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));

            // Convert 16x fp16 to 16x fp32 using F16C intrinsic
            __m512 a_f32 = _mm512_cvtph_ps(a_vec);
            __m512 b_f32 = _mm512_cvtph_ps(b_vec);

            // Perform floating-point addition
            __m512 c_f32 = _mm512_add_ps(a_f32, b_f32);

            // Convert result back to fp16 with round-to-nearest-even
            __m256i c_vec = _mm512_cvtps_ph(c_f32, _MM_FROUND_TO_NEAREST_INT);

            // Store 16 fp16 results back to memory
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(c + i), c_vec);
        }
    }

    // Single-threaded tail processing
    for (size_t i = align; i < numel; ++i) {
        float f_a = llaisys::utils::cast<float>(a[i]);
        float f_b = llaisys::utils::cast<float>(b[i]);
        c[i] = llaisys::utils::cast<llaisys::fp16_t>(f_a + f_b);
    }

#else
    // Fallback: scalar implementation with OpenMP parallelization for platforms without F16C support
    if (numel < PARALLEL_THRESHOLD) {
        for (size_t i = 0; i < numel; ++i) {
            float f_a = llaisys::utils::cast<float>(a[i]);
            float f_b = llaisys::utils::cast<float>(b[i]);
            c[i] = llaisys::utils::cast<llaisys::fp16_t>(f_a + f_b);
        }
    } else {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numel; ++i) {
            float f_a = llaisys::utils::cast<float>(a[i]);
            float f_b = llaisys::utils::cast<float>(b[i]);
            c[i] = llaisys::utils::cast<llaisys::fp16_t>(f_a + f_b);
        }
    }
#endif
}

void add_f32(float *c, const float *a, const float *b, size_t numel) {
    // Fast path: tiny data - pure scalar is fastest
    if (numel <= SIMD_THRESHOLD) {
        for (size_t i = 0; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
        return;
    }

#if defined(__AVX512F__)
    const size_t vec_size = 16;
    size_t align = numel & ~(vec_size - 1); // Bit-wise modulo

    // Medium scale: single-threaded SIMD
    if (numel < PARALLEL_THRESHOLD) {
        size_t i = 0;

        // Optional: 2x loop unrolling to reduce branch prediction overhead
        for (; i + 31 < align; i += 32) {
            // Process first 16 elements
            __m512 va1 = _mm512_loadu_ps(a + i);  // Load 16 unaligned floats
            __m512 vb1 = _mm512_loadu_ps(b + i);  // Load 16 unaligned floats
            __m512 vc1 = _mm512_add_ps(va1, vb1); // Vectorized addition
            _mm512_storeu_ps(c + i, vc1);         // Store result back

            // Process second 16 elements
            __m512 va2 = _mm512_loadu_ps(a + i + 16);
            __m512 vb2 = _mm512_loadu_ps(b + i + 16);
            __m512 vc2 = _mm512_add_ps(va2, vb2);
            _mm512_storeu_ps(c + i + 16, vc2);
        }

        // Process remaining 16-element blocks
        for (; i < align; i += vec_size) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 vc = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(c + i, vc);
        }

        // Handle remaining elements (< 16) with scalar fallback
        for (; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
        return;
    }

// Large scale: multi-threaded SIMD with manual partitioning
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Manual chunking: ensure each thread processes multiples of vec_size
        size_t chunk_size = align / nthreads;
        chunk_size = (chunk_size / vec_size) * vec_size;

        size_t start = tid * chunk_size;
        size_t end = (tid == nthreads - 1) ? align : start + chunk_size;

        // Process 16 single-precision floats per iteration
        for (size_t i = start; i < end; i += vec_size) {
            __m512 va = _mm512_loadu_ps(a + i); // Load 16 unaligned floats
            __m512 vb = _mm512_loadu_ps(b + i); // Load 16 unaligned floats
            __m512 vc = _mm512_add_ps(va, vb);  // Vectorized addition
            _mm512_storeu_ps(c + i, vc);        // Store result back
        }
    }

    // Single-threaded tail processing
    for (size_t i = align; i < numel; ++i) {
        c[i] = a[i] + b[i];
    }

#else
    // Fallback: scalar loop with OpenMP SIMD directive for automatic vectorization
    // Compiler will auto-vectorize if target supports SSE/AVX and optimizations enabled
    if (numel < PARALLEL_THRESHOLD) {
#pragma omp simd
        for (size_t i = 0; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
    } else {
#pragma omp parallel for schedule(static) simd
        for (size_t i = 0; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
    }
#endif
}