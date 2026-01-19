#include "matmul_quantized.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define MR 14
#define NR 32
#define NTHREADS 24
#define MC (MR * NTHREADS * 5)
#define NC (NR * NTHREADS * 30) 
#define KC 512

#define OMP_PRAGMA_PARALLEL _Pragma("omp parallel for num_threads(NTHREADS)")

static float blockA_packed[MC * KC] __attribute__((aligned(64)));

static int8_t blockB_packed[NC * KC] __attribute__((aligned(64)));

static inline __mmask16 create_mask(int nr) {
    nr = (nr < 0) ? 0 : (nr > 16) ? 16 : nr;
    return _cvtu32_mask16((1u << nr) - 1);
}

static inline void pack_panelA(const float *A, float *blockA_packed, int mr, int kc, int K) {
    for (int p = 0; p < kc; ++p) {
        for (int i = 0; i < mr; ++i) {
            *blockA_packed++ = A[i * K + p];
        }
        for (int i = mr; i < MR; ++i) {
            *blockA_packed++ = 0.0f;
        }
    }
}

static inline void pack_blockA(const float *A, float *blockA_packed, int mc, int kc, int K) {
    OMP_PRAGMA_PARALLEL
    for (int i = 0; i < mc; i += MR) {
        int mr = min(MR, mc - i);
        pack_panelA(&A[i * K], &blockA_packed[i * kc], mr, kc, K);
    }
}

static inline void pack_panelB(const int8_t *B, int8_t *blockB_packed, int nr, int kc, int K) {
    for (int p = 0; p < kc; ++p) {
        for (int j = 0; j < nr; ++j) {
            *blockB_packed++ = B[j * K + p];
        }
        // Padding
        for (int j = nr; j < NR; ++j) {
            *blockB_packed++ = 0;
        }
    }
}

static inline void pack_blockB(const int8_t *B, int8_t *blockB_packed, int nc, int kc, int K) {
    OMP_PRAGMA_PARALLEL
    for (int j = 0; j < nc; j += NR) {
        int nr = min(NR, nc - j);
        pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
    }
}

static inline void load_accum(float *C, __m512 C_accum[MR][2], int N, int mr) {
    for (int i = 0; i < mr; ++i) {
        C_accum[i][0] = _mm512_loadu_ps(&C[i * N]);
        C_accum[i][1] = _mm512_loadu_ps(&C[i * N + 16]);
    }
}
static inline void maskload_accum(float *C, __m512 C_accum[MR][2], int N, int mr, __mmask16 packed_mask_0, __mmask16 packed_mask_1) {
    for (int i = 0; i < mr; ++i) {
        C_accum[i][0] = _mm512_maskz_loadu_ps(packed_mask_0, &C[i * N]);
        C_accum[i][1] = _mm512_maskz_loadu_ps(packed_mask_1, &C[i * N + 16]);
    }
}
static inline void store_accum(float *C, __m512 C_accum[MR][2], int N, int mr) {
    for (int i = 0; i < mr; ++i) {
        _mm512_storeu_ps(&C[i * N], C_accum[i][0]);
        _mm512_storeu_ps(&C[i * N + 16], C_accum[i][1]);
    }
}
static inline void maskstore_accum(float *C, __m512 C_accum[MR][2], int N, int mr, __mmask16 packed_mask_0, __mmask16 packed_mask_1) {
    for (int i = 0; i < mr; ++i) {
        _mm512_mask_storeu_ps(&C[i * N], packed_mask_0, C_accum[i][0]);
        _mm512_mask_storeu_ps(&C[i * N + 16], packed_mask_1, C_accum[i][1]);
    }
}

/**
 * Inner fused-multiply-add (FMA) loop over a kc slice.
 *
 * Purpose:
 * - Consumes packed panels of A (float32) and B (int8) for one kc step,
 *   dequantizes B using per-column scales, and accumulates into `C_accum`.
 * - Processes 32 output columns per step (NR = 32) as two 16-lane vectors.
 *
 * Parameters:
 * - blockA_packed: Pointer to packed A panel, laid out as kc consecutive rows,
 *   each of length MR (broadcast per row).
 * - blockB_packed: Pointer to packed B block, laid out as kc consecutive rows,
 *   each of length NR (int8), split into two halves (0..15, 16..31).
 * - C_accum: Accumulators for MR rows and two 16-wide column halves.
 * - a_packedFloat16: Scratch register used to broadcast one A value across 16 lanes.
 * - b0_packedFloat16/b1_packedFloat16: Scratch registers holding dequantized B halves.
 * - kc: Length of the current K-slice.
 * - scale0/scale1: Per-column dequantization scales for columns 0..15 and 16..31.
 */
static inline void fma_loop(float *blockA_packed, int8_t *blockB_packed,
                            __m512 C_accum[MR][2], __m512 *a_packedFloat16,
                            __m512 *b0_packedFloat16, __m512 *b1_packedFloat16, 
                            int kc, __m512 scale0, __m512 scale1) {
    // Iterate over the K-dimension chunk (kc) and accumulate into C_accum.
    for (int p = 0; p < kc; ++p) {
        // Load 32 int8 elements of B for this p (NR = 32) into a 256-bit vector.
        __m256i b_int8_vec = _mm256_loadu_si256((const __m256i*)blockB_packed);

        // Split the 32 int8 values into two 16-lane halves (columns 0..15 and 16..31).
        __m128i b0_int8 = _mm256_castsi256_si128(b_int8_vec);           // columns 0..15
        __m128i b1_int8 = _mm256_extracti128_si256(b_int8_vec, 1);      // columns 16..31
        
        // Sign-extend int8 to int32 per lane for AVX-512 float conversion.
        __m512i b0_int32 = _mm512_cvtepi8_epi32(b0_int8);
        __m512i b1_int32 = _mm512_cvtepi8_epi32(b1_int8);

        // Convert to float32 and apply per-column quantization scales.
        *b0_packedFloat16 = _mm512_cvtepi32_ps(b0_int32);
        *b1_packedFloat16 = _mm512_cvtepi32_ps(b1_int32);

        *b0_packedFloat16 = _mm512_mul_ps(*b0_packedFloat16, scale0);   // scale for cols 0..15
        *b1_packedFloat16 = _mm512_mul_ps(*b1_packedFloat16, scale1);   // scale for cols 16..31

#define UNROLL_FMA(i)                                                                    \
    *a_packedFloat16 = _mm512_set1_ps(blockA_packed[i]);                                 \
    C_accum[i][0] = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, C_accum[i][0]); \
    C_accum[i][1] = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, C_accum[i][1]);

        // Unrolled per-row FMA: broadcast A(i,p), multiply with B(p, j) halves and accumulate.
        UNROLL_FMA(0)
        UNROLL_FMA(1)
        UNROLL_FMA(2)
        UNROLL_FMA(3)
        UNROLL_FMA(4)
        UNROLL_FMA(5)
        UNROLL_FMA(6)
        UNROLL_FMA(7)
        UNROLL_FMA(8)
        UNROLL_FMA(9)
        UNROLL_FMA(10)
        UNROLL_FMA(11)
        UNROLL_FMA(12)
        UNROLL_FMA(13)

#undef UNROLL_FMA

        // Advance packed pointers to the next p (kc-step):
        // A advances by MR elements; B advances by NR elements.
        blockA_packed += MR;
        blockB_packed += NR;
    }
}

/**
 * Micro-kernel to update a C submatrix of size (mr x nr).
 *
 * Purpose:
 * - Accumulates contributions over a kc slice using packed panels of A and B.
 * - Handles mask load/store when `nr < NR` to avoid writing out of bounds.
 *
 * Parameters:
 * - blockA_packed: Packed A panel (kc x MR) in row-major per kc step.
 * - blockB_packed: Packed B block (kc x NR) with int8 entries.
 * - C: Pointer to the top-left of the output submatrix to update.
 * - mr/nr: Actual sizes of the micro-tile along M and N (<= MR/NR).
 * - kc: Current K-slice length.
 * - N: Leading dimension (row-major stride) of C.
 * - current_scales: Pointer to per-output-column float scales, starting at the
 *   column corresponding to this micro-tile; first 16 columns use `scale0`, next 16 use `scale1`.
 */
static inline void micro_kernel(float *blockA_packed, int8_t *blockB_packed,
                                float *C, int mr, int nr, int kc, int N, const float* current_scales) {
    __m512 C_accum[MR][2];
    __m512 a_packedFloat16 = {};
    __m512 b0_packedFloat16 = {};
    __m512 b1_packedFloat16 = {};
    __mmask16 packed_mask_0 = {};
    __mmask16 packed_mask_1 = {};

    __m512 scale0 = _mm512_loadu_ps(current_scales);      // Columns 0..15
    __m512 scale1 = _mm512_loadu_ps(current_scales + 16); // Columns 16..31

    if (likely(nr == NR)) {
        load_accum(C, C_accum, N, mr);
        fma_loop(blockA_packed, blockB_packed, C_accum, &a_packedFloat16,
                 &b0_packedFloat16, &b1_packedFloat16, kc, scale0, scale1);
        store_accum(C, C_accum, N, mr);
    } else {
        packed_mask_0 = create_mask(nr);
        packed_mask_1 = create_mask(nr - 16);
        maskload_accum(C, C_accum, N, mr, packed_mask_0, packed_mask_1);
        fma_loop(blockA_packed, blockB_packed, C_accum, &a_packedFloat16,
                 &b0_packedFloat16, &b1_packedFloat16, kc, scale0, scale1);
        maskstore_accum(C, C_accum, N, mr, packed_mask_0, packed_mask_1);
    }
}

/**
 * Quantized matrix multiplication (GEMM) with per-column dequantization.
 *
 * Purpose:
 * - Computes C[M x N] = A[M x K] * Bt[N x K], where Bt is an int8 matrix
 *   provided in row-major (N x K) and dequantized per output column using `scales`.
 * - Uses cache-friendly packing, AVX-512 vectorization, and OpenMP parallel tiling.
 *
 * Data layout and semantics:
 * - A: float32, row-major M x K.
 * - B: int8, row-major N x K (i.e., transposed compared to conventional K x N).
 * - C: float32, row-major M x N; updated (accumulated) by this routine.
 * - scales: float32 length-N array; one dequantization scale per output column.
 *
 * Parameters:
 * - A: Pointer to A[M*K] in row-major.
 * - B: Pointer to B[N*K] in row-major (int8) representing Bt.
 * - C: Pointer to C[M*N] in row-major (output updated in-place).
 * - M: Number of rows in A and C.
 * - N: Number of columns in B and C.
 * - K: Shared inner dimension.
 * - scales: Per-output-column dequantization scales, length N.
 */
void matmul_quantized(const float *A, const int8_t *B, float *C, 
                      int M, int N, int K, const float *scales) {
    
    for (int j = 0; j < N; j += NC) {
        int nc = min(NC, N - j);
        
        for (int p = 0; p < K; p += KC) {
            int kc = min(KC, K - p);
            
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);

            for (int i = 0; i < M; i += MC) {
                int mc = min(MC, M - i);
                pack_blockA(&A[i * K + p], blockA_packed, mc, kc, K);
                
                OMP_PRAGMA_PARALLEL
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = min(NR, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = min(MR, mc - ir);
                        
                        micro_kernel(&blockA_packed[kc * ir], 
                                     &blockB_packed[kc * jr],
                                     &C[(i + ir) * N + (j + jr)], 
                                     mr, nr, kc, N, 
                                     &scales[j + jr]);
                    }
                }
            }
        }
    }
}