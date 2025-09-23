#include "matmul.hpp"
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

// #define MC 840
// #define NC 1024
// #define KC 384

#define NTHREADS 24
#define MC MR *NTHREADS * 5
#define NC NR *NTHREADS * 30
#define KC 512

#define OMP_PRAGMA_PARALLEL _Pragma("omp parallel for num_threads(NTHREADS)")

static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));

static inline __mmask16 create_mask(int nr) {
    nr = (nr < 0) ? 0 : (nr > 16) ? 16
                                  : nr;
    return _cvtu32_mask16((1u << nr) - 1);
}

static inline void pack_panelA(const float *A, float *blockA_packed, int mr, int kc, int K) {
    for (int p = 0; p < kc; ++p) {
        for (int i = 0; i < mr; ++i) {
            *blockA_packed++ = A[i * K + p];
        }
        for (int i = mr; i < MR; ++i) {
            *blockA_packed++ = 0;
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

static inline void pack_panelB(const float *B, float *blockB_packed, int nr, int kc, int K) {

    for (int p = 0; p < kc; ++p) {
        for (int j = 0; j < nr; ++j) {
            *blockB_packed++ = B[j * K + p];
        }
        for (int j = nr; j < NR; ++j) {
            *blockB_packed++ = 0;
        }
    }
}
static inline void pack_blockB(const float *B, float *blockB_packed, int nc, int kc, int K) {
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

static inline void fma_loop(float *blockA_packed, float *blockB_packed,
                            __m512 C_accum[MR][2], __m512 *a_packedFloat16,
                            __m512 *b0_packedFloat16, __m512 *b1_packedFloat16, int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

#define UNROLL_FMA(i)                                                                    \
    *a_packedFloat16 = _mm512_set1_ps(blockA_packed[i]);                                 \
    C_accum[i][0] = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, C_accum[i][0]); \
    C_accum[i][1] = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, C_accum[i][1]);

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

        blockA_packed += MR;
        blockB_packed += NR;
    }
}
static inline void micro_kernel(float *blockA_packed, float *blockB_packed,
                                float *C, int mr, int nr, int kc, int N) {
    __m512 C_accum[MR][2];
    __m512 a_packedFloat16 = {};
    __m512 b0_packedFloat16 = {};
    __m512 b1_packedFloat16 = {};
    __mmask16 packed_mask_0 = {};
    __mmask16 packed_mask_1 = {};

    if (likely(nr == NR)) {
        load_accum(C, C_accum, N, mr);
        fma_loop(blockA_packed, blockB_packed, C_accum, &a_packedFloat16,
                 &b0_packedFloat16, &b1_packedFloat16, kc);
        store_accum(C, C_accum, N, mr);
    } else {
        packed_mask_0 = create_mask(nr);
        packed_mask_1 = create_mask(nr - 16);
        maskload_accum(C, C_accum, N, mr, packed_mask_0, packed_mask_1);
        fma_loop(blockA_packed, blockB_packed, C_accum, &a_packedFloat16,
                 &b0_packedFloat16, &b1_packedFloat16, kc);
        maskstore_accum(C, C_accum, N, mr, packed_mask_0, packed_mask_1);
    }
}

// ==================== Scratch Impl ==================== //
// C = A * B^T + C, all row-major
// C: [M, N]
// A: [M, K]
// B: [N, K]
void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
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
                        micro_kernel(&blockA_packed[kc * ir], &blockB_packed[kc * jr],
                                     &C[(i + ir) * N + (j + jr)], mr, nr, kc, N);
                    }
                }
            }
        }
    }
}
