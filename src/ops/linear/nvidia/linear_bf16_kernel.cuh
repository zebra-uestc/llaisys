#pragma once

#include "../../../device/nvidia/nvidia_common.cuh"
#include <mma.h>

using namespace nvcuda;

/**
 * HGEMM Kernel with:
 * 1. Non-aligned M, N, K support (boundary handling)
 * 2. Transposed B matrix: A[M,K] * B^T = C[M,N], where B is stored as [N,K] row-major
 * 3. Bias addition: C[m,n] += bias[n] for all m
 *
 * Mathematical operation: C[m,n] = sum_k(A[m,k] * B[n,k]) + bias[n]
 *
 * Memory layout:
 *   A: [M, K] row-major
 *   B: [N, K] row-major (transposed storage)
 *   C: [M, N] row-major
 *   bias: [N]
 */
__global__ void linear_bf16_kernel_128x256(
    cuda_bfloat16 *__restrict__ C,          // [M, N] row-major
    const cuda_bfloat16 *__restrict__ A,    // [M, K] row-major
    const cuda_bfloat16 *__restrict__ B,    // [N, K] row-major (transposed)
    const cuda_bfloat16 *__restrict__ bias, // [N]
    const size_t M, const size_t N, const size_t K) {

    // Block tile sizes
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    // Padding to avoid bank conflicts
    const int APAD = 8;
    const int BPAD = 8;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;

    // Shared memory
    // s_a stores A tile: [BM][BK] as A[m][k]
    // s_b stores B tile: [BN][BK] as B[n][k]
    // __shared__ cuda_bfloat16 s_a[BM][BK + APAD];
    // __shared__ cuda_bfloat16 s_b[BN][BK + BPAD];
    extern __shared__ cuda_bfloat16 smem_bf16[];
    cuda_bfloat16 *s_a = smem_bf16;
    cuda_bfloat16 *s_b = s_a + 2 * BM * (BK + APAD);
    size_t s_a_db_offset = BM * (BK + APAD);
    size_t s_b_db_offset = BN * (BK + APAD);

    // WMMA fragments - both row_major since we transpose B during load
    wmma::fragment<wmma::matrix_a, 16, 16, 16, cuda_bfloat16, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, cuda_bfloat16, wmma::col_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    // Loading indices for A: same as original
    // 256 threads load BM*BK = 128*32 = 4096 elements
    // Each thread loads 2 rows * 8 cols = 16 elements
    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;

    // Loading indices for B: same as original
    // 256 threads load BN*BK = 256*32 = 8192 elements
    // Each thread loads 4 rows * 8 cols = 32 elements
    int load_b_smem_n = (tid >> 2) << 2;
    int load_b_smem_k = (tid & 3) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    {
        // ==================== Load A tile ====================
        // A[m][k] -> s_a[m][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = load_a_smem_k;
            int smem_m = load_a_smem_m + i;

            const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            // uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[smem_m][load_a_smem_k]);
            uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD)]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_m < M ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_a_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = src_ptr[j];
                    } else {
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int gmem_n = load_b_gmem_n + i;
            int gmem_k = load_b_smem_k;
            int smem_n = load_b_smem_n + i;

            const cuda_bfloat16 *src_ptr = &B[OFFSET(gmem_n, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD)]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_n < N ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_b_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = src_ptr[j];
                    } else {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int num_k_tiles = div_ceil(K, BK);

    for (int bk = 1; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
        int curr_idx = (bk - 1) & 1;
        int next_idx = bk & 1;
        // ==================== Load A tile ====================
        // A[m][k] -> s_a[m][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = k_start + load_a_smem_k;
            int smem_m = load_a_smem_m + i;

            const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            // uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[smem_m][load_a_smem_k]);
            uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD) + next_idx * s_a_db_offset]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_m < M ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_a_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = src_ptr[j];
                    } else {
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int gmem_n = load_b_gmem_n + i;
            int gmem_k = k_start + load_b_smem_k;
            int smem_n = load_b_smem_n + i;

            const cuda_bfloat16 *src_ptr = &B[OFFSET(gmem_n, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD) + next_idx * s_b_db_offset]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_n < N ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_b_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = src_ptr[j];
                    } else {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // ==================== Load fragments and compute ====================
        // Load A fragments: s_a[m][k] with row_major

        wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 64, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 64, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

        // Load B fragments: s_b[n][k] with col_major

        wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 64, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 64, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

// Compute: C += A * B
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int curr_idx = (num_k_tiles - 1) & 1;

    // ==================== Load fragments and compute ====================
    // Load A fragments: s_a[m][k] with row_major

    wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 64, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 64, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

    // Load B fragments: s_b[n][k] with col_major

    wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 64, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 64, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

// Compute: C += A * B
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    // ==================== Store results with bias ====================
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;

    // Temp buffer for output
    // __shared__ float s_c_float[8][16][16]; // 8 warps, 16x16 each
    // __shared__ cuda_bfloat16 s_c_cuda_bfloat16[8][16][16];   // 8 warps, 16x16 each
    float *s_c_float = reinterpret_cast<float *>(s_b + 2 * BN * (BK + BPAD));
    cuda_bfloat16 *s_c_cuda_bfloat16 = reinterpret_cast<cuda_bfloat16 *>(s_c_float + 8 * 16 * 16);

    // Each warp handles a 64x64 output block, divided into 4x4 grid of 16x16 tiles.
    // Within each 16x16 tile, 32 threads (one warp) write 256 elements:
    //   - Each thread writes 8 elements in a single column (rows 0,2,4,6,8,10,12,14)
    //   - The column index is fixed per thread: local_n = lane_id & 15
    //
    // For the entire 64x64 block, each thread writes:
    //   - 4 (i-tiles) × 4 (j-tiles) × 8 (rows per tile) = 128 elements
    //   - Spanning 32 rows (8 rows × 4 i-tiles) × 4 columns (one per j-tile)
    //
    // Since each thread always accesses the same relative column (lane_id & 15),
    // we only need to preload 4 bias values (one for each j-tile).
    float bias_vals[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int global_n = store_c_gmem_n + j * 16 + (lane_id & 15);
        bias_vals[j] = (global_n < N) ? __bfloat162float(bias[global_n]) : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            // Store fragment to shared memory
            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c[i][j], 16, wmma::mem_row_major);
            __syncwarp();

            // Add bias
#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_m = idx >> 4;
                int local_n = idx & 15;

                s_c_cuda_bfloat16[wid * 256 + local_m * 16 + local_n] = __float2bfloat16(s_c_float[wid * 256 + local_m * 16 + local_n] + bias_vals[j]);
            }
            __syncwarp();

            // Write to global memory using FLOAT4 (128-bit = 8 cuda_bfloat16s)
            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;

            int global_m = tile_m + row;
            int global_n = tile_n + col;

            if (global_m < M && global_n + 7 < N) {
                // Vectorized 128-bit store (8 cuda_bfloat16s at once)
                FLOAT4(C[OFFSET(global_m, global_n, N)]) = FLOAT4(s_c_cuda_bfloat16[wid * 256 + row * 16 + col]);
            } else if (global_m < M) {
                // Boundary case: element-wise store
#pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (global_n + c < N) {
                        C[OFFSET(global_m, global_n + c, N)] = s_c_cuda_bfloat16[wid * 256 + row * 16 + col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}

/**
 * HGEMM Kernel with:
 * 1. Non-aligned M, N, K support (boundary handling)
 * 2. Transposed B matrix: A[M,K] * B^T = C[M,N], where B is stored as [N,K] row-major
 * 3. Bias addition: C[m,n] += bias[n] for all m
 *
 * Mathematical operation: C[m,n] = sum_k(A[m,k] * B[n,k]) + bias[n]
 *
 * Memory layout:
 *   A: [M, K] row-major
 *   B: [N, K] row-major (transposed storage)
 *   C: [M, N] row-major
 *   bias: [N]
 */
__global__ void linear_bf16_kernel_128x128(
    cuda_bfloat16 *__restrict__ C,          // [M, N] row-major
    const cuda_bfloat16 *__restrict__ A,    // [M, K] row-major
    const cuda_bfloat16 *__restrict__ B,    // [N, K] row-major (transposed)
    const cuda_bfloat16 *__restrict__ bias, // [N]
    const size_t M, const size_t N, const size_t K) {

    // Block tile sizes
    const int BM = 128;
    const int BN = 128;
    const int BK = 32;

    // Padding to avoid bank conflicts
    const int APAD = 8;
    const int BPAD = 8;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;

    // Shared memory
    // s_a stores A tile: [BM][BK] as A[m][k]
    // s_b stores B tile: [BN][BK] as B[n][k]
    // __shared__ cuda_bfloat16 s_a[BM][BK + APAD];
    // __shared__ cuda_bfloat16 s_b[BN][BK + BPAD];
    extern __shared__ cuda_bfloat16 smem_bf16[];
    cuda_bfloat16 *s_a = smem_bf16;
    cuda_bfloat16 *s_b = s_a + 2 * BM * (BK + APAD);
    size_t s_a_db_offset = BM * (BK + APAD);
    size_t s_b_db_offset = BN * (BK + APAD);

    // WMMA fragments - both row_major since we transpose B during load
    wmma::fragment<wmma::matrix_a, 16, 16, 16, cuda_bfloat16, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, cuda_bfloat16, wmma::col_major> frag_b[2][2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][2];

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    // Loading indices for A: same as original
    // 256 threads load BM*BK = 128*32 = 4096 elements
    // Each thread loads 1 rows * 8 cols = 16 elements
    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;

    // Loading indices for B: same as original
    // 256 threads load BN*BK = 128*32 = 4096 elements
    // Each thread loads 2 rows * 8 cols = 16 elements
    int load_b_smem_n = (tid >> 2) << 1;
    int load_b_smem_k = (tid & 3) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    {
// ==================== Load A tile ====================
// A[m][k] -> s_a[m][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = load_a_smem_k;
            int smem_m = load_a_smem_m + i;

            const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD)]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_m < M ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_a_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = src_ptr[j];
                    } else {
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_n = load_b_gmem_n + i;
            int gmem_k = load_b_smem_k;
            int smem_n = load_b_smem_n + i;

            const cuda_bfloat16 *src_ptr = &B[OFFSET(gmem_n, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD)]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_n < N ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_b_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = src_ptr[j];
                    } else {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int num_k_tiles = div_ceil(K, BK);

    for (int bk = 1; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
        int curr_idx = (bk - 1) & 1;
        int next_idx = bk & 1;
// ==================== Load A tile ====================
// A[m][k] -> s_a[m][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = k_start + load_a_smem_k;
            int smem_m = load_a_smem_m + i;

            const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD) + next_idx * s_a_db_offset]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_m < M ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_a_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = src_ptr[j];
                    } else {
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_n = load_b_gmem_n + i;
            int gmem_k = k_start + load_b_smem_k;
            int smem_n = load_b_smem_n + i;

            const cuda_bfloat16 *src_ptr = &B[OFFSET(gmem_n, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD) + next_idx * s_b_db_offset]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_n < N ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_b_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = src_ptr[j];
                    } else {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // ==================== Load fragments and compute ====================
        // Load A fragments: s_a[m][k] with row_major
        wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 64, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 64, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

        // Load B fragments: s_b[n][k] with col_major
        wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 32, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 32 + 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 32, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 32 + 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

// Compute: C += A * B
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int curr_idx = (num_k_tiles - 1) & 1;

    // ==================== Load fragments and compute ====================
    // Load A fragments: s_a[m][k] with row_major
    wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 64, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 64, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

    // Load B fragments: s_b[n][k] with col_major

    wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 32, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 32 + 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 32, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 32 + 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

// Compute: C += A * B
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    // ==================== Store results with bias ====================
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 32;

    // Temp buffer for output
    // __shared__ float s_c_float[8][16][16]; // 8 warps, 16x16 each
    // __shared__ cuda_bfloat16 s_c_cuda_bfloat16[8][16][16];   // 8 warps, 16x16 each
    float *s_c_float = reinterpret_cast<float *>(s_b + 2 * BN * (BK + BPAD));
    cuda_bfloat16 *s_c_cuda_bfloat16 = reinterpret_cast<cuda_bfloat16 *>(s_c_float + 8 * 16 * 16);

    // Each warp handles a 32x32 output block, divided into 2x2 grid of 16x16 tiles.
    // Within each 16x16 tile, 32 threads (one warp) write 256 elements:
    //   - Each thread writes 8 elements in a single column (rows 0,2,4,6,8,10,12,14)
    //   - The column index is fixed per thread: local_n = lane_id & 15
    //
    // For the entire 32x32 block, each thread writes:
    //   - 2 (i-tiles) × 2 (j-tiles) × 8 (rows per tile) = 32 elements
    //   - Spanning 16 rows (8 rows × 2 i-tiles) × 2 columns (one per j-tile)
    //
    // Since each thread always accesses the same relative column (lane_id & 15),
    // we only need to preload 2 bias values (one for each j-tile).
    float bias_vals[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        int global_n = store_c_gmem_n + j * 16 + (lane_id & 15);
        bias_vals[j] = (global_n < N) ? __bfloat162float(bias[global_n]) : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            // Store fragment to shared memory
            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c[i][j], 16, wmma::mem_row_major);
            __syncwarp();

            // Add bias
#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_m = idx >> 4;
                int local_n = idx & 15;

                s_c_cuda_bfloat16[wid * 256 + local_m * 16 + local_n] = __float2bfloat16(s_c_float[wid * 256 + local_m * 16 + local_n] + bias_vals[j]);
            }
            __syncwarp();

            // Write to global memory using FLOAT4 (128-bit = 8 cuda_bfloat16s)
            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;

            int global_m = tile_m + row;
            int global_n = tile_n + col;

            if (global_m < M && global_n + 7 < N) {
                // Vectorized 128-bit store (8 cuda_bfloat16s at once)
                FLOAT4(C[OFFSET(global_m, global_n, N)]) = FLOAT4(s_c_cuda_bfloat16[wid * 256 + row * 16 + col]);
            } else if (global_m < M) {
                // Boundary case: element-wise store
#pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (global_n + c < N) {
                        C[OFFSET(global_m, global_n + c, N)] = s_c_cuda_bfloat16[wid * 256 + row * 16 + col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}

/**
 * HGEMM Kernel with:
 * 1. Non-aligned M, N, K support (boundary handling)
 * 2. Transposed B matrix: A[M,K] * B^T = C[M,N], where B is stored as [N,K] row-major
 * 3. Bias addition: C[m,n] += bias[n] for all m
 *
 * Mathematical operation: C[m,n] = sum_k(A[m,k] * B[n,k]) + bias[n]
 *
 * Memory layout:
 *   A: [M, K] row-major
 *   B: [N, K] row-major (transposed storage)
 *   C: [M, N] row-major
 *   bias: [N]
 */
__global__ void linear_bf16_kernel_64x128(
    cuda_bfloat16 *__restrict__ C,          // [M, N] row-major
    const cuda_bfloat16 *__restrict__ A,    // [M, K] row-major
    const cuda_bfloat16 *__restrict__ B,    // [N, K] row-major (transposed)
    const cuda_bfloat16 *__restrict__ bias, // [N]
    const size_t M, const size_t N, const size_t K) {

    // Block tile sizes
    const int BM = 64;
    const int BN = 128;
    const int BK = 32;

    // Padding to avoid bank conflicts
    const int APAD = 8;
    const int BPAD = 8;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;

    // Shared memory
    // s_a stores A tile: [BM][BK] as A[m][k]
    // s_b stores B tile: [BN][BK] as B[n][k]
    // __shared__ cuda_bfloat16 s_a[BM][BK + APAD];
    // __shared__ cuda_bfloat16 s_b[BN][BK + BPAD];
    extern __shared__ cuda_bfloat16 smem_bf16[];
    cuda_bfloat16 *s_a = smem_bf16;
    cuda_bfloat16 *s_b = s_a + 2 * BM * (BK + APAD);
    size_t s_a_db_offset = BM * (BK + APAD);
    size_t s_b_db_offset = BN * (BK + APAD);

    // WMMA fragments - both row_major since we transpose B during load
    wmma::fragment<wmma::matrix_a, 16, 16, 16, cuda_bfloat16, wmma::row_major> frag_a[2][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, cuda_bfloat16, wmma::col_major> frag_b[2][2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[2][2];

#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    // Loading indices for A: same as original
    // 256 threads load BM*BK = 64*32 = 2048 elements
    // Each thread loads 1 rows * 8 cols = 8 elements
    int load_a_smem_m = tid >> 2;
    int load_a_smem_k = (tid & 3) << 3;

    // Loading indices for B: same as original
    // 256 threads load BN*BK = 128*32 = 4096 elements
    // Each thread loads 2 rows * 8 cols = 16 elements
    int load_b_smem_n = (tid >> 2) << 1;
    int load_b_smem_k = (tid & 3) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    {
        // ==================== Load A tile ====================
        // A[m][k] -> s_a[m][k], straightforward

        int gmem_m = load_a_gmem_m;
        int gmem_k = load_a_smem_k;
        int smem_m = load_a_smem_m;

        const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
        bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

        uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD)]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(16, valid_bytes));
            src_size = gmem_m < M ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :
                : "r"(load_a_smem_addr),
                  "l"(src_ptr),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (gmem_k + j < K) {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = src_ptr[j];
                } else {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_n = load_b_gmem_n + i;
            int gmem_k = load_b_smem_k;
            int smem_n = load_b_smem_n + i;

            const cuda_bfloat16 *src_ptr = &B[OFFSET(gmem_n, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD)]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_n < N ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_b_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = src_ptr[j];
                    } else {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int num_k_tiles = div_ceil(K, BK);

    for (int bk = 1; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
        int curr_idx = (bk - 1) & 1;
        int next_idx = bk & 1;
        // ==================== Load A tile ====================
        // A[m][k] -> s_a[m][k], straightforward
        int gmem_m = load_a_gmem_m;
        int gmem_k = k_start + load_a_smem_k;
        int smem_m = load_a_smem_m;

        const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
        bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

        // uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[smem_m][load_a_smem_k]);
        uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD) + next_idx * s_a_db_offset]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(16, valid_bytes));
            src_size = gmem_m < M ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :
                : "r"(load_a_smem_addr),
                  "l"(src_ptr),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (gmem_k + j < K) {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = src_ptr[j];
                } else {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_n = load_b_gmem_n + i;
            int gmem_k = k_start + load_b_smem_k;
            int smem_n = load_b_smem_n + i;

            const cuda_bfloat16 *src_ptr = &B[OFFSET(gmem_n, gmem_k, K)];
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD) + next_idx * s_b_db_offset]);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_n < N ? src_size : 0;

                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                    :
                    : "r"(load_b_smem_addr),
                      "l"(src_ptr),
                      "r"(src_size));
            } else {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K) {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = src_ptr[j];
                    } else {
                        s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // ==================== Load fragments and compute ====================
        // Load A fragments: s_a[m][k] with row_major

        wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 32 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 32 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

        // Load B fragments: s_b[n][k] with col_major

        wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 32, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 32 + 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 32, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 32 + 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

// Compute: C += A * B
#pragma unroll
        for (int i = 0; i < 2; i++) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int curr_idx = (num_k_tiles - 1) & 1;

    // ==================== Load fragments and compute ====================
    // Load A fragments: s_a[m][k] with row_major

    wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 32 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 32 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

    // Load B fragments: s_b[n][k] with col_major

    wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 32, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 32 + 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 32, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 32 + 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

// Compute: C += A * B
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    // ==================== Store results with bias ====================
    int store_c_gmem_m = by * BM + comp_c_frag_m * 32;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 32;

    // Temp buffer for output
    // __shared__ float s_c_float[8][16][16]; // 8 warps, 16x16 each
    // __shared__ cuda_bfloat16 s_c_cuda_bfloat16[8][16][16];   // 8 warps, 16x16 each
    float *s_c_float = reinterpret_cast<float *>(s_b + 2 * BN * (BK + BPAD));
    cuda_bfloat16 *s_c_cuda_bfloat16 = reinterpret_cast<cuda_bfloat16 *>(s_c_float + 8 * 16 * 16);

    // Each warp handles a 32x32 output block, divided into 2x2 grid of 16x16 tiles.
    // Within each 16x16 tile, 32 threads (one warp) write 256 elements:
    //   - Each thread writes 8 elements in a single column (rows 0,2,4,6,8,10,12,14)
    //   - The column index is fixed per thread: local_n = lane_id & 15
    //
    // For the entire 32x32 block, each thread writes:
    //   - 2 (i-tiles) × 2 (j-tiles) × 8 (rows per tile) = 32 elements
    //   - Spanning 16 rows (8 rows × 2 i-tiles) × 2 columns (one per j-tile)
    //
    // Since each thread always accesses the same relative column (lane_id & 15),
    // we only need to preload 2 bias values (one for each j-tile).
    float bias_vals[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        int global_n = store_c_gmem_n + j * 16 + (lane_id & 15);
        bias_vals[j] = (global_n < N) ? __bfloat162float(bias[global_n]) : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            // Store fragment to shared memory
            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c[i][j], 16, wmma::mem_row_major);
            __syncwarp();

            // Add bias
#pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_m = idx >> 4;
                int local_n = idx & 15;

                s_c_cuda_bfloat16[wid * 256 + local_m * 16 + local_n] = __float2bfloat16(s_c_float[wid * 256 + local_m * 16 + local_n] + bias_vals[j]);
            }
            __syncwarp();

            // Write to global memory using FLOAT4 (128-bit = 8 cuda_bfloat16s)
            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;

            int global_m = tile_m + row;
            int global_n = tile_n + col;

            if (global_m < M && global_n + 7 < N) {
                // Vectorized 128-bit store (8 cuda_bfloat16s at once)
                FLOAT4(C[OFFSET(global_m, global_n, N)]) = FLOAT4(s_c_cuda_bfloat16[wid * 256 + row * 16 + col]);
            } else if (global_m < M) {
                // Boundary case: element-wise store
#pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (global_n + c < N) {
                        C[OFFSET(global_m, global_n + c, N)] = s_c_cuda_bfloat16[wid * 256 + row * 16 + col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}

/**
 * HGEMM Kernel with:
 * 1. Non-aligned M, N, K support (boundary handling)
 * 2. Transposed B matrix: A[M,K] * B^T = C[M,N], where B is stored as [N,K] row-major
 * 3. Bias addition: C[m,n] += bias[n] for all m
 *
 * Mathematical operation: C[m,n] = sum_k(A[m,k] * B[n,k]) + bias[n]
 *
 * Memory layout:
 *   A: [M, K] row-major
 *   B: [N, K] row-major (transposed storage)
 *   C: [M, N] row-major
 *   bias: [N]
 */
__global__ void linear_bf16_kernel_64x64(
    cuda_bfloat16 *__restrict__ C,          // [M, N] row-major
    const cuda_bfloat16 *__restrict__ A,    // [M, K] row-major
    const cuda_bfloat16 *__restrict__ B,    // [N, K] row-major (transposed)
    const cuda_bfloat16 *__restrict__ bias, // [N]
    const size_t M, const size_t N, const size_t K) {

    // Block tile sizes
    const int BM = 64;
    const int BN = 64;
    const int BK = 32;

    // Padding to avoid bank conflicts
    const int APAD = 8;
    const int BPAD = 8;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;

    // Shared memory
    // s_a stores A tile: [BM][BK] as A[m][k]
    // s_b stores B tile: [BN][BK] as B[n][k]
    // __shared__ cuda_bfloat16 s_a[BM][BK + APAD];
    // __shared__ cuda_bfloat16 s_b[BN][BK + BPAD];
    extern __shared__ cuda_bfloat16 smem_bf16[];
    cuda_bfloat16 *s_a = smem_bf16;
    cuda_bfloat16 *s_b = s_a + 2 * BM * (BK + APAD);
    size_t s_a_db_offset = BM * (BK + APAD);
    size_t s_b_db_offset = BN * (BK + APAD);

    // WMMA fragments - both row_major since we transpose B during load
    wmma::fragment<wmma::matrix_a, 16, 16, 16, cuda_bfloat16, wmma::row_major> frag_a[2][2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, cuda_bfloat16, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[2];

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        wmma::fill_fragment(frag_c[i], 0.0f);
    }

    // Loading indices for A: same as original
    // 256 threads load BM*BK = 64*32 = 2048 elements
    // Each thread loads 1 rows * 8 cols = 8 elements
    int load_a_smem_m = tid >> 2;
    int load_a_smem_k = (tid & 3) << 3;

    // Loading indices for B: same as original
    // 256 threads load BN*BK = 64*32 = 2048 elements
    // Each thread loads 1 rows * 8 cols = 8 elements
    int load_b_smem_n = tid >> 2;
    int load_b_smem_k = (tid & 3) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    {
        // ==================== Load A tile ====================
        // A[m][k] -> s_a[m][k], straightforward
        int gmem_m = load_a_gmem_m;
        int gmem_k = load_a_smem_k;
        int smem_m = load_a_smem_m;

        const cuda_bfloat16 *src_ptr_a = &A[OFFSET(gmem_m, gmem_k, K)];
        bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr_a) % 16 == 0);

        uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD)]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(16, valid_bytes));
            src_size = gmem_m < M ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :
                : "r"(load_a_smem_addr),
                  "l"(src_ptr_a),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (gmem_k + j < K) {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = src_ptr_a[j];
                } else {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
        int gmem_n = load_b_gmem_n;
        gmem_k = load_b_smem_k;
        int smem_n = load_b_smem_n;

        const cuda_bfloat16 *src_ptr_b = &B[OFFSET(gmem_n, gmem_k, K)];
        is_aligned = (reinterpret_cast<uint64_t>(src_ptr_b) % 16 == 0);

        uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD)]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(16, valid_bytes));
            src_size = gmem_n < N ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :
                : "r"(load_b_smem_addr),
                  "l"(src_ptr_b),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (gmem_k + j < K) {
                    s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = src_ptr_b[j];
                } else {
                    s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = __float2bfloat16(0.0f);
                }
            }
        }
    }

    asm("cp.async.commit_group;\n" ::);
    asm("cp.async.wait_group 0;\n" ::);

    __syncthreads();

    int num_k_tiles = div_ceil(K, BK);

    for (int bk = 1; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
        int curr_idx = (bk - 1) & 1;
        int next_idx = bk & 1;
        // ==================== Load A tile ====================
        // A[m][k] -> s_a[m][k], straightforward
        int gmem_m = load_a_gmem_m;
        int gmem_k = k_start + load_a_smem_k;
        int smem_m = load_a_smem_m;

        const cuda_bfloat16 *src_ptr_a = &A[OFFSET(gmem_m, gmem_k, K)];
        bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr_a) % 16 == 0);

        uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD) + next_idx * s_a_db_offset]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(16, valid_bytes));
            src_size = gmem_m < M ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :
                : "r"(load_a_smem_addr),
                  "l"(src_ptr_a),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (gmem_k + j < K) {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = src_ptr_a[j];
                } else {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
        int gmem_n = load_b_gmem_n;
        gmem_k = k_start + load_b_smem_k;
        int smem_n = load_b_smem_n;

        const cuda_bfloat16 *src_ptr_b = &B[OFFSET(gmem_n, gmem_k, K)];
        is_aligned = (reinterpret_cast<uint64_t>(src_ptr_b) % 16 == 0);

        uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD) + next_idx * s_b_db_offset]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(16, valid_bytes));
            src_size = gmem_n < N ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :
                : "r"(load_b_smem_addr),
                  "l"(src_ptr_b),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (gmem_k + j < K) {
                    s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = src_ptr_b[j];
                } else {
                    s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load fragments and compute ====================
        // Load A fragments: s_a[m][k] with row_major
        wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 16 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 16 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

        // Load B fragments: s_b[n][k] with col_major
        wmma::load_matrix_sync(frag_b[0], &s_b[OFFSET(comp_c_frag_n * 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1], &s_b[OFFSET(comp_c_frag_n * 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

// Compute: C += A * B
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            wmma::mma_sync(frag_c[i], frag_a[0][i], frag_b[0], frag_c[i]);
            wmma::mma_sync(frag_c[i], frag_a[1][i], frag_b[1], frag_c[i]);
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int curr_idx = (num_k_tiles - 1) & 1;

    // ==================== Load fragments and compute ====================
    // Load A fragments: s_a[m][k] with row_major
    wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 16 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 16 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

    // Load B fragments: s_b[n][k] with col_major
    wmma::load_matrix_sync(frag_b[0], &s_b[OFFSET(comp_c_frag_n * 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1], &s_b[OFFSET(comp_c_frag_n * 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

// Compute: C += A * B
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        wmma::mma_sync(frag_c[i], frag_a[0][i], frag_b[0], frag_c[i]);
        wmma::mma_sync(frag_c[i], frag_a[1][i], frag_b[1], frag_c[i]);
    }

    // ==================== Store results with bias ====================
    int store_c_gmem_m = by * BM + comp_c_frag_m * 32;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 16;

    // Temp buffer for output
    // __shared__ float s_c_float[8][16][16]; // 8 warps, 16x16 each
    // __shared__ cuda_bfloat16 s_c_cuda_bfloat16[8][16][16];   // 8 warps, 16x16 each
    float *s_c_float = reinterpret_cast<float *>(s_b + 2 * BN * (BK + BPAD));
    cuda_bfloat16 *s_c_cuda_bfloat16 = reinterpret_cast<cuda_bfloat16 *>(s_c_float + 8 * 16 * 16);

    // Each warp handles a 16x16 output block, divided into 1x1 grid of 16x16 tiles.
    // Within each 16x16 tile, 32 threads (one warp) write 256 elements:
    //   - Each thread writes 8 elements in a single column (rows 0,2,4,6,8,10,12,14)
    //   - The column index is fixed per thread: local_n = lane_id & 15

    // Since each thread always accesses the same relative column (lane_id & 15),
    // we only need to preload 1 bias values (one for each j-tile).
    float bias_val;
    int bias_global_n = store_c_gmem_n + (lane_id & 15);
    bias_val = (bias_global_n < N) ? __bfloat162float(bias[bias_global_n]) : 0.0f;

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int tile_m = store_c_gmem_m + i * 16;
        int tile_n = store_c_gmem_n;

        // Store fragment to shared memory
        wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c[i], 16, wmma::mem_row_major);
        __syncwarp();

        // Add bias
#pragma unroll
        for (int idx = lane_id; idx < 256; idx += 32) {
            int local_m = idx >> 4;
            int local_n = idx & 15;

            s_c_cuda_bfloat16[wid * 256 + local_m * 16 + local_n] = __float2bfloat16(s_c_float[wid * 256 + local_m * 16 + local_n] + bias_val);
        }
        __syncwarp();

        // Write to global memory using FLOAT4 (128-bit = 8 cuda_bfloat16s)
        int row = lane_id >> 1;
        int col = (lane_id & 1) << 3;

        int global_m = tile_m + row;
        int global_n = tile_n + col;

        if (global_m < M && global_n + 7 < N) {
            // Vectorized 128-bit store (8 cuda_bfloat16s at once)
            FLOAT4(C[OFFSET(global_m, global_n, N)]) = FLOAT4(s_c_cuda_bfloat16[wid * 256 + row * 16 + col]);
        } else if (global_m < M) {
            // Boundary case: element-wise store
#pragma unroll
            for (int c = 0; c < 8; c++) {
                if (global_n + c < N) {
                    C[OFFSET(global_m, global_n + c, N)] = s_c_cuda_bfloat16[wid * 256 + row * 16 + col + c];
                }
            }
        }
        __syncwarp();
    }
}

/**
 * HGEMM Kernel with:
 * 1. Non-aligned M, N, K support (boundary handling)
 * 2. Transposed B matrix: A[M,K] * B^T = C[M,N], where B is stored as [N,K] row-major
 * 3. Bias addition: C[m,n] += bias[n] for all m
 *
 * Mathematical operation: C[m,n] = sum_k(A[m,k] * B[n,k]) + bias[n]
 *
 * Memory layout:
 *   A: [M, K] row-major
 *   B: [N, K] row-major (transposed storage)
 *   C: [M, N] row-major
 *   bias: [N]
 */
__global__ void linear_bf16_kernel_32x64(
    cuda_bfloat16 *__restrict__ C,          // [M, N] row-major
    const cuda_bfloat16 *__restrict__ A,    // [M, K] row-major
    const cuda_bfloat16 *__restrict__ B,    // [N, K] row-major (transposed)
    const cuda_bfloat16 *__restrict__ bias, // [N]
    const size_t M, const size_t N, const size_t K) {

    // Block tile sizes
    const int BM = 32;
    const int BN = 64;
    const int BK = 32;

    // Padding to avoid bank conflicts
    const int APAD = 8;
    const int BPAD = 8;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;

    // Shared memory
    // s_a stores A tile: [BM][BK] as A[m][k]
    // s_b stores B tile: [BN][BK] as B[n][k]
    // __shared__ cuda_bfloat16 s_a[BM][BK + APAD];
    // __shared__ cuda_bfloat16 s_b[BN][BK + BPAD];
    extern __shared__ cuda_bfloat16 smem_bf16[];
    cuda_bfloat16 *s_a = smem_bf16;
    cuda_bfloat16 *s_b = s_a + 2 * BM * (BK + APAD);
    size_t s_a_db_offset = BM * (BK + APAD);
    size_t s_b_db_offset = BN * (BK + APAD);

    // WMMA fragments - both row_major since we transpose B during load
    wmma::fragment<wmma::matrix_a, 16, 16, 16, cuda_bfloat16, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, cuda_bfloat16, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    // Loading indices for A: same as original
    // 256 threads load BM*BK = 32*32 = 1024 elements
    // Each thread loads 1 rows * 4 cols = 4 elements
    int load_a_smem_m = tid >> 3;
    int load_a_smem_k = (tid & 7) << 2;

    // Loading indices for B: same as original
    // 256 threads load BN*BK = 64*32 = 2048 elements
    // Each thread loads 1 rows * 8 cols = 8 elements
    int load_b_smem_n = tid >> 2;
    int load_b_smem_k = (tid & 3) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    {
        // ==================== Load A tile ====================
        // A[m][k] -> s_a[m][k], straightforward
        int gmem_m = load_a_gmem_m;
        int gmem_k = load_a_smem_k;
        int smem_m = load_a_smem_m;

        const cuda_bfloat16 *src_ptr_a = &A[OFFSET(gmem_m, gmem_k, K)];
        bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr_a) % 8 == 0);

        uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD)]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(8, valid_bytes));
            src_size = gmem_m < M ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                :
                : "r"(load_a_smem_addr),
                  "l"(src_ptr_a),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (gmem_k + j < K) {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = src_ptr_a[j];
                } else {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
        int gmem_n = load_b_gmem_n;
        gmem_k = load_b_smem_k;
        int smem_n = load_b_smem_n;

        const cuda_bfloat16 *src_ptr_b = &B[OFFSET(gmem_n, gmem_k, K)];
        is_aligned = (reinterpret_cast<uint64_t>(src_ptr_b) % 16 == 0);

        uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD)]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(16, valid_bytes));
            src_size = gmem_n < N ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :
                : "r"(load_b_smem_addr),
                  "l"(src_ptr_b),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (gmem_k + j < K) {
                    s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = src_ptr_b[j];
                } else {
                    s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = __float2bfloat16(0.0f);
                }
            }
        }
    }

    asm("cp.async.commit_group;\n" ::);
    asm("cp.async.wait_group 0;\n" ::);

    __syncthreads();

    int num_k_tiles = div_ceil(K, BK);

    for (int bk = 1; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
        int curr_idx = (bk - 1) & 1;
        int next_idx = bk & 1;
        // ==================== Load A tile ====================
        // A[m][k] -> s_a[m][k], straightforward
        int gmem_m = load_a_gmem_m;
        int gmem_k = k_start + load_a_smem_k;
        int smem_m = load_a_smem_m;

        const cuda_bfloat16 *src_ptr_a = &A[OFFSET(gmem_m, gmem_k, K)];
        bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr_a) % 8 == 0);

        uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD) + next_idx * s_a_db_offset]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(8, valid_bytes));
            src_size = gmem_m < M ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                :
                : "r"(load_a_smem_addr),
                  "l"(src_ptr_a),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (gmem_k + j < K) {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = src_ptr_a[j];
                } else {
                    s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load B tile ====================
        // B[n][k] -> s_b[n][k], straightforward
        int gmem_n = load_b_gmem_n;
        gmem_k = k_start + load_b_smem_k;
        int smem_n = load_b_smem_n;

        const cuda_bfloat16 *src_ptr_b = &B[OFFSET(gmem_n, gmem_k, K)];
        is_aligned = (reinterpret_cast<uint64_t>(src_ptr_b) % 16 == 0);

        uint32_t load_b_smem_addr = __cvta_generic_to_shared(&s_b[OFFSET(smem_n, load_b_smem_k, BK + BPAD) + next_idx * s_b_db_offset]);

        if (is_aligned) {
            int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
            int src_size = max(0, min(16, valid_bytes));
            src_size = gmem_n < N ? src_size : 0;

            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                :
                : "r"(load_b_smem_addr),
                  "l"(src_ptr_b),
                  "r"(src_size));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (gmem_k + j < K) {
                    s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = src_ptr_b[j];
                } else {
                    s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load fragments and compute ====================
        // Load A fragments: s_a[m][k] with row_major
        wmma::load_matrix_sync(frag_a[0], &s_a[OFFSET(comp_c_frag_m * 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1], &s_a[OFFSET(comp_c_frag_m * 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

        // Load B fragments: s_b[n][k] with col_major
        wmma::load_matrix_sync(frag_b[0], &s_b[OFFSET(comp_c_frag_n * 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1], &s_b[OFFSET(comp_c_frag_n * 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

        // Compute: C += A * B
        wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
        wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int curr_idx = (num_k_tiles - 1) & 1;

    // ==================== Load fragments and compute ====================
    // Load A fragments: s_a[m][k] with row_major
    wmma::load_matrix_sync(frag_a[0], &s_a[OFFSET(comp_c_frag_m * 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1], &s_a[OFFSET(comp_c_frag_m * 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

    // Load B fragments: s_b[n][k] with col_major
    wmma::load_matrix_sync(frag_b[0], &s_b[OFFSET(comp_c_frag_n * 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1], &s_b[OFFSET(comp_c_frag_n * 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

    // Compute: C += A * B
    wmma::mma_sync(frag_c, frag_a[0], frag_b[0], frag_c);
    wmma::mma_sync(frag_c, frag_a[1], frag_b[1], frag_c);

    // ==================== Store results with bias ====================
    int store_c_gmem_m = by * BM + comp_c_frag_m * 16;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 16;

    // Temp buffer for output
    // __shared__ float s_c_float[8][16][16]; // 8 warps, 16x16 each
    // __shared__ cuda_bfloat16 s_c_cuda_bfloat16[8][16][16];   // 8 warps, 16x16 each
    float *s_c_float = reinterpret_cast<float *>(s_b + 2 * BN * (BK + BPAD));
    cuda_bfloat16 *s_c_cuda_bfloat16 = reinterpret_cast<cuda_bfloat16 *>(s_c_float + 8 * 16 * 16);

    // Each warp handles a 16x16 output block, divided into 1x1 grid of 16x16 tiles.
    // Within each 16x16 tile, 32 threads (one warp) write 256 elements:
    //   - Each thread writes 8 elements in a single column (rows 0,2,4,6,8,10,12,14)
    //   - The column index is fixed per thread: local_n = lane_id & 15

    // Since each thread always accesses the same relative column (lane_id & 15),
    // we only need to preload 1 bias values (one for each j-tile).
    float bias_val;
    int bias_global_n = store_c_gmem_n + (lane_id & 15);
    bias_val = (bias_global_n < N) ? __bfloat162float(bias[bias_global_n]) : 0.0f;

    int tile_m = store_c_gmem_m;
    int tile_n = store_c_gmem_n;

    // Store fragment to shared memory
    wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c, 16, wmma::mem_row_major);
    __syncwarp();

    // Add bias
#pragma unroll
    for (int idx = lane_id; idx < 256; idx += 32) {
        int local_m = idx >> 4;
        int local_n = idx & 15;

        s_c_cuda_bfloat16[wid * 256 + local_m * 16 + local_n] = __float2bfloat16(s_c_float[wid * 256 + local_m * 16 + local_n] + bias_val);
    }
    __syncwarp();

    // Write to global memory using FLOAT4 (128-bit = 8 cuda_bfloat16s)
    int row = lane_id >> 1;
    int col = (lane_id & 1) << 3;

    int global_m = tile_m + row;
    int global_n = tile_n + col;

    if (global_m < M && global_n + 7 < N) {
        // Vectorized 128-bit store (8 cuda_bfloat16s at once)
        FLOAT4(C[OFFSET(global_m, global_n, N)]) = FLOAT4(s_c_cuda_bfloat16[wid * 256 + row * 16 + col]);
    } else if (global_m < M) {
        // Boundary case: element-wise store
#pragma unroll
        for (int c = 0; c < 8; c++) {
            if (global_n + c < N) {
                C[OFFSET(global_m, global_n + c, N)] = s_c_cuda_bfloat16[wid * 256 + row * 16 + col + c];
            }
        }
    }
    __syncwarp();
}