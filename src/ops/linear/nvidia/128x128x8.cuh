#pragma once

#include "helper_cuda_ptx.cuh"
#include <cstdint>

// Matrix multiplication with bias: C = A * B^T + bias
// All matrices in row-major format:
// C: [M, N] - output matrix
// A: [M, K] - left input matrix
// B: [N, K] - right input matrix (transposed during computation)
// bias: [1, N] - bias vector broadcasted across rows
//
// Tile configuration: 128x128x8 (M x N x K per block)
// Thread block: 256 threads (8 warps)
// Each warp computes a 32x64 output tile
// Each thread computes an 8x8 output sub-tile
__global__
__launch_bounds__(256, 2) void linear_128x128x8_kernel(
    float *C,
    const float *A,
    const float *B,
    const float *bias,
    const size_t M,
    const size_t N,
    const size_t K) {
    // ===== Shared Memory Configuration =====
    // Using double buffering to hide memory latency
    // Buffer layout: [A_buffer0, A_buffer1, B_buffer0, B_buffer1]
    constexpr int smem_a_padding = 256;
    constexpr int smem_a_size = smem_a_padding * 8; // 8 rows per K-block
    constexpr int smem_a_ld = 132;                  // Leading dimension with padding to avoid bank conflicts

    constexpr int smem_b_padding = 256;
    constexpr int smem_b_size = smem_b_padding * 8;
    constexpr int smem_b_ld = 132;

    __shared__ float __align__(2 * smem_a_size * sizeof(float))
        smem_ptr[2 * (smem_a_size + smem_b_size)];

    // ===== Register Allocation =====
    float accumulator[8][8]{}; // Per-thread output tile (8x8)
    float ldg_a_buffer[4];     // Global->shared transfer buffer for A
    float ldg_b_buffer[4];     // Global->shared transfer buffer for B
    float frag_a[2][8];        // Double-buffered A fragments for computation
    float frag_b[2][8];        // Double-buffered B fragments for computation

    // Bitmasks to track valid global memory accesses (boundary checking)
    unsigned ldg_a_bitmask = 0x0;
    unsigned ldg_b_bitmask = 0x0;

    // Shared memory pointers for A and B tiles
    float *smem_a_ptr = smem_ptr;
    float *smem_b_ptr = smem_ptr + 2 * smem_a_size;

    // Thread mapping
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // ===== Global Memory Load Configuration for A =====
    // Each thread loads 4 consecutive floats (vectorized as float4)
    // 256 threads organized as 32 rows x 8 columns to cover 128x8 tile
    const int ldg_a_start_x = threadIdx.x % 8;                          // K dimension offset
    const int ldg_a_start_y = blockIdx.y * 128 + 4 * (threadIdx.x / 8); // M dimension offset
    const int ldg_a_start = ldg_a_start_x + ldg_a_start_y * K;
    const float *ldg_a_ptr = A + ldg_a_start;

    // Precompute offsets for the 4 elements each thread loads
    int ldg_a_offsets_y[4];
    int ldg_a_offsets[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_a_offsets_y[i] = i;
        ldg_a_offsets[i] = i * K;
    }

// Compute bitmask for valid M indices (boundary checking)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = ldg_a_start_y + ldg_a_offsets_y[i];
        if (m_idx < M) {
            ldg_a_bitmask ^= (0x1 << i);
        }
    }

    // ===== Global Memory Load Configuration for B =====
    const int ldg_b_start_x = threadIdx.x % 8;
    const int ldg_b_start_y = blockIdx.x * 128 + 4 * (threadIdx.x / 8);
    const int ldg_b_start = ldg_b_start_x + ldg_b_start_y * K;
    const float *ldg_b_ptr = B + ldg_b_start;

    int ldg_b_offsets_y[4];
    int ldg_b_offsets[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        ldg_b_offsets_y[i] = i;
        ldg_b_offsets[i] = i * K;
    }

// Compute bitmask for valid N indices
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int n_idx = ldg_b_start_y + ldg_b_offsets_y[i];
        if (n_idx < N) {
            ldg_b_bitmask ^= (0x1 << i);
        }
    }

    // ===== Shared Memory Store Configuration =====
    // Transpose layout: threads store data in column-major pattern
    const int sts_a_start_x = 4 * (threadIdx.x / 8);
    const int sts_a_start_y = threadIdx.x % 8;
    const int sts_a_start = sts_a_start_x + sts_a_start_y * smem_a_ld;
    float *sts_a_ptr = smem_a_ptr + sts_a_start;

    const int sts_b_start_x = 4 * (threadIdx.x / 8);
    const int sts_b_start_y = threadIdx.x % 8;
    const int sts_b_start = sts_b_start_x + sts_b_start_y * smem_b_ld;
    float *sts_b_ptr = smem_b_ptr + sts_b_start;

    uint64_t sts_a_addr;
    uint64_t sts_b_addr;
    CVTA_TO_SHARED_PTX(sts_a_addr, sts_a_ptr);
    CVTA_TO_SHARED_PTX(sts_b_addr, sts_b_ptr);

    // ===== K-dimension Loop Configuration =====
    // Handle non-multiple-of-8 K dimensions
    // If K % 8 == 0: n_blocks_k = K/8 - 1, first_block_k_size = 8
    // If K % 8 != 0: n_blocks_k = K/8, first_block_k_size = K % 8
    const int n_blocks_k = (K + 7) / 8 - 1;
    const int first_block_k_size = K - 8 * n_blocks_k;

// ===== Initial Load: First K-block to Shared Memory =====
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        bool guard_k = ldg_a_start_x < first_block_k_size;
        bool guard_m = ldg_a_bitmask & (0x1 << i);
        bool guard = guard_k && guard_m;
        LDG32_GUARD_MOV0_PTX(ldg_a_buffer[i], ldg_a_ptr + ldg_a_offsets[i], (unsigned)guard);
    }
    STS128_PTX(ldg_a_buffer[0], ldg_a_buffer[1], ldg_a_buffer[2], ldg_a_buffer[3], sts_a_addr);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        bool guard_k = ldg_b_start_x < first_block_k_size;
        bool guard_n = ldg_b_bitmask & (0x1 << i);
        bool guard = guard_k && guard_n;
        LDG32_GUARD_MOV0_PTX(ldg_b_buffer[i], ldg_b_ptr + ldg_b_offsets[i], (unsigned)guard);
    }
    STS128_PTX(ldg_b_buffer[0], ldg_b_buffer[1], ldg_b_buffer[2], ldg_b_buffer[3], sts_b_addr);

    __syncthreads();

    // ===== Shared Memory Load Configuration =====
    // Each warp processes a 64x64 output tile
    // Each thread in warp processes 8x8 output sub-tile
    uint64_t lds_a_addr;
    uint64_t lds_b_addr;

    // Swizzled lane mapping to improve memory coalescing
    const int lane_id_mapped_x = 2 * (lane_id / 8) + (lane_id % 2);
    const int lane_id_mapped_y = (lane_id / 2) % 4;
    const int warp_id_mapped_x = 64 * (warp_id % 2);
    const int warp_id_mapped_y = 32 * (warp_id / 2);

    const int lds_a_start = 4 * lane_id_mapped_y + warp_id_mapped_y;
    const int lds_b_start = 4 * lane_id_mapped_x + warp_id_mapped_x;
    float *lds_a_ptr = smem_a_ptr + lds_a_start;
    float *lds_b_ptr = smem_b_ptr + lds_b_start;

    CVTA_TO_SHARED_PTX(lds_a_addr, lds_a_ptr);
    CVTA_TO_SHARED_PTX(lds_b_addr, lds_b_ptr);

    // Load first fragments from shared memory for computation
    LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3], lds_a_addr);
    LDS128_PTX(frag_a[0][4], frag_a[0][5], frag_a[0][6], frag_a[0][7],
               lds_a_addr + 16 * sizeof(float));
    LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3], lds_b_addr);
    LDS128_PTX(frag_b[0][4], frag_b[0][5], frag_b[0][6], frag_b[0][7],
               lds_b_addr + 32 * sizeof(float));

    // Advance pointers to next K-block
    ldg_a_ptr += first_block_k_size;
    ldg_b_ptr += first_block_k_size;

    // Switch to second shared memory buffer (double buffering)
    sts_a_addr ^= 8192;
    sts_b_addr ^= 8192;

    // ===== Main K-dimension Loop =====
    for (int block_k = 0; block_k < n_blocks_k; block_k++) {

// Prefetch next K-block from global memory
#pragma unroll
        for (int i = 0; i < 4; i++) {
            bool guard_m = (ldg_a_bitmask & (0x1 << i));
            LDG32_GUARD_PTX(ldg_a_buffer[i], ldg_a_ptr + ldg_a_offsets[i], (unsigned)guard_m);

            bool guard_n = (ldg_b_bitmask & (0x1 << i));
            LDG32_GUARD_PTX(ldg_b_buffer[i], ldg_b_ptr + ldg_b_offsets[i], (unsigned)guard_n);
        }

// Inner loop: compute 8 outer products for current K-block
#pragma unroll
        for (int warp_k = 0; warp_k < 8; warp_k += 1) {
            const int prefetch = (warp_k + 1) % 8;
            const int frag_idx = warp_k & 1;
            const int frag_next_idx = (warp_k + 1) & 1;

            // Prefetch next fragments from shared memory (double buffering)
            LDS128_PTX(frag_a[frag_next_idx][0], frag_a[frag_next_idx][1],
                       frag_a[frag_next_idx][2], frag_a[frag_next_idx][3],
                       lds_a_addr + prefetch * smem_a_ld * sizeof(float));
            LDS128_PTX(frag_a[frag_next_idx][4], frag_a[frag_next_idx][5],
                       frag_a[frag_next_idx][6], frag_a[frag_next_idx][7],
                       lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
            LDS128_PTX(frag_b[frag_next_idx][0], frag_b[frag_next_idx][1],
                       frag_b[frag_next_idx][2], frag_b[frag_next_idx][3],
                       lds_b_addr + prefetch * smem_b_ld * sizeof(float));
            LDS128_PTX(frag_b[frag_next_idx][4], frag_b[frag_next_idx][5],
                       frag_b[frag_next_idx][6], frag_b[frag_next_idx][7],
                       lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));

// Compute outer product and accumulate (8x8 sub-tile)
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
                }
            }
        }

        // Store prefetched global data to shared memory
        // Note: shared memory is still being read during accumulation above
        STS128_PTX(ldg_a_buffer[0], ldg_a_buffer[1], ldg_a_buffer[2], ldg_a_buffer[3], sts_a_addr);
        STS128_PTX(ldg_b_buffer[0], ldg_b_buffer[1], ldg_b_buffer[2], ldg_b_buffer[3], sts_b_addr);

        __syncthreads();

        // Swap double buffers
        sts_a_addr ^= 8192;
        sts_b_addr ^= 8192;
        lds_a_addr ^= 8192;
        lds_b_addr ^= 8192;

        // Advance to next K-block
        ldg_a_ptr += 8;
        ldg_b_ptr += 8;

        // Load first fragments from new shared memory buffer
        LDS128_PTX(frag_a[0][0], frag_a[0][1], frag_a[0][2], frag_a[0][3], lds_a_addr);
        LDS128_PTX(frag_a[0][4], frag_a[0][5], frag_a[0][6], frag_a[0][7],
                   lds_a_addr + 16 * sizeof(float));
        LDS128_PTX(frag_b[0][0], frag_b[0][1], frag_b[0][2], frag_b[0][3], lds_b_addr);
        LDS128_PTX(frag_b[0][4], frag_b[0][5], frag_b[0][6], frag_b[0][7],
                   lds_b_addr + 32 * sizeof(float));
    }

// ===== Process Last K-block =====
#pragma unroll
    for (int warp_k = 0; warp_k < 8; warp_k += 1) {
        const int prefetch = (warp_k + 1) % 8;
        const int frag_idx = warp_k & 1;
        const int frag_next_idx = (warp_k + 1) & 1;

        LDS128_PTX(frag_a[frag_next_idx][0], frag_a[frag_next_idx][1],
                   frag_a[frag_next_idx][2], frag_a[frag_next_idx][3],
                   lds_a_addr + prefetch * smem_a_ld * sizeof(float));
        LDS128_PTX(frag_a[frag_next_idx][4], frag_a[frag_next_idx][5],
                   frag_a[frag_next_idx][6], frag_a[frag_next_idx][7],
                   lds_a_addr + (prefetch * smem_a_ld + 16) * sizeof(float));
        LDS128_PTX(frag_b[frag_next_idx][0], frag_b[frag_next_idx][1],
                   frag_b[frag_next_idx][2], frag_b[frag_next_idx][3],
                   lds_b_addr + prefetch * smem_b_ld * sizeof(float));
        LDS128_PTX(frag_b[frag_next_idx][4], frag_b[frag_next_idx][5],
                   frag_b[frag_next_idx][6], frag_b[frag_next_idx][7],
                   lds_b_addr + (prefetch * smem_b_ld + 32) * sizeof(float));

#pragma unroll
        for (int i = 0; i < 8; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                accumulator[i][j] += frag_a[frag_idx][i] * frag_b[frag_idx][j];
            }
        }
    }

    // ===== Write Results to Global Memory =====
    // Reuse shared memory for output staging (previous data no longer needed)
    uint64_t sts_c_addr;
    const int sts_c_offset = 512 * warp_id + 4 * 32 * lane_id_mapped_y + 4 * lane_id_mapped_x;
    CVTA_TO_SHARED_PTX(sts_c_addr, smem_ptr + sts_c_offset);

    float *lds_c_ptr = smem_ptr + 512 * warp_id + lane_id;

    const int m_idx = blockIdx.y * 128 + warp_id_mapped_y;
    const int n_idx = blockIdx.x * 128 + warp_id_mapped_x + lane_id;
    float *stg_c_ptr = C + m_idx * N + n_idx;

    // Write output tile warp by warp with bias addition
    if (m_idx < M) {
// Process 32x64 warp tile as 4 16x32 sub-tiles
#pragma unroll 1
        for (int i = 0; i < 2; ++i) {
#pragma unroll 1
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

// Stage 4 rows (16 elements total) to shared memory
#pragma unroll 2
                for (int p = 0; p < 4; ++p) {
                    STS128_PTX(accumulator[i * 4 + p][j * 4],
                               accumulator[i * 4 + p][j * 4 + 1],
                               accumulator[i * 4 + p][j * 4 + 2],
                               accumulator[i * 4 + p][j * 4 + 3],
                               sts_c_addr + p * 8 * sizeof(float4));
                }
                __syncthreads();

// Write 16 rows to global memory with bias and boundary checking
#pragma unroll 4
                for (int p = 0; p < 16; ++p) {
                    const int m_edge = M - (m_idx + i * 16);
                    const int n_pos = n_idx + j * 32;
                    const float c = n_pos < N ? bias[n_pos] : 0.0f;
                    const bool guard = p < m_edge && n_pos < N;
                    STG32_GUARD_PTX(lds_c_ptr[p * 32] + c,
                                    stg_c_ptr + (i * 16 + p) * N + j * 32,
                                    (unsigned)guard);
                }
            }
        }
    }
}