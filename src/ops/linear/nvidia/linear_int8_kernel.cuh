#include "../../../device/nvidia/nvidia_common.cuh"

template<int BM, int BN, int BK, int APAD, int BPAD_BF16, int BPAD_INT8>
__global__ void linear_w8a16_kernel(
    cuda_bfloat16 *__restrict__ C,       // [M, N]
    const cuda_bfloat16 *__restrict__ A, // [M, K]
    const int8_t *__restrict__ B,        // [N, K] - INT8
    const cuda_bfloat16 *__restrict__ bias,
    const cuda_bfloat16 *__restrict__ scale,   // [N] - Per-channel scale
    const size_t M, const size_t N, const size_t K) {

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;
    int num_threads = blockDim.x;

    // === Shared Memory Layout ===
    // 1. Raw Shared Memory Pointer
    extern __shared__ int8_t raw_smem[];
    
    // 2. BF16 Operands (For WMMA) - Double Buffered
    // Size: 2 * (BM + BN) * (BK + PAD) * sizeof(bf16)
    cuda_bfloat16 *s_a = reinterpret_cast<cuda_bfloat16*>(raw_smem);
    cuda_bfloat16 *s_b = s_a + 2 * BM * (BK + APAD); 
    
    // 3. Int8 Staging Buffer (For cp.async prefetch) - Double Buffered
    // Size: 2 * BN * (BK + PAD) * sizeof(int8)
    // Placed right after s_b
    int8_t *s_b_int8 = reinterpret_cast<int8_t*>(s_b + 2 * BN * (BK + BPAD_BF16));

    // Offsets for Double Buffering
    const size_t s_a_db_offset = BM * (BK + APAD);
    const size_t s_b_db_offset = BN * (BK + BPAD_BF16);      
    const size_t s_b_int8_db_offset = BN * (BK + BPAD_INT8);

    // WMMA Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, cuda_bfloat16, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, cuda_bfloat16, wmma::col_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

    // Initialize Accumulators
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    int warps_n = BN / 64;
    int comp_c_frag_m = wid / warps_n; // Warp M-index
    int comp_c_frag_n = wid % warps_n; // Warp N-index

    // Lambda: Robust Load A (Stride Loop)
    auto load_a_gmem_to_smem = [&](int k_start, int buffer_idx) {
        int smem_base = buffer_idx * s_a_db_offset;
        // Total elements to load: BM * BK
        // Each thread loads 8 elements (128 bits) -> use int4
        // Total threads needed: (BM * BK * 2) / 16 = (BM * BK) / 8
        // Using loop to cover
        
        // Use vectorized load int4 (8 x bf16)
        int total_vec_loads = (BM * BK) / 8;
        
        #pragma unroll 1 // No unroll for variable loop
        for (int idx = tid; idx < total_vec_loads; idx += num_threads) {
            int row = idx / (BK / 8); 
            int col_vec = idx % (BK / 8);
            int col = col_vec * 8;
            
            int gmem_m = by * BM + row;
            int gmem_k = k_start + col;
            
            uint32_t smem_addr = __cvta_generic_to_shared(&s_a[smem_base + OFFSET(row, col, BK + APAD)]);
            
            const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
            // Check Bounds (Vectorized)
            // Assuming K is multiple of 8 (BK=32)
            bool valid = (gmem_m < M) && (gmem_k < K);
            // cp.async handles predicate 0 byte copy safely? No, need to pass 0 bytes.
            int copy_bytes = valid ? 16 : 0;
            
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" 
                :: "r"(smem_addr), "l"(src_ptr), "r"(copy_bytes));
        }
    };
 
    // Lambda: Robust Load B (Int8) (Stride Loop)
    auto load_b_gmem_to_smem = [&](int k_start, int buffer_idx) {
        int smem_base = buffer_idx * s_b_int8_db_offset;
        // Total Int8 elements: BN * BK
        // Vector load 16 bytes (int4)
        int total_vec_loads = (BN * BK) / 16;
        
        #pragma unroll 1
        for (int idx = tid; idx < total_vec_loads; idx += num_threads) {
            int row = idx / (BK / 16); // Row in B (0..BN-1)
            int col_vec = idx % (BK / 16);
            int col = col_vec * 16;
            
            int gmem_n = bx * BN + row;
            int gmem_k = k_start + col;
            
            uint32_t smem_addr = __cvta_generic_to_shared(&s_b_int8[smem_base + row * (BK + BPAD_INT8) + col]);
            const int8_t *src_ptr = &B[OFFSET(gmem_n, gmem_k, K)];
            
            bool valid = (gmem_n < N) && (gmem_k < K);
            int copy_bytes = valid ? 16 : 0;
            
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" 
                :: "r"(smem_addr), "l"(src_ptr), "r"(copy_bytes));
        }
    };
 
    // Lambda: Robust Expand
    auto expand_int8_to_bf16 = [&](int buffer_idx) {
        // Each thread converts a chunk. 
        // Total rows to convert = BN.
        // Each thread processes 1 row? 
        // If threads < BN, we need stride.
        
        for (int row = tid; row < BN; row += num_threads) {
            int global_n_idx = bx * BN + row;
            float row_scale_f = (global_n_idx < N) ? __bfloat162float(scale[global_n_idx]) : 0.0f;
            
            int8_t* src_ptr = s_b_int8 + buffer_idx * s_b_int8_db_offset + row * (BK + BPAD_INT8);
            cuda_bfloat16* dst_ptr = s_b + buffer_idx * s_b_db_offset + row * (BK + BPAD_BF16);
            
            // 32 elements per row (BK=32). Load 2 x int4 (16 bytes each)
            int4* src_vec = reinterpret_cast<int4*>(src_ptr);
            float4* dst_vec = reinterpret_cast<float4*>(dst_ptr);
 
            #pragma unroll
            for(int k=0; k<2; k++) {
                int4 loaded_bytes = src_vec[k];
                int8_t* bytes = reinterpret_cast<int8_t*>(&loaded_bytes);
                cuda_bfloat16 buffer[16];
                #pragma unroll
                for(int x=0; x<16; x++) {
                    buffer[x] = __float2bfloat16((float)bytes[x] * row_scale_f);
                }
                dst_vec[k*2 + 0] = reinterpret_cast<float4*>(buffer)[0];
                dst_vec[k*2 + 1] = reinterpret_cast<float4*>(buffer)[1];
            }
        }
    };

    // === Prologue: Load First Tile ===
    {
        load_a_gmem_to_smem(0, 0);
        load_b_gmem_to_smem(0, 0);
        
        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        
        expand_int8_to_bf16(0);
        __syncthreads();
    }

    int num_k_tiles = div_ceil(K, BK);

    // === Main Loop ===
    for (int bk = 1; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
        int curr_idx = (bk - 1) & 1;
        int next_idx = bk & 1;

        // Async Load Next
        load_a_gmem_to_smem(k_start, next_idx);
        load_b_gmem_to_smem(k_start, next_idx);
        asm("cp.async.commit_group;\n" ::);

        // Compute Current
        // Offsets must use comp_c_frag_m/n which are calculated robustly
        int base_m = comp_c_frag_m * 64;
        int base_n = comp_c_frag_n * 64;
        
        wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(base_m, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(base_m + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(base_m + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(base_m + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(base_m, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(base_m + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(base_m + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(base_m + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
 
        wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(base_n, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(base_n + 16, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[OFFSET(base_n + 32, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[OFFSET(base_n + 48, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(base_n, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(base_n + 16, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[OFFSET(base_n + 32, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[OFFSET(base_n + 48, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);


        // WMMA Compute
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        // 3. Wait for Next Tile
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        // 4. Dequantize Next Tile (Staging -> BF16 Buffer)
        expand_int8_to_bf16(next_idx);
        __syncthreads();
    }

    // === Process Last Tile (Compute Only) ===
    int curr_idx = (num_k_tiles - 1) & 1;

    int base_m = comp_c_frag_m * 64;
    int base_n = comp_c_frag_n * 64;

    // Load Last A
    wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(base_m, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(base_m + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(base_m + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(base_m + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(base_m, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(base_m + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(base_m + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(base_m + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

    // Load Last B
    wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(base_n, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(base_n + 16, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[OFFSET(base_n + 32, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[OFFSET(base_n + 48, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(base_n, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(base_n + 16, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[OFFSET(base_n + 32, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[OFFSET(base_n + 48, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    // ==================== Store Result ====================
    int store_c_gmem_m = by * BM + base_m;
    int store_c_gmem_n = bx * BN + base_n;

    // Reuse s_b memory area for float accumulators (save Shared Memory)
    // NOTE: s_b (BF16) size is enough to hold float accumulators.
    float *s_c_float = reinterpret_cast<float *>(s_b); 
    // Reuse s_a memory for final BF16 output tile (s_a is no longer needed)
    cuda_bfloat16 *s_c_bf16 = s_a;

    // Preload Bias
    float bias_vals[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int global_n = store_c_gmem_n + j * 16 + (lane_id & 15);
        bias_vals[j] = (global_n < N && bias) ? __bfloat162float(bias[global_n]) : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            // Store accumulator to Shared Memory (float)
            // Use thread-specific offset to avoid overwriting if reusing buffer aggressively
            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c[i][j], 16, wmma::mem_row_major);
            __syncwarp();

            // Add Bias & Convert to BF16
            #pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_m = idx >> 4;
                int local_n = idx & 15;
                
                float val = s_c_float[wid * 256 + local_m * 16 + local_n] + bias_vals[j];
                s_c_bf16[wid * 256 + local_m * 16 + local_n] = __float2bfloat16(val);
            }
            __syncwarp();

            // Write to Global Memory
            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;

            int global_m = tile_m + row;
            int global_n = tile_n + col;

            if (global_m < M && global_n + 7 < N) {
                // Vectorized store: 128-bit (8 x bf16)
                int4 v = *reinterpret_cast<int4*>(&s_c_bf16[wid * 256 + row * 16 + col]);
                reinterpret_cast<int4*>(&C[OFFSET(global_m, global_n, N)])[0] = v;
            } else if (global_m < M) {
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (global_n + c < N) {
                        C[OFFSET(global_m, global_n + c, N)] = s_c_bf16[wid * 256 + row * 16 + col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}