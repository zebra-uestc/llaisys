#include "../../../device/nvidia/nvidia_common.cuh"


// Helper to convert float to bfloat16
__device__ __forceinline__ cuda_bfloat16 __float2bfloat16_opt(float f) {
    return __float2bfloat16(f);
}

__global__ void linear_w8a16_kernel(
    cuda_bfloat16 *__restrict__ C,       // [M, N]
    const cuda_bfloat16 *__restrict__ A, // [M, K]
    const int8_t *__restrict__ B,        // [N, K] - INT8
    const cuda_bfloat16 *__restrict__ bias,
    const cuda_bfloat16 *__restrict__ scale,   // [N] - Per-channel scale
    const size_t M, const size_t N, const size_t K) {

    // Block tile sizes
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;
    const int APAD = 8;
    const int BPAD_BF16 = 8;
    const int BPAD_INT8 = 16;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;

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

    // === Address Calculation ===
    // Thread mapping for A (BF16)
    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_a_gmem_m = by * BM + load_a_smem_m;
    
    // Thread mapping for B (Int8 Staging)
    // We need to load BN(256) * BK(32) bytes using 256 threads.
    // Each thread loads 16 bytes (int4) per step.
    // Total steps needed: (256 * 32) / (256 * 16) = 2 steps.
    int load_b_int8_n = tid / 2;     // 0..127
    int load_b_int8_k = (tid % 2) * 16; // 0 or 16

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    // === Helper Lambda: Expand Int8 (SMEM) -> BF16 (SMEM) ===
    auto expand_int8_to_bf16 = [&](int buffer_idx) {
        // Each thread handles one row of B tile (BK=32 elements)
        // Since blockDim.x = 256 and BN = 256, mapping is 1-to-1 for rows.
        int row = tid; 
        
        if (row < BN) {
            // Load Scale for this row (Per-Channel)
            int global_n_idx = bx * BN + row;
            float row_scale_f = (global_n_idx < N) ? __bfloat162float(scale[global_n_idx]) : 0.0f;
            
            // Pointers
            int8_t* src_ptr = s_b_int8 + buffer_idx * s_b_int8_db_offset + row * (BK + BPAD_INT8);
            cuda_bfloat16* dst_ptr = s_b + buffer_idx * s_b_db_offset + row * (BK + BPAD_BF16);
            
            // Vectorized Conversion: 32 elements = 2 x int4 (16 bytes)
            // We use int4 to load 16 bytes at once
            int4* src_vec = reinterpret_cast<int4*>(src_ptr);
            // We write 2 * float4 (where each float4 is interpreted as 8 bf16s = 16 bytes)
            float4* dst_vec = reinterpret_cast<float4*>(dst_ptr);

            #pragma unroll
            for(int k=0; k<2; k++) {
                int4 loaded_bytes = src_vec[k];
                int8_t* bytes = reinterpret_cast<int8_t*>(&loaded_bytes);
                
                // Temp buffer for 16 bf16s
                cuda_bfloat16 buffer[16];
                
                #pragma unroll
                for(int x=0; x<16; x++) {
                    buffer[x] = __float2bfloat16((float)bytes[x] * row_scale_f);
                }
                
                // Write back as vector (16 bytes = 1 float4)
                dst_vec[k*2 + 0] = reinterpret_cast<float4*>(buffer)[0];
                dst_vec[k*2 + 1] = reinterpret_cast<float4*>(buffer)[1];
            }
        }
    };

    // === Prologue: Load First Tile ===
    {
        // 1. Load A (BF16)
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = load_a_smem_k; // k=0
            int smem_m = load_a_smem_m + i;
            
            const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
            uint32_t smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD)]);
            
            int src_bytes = (gmem_m < M && gmem_k < K) ? min(16, (int)(K - gmem_k) * 2) : 0;
            
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" 
                :: "r"(smem_addr), "l"(src_ptr), "r"(src_bytes));
        }

        // 2. Load B (Int8) into Staging Buffer
        #pragma unroll
        for(int k_iter=0; k_iter<2; k_iter++) {
             int my_n = load_b_int8_n + k_iter * 128; // stride 128
             int my_k = load_b_int8_k;
             
             int gmem_n = bx * BN + my_n;
             int gmem_k = my_k; // k=0
             
             const int8_t *src_ptr = &B[OFFSET(gmem_n, gmem_k, K)];
             uint32_t smem_addr = __cvta_generic_to_shared(&s_b_int8[my_n * (BK + BPAD_INT8) + my_k]);
             
             int src_bytes = (gmem_n < N && gmem_k < K) ? min(16, (int)(K - gmem_k)) : 0;
             
             asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" 
                :: "r"(smem_addr), "l"(src_ptr), "r"(src_bytes));
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        
        // 3. Dequantize First Tile (Int8 -> BF16)
        expand_int8_to_bf16(0);
        __syncthreads();
    }

    int num_k_tiles = div_ceil(K, BK);

    // === Main Loop ===
    for (int bk = 1; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
        int curr_idx = (bk - 1) & 1;
        int next_idx = bk & 1;

        // 1. Issue Async Load for NEXT Tile
        
        // Load Next A
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = k_start + load_a_smem_k;
            int smem_m = load_a_smem_m + i;
            
            const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
            uint32_t smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD) + next_idx * s_a_db_offset]);
            
            int src_bytes = (gmem_m < M && gmem_k < K) ? min(16, (int)(K - gmem_k) * 2) : 0;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_addr), "l"(src_ptr), "r"(src_bytes));
        }

        // Load Next B (Int8)
        #pragma unroll
        for(int k_iter=0; k_iter<2; k_iter++) {
             int my_n = load_b_int8_n + k_iter * 128; 
             int my_k = load_b_int8_k;
             int gmem_n = bx * BN + my_n;
             int gmem_k = k_start + my_k;
             
             const int8_t *src_ptr = &B[OFFSET(gmem_n, gmem_k, K)];
             // Write to Staging Buffer [Next]
             uint32_t smem_addr = __cvta_generic_to_shared(&s_b_int8[my_n * (BK + BPAD_INT8) + my_k + next_idx * s_b_int8_db_offset]);
             
             int src_bytes = (gmem_n < N && gmem_k < K) ? min(16, (int)(K - gmem_k)) : 0;
             asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_addr), "l"(src_ptr), "r"(src_bytes));
        }

        asm("cp.async.commit_group;\n" ::);

        // 2. Compute CURRENT Tile
        // Use s_b (BF16) which is already populated
        
        // Load A Fragments
        wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 64, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 64, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

        // Load B Fragments
        wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 64, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 64, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);

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

    // Load Last A
    wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 64, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 64, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

    // Load Last B
    wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 64, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 0, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 64, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 16, BK + BPAD_BF16) + curr_idx * s_b_db_offset], BK + BPAD_BF16);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    // ==================== Store Result ====================
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;

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