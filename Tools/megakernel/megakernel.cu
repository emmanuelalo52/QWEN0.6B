#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "rmsnorm.cuh"
#include "swiglu.cuh"
#include "swiglu.cu"
namespace cg = cooperative_groups;

// Model constants 
constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;   // 2048
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM; // 1024


#ifndef LDG_NUM_BLOCKS
#define LDG_NUM_BLOCKS 28
#endif

#ifndef LDG_BLOCK_SIZE
#define LDG_BLOCK_SIZE 256
#endif

#ifndef LDG_LM_NUM_BLOCKS
#define LDG_LM_NUM_BLOCKS 1184
#endif

#ifndef LDG_LM_BLOCK_SIZE
#define LDG_LM_BLOCK_SIZE 256
#endif

#ifndef LDG_LM_ROWS_PER_WARP
#define LDG_LM_ROWS_PER_WARP 2
#endif

#ifndef LDG_ATTN_BLOCKS
#define LDG_ATTN_BLOCKS NUM_Q_HEADS
#endif

#ifndef LDG_PREFETCH_QK
#define LDG_PREFETCH_QK 1
#endif

#ifndef LDG_PREFETCH_DOWN 
#define LDG_PREFETCH_DOWN 1
#endif

#ifndef LDG_PREFETCH_THREAD_STRIDE
#define LDG_PREFETCH_THREAD_STRIDE 1
#endif

#ifndef LDG_PREFETCH_ELEM_STRIDE
#define LDG_PREFETCH_ELEM_STRIDE 1
#endif

#ifndef LDG_PREFETCH_BLOCK_STRIDE
#define LDG_PREFETCH_BLOCK_STRIDE 1
#endif

#ifndef LDG_PREFETCH_GATE
#define LDG_PREFETCH_GATE 1
#endif

#ifndef LDG_PREFETCH_UP
#define LDG_PREFETCH_UP 1
#endif

constexpr int LDG_NUM_WARPS = LDG_LM_BLOCK_SIZE/WARP_SIZE;
constexpr float LDG_RMS_EPS = 1e-6f;
// LM_head
constexpr int LDG_VOCAB_SIZE = 151936;

struct LDGLayerWeight{
    const half* input_layernorm_weight;      // [HIDDEN_SIZE]
    const half* q_proj_weight;               // [Q_SIZE, HIDDEN_SIZE]
    const half* k_proj_weight;               // [KV_SIZE, HIDDEN_SIZE]
    const half* v_proj_weight;               // [KV_SIZE, HIDDEN_SIZE]
    const half* q_norm_weight;               // [HEAD_DIM]  <-- was missing
    const half* k_norm_weight;               // [HEAD_DIM]
    const half* o_proj_weight;               // [HIDDEN_SIZE, Q_SIZE]
    const half* post_attn_layernorm_weight;  // [HIDDEN_SIZE]
    const half* gate_proj_weight;            // [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    const half* up_proj_weight;              // [INTERMEDIATE_SIZE, HIDDEN_SIZE]  <-- was missing
    const half* down_proj_weight;            // [HIDDEN_SIZE, INTERMEDIATE_SIZE]
};


struct AtomicGridSync {
  unsigned int *counter;
  unsigned int *generation;
  unsigned int nblocks;
  unsigned int local_gen;

  __device__ void sync() {
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned int my_gen = local_gen;
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      unsigned int arrived = atomicAdd(counter, 1);
      if (arrived == nblocks - 1) {
        *counter = 0;
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        atomicAdd(generation, 1);
      } else {
        volatile unsigned int *vgen = (volatile unsigned int *)generation;
        while (*vgen <= my_gen) {
        }
      }
      local_gen = my_gen + 1;
    }
    __syncthreads();
  }
};

__device__ __forceinline__ float ldg_warp_reduce_sum(float val){
    #pragma unroll
    for(int offset = WARP_SIZE/2; offset > 0; offset/=2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

#define LOG2E_HALF __float2half(1.44269504088896340736f)

__device__ __forceinline__ __half ptx_hrcp(__half x) {
    __half y;
    unsigned short xbits = __half_as_ushort(x);
    unsigned short ybits;
    asm volatile("rcp.approx.ftz.f16 %0, %1;" : "=h"(ybits) : "h"(xbits));
    y = __ushort_as_half(ybits);
    return y;
}

__device__ __forceinline__ __half ptx_hexp2(__half x) {
    __half y;
    unsigned short xbits = __half_as_ushort(x);
    unsigned short ybits;
    asm volatile("ex2.approx.ftz.f16 %0, %1;" : "=h"(ybits) : "h"(xbits));
    y = __ushort_as_half(ybits);
    return y;
}

__device__ __forceinline__ __half fast_exp(__half x) {
    return ptx_hexp2(__hmul(x, LOG2E_HALF));
}

__device__ __forceinline__ __half ldg_silu(__half x){
    return (__hmul(x,ptx_hrcp(__hadd(__float2half(1.0f), fast_exp(__hneg(x))))));
}

__device__ __forceinline__ uint2 ldg_load_weights_u2(const uint2* ptr) {
    uint2 res;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];" 
                 : "=r"(res.x), "=r"(res.y) : "l"(ptr));
    return res;
}

__device__ __forceinline__ uint4 ldg_load_weights_u4(const uint4* ptr) {
    uint4 res;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" 
                 : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w) : "l"(ptr));
    return res;
}

__device__ __forceinline__ void device_rmsnorm_step(
    half* s_norm_out,
    const half* input,
    half* residual,
    const uint2* weight,
    float eps,
    int block_id
){
    int tx = threadIdx.x;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32>warp = cg::tiled_partition<32>(block);

    __shared__ float s_rms_inv;
    __shared__ float shared_reduce[LDG_BLOCK_SIZE/WARP_SIZE];

    float thread_sum_sq = 0.0f;

    // Fused Add + sum of squares(VEC SIZE is 4)
    #pragma unroll
    for(int i = tx; i<HIDDEN_SIZE/4; i+=LDG_BLOCK_SIZE){
        uint2 v_in = reinterpret_cast<const uint2*>(input)[i];
        uint2 v_res = reinterpret_cast<uint2*>(residual)[i];

        half2* h2_in = reinterpret_cast<half2*>(&v_in);
        half2* h2_res = reinterpret_cast<half2*>(&v_res);

        // Residual = input + residual
        h2_res[0] = __hadd2(h2_in[0], h2_res[0]);
        h2_res[1] = __hadd2(h2_in[1], h2_res[1]);

        if(block_id == 0) reinterpret_cast<uint2*>(residual)[i] = v_res;

        // Cache for normalization
        reinterpret_cast<uint2*>(s_norm_out)[i] = v_res;

        float2 f0 = __half22float2(h2_res[0]);
        float2 f1 = __half22float2(h2_res[1]);
        thread_sum_sq += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y *f1.y;
    }
    // Reduction
    float warp_sum=  cg::reduce(warp,thread_sum_sq,cg::plus<float>());
    if(warp.thread_rank() == 0) shared_reduce[tx/32] = warp_sum;
    block.sync();

    float block_sum = 0.0f;
    if(tx<(LDG_BLOCK_SIZE/WARP_SIZE)) block_sum = shared_reduce[tx];
    block_sum = cg::reduce(warp,block_sum,cg::plus<float>());

    if (tx == 0) s_rms_inv = rsqrt(block_sum/(float)HIDDEN_SIZE + eps);
    block.sync();

    // Normalise and write to s_norm_out
    float inv_rms = s_rms_inv;
    #pragma unroll
    for(int i = tx; i < HIDDEN_SIZE/4; i+=LDG_BLOCK_SIZE){
        uint2 v_val = reinterpret_cast<uint2*>(s_norm_out)[i];
        uint2 v_weight = weight[i];
        half2* h2_val = reinterpret_cast<half2*>(&v_val);
        half2* h2_w = reinterpret_cast<half2*>(&v_weight);

        #pragma unroll
        for(int j = 0; j < 2; j++){
            float2 f_v = __half22float2(h2_val[j]);
            float2 f_weight = __half22float2(h2_w[j]);
            f_v.x *= (inv_rms * f_weight.x);
            f_v.y *= (inv_rms * f_weight.y);
            h2_val[j] = __float22half2_rn(f_v);
        }
        reinterpret_cast<uint2*>(s_norm_out)[i] = v_val;
    }
    block.sync();
}
__device__ void ldg_matvec_qkv_fp16(
    AtomicGridSync &grid, 
    half* s_norm,               // Pre-computed in shared memory
    const half* q_weight, const half* k_weight, const half* v_weight,
    half* q_out, half* k_out, half* v_out
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m = row_start + warp_id; m < row_end; m += (LDG_BLOCK_SIZE / 32)) {
        const half* weight_row;
        half* out_ptr;

        // Pointer assignment (Same as your kernel.cu logic)
        if (m < Q_SIZE) { weight_row = q_weight + m * HIDDEN_SIZE; out_ptr = q_out + m; }
        else if (m < Q_SIZE + KV_SIZE) { weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE; out_ptr = k_out + (m - Q_SIZE); }
        else { weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE; out_ptr = v_out + (m - Q_SIZE - KV_SIZE); }

        float sum = 0.0f;
        // Vectorized: 128-bit load = 8 halfs. 1024 / 8 = 128 iterations total.
        // Each warp (32 threads) covers 32*8 = 256 elements. 1024 / 256 = 4 iterations.
        #pragma unroll
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += 32 * 8) {
            uint4 w_u4 = ldg_load_weights_u4(reinterpret_cast<const uint4*>(weight_row + k));
            uint4 a_u4 = *reinterpret_cast<uint4*>(s_norm + k);

            half2* w_h2 = reinterpret_cast<half2*>(&w_u4);
            half2* a_h2 = reinterpret_cast<half2*>(&a_u4);

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 fw = __half22float2(w_h2[j]);
                float2 fa = __half22float2(a_h2[j]);
                sum += (fw.x * fa.x) + (fw.y * fa.y);
            }
        }

        // Warp reduction using your ldg_warp_reduce_sum (converted to float for 1650 accuracy)
        sum = ldg_warp_reduce_sum(sum); 
        if (lane_id == 0) *out_ptr = __float2half(sum);
    }
}
__device__ void ldg_prefetch_weights_l2(const half* weights, int num_elements) {
    int tx = threadIdx.x;
    // Calculate how many 128-bit (8-element) chunks we have
    int num_vec = num_elements / 8;
    for (int i = tx; i < num_vec; i += LDG_BLOCK_SIZE) {
        uint4 dummy_vec = ldg_load_weights_u4(reinterpret_cast<const uint4*>(weights) + i);
        // This assembly prevents the compiler from optimizing away the load
        asm volatile("" : : "r"(dummy_vec.x), "r"(dummy_vec.y), "r"(dummy_vec.z), "r"(dummy_vec.w));
    }
}

__device__ void ldg_attention(
    AtomicGridSync &grid, 
    half* q, 
    half* k, 
    const half* v, 
    half* k_cache, 
    half* v_cache, 
    half* attn_out,
    int cache_len, 
    int max_seq_len, 
    float attn_scale,
    const half* q_norm_weight, 
    const half* k_norm_weight,
    const half* cos_table, 
    const half* sin_table, 
    int position,
    // Weights for prefetching
    const half* o_w, 
    const half* g_w, 
    const half* u_w, 
    const half* d_w
){
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const half* cos_pos = cos_table + position * HEAD_DIM;
    const half* sin_pos = sin_table + position * HEAD_DIM;

    // --- 1. QK Norm & RoPE (Fused) ---
    // Block 0: Handles K norm + RoPE + Cache Write
    if(block_id == 0){
        for(int h = warp_id; h < NUM_KV_HEADS; h += LDG_NUM_WARPS){
            half* k_ptr = k + h * HEAD_DIM;
            half* kc_ptr = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
            half* vc_ptr = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

            float ss = 0.0f;
            for(int i = lane_id; i < HEAD_DIM; i += WARP_SIZE){
                float val = __half2float(k_ptr[i]);
                ss += val * val;
            }
            ss = ldg_warp_reduce_sum(ss);
            float inv_rms = rsqrtf(ss / (float)HEAD_DIM + 1e-6f);
            inv_rms = __shfl_sync(0xffffffff, inv_rms, 0);

            for(int i = lane_id; i < HEAD_DIM; i += WARP_SIZE){
                float k_val = __half2float(k_ptr[i]) * inv_rms * __half2float(k_norm_weight[i]);
                float c = __half2float(cos_pos[i]);
                float s = __half2float(sin_pos[i]);
                
                // RoPE: [x, y] -> [x*c - y*s, x*s + y*c] using half-dim swap
                int po = (i < HEAD_DIM / 2) ? (HEAD_DIM / 2) : -(HEAD_DIM / 2);
                (void)po;
                float k_neighbor = __shfl_xor_sync(0xffffffff, k_val, 16); // Works if HEAD_DIM iterations are managed
                // For sm_75, we'll use a simpler paired register approach:
                float k_rot = (i < HEAD_DIM / 2) ? (k_val * c - __shfl_sync(0xffffffff, k_val, lane_id + 16) * s) : (k_val * c + __shfl_sync(0xffffffff, k_val, lane_id - 16) * s);
                
                kc_ptr[i] = __float2half(k_rot);
                vc_ptr[i] = v[h * HEAD_DIM + i];
            }
        }
    }

    // Attention blocks: Handle Q norm + RoPE
    if(block_id < LDG_ATTN_BLOCKS && warp_id == 0){
        int heads_per_block = (NUM_Q_HEADS + LDG_ATTN_BLOCKS - 1) / LDG_ATTN_BLOCKS;
        int q_start = block_id * heads_per_block;
        int q_end = min(q_start + heads_per_block, NUM_Q_HEADS);

        for(int qh = q_start; qh < q_end; qh++){
            half* q_ptr = q + qh * HEAD_DIM;
            float ss = 0.0f;
            for(int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
                float v = __half2float(q_ptr[i]);
                ss += v * v;
            }
            ss = ldg_warp_reduce_sum(ss);
            float inv_rms = rsqrtf(ss / (float)HEAD_DIM + 1e-6f);
            inv_rms = __shfl_sync(0xffffffff, inv_rms, 0);

            for(int i = lane_id; i < HEAD_DIM; i += WARP_SIZE){
                float q_val = __half2float(q_ptr[i]) * inv_rms * __half2float(q_norm_weight[i]);
                float c = __half2float(cos_pos[i]);
                float s = __half2float(sin_pos[i]);
                // Simplified rotation for GTX 1650
                float q_rot = (i < HEAD_DIM / 2) ? (q_val * c - __shfl_xor_sync(0xffffffff, q_val, 16) * s) : (q_val * c + __shfl_xor_sync(0xffffffff, q_val, 16) * s);
                q_ptr[i] = __float2half(q_rot);
            }
        }
    }

    // --- 2. Prefetching (Idle Blocks) ---
    if (block_id >= LDG_ATTN_BLOCKS) {
        int prefetch_id = block_id - LDG_ATTN_BLOCKS;
        int num_prefetch_blocks = LDG_NUM_BLOCKS - LDG_ATTN_BLOCKS;
        // Divide weights across remaining blocks to warm L2 cache
        int total_elements = (HIDDEN_SIZE * Q_SIZE) + (HIDDEN_SIZE * INTERMEDIATE_SIZE * 3); 
        int per_block = (total_elements + num_prefetch_blocks - 1) / num_prefetch_blocks;
        int start = prefetch_id * per_block;
        int end = min(start + per_block, total_elements);
        
        for (int i = start + threadIdx.x; i < end; i += LDG_BLOCK_SIZE) {
            // Casting to void* to use generic prefetch PTX
            const half* ptr = (i < HIDDEN_SIZE * Q_SIZE) ? (o_w + i) : (g_w + (i - HIDDEN_SIZE * Q_SIZE));
            (void)ptr;
        }
    }

    grid.sync();

    // --- 3. Attention Computation ---
    __shared__ float s_max_score[LDG_NUM_WARPS];
    __shared__ float s_sum_exp[LDG_NUM_WARPS];
    __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

    if(block_id < LDG_ATTN_BLOCKS){
        int heads_per_block = (NUM_Q_HEADS + LDG_ATTN_BLOCKS - 1) / LDG_ATTN_BLOCKS;
        int q_start = block_id * heads_per_block;
        int q_end = min(q_start + heads_per_block, NUM_Q_HEADS);

        for(int qh = q_start; qh < q_end; qh++){
            half* q_head = q + qh * HEAD_DIM;
            int kv_head  = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
            
            float max_score = -INFINITY;
            float sum_exp = 0.0f;
            float acc[4] = {0.0f}; // Local accumulation for 4 elements per thread

            for(int t = warp_id; t < cache_len; t += LDG_NUM_WARPS){
                half* kc = k_cache + kv_head * max_seq_len * HEAD_DIM + t * HEAD_DIM;
                float score = 0.0f;
                #pragma unroll
                for(int i = lane_id; i < HEAD_DIM; i += 32) score += __half2float(q_head[i]) * __half2float(kc[i]);
                score = ldg_warp_reduce_sum(score) * attn_scale;
                score = __shfl_sync(0xffffffff, score, 0);

                float old_max = max_score;
                max_score = fmaxf(max_score, score);
                float e = expf(score - max_score);
                float e_old = expf(old_max - max_score);
                sum_exp = sum_exp * e_old + e;

                half* vc = v_cache + kv_head * max_seq_len * HEAD_DIM + t * HEAD_DIM;
                #pragma unroll
                for (int i = 0; i < 4; i++) acc[i] = acc[i] * e_old + e * __half2float(vc[lane_id + i * 32]);
            }

            // Final Reduction across warps
            if (lane_id == 0) { s_max_score[warp_id] = max_score; s_sum_exp[warp_id] = sum_exp; }
            #pragma unroll
            for (int i = 0; i < 4; i++) s_out_acc[warp_id][lane_id + i * 32] = acc[i];
            __syncthreads();

            if (warp_id == 0) {
                float global_max = s_max_score[0];
                for (int w = 1; w < LDG_NUM_WARPS; w++) global_max = fmaxf(global_max, s_max_score[w]);
                
                float total_sum = 0.0f;
                float final_acc[4] = {0.0f};
                for (int w = 0; w < LDG_NUM_WARPS; w++) {
                    float scale = expf(s_max_score[w] - global_max);
                    total_sum += s_sum_exp[w] * scale;
                    #pragma unroll
                    for (int i = 0; i < 4; i++) final_acc[i] += s_out_acc[w][lane_id + i * 32] * scale;
                }
                
                half* out_head = attn_out + qh * HEAD_DIM;
                #pragma unroll
                for (int i = 0; i < 4; i++) out_head[lane_id + i * 32] = __float2half(final_acc[i] / total_sum);
            }
            __syncthreads();
        }
    }
    grid.sync();
}
// O projection + Residual + PostNorm + MLP(_ldg)
__device__ void ldg_o_proj_postnorm_mlp(
    AtomicGridSync &grid, 
    const __half *__restrict__ o_weight,
    const __half *__restrict__ post_norm_weight,
    const __half *__restrict__ gate_weight,
    const __half *__restrict__ up_weight,
    const __half *__restrict__ down_weight,
    const float *__restrict__ attn_out, 
    float *__restrict__ g_residual,
    float *__restrict__ g_activations, 
    float *__restrict__ g_mlp_intermediate,
    __half *__restrict__ hidden_out
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    __shared__ __align__(16) __half s_attn[Q_SIZE];
    __shared__ __align__(16) __half s_act[HIDDEN_SIZE];
    __shared__ __align__(16) __half s_mlp[INTERMEDIATE_SIZE];

    // Cache attention output (Input is float, store as half in smem)
    for (int i = threadIdx.x; i < Q_SIZE; i += LDG_BLOCK_SIZE) {
        s_attn[i] = __float2half(attn_out[i]);
    }
    __syncthreads();

    // 2. O Projection + Residual
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    for (int m = hid_start + warp_id; m < hid_end; m += LDG_NUM_WARPS) {
        float sum = 0.0f;
        const uint4* o_row_ptr = reinterpret_cast<const uint4*>(o_weight + m * Q_SIZE);

        #pragma unroll 4
        for (int k = lane_id; k < Q_SIZE / 8; k += WARP_SIZE) {
            // 128-bit optimized load from megakernel.cu
            uint4 w_u4 = ldg_load_weights_u4(o_row_ptr + k);
            uint4 a_u4 = *reinterpret_cast<const uint4*>(s_attn + k * 8);
            
            half2* wh = reinterpret_cast<half2*>(&w_u4);
            half2* ah = reinterpret_cast<half2*>(&a_u4);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 fw = __half22float2(wh[i]);
                float2 fa = __half22float2(ah[i]);
                sum += (fw.x * fa.x) + (fw.y * fa.y);
            }
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0) {
            g_activations[m] = sum + g_residual[m];
        }
    }

    grid.sync(); 
    
    // Post-Attention RMSNorm (Redundant across blocks to avoid extra syncs)
    {
        __shared__ float s_sum_sq;
        if (threadIdx.x == 0) s_sum_sq = 0.0f;
        __syncthreads();

        float local_ss = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = g_activations[i];
            s_act[i] = __float2half(v); 
            local_ss += v * v;
            if (block_id == 0) g_residual[i] = v; // Save residual for next layer
        }
        local_ss = ldg_warp_reduce_sum(local_ss);
        if (lane_id == 0) atomicAdd(&s_sum_sq, local_ss);
        __syncthreads();

        float rstd = rsqrtf(s_sum_sq / (float)HIDDEN_SIZE + LDG_RMS_EPS);

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __half2float(post_norm_weight[i]);
            s_act[i] = __float2half(__half2float(s_act[i]) * rstd * w);
        }
        __syncthreads();
    }

    // 4. Gate + Up + SiLU (SwiGLU)
    int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start = block_id * int_per_block;
    int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

    for (int m = int_start + warp_id; m < int_end; m += LDG_NUM_WARPS) {
        float gate_sum = 0.0f, up_sum = 0.0f;
        const uint4* g_row = reinterpret_cast<const uint4*>(gate_weight + m * HIDDEN_SIZE);
        const uint4* u_row = reinterpret_cast<const uint4*>(up_weight + m * HIDDEN_SIZE);

        #pragma unroll 4
        for (int k = lane_id; k < HIDDEN_SIZE / 8; k += WARP_SIZE) {
            uint4 g_u4 = ldg_load_weights_u4(g_row + k);
            uint4 u_u4 = ldg_load_weights_u4(u_row + k);
            uint4 a_u4 = *reinterpret_cast<const uint4*>(s_act + k * 8);

            half2* gh = reinterpret_cast<half2*>(&g_u4);
            half2* uh = reinterpret_cast<half2*>(&u_u4);
            half2* ah = reinterpret_cast<half2*>(&a_u4);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 fa = __half22float2(ah[i]);
                gate_sum += (__half2float(gh[i].x) * fa.x) + (__half2float(gh[i].y) * fa.y);
                up_sum   += (__half2float(uh[i].x) * fa.x) + (__half2float(uh[i].y) * fa.y);
            }
        }
        gate_sum = ldg_warp_reduce_sum(gate_sum);
        up_sum = ldg_warp_reduce_sum(up_sum);

        if (lane_id == 0) {
            // Optimized silu from megakernel.cu
            float activated = __half2float(ldg_silu(__float2half(gate_sum))) * up_sum;
            g_mlp_intermediate[m] = activated;
        }
    }

    grid.sync();

    // 5. Down projection + residual
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += LDG_BLOCK_SIZE) {
        s_mlp[i] = __float2half(g_mlp_intermediate[i]);
    }
    __syncthreads();

    for (int m = hid_start + warp_id; m < hid_end; m += LDG_NUM_WARPS) {
        float sum = 0.0f;
        const uint4* d_row = reinterpret_cast<const uint4*>(down_weight + m * INTERMEDIATE_SIZE);

        #pragma unroll 4
        for (int k = lane_id; k < INTERMEDIATE_SIZE / 8; k += WARP_SIZE) {
            uint4 d_u4 = ldg_load_weights_u4(d_row + k);
            uint4 m_u4 = *reinterpret_cast<const uint4*>(s_mlp + k * 8);

            half2* dh = reinterpret_cast<half2*>(&d_u4);
            half2* mh = reinterpret_cast<half2*>(&m_u4);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 fd = __half22float2(dh[i]);
                float2 fm = __half22float2(mh[i]);
                sum += (fd.x * fm.x) + (fd.y * fm.y);
            }
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0) {
            hidden_out[m] = __float2half(sum + g_residual[m]);
        }
    }
    grid.sync();
}

// Global variables and helper functions for persistent kernel infrastructure

static unsigned int *d_barrier_counter = nullptr;
static unsigned int *d_barrier_sense = nullptr;
static unsigned int *d_kv_flag = nullptr;
static unsigned int *d_attn_flag = nullptr;
static int *d_mutable_position = nullptr;
static int *d_mutable_token_id = nullptr;
static int *h_pinned_position = nullptr;
static int *h_pinned_token_id = nullptr;

static void ensure_barrier_alloc() {
  if (!d_barrier_counter) {
    cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
    cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
    cudaMalloc(&d_kv_flag, sizeof(unsigned int));
    cudaMalloc(&d_attn_flag, sizeof(unsigned int));
    cudaMalloc(&d_mutable_position, sizeof(int));
    cudaMalloc(&d_mutable_token_id, sizeof(int));
    cudaHostAlloc(&h_pinned_position, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_pinned_token_id, sizeof(int), cudaHostAllocDefault);
    cudaMemset(d_barrier_counter, 0, sizeof(unsigned int));
    cudaMemset(d_barrier_sense, 0, sizeof(unsigned int));
    cudaMemset(d_kv_flag, 0, sizeof(unsigned int));
    cudaMemset(d_attn_flag, 0, sizeof(unsigned int));
  }
}

// Device-side step update kernel
__global__ void ldg_update_step(
    const int *__restrict__ lm_output,
    int *__restrict__ d_token_id,
    int *__restrict__ d_position,
    int *__restrict__ output_log,
    int *__restrict__ d_step_counter) {
  int tok = *lm_output;
  int step = *d_step_counter;
  *d_token_id = tok;
  *d_position = *d_position + 1;
  output_log[step] = tok;
  *d_step_counter = step + 1;
}

// Forward declarations
static inline void ldg_configure_kernel_attributes();

// Decoder kernels (must be defined in your main file)
// Shared decode body called by both direct and persistent kernels
__device__ void ldg_decode_body(
    const half *embed_weight,
    const LDGLayerWeight *layer_weights,
    const half *final_norm_weight,
    const half *cos_table,
    const half *sin_table,
    half *k_cache, half *v_cache,
    half *hidden_buffer,
    float *g_activations, float *g_residual,
    half *g_q, half *g_k, half *g_v,
    half *g_attn_out, float *g_mlp_intermediate, float *g_normalized,
    int num_layers, int position, int input_token_id,
    int max_seq_len, float attn_scale,
    AtomicGridSync &grid)
{
    int tid = threadIdx.x;

    // 1. Embedding lookup -- block 0 writes residual, all blocks read it after sync
    if (blockIdx.x == 0) {
        const half *embed_row = embed_weight + (long long)input_token_id * HIDDEN_SIZE;
        for (int i = tid; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            hidden_buffer[i] = embed_row[i];
            g_residual[i]    = __half2float(embed_row[i]);
        }
    }
    grid.sync();

    // 2. Transformer layers
    __shared__ half s_norm[HIDDEN_SIZE];

    for (int layer = 0; layer < num_layers; layer++) {
        const LDGLayerWeight &lw = layer_weights[layer];

        // Input RMSNorm (fused residual add inside)
        device_rmsnorm_step(
            s_norm,
            hidden_buffer,
            reinterpret_cast<half*>(g_residual),
            reinterpret_cast<const uint2*>(lw.input_layernorm_weight),
            LDG_RMS_EPS,
            (int)blockIdx.x
        );
        grid.sync();

        // QKV projection
        ldg_matvec_qkv_fp16(
            grid, s_norm,
            lw.q_proj_weight, lw.k_proj_weight, lw.v_proj_weight,
            g_q, g_k, g_v
        );
        grid.sync();

        // Q/K norm + RoPE + attention
        half *layer_k_cache = k_cache + (long long)layer * NUM_KV_HEADS * max_seq_len * HEAD_DIM;
        half *layer_v_cache = v_cache + (long long)layer * NUM_KV_HEADS * max_seq_len * HEAD_DIM;

        ldg_attention(
            grid,
            g_q, g_k, g_v,
            layer_k_cache, layer_v_cache,
            g_attn_out,
            position + 1,   // cache_len = tokens including current
            max_seq_len,
            attn_scale,
            lw.q_norm_weight,
            lw.k_norm_weight,
            cos_table, sin_table,
            position,
            lw.o_proj_weight,
            lw.gate_proj_weight,
            lw.up_proj_weight,
            lw.down_proj_weight
        );
        // grid.sync() already called inside ldg_attention at end

        // O proj + post-norm + MLP: feed attn_out as float to match signature
        // Convert half g_attn_out -> float g_activations for the function
        for (int i = tid; i < Q_SIZE; i += LDG_BLOCK_SIZE)
            g_activations[i] = __half2float(g_attn_out[i]);
        grid.sync();

        ldg_o_proj_postnorm_mlp(
            grid,
            lw.o_proj_weight,
            lw.post_attn_layernorm_weight,
            lw.gate_proj_weight,
            lw.up_proj_weight,
            lw.down_proj_weight,
            g_activations,
            g_residual,
            g_activations,
            g_mlp_intermediate,
            hidden_buffer
        );
        // grid.sync() already called inside ldg_o_proj_postnorm_mlp at end
    }

    // 3. Final RMSNorm -> g_normalized (float, for LM head)
    {
        __shared__ float s_inv_rms;
        __shared__ float s_warp_ss[LDG_BLOCK_SIZE / WARP_SIZE];
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        float local_ss = 0.0f;
        for (int i = tid; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = __half2float(hidden_buffer[i]);
            local_ss += v * v;
        }
        local_ss = ldg_warp_reduce_sum(local_ss);
        if (lane_id == 0) s_warp_ss[warp_id] = local_ss;
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane_id < LDG_BLOCK_SIZE / WARP_SIZE) ? s_warp_ss[lane_id] : 0.0f;
            v = ldg_warp_reduce_sum(v);
            if (lane_id == 0) s_inv_rms = rsqrtf(v / (float)HIDDEN_SIZE + LDG_RMS_EPS);
        }
        __syncthreads();

        float inv_rms = s_inv_rms;
        for (int i = tid; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __half2float(final_norm_weight[i]);
            float v = __half2float(hidden_buffer[i]);
            g_normalized[i] = v * inv_rms * w;
        }
    }
    grid.sync();
}

__global__ void __launch_bounds__(LDG_BLOCK_SIZE) ldg_decode_kernel_direct(
    const half *embed_weight,
    const LDGLayerWeight *layer_weights,
    const half *final_norm_weight,
    const half *cos_table,
    const half *sin_table,
    half *k_cache, half *v_cache,
    half *hidden_buffer,
    float *g_activations, float *g_residual,
    half *g_q, half *g_k, half *g_v,
    half *g_attn_out, float *g_mlp_intermediate, float *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_sense,
    unsigned int *kv_flag, unsigned int *attn_flag,
    int num_layers, int position, int input_token_id,
    int max_seq_len, float attn_scale)
{
    AtomicGridSync grid;
    grid.counter    = barrier_counter;
    grid.generation = barrier_sense;
    grid.nblocks    = gridDim.x;
    grid.local_gen  = 0;

    ldg_decode_body(
        embed_weight, layer_weights, final_norm_weight,
        cos_table, sin_table,
        k_cache, v_cache, hidden_buffer,
        g_activations, g_residual,
        g_q, g_k, g_v, g_attn_out, g_mlp_intermediate, g_normalized,
        num_layers, position, input_token_id, max_seq_len, attn_scale,
        grid
    );
}

__global__ void __launch_bounds__(LDG_BLOCK_SIZE) ldg_decode_kernel_persistent(
    const half *embed_weight,
    const LDGLayerWeight *layer_weights,
    const half *final_norm_weight,
    const half *cos_table,
    const half *sin_table,
    half *k_cache, half *v_cache,
    half *hidden_buffer,
    float *g_activations, float *g_residual,
    half *g_q, half *g_k, half *g_v,
    half *g_attn_out, float *g_mlp_intermediate, float *g_normalized,
    unsigned int *barrier_counter, unsigned int *barrier_sense,
    unsigned int *kv_flag, unsigned int *attn_flag,
    int num_layers, const int *d_position,
    const int *d_token_id, int max_seq_len, float attn_scale)
{
    AtomicGridSync grid;
    grid.counter    = barrier_counter;
    grid.generation = barrier_sense;
    grid.nblocks    = gridDim.x;
    grid.local_gen  = 0;

    ldg_decode_body(
        embed_weight, layer_weights, final_norm_weight,
        cos_table, sin_table,
        k_cache, v_cache, hidden_buffer,
        g_activations, g_residual,
        g_q, g_k, g_v, g_attn_out, g_mlp_intermediate, g_normalized,
        num_layers, *d_position, *d_token_id, max_seq_len, attn_scale,
        grid
    );
}
/**
 * Phase 1: Distributed vocab projection and block-level argmax
 */
__global__ void __launch_bounds__(LDG_LM_BLOCK_SIZE, 1) ldg_lm_head_phase1(
    const float *__restrict__ normalized,
    const half *__restrict__ weight,
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs)
{
    __shared__ __align__(128) float s_hidden[HIDDEN_SIZE];
    
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE) {
        s_hidden[i] = normalized[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    int warp_stride = LDG_LM_BLOCK_SIZE / WARP_SIZE;
    int base = row_start + warp_id * LDG_LM_ROWS_PER_WARP;

    for (int m_base = base; m_base < row_end; m_base += warp_stride * LDG_LM_ROWS_PER_WARP) {
        int rows[LDG_LM_ROWS_PER_WARP];
        bool valid[LDG_LM_ROWS_PER_WARP];
        float sum[LDG_LM_ROWS_PER_WARP];

        #pragma unroll
        for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
            rows[r] = m_base + r;
            valid[r] = rows[r] < row_end;
            sum[r] = 0.0f;
        }

        #pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            float4 a1 = *reinterpret_cast<const float4 *>(s_hidden + k);
            float4 a2 = *reinterpret_cast<const float4 *>(s_hidden + k + 4);

            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) continue;

                const half *w_row_ptr = weight + rows[r] * HIDDEN_SIZE + k;
                uint4 w_u4 = ldg_load_weights_u4(reinterpret_cast<const uint4 *>(w_row_ptr));
                const half2 *w_h2 = reinterpret_cast<const half2 *>(&w_u4);

                float2 wf0 = __half22float2(w_h2[0]);
                float2 wf1 = __half22float2(w_h2[1]);
                float2 wf2 = __half22float2(w_h2[2]);
                float2 wf3 = __half22float2(w_h2[3]);

                sum[r] += wf0.x * a1.x + wf0.y * a1.y +
                          wf1.x * a1.z + wf1.y * a1.w +
                          wf2.x * a2.x + wf2.y * a2.y +
                          wf3.x * a2.z + wf3.y * a2.w;
            }
        }

        #pragma unroll
        for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
            if (!valid[r]) continue;
            float reduced = ldg_warp_reduce_sum(sum[r]);
            if (lane_id == 0 && reduced > local_max) {
                local_max = reduced;
                local_max_idx = rows[r];
            }
        }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_max_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_max_idx = other_idx;
        }
    }
    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ struct { float val; int idx; } s_warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];

    if (lane_id == 0) {
        s_warp_max[warp_id].val = local_max;
        s_warp_max[warp_id].idx = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = LDG_LM_BLOCK_SIZE / WARP_SIZE;
        float final_max = (lane_id < num_warps) ? s_warp_max[lane_id].val : -INFINITY;
        int final_idx = (lane_id < num_warps) ? s_warp_max[lane_id].idx : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float v = __shfl_down_sync(0xffffffff, final_max, offset);
            int i = __shfl_down_sync(0xffffffff, final_idx, offset);
            if (v > final_max) {
                final_max = v;
                final_idx = i;
            }
        }

        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = final_max;
            block_max_idxs[blockIdx.x] = final_idx;
        }
    }
}

/**
 * Phase 2: Global argmax reduction
 */
__global__ void __launch_bounds__(LDG_LM_BLOCK_SIZE, 1) ldg_lm_head_phase2(
    const float *__restrict__ block_max_vals,
    const int *__restrict__ block_max_idxs,
    int *__restrict__ output_token,
    int num_blocks)
{
    __shared__ struct { float val; int idx; } s_data[LDG_LM_BLOCK_SIZE];

    float thread_max = -INFINITY;
    int thread_idx = -1;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        float val = block_max_vals[i];
        if (val > thread_max) {
            thread_max = val;
            thread_idx = block_max_idxs[i];
        }
    }

    s_data[threadIdx.x].val = thread_max;
    s_data[threadIdx.x].idx = thread_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_data[threadIdx.x + stride].val > s_data[threadIdx.x].val) {
                s_data[threadIdx.x] = s_data[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output_token = s_data[0].idx;
    }
}

// Launch functions

extern "C" void launch_ldg_decode_direct(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeight *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int max_seq_len,
    float attn_scale, cudaStream_t stream) {
  
  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();

  ldg_decode_kernel_direct<<<LDG_NUM_BLOCKS, LDG_BLOCK_SIZE, 0, stream>>>(
      (const half *)embed_weight, layer_weights,
      (const half *)final_norm_weight,
      (const half *)cos_table, (const half *)sin_table,
      (half *)k_cache, (half *)v_cache,
      (half *)hidden_buffer, (float *)g_activations,
      (float *)g_residual, (half *)g_q, (half *)g_k, (half *)g_v,
      (half *)g_attn_out, (float *)g_mlp_intermediate, (float *)g_normalized,
      d_barrier_counter, d_barrier_sense, d_kv_flag, d_attn_flag, num_layers,
      position, input_token_id, max_seq_len, attn_scale);

  ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
      (const float *)g_normalized, 
      (const half *)lm_head_weight,
      (float *)block_max_vals, 
      (int *)block_max_idxs);

  ldg_lm_head_phase2<<<1, LDG_LM_BLOCK_SIZE, 0, stream>>>(
      (const float *)block_max_vals,
      (const int *)block_max_idxs,
      output_token_id,
      LDG_LM_NUM_BLOCKS);
}

extern "C" void launch_ldg_decode_persistent(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeight *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int cache_len,
    int max_seq_len, float attn_scale, cudaStream_t stream) {
  
  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();

  *h_pinned_position = position;
  *h_pinned_token_id = input_token_id;
  cudaMemcpyAsync(d_mutable_position, h_pinned_position, sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_mutable_token_id, h_pinned_token_id, sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  ldg_decode_kernel_persistent<<<LDG_NUM_BLOCKS, LDG_BLOCK_SIZE, 0, stream>>>(
      (const half *)embed_weight, layer_weights,
      (const half *)final_norm_weight,
      (const half *)cos_table, (const half *)sin_table,
      (half *)k_cache, (half *)v_cache,
      (half *)hidden_buffer, (float *)g_activations,
      (float *)g_residual, (half *)g_q, (half *)g_k, (half *)g_v,
      (half *)g_attn_out, (float *)g_mlp_intermediate, (float *)g_normalized,
      d_barrier_counter, d_barrier_sense, d_kv_flag, d_attn_flag, num_layers,
      d_mutable_position, d_mutable_token_id, max_seq_len, attn_scale);

  ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
      (const float *)g_normalized,
      (const half *)lm_head_weight,
      (float *)block_max_vals,
      (int *)block_max_idxs);

  ldg_lm_head_phase2<<<1, LDG_LM_BLOCK_SIZE, 0, stream>>>(
      (const float *)block_max_vals,
      (const int *)block_max_idxs,
      output_token_id,
      LDG_LM_NUM_BLOCKS);
}

extern "C" void launch_ldg_generate_nosync(
    int first_token_id, int num_steps, const void *embed_weight,
    const LDGLayerWeight *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int *output_log, 
    int num_layers, int start_position, int max_seq_len, float attn_scale,
    cudaStream_t stream) {

  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();

  static int *d_step_counter = nullptr;
  static int *d_output_token = nullptr;
  if (!d_step_counter) {
    cudaMalloc(&d_step_counter, sizeof(int));
    cudaMalloc(&d_output_token, sizeof(int));
  }
  cudaMemsetAsync(d_step_counter, 0, sizeof(int), stream);

  *h_pinned_position = start_position;
  *h_pinned_token_id = first_token_id;
  cudaMemcpyAsync(d_mutable_position, h_pinned_position, sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_mutable_token_id, h_pinned_token_id, sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  for (int step = 0; step < num_steps; step++) {
    ldg_decode_kernel_persistent<<<LDG_NUM_BLOCKS, LDG_BLOCK_SIZE, 0, stream>>>(
        (const half *)embed_weight, layer_weights,
        (const half *)final_norm_weight,
        (const half *)cos_table, (const half *)sin_table,
        (half *)k_cache, (half *)v_cache,
        (half *)hidden_buffer, (float *)g_activations,
        (float *)g_residual, (half *)g_q, (half *)g_k, (half *)g_v,
        (half *)g_attn_out, (float *)g_mlp_intermediate, (float *)g_normalized,
        d_barrier_counter, d_barrier_sense, d_kv_flag, d_attn_flag, num_layers,
        d_mutable_position, d_mutable_token_id, max_seq_len, attn_scale);

    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const half *)lm_head_weight,
        (float *)block_max_vals,
        (int *)block_max_idxs);

    ldg_lm_head_phase2<<<1, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)block_max_vals,
        (const int *)block_max_idxs,
        d_output_token,
        LDG_LM_NUM_BLOCKS);

    ldg_update_step<<<1, 1, 0, stream>>>(
        d_output_token, d_mutable_token_id,
        d_mutable_position, output_log,
        d_step_counter);
  }
}

static inline void ldg_configure_kernel_attributes() {
  static bool configured = false;
  if (configured) return;
  configured = true;
  
  cudaFuncSetAttribute(ldg_decode_kernel_persistent,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  cudaFuncSetAttribute(ldg_lm_head_phase1,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
  cudaFuncSetAttribute(ldg_lm_head_phase2,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
}
