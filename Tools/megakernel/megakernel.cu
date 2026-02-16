#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "rmsnorm.cuh"

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
#define LDG_NUM_BLOCK 28
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
#define LDG_ATTN_BOCKS NUM_Q_HEADS
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

// LM_head
constexpr int LDG_VOCAB_SIZE = 151936;

struct LDGLayerWeight{
    const uint2* input_layernorm_weight;
    const uint2* q_proj_weight;
    const uint2* k_proj_weight;
    const uint2* v_proj_weight;
    const uint2* k_norm_weight;
    const uint2* o_proj_weight;
    const uint2* post_attn_layernorm_weight;
    const uint2* gate_proj_weight;
    const uint2* down_proj_weight;
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

__device__ const __half LOG2E_HALF = __float2half(1.44269504088896340736f);

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
    return (__hmul(x,ptx_hrcp(__float2half(1.0f)+fast_exp(-x))));
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

    // QK Norm & RoPE (Fused)
    // Handles K norm + RoPE + Cache write (Block 0)
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
                float k_neighbor = __shfl_xor_sync(0xffffffff, k_val, 16);
                float k_rot = (i < HEAD_DIM / 2) ? (k_val * c - __shfl_sync(0xffffffff, k_val, lane_id + 16) * s) : (k_val * c + __shfl_sync(0xffffffff, k_val, lane_id - 16) * s);
                
                kc_ptr[i] = __float2half(k_rot);
                vc_ptr[i] = v[h * HEAD_DIM + i];
            }
        }
    }

    // Handle Q norm + RoPE
    if(block_id < LDG_ATTN_BOCKS && warp_id == 0){
        int heads_per_block = (NUM_Q_HEADS + LDG_ATTN_BOCKS - 1) / LDG_ATTN_BOCKS;
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
                float q_rot = (i < HEAD_DIM / 2) ? (q_val * c - __shfl_xor_sync(0xffffffff, q_val, 16) * s) : (q_val * c + __shfl_xor_sync(0xffffffff, q_val, 16) * s);
                q_ptr[i] = __float2half(q_rot);
            }
        }
    }

    // Prefetching (Idle Blocks)
    if (block_id >= LDG_ATTN_BOCKS) {
        int prefetch_id = block_id - LDG_ATTN_BOCKS;
        int num_prefetch_blocks = LDG_NUM_BLOCK - LDG_ATTN_BOCKS;
        // Divide weights across remaining blocks to warm L2 cache
        int total_elements = (HIDDEN_SIZE * Q_SIZE) + (HIDDEN_SIZE * INTERMEDIATE_SIZE * 3); 
        int per_block = (total_elements + num_prefetch_blocks - 1) / num_prefetch_blocks;
        int start = prefetch_id * per_block;
        int end = min(start + per_block, total_elements);
        
        for (int i = start + threadIdx.x; i < end; i += LDG_BLOCK_SIZE) {
            // Casting to void* to use generic prefetch PTX
            const half* ptr = (i < HIDDEN_SIZE * Q_SIZE) ? (o_w + i) : (g_w + (i - HIDDEN_SIZE * Q_SIZE));
            __builtin_prefetch(ptr, 0, 0); // L2 prefetch
        }
    }

    grid.sync();

    // Attention Computation
    __shared__ float s_max_score[LDG_NUM_WARPS];
    __shared__ float s_sum_exp[LDG_NUM_WARPS];
    __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

    if(block_id < LDG_ATTN_BOCKS){
        int heads_per_block = (NUM_Q_HEADS + LDG_ATTN_BOCKS - 1) / LDG_ATTN_BOCKS;
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
