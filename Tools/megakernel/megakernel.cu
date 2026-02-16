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
