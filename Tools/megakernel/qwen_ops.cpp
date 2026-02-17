// qwen_ops.cpp - PyTorch C++ Extension for Qwen Megakernel
#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declarations of C functions from megakernel.cu
extern "C" void launch_ldg_decode_direct(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const void *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int max_seq_len,
    float attn_scale, cudaStream_t stream);

extern "C" void launch_ldg_generate_nosync(
    int first_token_id, int num_steps, const void *embed_weight,
    const void *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int *output_log, 
    int num_layers, int start_position, int max_seq_len, float attn_scale,
    cudaStream_t stream);

// PyTorch wrapper for decode
torch::Tensor decode(
    torch::Tensor out_token,           // [1] int32 - output
    int64_t token_id,                  // Input token ID
    torch::Tensor embed_weight,        // [VOCAB_SIZE, HIDDEN_SIZE] fp16
    torch::Tensor layer_weights_packed, // Packed layer weights
    torch::Tensor final_norm_weight,   // [HIDDEN_SIZE] fp16
    torch::Tensor lm_head_weight,      // [VOCAB_SIZE, HIDDEN_SIZE] fp16
    torch::Tensor cos_table,           // [MAX_SEQ_LEN, HEAD_DIM] fp16
    torch::Tensor sin_table,           // [MAX_SEQ_LEN, HEAD_DIM] fp16
    torch::Tensor k_cache,             // [NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM] fp16
    torch::Tensor v_cache,             // [NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM] fp16
    torch::Tensor hidden,              // [HIDDEN_SIZE] fp16 - workspace
    torch::Tensor act,                 // [HIDDEN_SIZE] fp16 - workspace
    torch::Tensor res,                 // [HIDDEN_SIZE] fp16 - workspace
    torch::Tensor q,                   // [Q_SIZE] fp16 - workspace
    torch::Tensor k,                   // [KV_SIZE] fp16 - workspace
    torch::Tensor v,                   // [KV_SIZE] fp16 - workspace
    torch::Tensor attn_out,            // [Q_SIZE] fp16 - workspace
    torch::Tensor mlp_inter,           // [INTERMEDIATE_SIZE] fp16 - workspace
    torch::Tensor norm_out,            // [HIDDEN_SIZE] fp16 - workspace
    torch::Tensor bmax_vals,           // [4096] fp16 - workspace
    torch::Tensor bmax_idxs,           // [4096] int32 - workspace
    int64_t num_layers,
    int64_t position,
    int64_t max_seq_len,
    double attn_scale)
{
    // Call C function
    launch_ldg_decode_direct(
        (int)token_id,
        out_token.data_ptr<int>(),
        embed_weight.data_ptr(),
        layer_weights_packed.data_ptr(),
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden.data_ptr(),
        act.data_ptr(),
        res.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        attn_out.data_ptr(),
        mlp_inter.data_ptr(),
        norm_out.data_ptr(),
        bmax_vals.data_ptr(),
        bmax_idxs.data_ptr(),
        (int)num_layers,
        (int)position,
        (int)max_seq_len,
        (float)attn_scale,
        0  // default stream
    );
    
    return out_token;
}

// PyTorch wrapper for generate_nosync
torch::Tensor generate_nosync(
    int64_t first_token_id,
    int64_t num_steps,
    torch::Tensor embed_weight,
    torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor hidden,
    torch::Tensor act,
    torch::Tensor res,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attn_out,
    torch::Tensor mlp_inter,
    torch::Tensor norm_out,
    torch::Tensor bmax_vals,
    torch::Tensor bmax_idxs,
    int64_t num_layers,
    int64_t start_position,
    int64_t max_seq_len,
    double attn_scale)
{
    // Allocate output tensor
    auto output_log = torch::empty(
        {num_steps}, 
        torch::dtype(torch::kInt32).device(torch::kCUDA)
    );
    
    // Call C function
    launch_ldg_generate_nosync(
        (int)first_token_id,
        (int)num_steps,
        embed_weight.data_ptr(),
        layer_weights_packed.data_ptr(),
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden.data_ptr(),
        act.data_ptr(),
        res.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        attn_out.data_ptr(),
        mlp_inter.data_ptr(),
        norm_out.data_ptr(),
        bmax_vals.data_ptr(),
        bmax_idxs.data_ptr(),
        output_log.data_ptr<int>(),
        (int)num_layers,
        (int)start_position,
        (int)max_seq_len,
        (float)attn_scale,
        0  // default stream
    );
    
    return output_log;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode, "Qwen3 single token decode");
    m.def("generate_nosync", &generate_nosync, "Qwen3 batched generation");
    // Export a tiny ABI marker so Python can reject stale extension binaries.
    // Bump this whenever pointer packing or kernel argument conventions change.
    m.def("abi_version", []() { return 2; }, "Megakernel extension ABI version");
    m.def("built_torch_version", []() { return std::string(TORCH_VERSION); },
          "Torch version used when building this extension");
}
