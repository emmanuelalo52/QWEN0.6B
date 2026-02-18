// qwen_ops.cpp - PyTorch C++ Extension for Qwen Megakernel
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

struct LDGLayerWeight;

// Forward declarations of C functions from megakernel.cu
extern "C" void launch_ldg_decode_direct(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeight *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int max_seq_len,
    float attn_scale, cudaStream_t stream);

extern "C" void launch_ldg_generate_nosync(
    int first_token_id, int num_steps, const void *embed_weight,
    const LDGLayerWeight *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int *output_log, int num_layers, int start_position,
    int max_seq_len, float attn_scale, cudaStream_t stream);

// ABI Version 2: Matches 12-pointer (96-byte) alignment for GTX 1650
int abi_version() {
    return 2;
}

torch::Tensor decode(
    int input_token_id,
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
    int num_layers,
    int position,
    int max_seq_len,
    float attn_scale
) {
    auto output_token_id = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    
    // GTX 1650 HARDWARE GUARD: Ensure the pointer table is 16-byte aligned
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(layer_weights_packed.data_ptr());
    TORCH_CHECK(ptr_val % 16 == 0, "GTX 1650 Alignment Error: layer_weights_packed must be 16-byte aligned. Current address ends in: ", ptr_val % 16);

    launch_ldg_decode_direct(
        input_token_id,
        output_token_id.data_ptr<int>(),
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeight*>(layer_weights_packed.data_ptr()),
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
        num_layers,
        position,
        max_seq_len,
        attn_scale,
        0 // default stream
    );

    return output_token_id;
}

torch::Tensor generate_nosync(
    int first_token_id,
    int num_steps,
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
    int num_layers,
    int start_position,
    int max_seq_len,
    float attn_scale
) {
    auto output_log = torch::zeros({num_steps}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    
    // GTX 1650 HARDWARE GUARD
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(layer_weights_packed.data_ptr());
    TORCH_CHECK(ptr_val % 16 == 0, "GTX 1650 Alignment Error: layer_weights_packed must be 16-byte aligned.");

    launch_ldg_generate_nosync(
        first_token_id,
        num_steps,
        embed_weight.data_ptr(),
        reinterpret_cast<const LDGLayerWeight*>(layer_weights_packed.data_ptr()),
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
        num_layers,
        start_position,
        max_seq_len,
        attn_scale,
        0 // default stream
    );
    
    return output_log;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode, "Qwen3 single token decode");
    m.def("generate_nosync", &generate_nosync, "Qwen3 batched generation");
    m.def("abi_version", &abi_version, "Get the ABI version of the extension");
}