"""Weight loading and high-level decode API for Qwen3-0.6B."""

import math
import torch
import os

try:
    import qwen_megakernel_C
except ImportError:
    qwen_megakernel_C = None

EXTENSION_ABI_VERSION = 2

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM   # 2048
KV_SIZE = 8 * HEAD_DIM   # 1024
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 151936

# Qwen3-0.6B config.json: "rope_theta": 1000000
# Using 10000.0 (the LLaMA default) causes RoPE frequencies to be ~100x too high,
# making every position > 0 produce completely wrong rotations.
# Position 0 is always identity (cos(0)=1, sin(0)=0) so the first token is
# unaffected — this is why token 1 matched HF but all subsequent tokens diverged.
ROPE_THETA = 1_000_000.0


def _require_megakernel_op(op_name: str):
    """Return an op from the extension module or torch.ops namespace."""
    if qwen_megakernel_C is not None and hasattr(qwen_megakernel_C, op_name):
        return getattr(qwen_megakernel_C, op_name)

    namespace = getattr(torch.ops, "qwen_megakernel_C", None)
    if namespace is not None and hasattr(namespace, op_name):
        return getattr(namespace, op_name)

    available_ops = []
    if qwen_megakernel_C is not None:
        available_ops.extend(
            op for op in ("decode", "generate_nosync")
            if hasattr(qwen_megakernel_C, op)
        )
    if namespace is not None:
        available_ops.extend(
            op for op in ("decode", "generate_nosync")
            if hasattr(namespace, op)
        )

    available_ops_text = ", ".join(sorted(set(available_ops))) or "none"
    raise RuntimeError(
        f"qwen_megakernel_C op '{op_name}' is unavailable. "
        "The C++/CUDA extension is not loaded for this PyTorch build. "
        "Rebuild/install the megakernel extension against your current "
        f"torch version ({torch.__version__}). "
        f"Available extension ops: {available_ops_text}."
    )


def _assert_extension_compatibility() -> None:
    """Fail fast on stale extension builds that can crash with CUDA misalignment."""
    if qwen_megakernel_C is None:
        return

    ext_abi = None
    if hasattr(qwen_megakernel_C, "abi_version"):
        try:
            ext_abi = int(qwen_megakernel_C.abi_version())
        except Exception:
            ext_abi = None

    if ext_abi != EXTENSION_ABI_VERSION:
        built_torch = "unknown"
        if hasattr(qwen_megakernel_C, "built_torch_version"):
            try:
                built_torch = str(qwen_megakernel_C.built_torch_version())
            except Exception:
                built_torch = "unknown"

        raise RuntimeError(
            "Incompatible qwen_megakernel_C extension binary detected. "
            f"Expected ABI {EXTENSION_ABI_VERSION}, got {ext_abi}. "
            "This usually means Python code and CUDA extension are out-of-sync "
            "(e.g. pointer packing changed) and can trigger CUDA 'misaligned address'. "
            f"Rebuild/reinstall qwen_megakernel_C for torch {torch.__version__} "
            f"(extension was built against {built_torch})."
        )


_assert_extension_compatibility()
_decode = _require_megakernel_op("decode")


def load_weights(model_name="Qwen/Qwen3-0.6B", verbose: bool = True):
    """Load Qwen3-0.6B weights from HuggingFace into GPU tensors."""
    if not verbose:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass
        try:
            from huggingface_hub import logging as hf_hub_logging
            hf_hub_logging.set_verbosity_error()
        except Exception:
            pass

    if verbose:
        print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    # RoPE tables
    # Compute inv_freq in float32 for numerical accuracy, then build the
    # cos/sin table in float32 before casting to float16.
    # Using float16 throughout causes large errors for theta=1e6 because very
    # small inv_freq values (~ 1e-6) underflow to zero in fp16, making all
    # high-frequency dimensions identical and destroying positional information.
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)          # [MAX_SEQ_LEN, HEAD_DIM//2]
    # repeat(1, 2) → [MAX_SEQ_LEN, HEAD_DIM]: dims i and i+HEAD_DIM//2 share the
    # same frequency, matching HuggingFace's rotate_half convention.
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.float16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.float16).cuda().contiguous()

    # Per-layer weights (11 tensors per layer, flattened)
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        layer_weights.extend([
            state[p + "input_layernorm.weight"].contiguous(),
            state[p + "self_attn.q_proj.weight"].contiguous(),
            state[p + "self_attn.k_proj.weight"].contiguous(),
            state[p + "self_attn.v_proj.weight"].contiguous(),
            state[p + "self_attn.q_norm.weight"].contiguous(),
            state[p + "self_attn.k_norm.weight"].contiguous(),
            state[p + "self_attn.o_proj.weight"].contiguous(),
            state[p + "post_attention_layernorm.weight"].contiguous(),
            state[p + "mlp.gate_proj.weight"].contiguous(),
            state[p + "mlp.up_proj.weight"].contiguous(),
            state[p + "mlp.down_proj.weight"].contiguous(),
        ])

    embed_weight = state["model.embed_tokens.weight"].contiguous()
    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=state["model.norm.weight"].contiguous(),
        lm_head_weight=embed_weight,  # tied embeddings
        cos_table=cos_table,
        sin_table=sin_table,
    )

    del model
    torch.cuda.empty_cache()
    return weights, tokenizer


def _pack_layer_weights(layer_weights: list) -> torch.Tensor:
    N = 11  # weights per layer — must match LDGLayerWeight field order in megakernel.cu:
            # [0] input_layernorm, [1] q_proj, [2] k_proj, [3] v_proj,
            # [4] q_norm,          [5] k_norm, [6] o_proj, [7] post_attn_norm,
            # [8] gate_proj,       [9] up_proj,[10] down_proj
    n_layers = len(layer_weights) // N
    assert len(layer_weights) == n_layers * N, (
        f"Expected {n_layers * N} weight tensors ({N} per layer × {n_layers} layers), "
        f"got {len(layer_weights)}. Check that load_weights() appends exactly {N} "
        "tensors per layer in the same order as the LDGLayerWeight struct in megakernel.cu."
    )
    all_ptrs = []
    for li in range(n_layers):
        base = li * N
        for j in range(N):
            all_ptrs.append(layer_weights[base + j].data_ptr())
        all_ptrs.append(0)  # padding: struct is 12 pointers (96 bytes) on C++ side

    t = torch.zeros(len(all_ptrs) + 2, dtype=torch.int64, device="cuda")
    base_ptr = t.data_ptr()
    offset = (16 - (base_ptr % 16)) % 16
    offset_elems = offset // 8
    t_aligned = t[offset_elems : offset_elems + len(all_ptrs)]
    t_aligned.copy_(torch.tensor(all_ptrs, dtype=torch.int64))
    assert t_aligned.data_ptr() % 16 == 0, "Alignment failed!"
    return t_aligned


class Decoder:
    """Stateful decoder wrapping the Qwen megakernel ops."""

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3-0.6B",
        verbose: bool = True,
    ):
        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)

        self.tokenizer = tokenizer
        self._position = 0
        self._weights = weights  # keep alive to prevent GC of weight memory

        self._embed_weight        = weights["embed_weight"]
        self._final_norm_weight   = weights["final_norm_weight"]
        self._lm_head_weight      = weights["lm_head_weight"]
        self._cos_table           = weights["cos_table"]
        self._sin_table           = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
        self._check_weight_alignment(weights["layer_weights"])
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._attn_scale          = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.float16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers
        f16 = dict(dtype=torch.float16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32,   device="cuda")
        self._hidden    = torch.empty(HIDDEN_SIZE,        **f16)
        # act/mlp_inter/norm_out are float32 (used as float* in kernel)
        self._act    = torch.empty(Q_SIZE,            **f32)  # 2048 floats - o_proj input
        self._res    = torch.empty(HIDDEN_SIZE,       **f32)  # 1024 floats - residual (float32 not float16!)
        self._q         = torch.empty(Q_SIZE,             **f16)
        self._k         = torch.empty(KV_SIZE,            **f16)
        self._v         = torch.empty(KV_SIZE,            **f16)
        self._attn_out  = torch.empty(Q_SIZE,             **f16)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE,  **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE,        **f32)
        self._fmax_vals = torch.empty(4096,               **f32)
        self._fmax_idxs = torch.empty(4096,               **i32)
        self._out_token = torch.empty(1,                  **i32)
        
    def _check_weight_alignment(self, layer_weights_list):
        N = 11
        names = [
            "input_layernorm", "q_proj", "k_proj", "v_proj",
            "q_norm", "k_norm", "o_proj", "post_attn_norm",
            "gate_proj", "up_proj", "down_proj"
        ]
        bad = []
        for li in range(NUM_LAYERS):
            for j in range(N):
                ptr = layer_weights_list[li * N + j].data_ptr()
                if ptr % 16 != 0:
                    bad.append(f"  Layer {li} [{names[j]}]: ptr={ptr:#x}, rem={ptr%16}")
        if bad:
            print("MISALIGNED WEIGHT TENSORS:")
            for b in bad:
                print(b)
        else:
            print("All weight pointers 16-byte aligned OK")
        
        

    def step(self, input_token_id: int | torch.Tensor) -> int:
        buffers = {
        "hidden":    self._hidden,
        "act":       self._act,
        "res":       self._res,
        "q":         self._q,
        "k":         self._k,
        "v":         self._v,
        "attn_out":  self._attn_out,
        "mlp_inter": self._mlp_inter,
        "norm_out":  self._norm_out,
        "fmax_vals": self._fmax_vals,
        "fmax_idxs": self._fmax_idxs,
        "k_cache":   self._k_cache,
        "v_cache":   self._v_cache,
    }
        for name, buf in buffers.items():
            ptr = buf.data_ptr()
            if ptr % 16 != 0:
                print(f"MISALIGNED SCRATCH: {name} ptr={ptr:#x} rem={ptr%16}")

        # result_tensor = _decode(...)
        # ptr = self._layer_weights_packed.data_ptr()
        # assert ptr % 16 == 0, f"Misaligned! ptr={ptr:#x}, remainder={ptr % 16}"

        result_tensor = _decode(
            input_token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._fmax_vals,
            self._fmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        torch.cuda.synchronize()
        self._out_token.copy_(result_tensor)
        self._position += 1
        return int(result_tensor.item())
    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)

        _gen = _require_megakernel_op("generate_nosync")
        output_ids = _gen(
            ids[-1],
            max_tokens,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._fmax_vals,
            self._fmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += max_tokens
        out = output_ids.cpu().tolist()
        eos = self.tokenizer.eos_token_id
        if eos in out:
            out = out[: out.index(eos)]
        return self.tokenizer.decode(out, skip_special_tokens=True)


def generate(prompt: str, max_tokens: int = 50, verbose: bool = True) -> str:
    """One-shot convenience: load model, generate, return text."""
    return Decoder(verbose=verbose).generate(prompt, max_tokens)