"""Weight loading and high-level decode API for Qwen3-0.6B."""

import math
import torch
import os

try:
    import qwen_megakernel_C
except ImportError:
    qwen_megakernel_C = None

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM   # 2048
KV_SIZE = 8 * HEAD_DIM   # 1024
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 151936


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
        model_name, dtype=torch.float16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    # RoPE tables
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float16) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float16)
    freqs = torch.outer(positions, inv_freq)
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


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    ptrs: list[int] = []
    """Pack per-layer weight pointers to match LDGLayerWeight in CUDA.

    By default we emit 12 pointers/layer (11 real pointers + 1 padding slot)
    because `LDGLayerWeight` in `megakernel.cu` includes a trailing padding
    field for ABI-safe 16-byte alignment.

    Set `QWEN_MEGAKERNEL_PTRS_PER_LAYER=11` only as a temporary compatibility
    workaround when running against an older extension binary.
    """
    ptrs_per_layer = int(os.environ.get("QWEN_MEGAKERNEL_PTRS_PER_LAYER", "12"))
    if ptrs_per_layer not in (11, 12):
        raise ValueError(
            "QWEN_MEGAKERNEL_PTRS_PER_LAYER must be 11 or 12, "
            f"got {ptrs_per_layer}."
        )

    real_ptrs_per_layer = 11
    ptrs: list[int] = []
    for i in range(NUM_LAYERS):
        base = i * real_ptrs_per_layer
        for j in range(real_ptrs_per_layer):
            ptrs.append(layer_weights[base + j].data_ptr())

        if ptrs_per_layer == 11:
            # Extension ABI expects 11 fields/layer (legacy).
            continue

        # Extension ABI expects 12 fields/layer (current):
        # 11 real pointers + 1 trailing padding pointer.
        ptrs.append(0)
        
    return torch.tensor(ptrs, dtype=torch.int64, device="cuda").contiguous()


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
        # These buffers are consumed as float* inside the CUDA megakernel.
        self._act       = torch.empty(HIDDEN_SIZE,        **f32)
        self._res       = torch.empty(HIDDEN_SIZE,        **f32)
        self._q         = torch.empty(Q_SIZE,             **f16)
        self._k         = torch.empty(KV_SIZE,            **f16)
        self._v         = torch.empty(KV_SIZE,            **f16)
        self._attn_out  = torch.empty(Q_SIZE,             **f16)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE,  **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE,        **f32)
        self._fmax_vals = torch.empty(4096,               **f32)
        self._fmax_idxs = torch.empty(4096,               **i32)
        self._out_token = torch.empty(1,                  **i32)

    def step(self, token_id: int) -> int:
        """Decode one token. Returns the next token id."""
        try:
            _decode(
                self._out_token,
                token_id,
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
            # Surface kernel faults at the actual callsite; CUDA reports async
            # errors late by default which hides the true source.
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "misaligned address" in str(e).lower():
                raise RuntimeError(
                    "CUDA misaligned address during megakernel decode. "
                    "This usually indicates an ABI mismatch between Python "
                    "pointer packing and the compiled qwen_megakernel_C binary "
                    "(stale extension after source changes). Rebuild the CUDA "
                    "extension against the current source and torch build. "
                    "If you must run against an older binary, try "
                    "QWEN_MEGAKERNEL_PTRS_PER_LAYER=11 as a temporary fallback."
                ) from e
            raise
        self._position += 1
        return self._out_token.item()

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
