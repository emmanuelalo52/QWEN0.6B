# Qwen0.6B — Single-GPU Megakernel Inference Engine

A hand-written CUDA megakernel for running **Qwen3-0.6B** inference end-to-end on a single GPU, with a PyTorch C++ extension wrapper and a Python decode API. The design goal is to minimise host–device synchronisation and memory traffic by fusing all transformer operations (embedding lookup, RMSNorm, RoPE, multi-head attention with KV cache, SwiGLU MLP, and top-1 argmax for next-token selection) into a single persistent kernel.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [File Structure](#file-structure)
- [Model Configuration](#model-configuration)
- [Hardware Target](#hardware-target)
- [Installation](#installation)
- [Usage](#usage)
  - [Python API](#python-api)
  - [Benchmarking](#benchmarking)
- [Internals](#internals)
  - [megakernel.cu](#megakernelcu)
  - [qwen_ops.cpp](#qwen_opscpp)
  - [Qwen06B_architecture.py](#qwen06b_architecturepy)
  - [benchmark.py](#benchmarkpy)
- [Weight Layout](#weight-layout)
- [KV Cache](#kv-cache)
- [RoPE Implementation Notes](#rope-implementation-notes)
- [ABI Versioning](#abi-versioning)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   Python (host)                      │
│  Decoder.step(token_id)  /  Decoder.generate(prompt) │
└───────────────┬──────────────────────────────────────┘
                │  pybind11 C++ extension
┌───────────────▼──────────────────────────────────────┐
│              qwen_ops.cpp                            │
│   decode()  /  generate_nosync()                     │
│   · alignment guard (16-byte, GTX 1650)              │
│   · passes raw device pointers to CUDA               │
└───────────────┬──────────────────────────────────────┘
                │  cudaStream_t
┌───────────────▼──────────────────────────────────────┐
│              megakernel.cu                           │
│   launch_ldg_decode_direct()                         │
│   launch_ldg_generate_nosync()                       │
│                                                      │
│   Per token:                                         │
│   embed → [layerNorm → QKV proj → q/k norm →         │
│            RoPE → attn (KV cache) → o_proj →         │
│            layerNorm → gate/up → SiLU → down] × 28  │
│   → final norm → lm_head → argmax                   │
└──────────────────────────────────────────────────────┘
```

The `generate_nosync` path runs multiple decode steps without any host–device sync between them, returning all generated token IDs in a single GPU-side buffer. This eliminates Python overhead for each token.

---

## File Structure

```
Qwen0.6B/
├── megakernel.cu            # Core CUDA implementation (all transformer ops)
├── qwen_ops.cpp             # PyTorch C++ / pybind11 extension
├── Qwen06B_architecture.py  # Weight loading, packing, Python Decoder class
└── benchmark.py             # Correctness check + speed comparison vs HF
```

---

## Model Configuration

| Parameter           | Value                         |
|---------------------|-------------------------------|
| Model               | Qwen/Qwen3-0.6B               |
| Hidden size         | 1 024                         |
| Intermediate size   | 3 072                         |
| Number of layers    | 28                            |
| Query heads         | 16                            |
| KV heads            | 8 (GQA)                       |
| Head dimension      | 128                           |
| Q projection size   | 16 × 128 = 2 048              |
| KV projection size  | 8  × 128 = 1 024              |
| Vocabulary size     | 151 936                       |
| Max sequence length | 2 048                         |
| RoPE θ              | 1 000 000.0                   |
| Attention scale     | 1 / √128 ≈ 0.0884             |
| Weight dtype        | float16                       |
| Embeddings          | Tied (embed_tokens = lm_head) |

---

## Hardware Target

The primary development target is an **NVIDIA GTX 1650** (Turing, `sm_75`, 4 GB VRAM). Several guards in the code are tuned for this card.

- **16-byte pointer alignment** is enforced on the packed layer-weight table (`TORCH_CHECK(ptr % 16 == 0, …)`) because misaligned `LDG` instructions cause a `cudaErrorMisalignedAddress` on Turing and older architectures.
- The ABI version constant (`EXTENSION_ABI_VERSION = 2`) encodes the 12-pointer (96-byte) `LDGLayerWeight` struct layout. If this struct changes, bump the version and rebuild.
- The code should run on any Pascal or newer GPU with sufficient VRAM, but has only been validated on GTX 1650.

---

## Installation

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA (must match the GPU driver)
- CUDA Toolkit (matching the PyTorch build)
- `transformers` and `huggingface_hub` (for weight download and correctness check)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers huggingface_hub
```

### Building the C++ / CUDA Extension

```bash
python setup.py build_ext --inplace
# or with pip:
pip install -e .
```

The build produces `qwen_megakernel_C.so` (or `.pyd` on Windows). The Python layer imports it as `qwen_megakernel_C`.

> **Important:** Always rebuild after updating `megakernel.cu` or `qwen_ops.cpp`. Running a stale `.so` against updated Python code will trigger the ABI version check and raise a `RuntimeError` before any CUDA code executes.

### Verifying the Build

```python
import qwen_megakernel_C
print(qwen_megakernel_C.abi_version())   # should print 2
print(dir(qwen_megakernel_C))            # should include 'decode', 'generate_nosync'
```

---

## Usage

### Python API

#### One-shot generation

```python
from Qwen06B_architecture import generate

text = generate("The capital of France is", max_tokens=50)
print(text)
```

#### Stateful `Decoder` (recommended for benchmarking)

```python
from Qwen06B_architecture import Decoder

dec = Decoder(verbose=True)          # downloads weights on first run
text = dec.generate("Hello, world!", max_tokens=200)
print(text)

# Re-use without reloading weights:
dec.reset()
text2 = dec.generate("Once upon a time", max_tokens=100)
```

`Decoder.reset()` zeroes the KV cache and resets the position counter. Weights stay resident in GPU memory.

#### Token-by-token stepping

```python
dec = Decoder(verbose=False)
ids = dec.tokenizer.encode("Hello", add_special_tokens=True)

for tid in ids[:-1]:         # prefill
    dec.step(tid)

tok = ids[-1]
for _ in range(50):          # generate
    tok = dec.step(tok)
    print(dec.tokenizer.decode([tok]), end="", flush=True)
```

`Decoder.step(token_id)` runs one full forward pass through all 28 layers and returns the next token id as a Python `int`.

#### Low-level C++ ops (advanced)

```python
import qwen_megakernel_C

# Single decode step
output_tensor = qwen_megakernel_C.decode(
    input_token_id,
    embed_weight, layer_weights_packed, final_norm_weight, lm_head_weight,
    cos_table, sin_table, k_cache, v_cache,
    hidden, act, res, q, k, v, attn_out, mlp_inter, norm_out,
    bmax_vals, bmax_idxs,
    num_layers, position, max_seq_len, attn_scale,
)

# Multi-step generation without Python sync overhead
output_log = qwen_megakernel_C.generate_nosync(
    first_token_id, num_steps,
    embed_weight, layer_weights_packed, ...,
    num_layers, start_position, max_seq_len, attn_scale,
)
```

All tensor arguments must live on CUDA and be contiguous. The `layer_weights_packed` tensor must be 16-byte aligned (guaranteed by `_pack_layer_weights()`).

---

### Benchmarking

```bash
python benchmark.py
```

The script runs three measurements:

1. **Correctness check** — prefills a short prompt with both HuggingFace and the megakernel, then compares 8 generated tokens. Prints accuracy as a percentage.
2. **Generation benchmark** — measures tokens/second and ms/token for a ~1 000-token prefill followed by 1 000 generated tokens, averaged over 5 runs with 3 warmup runs.
3. **Prefill latency** — times how long the megakernel takes to process the ~1 000-token prompt alone.

To also compare against the HuggingFace baseline, call `bench_pytorch_hf()` directly from the script (it is defined but gated behind the megakernel availability check in `__main__`).

---

## Internals

### `megakernel.cu`

The heart of the project. Exports two C-linkage functions:

#### `launch_ldg_decode_direct`

Runs a single decode step for one input token at a given sequence position. All 28 transformer layers are processed sequentially by the kernel(s) launched here. The name `LDG` refers to the CUDA `LDG` (Load from Global memory) instruction family used for read-only weight access, which routes through the texture cache and avoids polluting the L1 data cache.

Argument summary:

| Argument         | Description                                          |
|------------------|------------------------------------------------------|
| `input_token_id` | Integer token to look up in the embedding table      |
| `output_token_id`| GPU pointer; kernel writes the next token id here    |
| `embed_weight`   | `float16` embedding matrix `[vocab, hidden]`         |
| `layer_weights`  | Pointer to packed `LDGLayerWeight[]` array           |
| `final_norm_weight` | RMSNorm weight for the output norm              |
| `lm_head_weight` | `float16` unembedding matrix (tied to embeddings)    |
| `cos_table / sin_table` | Precomputed RoPE tables `[max_seq, head_dim]` |
| `k_cache / v_cache` | KV cache `[layers, kv_heads, max_seq, head_dim]` |
| `hidden_buffer`  | Scratch: current hidden state `[hidden_size]`        |
| `g_activations`  | Scratch: Q-size float32 (o_proj accumulator)         |
| `g_residual`     | Scratch: float32 residual stream                     |
| `g_q/k/v`        | Scratch: projected query/key/value                   |
| `g_attn_out`     | Scratch: attention output                            |
| `g_mlp_intermediate` | Scratch: MLP intermediate activations          |
| `g_normalized`   | Scratch: float32 post-norm buffer                    |
| `block_max_vals/idxs` | Scratch: block-level reduce for argmax        |
| `num_layers`     | 28 for Qwen3-0.6B                                    |
| `position`       | Current sequence position (0-indexed)                |
| `max_seq_len`    | 2 048                                                |
| `attn_scale`     | 1/√head_dim                                          |

#### `launch_ldg_generate_nosync`

Wraps `launch_ldg_decode_direct` in a loop that runs for `num_steps` tokens without returning control to the host between steps. Generated token ids are written to a GPU-side `int[]` log. The result is transferred to the host in one shot after all steps complete. This is the fast path for bulk generation.

#### `LDGLayerWeight` struct

A 12-pointer (96-byte) struct holding device pointers to the 11 weight tensors for a single transformer layer:

```
[0]  input_layernorm.weight      float16 [hidden]
[1]  self_attn.q_proj.weight     float16 [q_size, hidden]
[2]  self_attn.k_proj.weight     float16 [kv_size, hidden]
[3]  self_attn.v_proj.weight     float16 [kv_size, hidden]
[4]  self_attn.q_norm.weight     float16 [head_dim]
[5]  self_attn.k_norm.weight     float16 [head_dim]
[6]  self_attn.o_proj.weight     float16 [hidden, q_size]
[7]  post_attention_layernorm.weight  float16 [hidden]
[8]  mlp.gate_proj.weight        float16 [intermediate, hidden]
[9]  mlp.up_proj.weight          float16 [intermediate, hidden]
[10] mlp.down_proj.weight        float16 [hidden, intermediate]
[11] (padding — zero)
```

The 12th pointer slot is zero-padding to keep each entry 96 bytes and 16-byte aligned in the array.

---

### `qwen_ops.cpp`

A thin PyTorch C++ extension that:

1. Receives PyTorch tensors from Python.
2. Enforces 16-byte alignment on the packed layer-weight tensor (required for GTX 1650 `LDG` instructions).
3. Extracts raw `void*` device pointers.
4. Passes them to the C-linkage functions in `megakernel.cu` along with the current CUDA stream (`c10::cuda::getCurrentCUDAStream()`).
5. Returns results as PyTorch tensors.

Exports via pybind11:

| Function           | Description                              |
|--------------------|------------------------------------------|
| `decode`           | Single token decode, returns `int32[1]`  |
| `generate_nosync`  | Multi-step decode, returns `int32[N]`    |
| `abi_version`      | Returns `2` (compile-time constant)      |

---

### `Qwen06B_architecture.py`

High-level Python layer. Key responsibilities:

**`load_weights(model_name)`** — Downloads Qwen3-0.6B from HuggingFace, extracts the 28-layer weight dictionary, and builds the precomputed RoPE cosine/sine tables. Returns a `dict` and a tokenizer.

**`_pack_layer_weights(layer_weights)`** — Converts a flat Python list of 308 tensors (11 per layer × 28 layers) into a GPU `int64` tensor of device pointers arranged as `LDGLayerWeight[]`. Applies an offset to ensure 16-byte alignment. The resulting tensor must remain alive for the lifetime of any inference call (it is stored on `self._weights`).

**`Decoder`** — Stateful wrapper class:
- Allocates and owns the KV cache and all scratch buffers.
- Exposes `step(token_id) -> int` for single-step inference.
- Exposes `generate(prompt, max_tokens) -> str` which uses `generate_nosync` for the bulk of generation.
- `reset()` zeros the KV cache and position counter without re-allocating.

**`_assert_extension_compatibility()`** — Called at module import time. Compares the `abi_version()` reported by the loaded `.so` against `EXTENSION_ABI_VERSION = 2`. Raises `RuntimeError` on mismatch so that stale builds fail loudly rather than silently corrupting memory.

---

### `benchmark.py`

Standalone script. Does not import from `Qwen06B_architecture` at module scope (imports happen inside functions), so it can be run even if the extension is absent — it will print an informative error.

Functions:

| Function                 | Description                                              |
|--------------------------|----------------------------------------------------------|
| `megakernel_available()` | Probes for `qwen_megakernel_C` without raising           |
| `correctness_check()`    | Compares megakernel vs HuggingFace token-by-token        |
| `bench_pytorch_hf()`     | Times HF `model.generate` over 5 runs                   |
| `bench_megakernel()`     | Times `Decoder.generate` over 5 runs                    |
| `bench_megakernel_prefill()` | Times prefill only (no new tokens generated)         |

---

## Weight Layout

Weights are loaded from the HuggingFace `state_dict` in float16 and kept on GPU. They are **not** quantized. The embedding and unembedding matrices are tied (share the same tensor pointer).

Memory footprint (approximate):

| Component             | Parameters | VRAM (fp16) |
|-----------------------|-----------|-------------|
| Embedding / lm_head   | 155.6 M   | 311 MB      |
| Per-layer (× 28)      | ~15.6 M   | ~31 MB each |
| KV cache (2048 ctx)   | —         | ~28 MB      |
| Scratch buffers       | —         | < 1 MB      |
| **Total**             | **~595 M** | **~1.2 GB** |

The model fits comfortably within 4 GB of VRAM.

---

## KV Cache

Shape: `[num_layers, num_kv_heads, max_seq_len, head_dim]` = `[28, 8, 2048, 128]` in float16.

Two separate tensors are allocated: `k_cache` and `v_cache`. At each decode step `position`, the kernel writes the new KV pair at index `[:, :, position, :]` and reads all positions `0..position` during attention. This is a simple non-paged cache — it does not support sequences longer than `MAX_SEQ_LEN = 2048`. Call `Decoder.reset()` between unrelated sequences.

---

## RoPE Implementation Notes

**`ROPE_THETA = 1_000_000.0`** (not the LLaMA default of 10 000.0). Using the wrong θ causes correct output only at position 0 (where `cos(0) = 1, sin(0) = 0`) and garbled output at all subsequent positions.

**Precision:** Inverse frequencies and the cos/sin tables are computed in `float32` before being cast to `float16` for storage. With θ = 1 000 000 the smallest inverse frequency is `θ^(-(HEAD_DIM-2)/HEAD_DIM) ≈ 6×10⁻⁶`, which underflows to zero in float16. Computing in float32 first avoids this.

**Convention:** The tables are built with `torch.cos(freqs).repeat(1, 2)` which replicates each frequency so that dimension `i` and dimension `i + HEAD_DIM/2` share the same frequency value. This matches HuggingFace's `rotate_half` RoPE convention used in Qwen3.

---

## ABI Versioning

Any change to the `LDGLayerWeight` struct layout (pointer count, padding, or field order) requires:

1. Incrementing `EXTENSION_ABI_VERSION` in `Qwen06B_architecture.py`.
2. Updating `abi_version()` in `qwen_ops.cpp` to return the same value.
3. Rebuilding the extension.

The version check runs at Python import time and raises a descriptive error before any kernel is launched.

Current version: **2** (12-pointer, 96-byte struct with one zero-padding slot).

---

## Benchmark Results

All results measured on a **GTX 1650 (4 GB VRAM)** running PyTorch 2.10.0+cu130. The megakernel achieves 100% token-level accuracy against the HuggingFace reference implementation across all runs.

### Correctness

The correctness check prefills the prompt `"Hello"` (token id `9707`) and compares 8 greedily-decoded tokens between the megakernel and HuggingFace `model.generate`. All runs pass with **100% accuracy**.

```
HF tokens: [21806, 0, 358, 2776, 264, 2699, 21815, 911]
MK tokens: [21806, 0, 358, 2776, 264, 2699, 21815, 911]
Accuracy:  100.00% (8/8 tokens matched)
✅ SUCCESS: Megakernel output matches Hugging Face exactly.
```

The KV cache values also match closely — HF reports `[1.2685, 1.3095, -2.5937, -1.8886]` for the first-token K cache and the megakernel writes `[1.2695, 1.3095, -2.5957, -1.8886]` (differences are float16 rounding, not a correctness issue).

### Generation Throughput (short prompt)

Measured with the default short prompt (`"Hello"`), 1 000 new tokens, 3 warmup runs, 5 timed runs:

| Run | tok/s | ms/tok |
|-----|------:|-------:|
| 1   | 130.3 |   7.68 |
| 2   | 111.7 |   8.95 |

Variance between runs reflects GPU clock boost state. Steady-state throughput is approximately **~110–130 tok/s** at this context length.

### Generation Throughput (long prompt — 1 000-token prefill + 1 000 generated tokens)

With a ~1 000-token prefill (the prompt is repeated sentences until the tokenizer produces exactly 1 000 tokens), the attention computation per step grows significantly because the kernel must attend over the full KV cache history:

| Run | tok/s | ms/tok |
|-----|------:|-------:|
| 1   | 38.1  |  26.25 |
| 2   | 38.0  |  26.29 |

Throughput is stable at **~38 tok/s** under long-context load. The ~3× slowdown vs. short-prompt is expected: attention cost scales linearly with sequence length and dominates at 1 000+ tokens on a card without tensor cores operating in WMMA mode.

### Prefill Latency (1 000-token prompt, no generation)

Processing the full 1 000-token prompt via sequential `Decoder.step()` calls:

| Metric     | Value        |
|------------|-------------|
| Total time | 11.48 s     |
| Per token  | 11.48 ms/tok|

Prefill runs each token through all 28 layers with a growing attention window (position 0 attends over 1 token, position 999 attends over 1 000 tokens). The per-token cost therefore increases throughout prefill; the reported figure is the amortised average. A parallel prefill kernel (processing all prompt tokens simultaneously) would reduce this substantially and is a natural next step.

### Summary

| Scenario                         | tok/s  | ms/tok |
|----------------------------------|-------:|-------:|
| Short prompt generation          | ~120   |  ~8.3  |
| Long prompt generation (1k ctx)  | ~38    | ~26.3  |
| Prefill (1k tokens, sequential)  | ~87    | ~11.5  |

---

## Known Limitations

- **No batching.** The kernel processes exactly one token per call. Batch size is always 1.
- **No quantization.** Weights are stored and computed in float16. INT8/INT4 is not implemented.
- **No paged attention.** The KV cache is a fixed-size contiguous buffer. Sequences exceeding `MAX_SEQ_LEN = 2048` tokens will corrupt the cache.
- **No beam search.** Only greedy decoding (top-1 argmax) is implemented.
- **No sampling.** Temperature, top-p, and top-k are not available.
- **GTX 1650 only validated.** The 16-byte alignment requirement and block sizes are tuned for `sm_75`. Other architectures should work but have not been tested.
- **Linux / POSIX assumed.** The build system and paths are not tested on Windows.

---

## License

See `LICENSE` in the repository root.
