"""
profile_megakernel.py — Profile the Qwen3-0.6B megakernel with Nsight.

This script inserts NVTX ranges around every logical phase so that
Nsight Systems shows a clean, labeled timeline:

    ┌─────────────────────────────────────────────────────┐
    │  NVTX Ranges visible in Nsight Systems              │
    ├─────────────────────────────────────────────────────┤
    │  [Warmup]                                           │
    │  [Prefill]  → one range per prefill token           │
    │    └─ [Prefill token 0]                             │
    │    └─ [Prefill token 1]  ...                        │
    │  [Generation]                                       │
    │    └─ [Decode step 0]                               │
    │    └─ [Decode step 1]  ...                          │
    │  [LM Head] — automatically inside each decode       │
    └─────────────────────────────────────────────────────┘

How to run under Nsight Systems (WSL or Windows):
─────────────────────────────────────────────────
  # From WSL terminal:
  nsys profile \
      --output=qwen_megakernel \
      --trace=cuda,nvtx \
      --capture-range=nvtx \
      --nvtx-capture="Profile" \
      python profile_megakernel.py

  # Then open qwen_megakernel.nsys-rep in Nsight Systems GUI on Windows.

How to run under Nsight Compute (single-kernel deep dive):
───────────────────────────────────────────────────────────
  ncu --target-processes all \
      --kernel-name ldg_decode_kernel_persistent \
      --launch-count 1 \
      --set full \
      --export qwen_ncu \
      python profile_megakernel.py --steps 1

  # Open qwen_ncu.ncu-rep in Nsight Compute GUI.

Script options:
  --prompt      Prompt string for generation     (default: "What is a neural network?")
  --prefill     Number of prefill tokens         (default: 10)
  --steps       Number of decode steps to trace  (default: 20)
  --warmup      Warmup decode steps before range (default: 5)
  --model       HuggingFace model name/path
"""

import ctypes
_libcuda = ctypes.CDLL("libcuda.so")
import argparse
import sys
import torch

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Nsight profiling for Qwen megakernel")
parser.add_argument("--prompt",   type=str, default="What is a neural network?")
parser.add_argument("--prefill",  type=int, default=10,
                    help="Max prefill tokens (truncates prompt if longer)")
parser.add_argument("--steps",    type=int, default=20,
                    help="Number of decode steps to capture in Nsight range")
parser.add_argument("--warmup",   type=int, default=5,
                    help="Decode steps to run before opening the Nsight capture range")
parser.add_argument("--model",    type=str, default="Qwen/Qwen3-0.6B")
args = parser.parse_args()

# ── NVTX availability ─────────────────────────────────────────────────────────
try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False

def nvtx_range(name: str):
    """Context manager: push/pop NVTX range (no-op if NVTX unavailable)."""
    class _NVTXCtx:
        def __enter__(self):
            if HAS_NVTX:
                nvtx.range_push(name)
        def __exit__(self, *_):
            if HAS_NVTX:
                nvtx.range_pop()
    return _NVTXCtx()

# ── Load decoder ──────────────────────────────────────────────────────────────
print("Loading Qwen3-0.6B megakernel decoder…")
try:
    from Model.Qwen06B_architecture import Decoder
except ImportError as e:
    sys.exit(f"[ERROR] {e}\nMake sure qwen_megakernel_C is built and on PYTHONPATH.")

decoder = Decoder(model_name=args.model, verbose=True)
tokenizer = decoder.tokenizer
print(f"Model ready. NVTX available: {HAS_NVTX}\n")

# ── Tokenise prompt ───────────────────────────────────────────────────────────
all_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
prefill_ids = all_ids[: args.prefill]          # trim to requested prefill length
print(f"Prompt  : '{args.prompt}'")
print(f"Prefill : {len(prefill_ids)} tokens  {prefill_ids}")
print(f"Warmup  : {args.warmup} decode steps  (outside Nsight capture range)")
print(f"Capture : {args.steps} decode steps   (inside 'Profile' NVTX range)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Warmup (outside any Nsight capture range so it doesn't pollute
#            the timeline with first-launch overheads)
# ─────────────────────────────────────────────────────────────────────────────
print("── Warmup ──────────────────────────────────────────────")
decoder.reset()

with nvtx_range("Warmup"):
    for _ in range(args.warmup):
        tok = all_ids[0] if all_ids else 0
        decoder.step(tok)

decoder.reset()
torch.cuda.synchronize()
print(f"Warmup complete ({args.warmup} steps).\n")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1+2 inside the top-level "Profile" NVTX range
# Nsight Systems --capture-range=nvtx --nvtx-capture="Profile" will record
# only the GPU activity within this range, giving a tight, readable report.
# ─────────────────────────────────────────────────────────────────────────────
with nvtx_range("Profile"):

    # ── Prefill ──────────────────────────────────────────────────────────────
    print("── Prefill ─────────────────────────────────────────────")
    with nvtx_range("Prefill"):
        for i, tid in enumerate(prefill_ids):
            with nvtx_range(f"Prefill token {i}  id={tid}"):
                out = decoder.step(tid)
            print(f"  prefill[{i:>3}] input={tid:>6}  output={out:>6}")
    torch.cuda.synchronize()
    print()

    # ── Generation ───────────────────────────────────────────────────────────
    print("── Generation ──────────────────────────────────────────")
    tok = prefill_ids[-1] if prefill_ids else tokenizer.bos_token_id
    generated_ids = []
    with nvtx_range("Generation"):
        _libcuda.cuProfilerStart()  
        for step in range(args.steps):
            with nvtx_range(f"Decode step {step}"):
                tok = decoder.step(tok)
            generated_ids.append(tok)
            word = tokenizer.decode([tok], skip_special_tokens=False)
            print(f"  step[{step:>3}] token={tok:>6}  '{word}'")
            if tok == tokenizer.eos_token_id:
                print("  [EOS reached — stopping early]")
                break
        _libcuda.cuProfilerStop()

    torch.cuda.synchronize()

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("Profiling capture complete.")
text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Generated text: '{text}'")
print()
print("Next steps:")
print("  • Open the .nsys-rep in Nsight Systems to see per-kernel timelines.")
print("  • Look for  ldg_decode_kernel_persistent,  ldg_lm_head_phase1/2")
print("    under the CUDA row, labelled by their NVTX parent ranges.")
print("  • Use Nsight Compute (ncu) for warp occupancy, memory throughput,")
print("    and instruction-level stats on ldg_decode_kernel_persistent.")
