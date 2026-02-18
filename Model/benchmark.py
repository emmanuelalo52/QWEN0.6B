"""Benchmark: Qwen megakernel vs PyTorch HuggingFace baseline."""

import gc
import time
import warnings

import torch

warnings.filterwarnings("ignore")

# Long prompt generation (≈1000 tokens)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
sentence = "The quick brown fox jumps over the lazy dog. "
target_tokens = 1000

prompt_long = ""
tokens = []
while len(tokens) < target_tokens:
    prompt_long += sentence
    tokens = tokenizer.encode(prompt_long)
# Trim to exactly target_tokens
prompt_long = tokenizer.decode(tokens[:target_tokens], skip_special_tokens=True)
print(f"Long prompt generated with {len(tokenizer.encode(prompt_long))} tokens.")


TOKENS = 1000
WARMUP = 3
RUNS = 5
PROMPT = "Hello"                # short prompt for correctness check
LONG_PROMPT = prompt_long       # long prompt for performance benchmarks
CHECK_TOKENS = 8
RUN_CORRECTNESS = True


def megakernel_available() -> tuple[bool, str]:
    try:
        import qwen_megakernel_C
        missing = [op for op in ("decode", "generate_nosync") if not hasattr(qwen_megakernel_C, op)]
        if missing:
            return False, f"missing ops: {', '.join(missing)}"
        return True, "ok"
    except ImportError as e:
        return False, str(e)


def bench_pytorch_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    input_ids = tokenizer(LONG_PROMPT, return_tensors="pt").input_ids.cuda()

    def run():
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return TOKENS / avg, avg * 1000 / TOKENS


def bench_megakernel():
    from Qwen06B_architecture import Decoder

    dec = Decoder(verbose=False)

    def run():
        dec.reset()
        dec.generate(LONG_PROMPT, max_tokens=TOKENS)

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return TOKENS / avg, avg * 1000 / TOKENS


def bench_megakernel_prefill():
    """Measure prefill latency for the long prompt (no generation)."""
    from Qwen06B_architecture import Decoder

    dec = Decoder(verbose=False)
    ids = dec.tokenizer.encode(LONG_PROMPT)
    dec.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for tid in ids:
        dec.step(tid)   # includes the first generated token after last prompt token
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return t1 - t0, len(ids)


def correctness_check():
    import os
    import torch
    import traceback

    # Disable HF Hub progress bars and warnings for a cleaner output
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except AttributeError:
        pass

    # Load HF model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    from Qwen06B_architecture import Decoder

    # Initialize your custom Megakernel decoder with error catching
    print("Initializing Megakernel decoder...")
    try:
        dec = Decoder(weights=None, tokenizer=tokenizer, verbose=False)
    except Exception as e:
        print(f"❌ Decoder initialization failed: {e}")
        traceback.print_exc()
        return

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=CHECK_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    hf_ids = output[0, -CHECK_TOKENS:].tolist()
    print(f"HF tokens: {hf_ids}")

    with torch.no_grad():
        # Run a single forward pass with the prompt to get past_key_values
        outputs = model(input_ids, use_cache=True, past_key_values=None)
        past_kv = outputs.past_key_values
    # past_kv is a tuple of tuples: ( (k_layer0, v_layer0), (k_layer1, v_layer1), ... )
    first_k_hf = past_kv[0][0][0, 0, 0, :4].cpu().float().tolist()   # layer 0, head 0, pos 0
    first_v_hf = past_kv[0][1][0, 0, 0, :4].cpu().float().tolist()
    print(f"HF first token K cache (first 4 dims): {first_k_hf}")
    print(f"HF first token V cache (first 4 dims): {first_v_hf}")

    dec.reset()
    prompt_ids = input_ids[0].tolist()
    print(f"Prompt IDs: {prompt_ids}")

    # Prefill the prompt (except the last token)
    for i, tid in enumerate(prompt_ids[:-1]):
        print(f"Prefill step {i}: input {tid}")
        try:
            out = dec.step(tid)
            print(f"  → output {out} (should be ignored)")
        except Exception as e:
            print(f"❌ Error during prefill step {i}: {e}")
            traceback.print_exc()
            return
    print(f"K cache pointer (Python): {dec._k_cache.data_ptr()}")

    mk_ids = []
    tok = prompt_ids[-1]

    # Print Megakernel cache after prefilling (should correspond to HF's cache above)
    first_k_cache_entry = dec._k_cache[0, 0, 0, :4].cpu().float().tolist()
    first_v_cache_entry = dec._v_cache[0, 0, 0, :4].cpu().float().tolist()
    print(f"Megakernel first token K cache (first 4 dims): {first_k_cache_entry}")
    print(f"Megakernel first token V cache (first 4 dims): {first_v_cache_entry}")

    # Print the pointer address of the K cache tensor
    print(f"K cache pointer (Python): {dec._k_cache.data_ptr()}")

    print(f"Starting generation from token {tok}")
    for step_idx in range(CHECK_TOKENS):
        print(f"Generation step {step_idx}: input {tok}")
        try:
            tok = dec.step(tok)
            # After the first generation step, check the cache content
            if step_idx == 0:
                cache_after_first = dec._k_cache[0, 0, 0, :4].cpu().float().tolist()
                print(f"After step 0, K cache (first 4): {cache_after_first}")
            mk_ids.append(tok)
            print(f"  → output {tok}")
        except Exception as e:
            print(f"❌ Error during generation step {step_idx}: {e}")
            traceback.print_exc()
            return

    # 4. Compare Results
    matches = sum(1 for h, m in zip(hf_ids, mk_ids) if h == m)
    accuracy = (matches / CHECK_TOKENS) * 100

    print("-" * 30)
    print(f"CORRECTNESS REPORT ({CHECK_TOKENS} tokens)")
    print("-" * 30)
    print(f"HF tokens: {hf_ids}")
    print(f"MK tokens: {mk_ids}")
    print(f"Accuracy:  {accuracy:.2f}% ({matches}/{CHECK_TOKENS} tokens matched)")

    if accuracy == 100:
        print("\n✅ SUCCESS: Megakernel output matches Hugging Face exactly.")
    else:
        print("\n❌ FAILURE: Mismatch detected. Check weight loading or kernel logic.")
        print(f"HF text: '{tokenizer.decode(hf_ids)}'")
        print(f"MK text: '{tokenizer.decode(mk_ids)}'")


if __name__ == "__main__":
    print("=" * 55)
    print("Qwen Megakernel Benchmark")
    print("=" * 55)
    print()

    print("Megakernel")
    available, reason = megakernel_available()
    if not available:
        print(f"Megakernel extension unavailable: {reason}")
        print("Rebuild/install qwen_megakernel_C for this torch version and rerun.")
    else:
        if RUN_CORRECTNESS:
            correctness_check()
            print()

        # Generation benchmark (long prompt + 1000 new tokens)
        mk_tok, mk_ms = bench_megakernel()
        print()
        print("=" * 55)
        print("Generation performance (prefill + 1000 generated tokens)")
        print("-" * 55)
        print(f"{'Megakernel':<25} {mk_tok:>8.1f} tok/s  {mk_ms:>8.2f} ms/tok")

        # Prefill latency measurement (long prompt only)
        prefill_time, num_tokens = bench_megakernel_prefill()
        print()
        print("=" * 55)
        print(f"Prefill latency for {num_tokens} tokens")
        print("-" * 55)
        print(f"Total time: {prefill_time:.3f} s")
        print(f"Per token:  {prefill_time*1000/num_tokens:.2f} ms/tok")