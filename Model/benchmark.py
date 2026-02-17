"""Benchmark: Qwen megakernel vs PyTorch HuggingFace baseline."""

import gc
import time
import warnings

import torch

warnings.filterwarnings("ignore")

TOKENS = 100
WARMUP = 3
RUNS = 5
PROMPT = "Hello"
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
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()

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
        dec.generate(PROMPT, max_tokens=TOKENS)

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


def correctness_check():
    import os
    import torch

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

    # Initialize your custom Megakernel decoder
    dec = Decoder(weights=None, tokenizer=tokenizer, verbose=False)

    # 1. Generate Baseline tokens with Hugging Face
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=CHECK_TOKENS,
            do_sample=False,      # Greedy decoding for deterministic comparison
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Extract only the newly generated tokens
    hf_ids = output[0, -CHECK_TOKENS:].tolist()

    # 2. Generate tokens with Megakernel (MK)
    dec.reset()
    prompt_ids = input_ids[0].tolist()
    
    # Prefill the prompt (except the last token)
    for tid in prompt_ids[:-1]:
        dec.step(tid)
    
    mk_ids = []
    tok = prompt_ids[-1]
    # Generate tokens one by one
    for _ in range(CHECK_TOKENS):
        tok = dec.step(tok)
        mk_ids.append(tok)

    # 3. Compare Results
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
        # Debugging: Show decoded text to see how far they diverged
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
        mk_tok, mk_ms = bench_megakernel()

        print()
        print("=" * 55)
        print(f"{'Backend':<25} {'tok/s':>8} {'ms/tok':>8}")
        print("-" * 55)
        print(f"{'Megakernel':<25} {mk_tok:>8.1f} {mk_ms:>8.2f}")
