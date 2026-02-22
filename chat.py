"""
chat.py — Interactive Q&A terminal using the Qwen3-0.6B megakernel decoder.

Usage:
    python chat.py
    python chat.py --max_tokens 200
    python chat.py --system "You are a helpful coding assistant."

Commands inside the chat:
    /reset   — wipe KV cache and start a fresh conversation
    /tokens  — show current token position counter
    /quit    — exit
"""

import argparse
import sys
import torch

# ── Arg parsing ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Qwen3-0.6B megakernel chat")
parser.add_argument("--max_tokens", type=int, default=150,
                    help="Max new tokens to generate per reply (default: 150)")
parser.add_argument("--system", type=str,
                    default="You are a helpful assistant.",
                    help="System prompt injected before the first user turn")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                    help="HuggingFace model name/path")
args = parser.parse_args()


# ── Load model ───────────────────────────────────────────────────────────────
print("Loading Qwen3-0.6B megakernel decoder…")
try:
    from Model.Qwen06B_architecture import Decoder
except ImportError as e:
    sys.exit(f"[ERROR] Could not import Qwen06B_architecture: {e}\n"
             "Make sure qwen_megakernel_C is built and on PYTHONPATH.")

decoder = Decoder(model_name=args.model, verbose=True)
tokenizer = decoder.tokenizer
print("Model ready.\n")


# ── Chat helpers ──────────────────────────────────────────────────────────────
def format_prompt(system: str, history: list[dict], new_question: str) -> str:
    """
    Build a simple Qwen3-style chat prompt.
    Qwen3 uses <|im_start|> / <|im_end|> chat tokens.
    """
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    for turn in history:
        parts.append(f"<|im_start|>user\n{turn['user']}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>")
    parts.append(f"<|im_start|>user\n{new_question}<|im_end|>")
    parts.append("<|im_start|>assistant\n")   # prompt the model to continue
    return "\n".join(parts)


def chat_once(question: str, history: list[dict]) -> str:
    """Run one Q→A turn; returns the assistant reply string."""
    prompt = format_prompt(args.system, history, question)
    reply = decoder.generate(prompt, max_tokens=args.max_tokens)
    # Strip anything the model appended after its own <|im_end|>
    if "<|im_end|>" in reply:
        reply = reply[: reply.index("<|im_end|>")].strip()
    return reply.strip()


# ── REPL ──────────────────────────────────────────────────────────────────────
history: list[dict] = []

print("=" * 60)
print(f" Qwen3-0.6B Megakernel Chat  (max_tokens={args.max_tokens})")
print("=" * 60)
print(" Commands: /reset  /tokens  /quit")
print("-" * 60)

try:
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ── Built-in commands ──
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye!")
            break

        if user_input.lower() == "/reset":
            decoder.reset()
            history.clear()
            print("[KV cache cleared — conversation reset]")
            continue

        if user_input.lower() == "/tokens":
            print(f"[Current position: {decoder.position} tokens]")
            continue

        # ── Normal generation ──
        print("\nAssistant: ", end="", flush=True)
        try:
            reply = chat_once(user_input, history)
            print(reply)
            history.append({"user": user_input, "assistant": reply})
        except RuntimeError as e:
            print(f"\n[KERNEL ERROR] {e}")
            print("Try /reset to clear the KV cache and retry.")

except Exception as e:
    print(f"\n[Fatal] {e}")
    raise
