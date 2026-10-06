"""Quick generation smoke test for a Mixtral plugin variant.

Loads the model and generates from a handful of real prompts to verify the
output is sensible English (not gibberish, not all-special-tokens, etc.).

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/bench/mixtral_smoke_test.py \\
        --model results/proxy_new/mixtral/slbf_k832_m8_g_u_gf_inference_bf16
"""
import argparse
from vllm import LLM, SamplingParams


PROMPTS = [
    "The capital of France is",
    "In a hole in the ground there lived",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Translate to French: 'How are you today?' →",
    "Q: Who wrote the play Hamlet?\nA:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        pipeline_parallel_size=args.pp,
        tensor_parallel_size=1,
        max_model_len=512,
        enforce_eager=True,
    )

    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )
    outs = llm.generate(prompts=PROMPTS, sampling_params=sp, use_tqdm=False)

    print("\n" + "=" * 70)
    print(f"SMOKE TEST: {args.model}")
    print("=" * 70)
    for o in outs:
        prompt = o.prompt
        gen = o.outputs[0].text
        print(f"\n>>> {prompt!r}")
        print(f"    {gen!r}")


if __name__ == "__main__":
    main()
