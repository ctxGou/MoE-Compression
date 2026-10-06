"""Throughput / VRAM benchmark for Mixtral plugins (naive vs SLBF gauge-fixed).

Runs two scenarios per model:
  prefill:  prompt_len=4096, output_len=1   — input-dominated (TTFT-like)
  decode:   prompt_len=128,  output_len=256 — generation-dominated

VRAM reporting captures vLLM's own log lines (the meaningful numbers):
  - "Model loading took X GiB memory"           → weights per PP rank
  - "Available KV cache memory: X GiB"          → KV cache budget per PP rank
  - "GPU KV cache size: X tokens"               → KV cache capacity (model total)
  - "Maximum concurrency for X tokens per req"  → max concurrent at max_model_len

A redirected log file is also tee'd so we can parse it. The pynvml peak is
informational only (it just tracks the gpu_memory_utilization preallocation).

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/bench/mixtral_throughput.py \\
        --label slbf_k832_gf \\
        --model results/proxy_new/mixtral/slbf_k832_m8_g_u_gf_inference_bf16 \\
        --out results/bench/mixtral_throughput.csv
"""
import argparse, csv, io, os, re, sys, time, random, threading
import contextlib
import pynvml
from vllm import LLM, SamplingParams


class MultiGpuMemPoller:
    """Polls memory.used on multiple physical GPUs in a background thread."""

    def __init__(self, gpu_indices: list[int], interval: float = 0.2):
        self.gpu_indices = gpu_indices
        self.interval = interval
        self.peak_bytes = {g: 0 for g in gpu_indices}
        self._stop = threading.Event()
        self._thread = None
        pynvml.nvmlInit()
        self._handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in gpu_indices}

    def _query(self):
        return {g: pynvml.nvmlDeviceGetMemoryInfo(h).used for g, h in self._handles.items()}

    def _run(self):
        while not self._stop.is_set():
            try:
                used = self._query()
                for g, b in used.items():
                    if b > self.peak_bytes[g]:
                        self.peak_bytes[g] = b
            except Exception:
                pass
            self._stop.wait(self.interval)

    def reset(self):
        try:
            self.peak_bytes = self._query()
        except Exception:
            self.peak_bytes = {g: 0 for g in self.gpu_indices}

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def per_gpu_gb(self) -> dict[int, float]:
        return {g: b / (1024 ** 3) for g, b in self.peak_bytes.items()}

    @property
    def total_gb(self) -> float:
        return sum(self.peak_bytes.values()) / (1024 ** 3)


def make_random_prompts(num: int, length: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(64, vocab_size - 1) for _ in range(length)] for _ in range(num)]


SCENARIO_SHAPES = {
    "prefill": dict(prompt_len=4096, output_len=1),
    "decode":  dict(prompt_len=128,  output_len=256),
}

# Concurrency sweep: 16, 32, 64 fit within naive's KV cap (~75 @ 4416 tok);
# 80, 96 should queue/error on naive but fit SLBF's cap (~115).
DEFAULT_CONCS = [16, 32, 64, 80, 96]


def run_scenario(llm, vocab_size, name, prompt_len, output_len, n_prompts,
                  poller, seed=12345, do_warmup=True):
    prompts = make_random_prompts(n_prompts, prompt_len, vocab_size, seed=seed)
    prompts_dicts = [{"prompt_token_ids": p} for p in prompts]
    sampling = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)

    if do_warmup:
        warmup = make_random_prompts(2, prompt_len, vocab_size, seed=999)
        warmup_dicts = [{"prompt_token_ids": p} for p in warmup]
        _ = llm.generate(prompts=warmup_dicts, sampling_params=sampling, use_tqdm=False)

    poller.reset()
    t0 = time.time()
    try:
        outputs = llm.generate(prompts=prompts_dicts, sampling_params=sampling, use_tqdm=False)
        wall = time.time() - t0
        status = "ok"
        total_input = sum(len(p) for p in prompts)
        total_output = sum(len(o.outputs[0].token_ids) for o in outputs)
    except Exception as e:
        wall = time.time() - t0
        status = f"error: {type(e).__name__}: {str(e)[:100]}"
        total_input = 0
        total_output = 0
        print(f"  ERROR: {status}", flush=True)
    peak_gb = poller.total_gb
    per_gpu = poller.per_gpu_gb
    return {
        "scenario": name,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "n_prompts": n_prompts,
        "wall_seconds": round(wall, 3),
        "status": status,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "input_tput_t_per_s": round(total_input / wall, 1) if wall > 0 else 0.0,
        "output_tput_t_per_s": round(total_output / wall, 1) if (wall > 0 and total_output > 0) else 0.0,
        "peak_vram_total_gb": round(peak_gb, 3),
        "peak_vram_per_gpu_gb": ",".join(f"{g}:{v:.2f}" for g, v in sorted(per_gpu.items())),
    }


def parse_vllm_load_log(log_path: str) -> dict:
    """Extract weights / KV cache metrics from a saved vLLM log.

    Returns dict with keys: weights_gib_per_rank, kv_cache_gib_per_rank,
    kv_cache_tokens, max_concurrency. Any missing key is set to None.
    """
    rx_weights = re.compile(r"Model loading took ([\d.]+) GiB memory")
    rx_kv      = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
    rx_kvtok   = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
    rx_conc    = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
    weights, kv, kv_tok, conc, max_seq = [], [], None, None, None
    try:
        with open(log_path) as f:
            for line in f:
                if (m := rx_weights.search(line)): weights.append(float(m.group(1)))
                elif (m := rx_kv.search(line)):   kv.append(float(m.group(1)))
                elif (m := rx_kvtok.search(line)):
                    kv_tok = int(m.group(1).replace(",", ""))
                elif (m := rx_conc.search(line)):
                    max_seq = int(m.group(1).replace(",", ""))
                    conc = float(m.group(2))
    except FileNotFoundError:
        pass
    return {
        "weights_gib_per_rank": weights[0] if weights else None,
        "kv_cache_gib_per_rank": kv[0] if kv else None,
        "kv_cache_tokens": kv_tok,
        "max_concurrency_at_max_seq": conc,
        "max_seq_for_conc": max_seq,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--scenarios", default="prefill,decode")
    ap.add_argument("--concurrencies", default=",".join(str(c) for c in DEFAULT_CONCS),
                    help="Comma-separated concurrencies (= n_prompts per scenario).")
    ap.add_argument("--load_log", default=None,
                    help="Path to the tee'd vLLM stdout log to parse for "
                         "weights/KV-cache metrics. If unset, the script "
                         "attempts to read from $VLLM_LOG_PATH.")
    args = ap.parse_args()

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_phys = [int(x) for x in cuda_visible.split(",")]
    print(f"=== Loading {args.label} ===", flush=True)
    print(f"  model:  {args.model}", flush=True)
    print(f"  GPUs:   {gpu_phys}  (PP={args.pp}, TP=1)", flush=True)

    poller = MultiGpuMemPoller(gpu_phys, interval=0.2)
    poller.start()
    poller.reset()

    t_load = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        pipeline_parallel_size=args.pp,
        tensor_parallel_size=1,
        max_model_len=4096 + 256 + 64,
        enforce_eager=True,
        enable_prefix_caching=False,  # avoid cross-run prefix hits in the conc sweep
    )
    load_wall = time.time() - t_load
    peak_after_load = poller.total_gb
    print(f"  load_wall = {load_wall:.1f}s   peak_after_load = {peak_after_load:.2f} GB total", flush=True)
    print(f"  per-gpu after load: {poller.per_gpu_gb}", flush=True)

    hf_config = llm.llm_engine.vllm_config.model_config.hf_config
    vocab_size = hf_config.vocab_size
    print(f"  vocab_size = {vocab_size}", flush=True)

    concs = [int(x) for x in args.concurrencies.split(",")]
    print(f"  concurrencies = {concs}", flush=True)

    rows = []
    for scen_name in args.scenarios.split(","):
        scen_name = scen_name.strip()
        if scen_name not in SCENARIO_SHAPES:
            print(f"  skip unknown scenario {scen_name}", flush=True)
            continue
        shape = SCENARIO_SHAPES[scen_name]
        warmup_done = False
        for conc in concs:
            print(f"\n--- scenario={scen_name}  conc={conc}  shape={shape} ---", flush=True)
            r = run_scenario(
                llm, vocab_size, scen_name,
                prompt_len=shape["prompt_len"], output_len=shape["output_len"],
                n_prompts=conc, poller=poller,
                do_warmup=not warmup_done,
            )
            warmup_done = True
            r["label"] = args.label
            r["pp"] = args.pp
            r["gpu_indices"] = ",".join(str(g) for g in gpu_phys)
            r["load_wall_s"] = round(load_wall, 2)
            r["peak_vram_after_load_gb"] = round(peak_after_load, 3)
            rows.append(r)
            print(f"  status={r['status']}  wall={r['wall_seconds']}s  "
                  f"in_tput={r['input_tput_t_per_s']}/s  out_tput={r['output_tput_t_per_s']}/s  "
                  f"peak={r['peak_vram_total_gb']:.2f}GB", flush=True)

    poller.stop()

    # Parse the vLLM stdout log for the real weights / KV-cache numbers.
    load_log = args.load_log or os.environ.get("VLLM_LOG_PATH")
    log_metrics = parse_vllm_load_log(load_log) if load_log else {
        "weights_gib_per_rank": None, "kv_cache_gib_per_rank": None,
        "kv_cache_tokens": None, "max_concurrency_at_max_seq": None,
        "max_seq_for_conc": None,
    }
    print("\nParsed vLLM load metrics:", log_metrics, flush=True)
    for r in rows:
        r.update(log_metrics)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    fieldnames = ["label", "scenario", "prompt_len", "output_len", "n_prompts", "pp",
                  "gpu_indices", "wall_seconds", "status", "input_tokens", "output_tokens",
                  "input_tput_t_per_s", "output_tput_t_per_s",
                  "weights_gib_per_rank", "kv_cache_gib_per_rank",
                  "kv_cache_tokens", "max_concurrency_at_max_seq", "max_seq_for_conc",
                  "peak_vram_total_gb", "peak_vram_per_gpu_gb",
                  "peak_vram_after_load_gb", "load_wall_s"]
    with open(args.out, "a" if not write_header else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
