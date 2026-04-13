"""Part 4: Benchmark

Measures:
1. Throughput (tokens/sec) vs number of concurrent requests
2. Batched scheduler vs one request at a time
"""

import time
import torch
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-0.6B")


def benchmark_batched(scheduler_class, sampling_params_class, num_requests: int, max_tokens: int = 50) -> float:
    """Run num_requests concurrently using the batched scheduler. Returns tokens/sec."""
    from nano_sglang.scheduler import Scheduler
    from nano_sglang.sampling import SamplingParams

    scheduler = Scheduler(MODEL_PATH)
    params = SamplingParams(temperature=0, max_tokens=max_tokens)

    prompts = [f"Write a short sentence about topic {i}" for i in range(num_requests)]
    for p in prompts:
        scheduler.add_request(p, params)

    start = time.time()
    results = scheduler.run_to_completion(params)
    elapsed = time.time() - start

    total_tokens = sum(len(scheduler.tokenizer.encode(r)) for r in results)
    throughput = total_tokens / elapsed
    return throughput, elapsed, total_tokens


def benchmark_sequential(engine_class, sampling_params_class, num_requests: int, max_tokens: int = 50) -> float:
    """Run num_requests one at a time sequentially. Returns tokens/sec."""
    from nano_sglang.engine import Engine
    from nano_sglang.sampling import SamplingParams

    engine = Engine(MODEL_PATH)
    params = SamplingParams(temperature=0, max_tokens=max_tokens)

    prompts = [f"Write a short sentence about topic {i}" for i in range(num_requests)]

    start = time.time()
    results = []
    for p in prompts:
        text = engine.generate(p, params)
        results.append(text)
    elapsed = time.time() - start

    total_tokens = sum(len(engine.tokenizer.encode(r)) for r in results)
    throughput = total_tokens / elapsed
    return throughput, elapsed, total_tokens


def run_benchmark():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run this on GPU via modal.")
        return

    print("=" * 60)
    print("Part 4: Benchmark Results")
    print("=" * 60)

    # ── Experiment 1: Throughput vs number of concurrent requests ──
    print("\n[Experiment 1] Batched Scheduler: Throughput vs Concurrent Requests")
    print(f"{'Requests':>10} {'Tokens':>10} {'Time(s)':>10} {'Tok/sec':>10}")
    print("-" * 45)

    batch_sizes = [1, 2, 4, 8, 16]
    batched_results = {}

    for n in batch_sizes:
        throughput, elapsed, total_tokens = benchmark_batched(None, None, num_requests=n)
        batched_results[n] = throughput
        print(f"{n:>10} {total_tokens:>10} {elapsed:>10.2f} {throughput:>10.1f}")

    # ── Experiment 2: Batched vs Sequential ──
    print("\n[Experiment 2] Batched vs Sequential (8 requests)")
    print(f"{'Method':>15} {'Tokens':>10} {'Time(s)':>10} {'Tok/sec':>10}")
    print("-" * 50)

    n = 8
    batched_tps, batched_time, batched_tokens = benchmark_batched(None, None, num_requests=n)
    print(f"{'Batched':>15} {batched_tokens:>10} {batched_time:>10.2f} {batched_tps:>10.1f}")

    seq_tps, seq_time, seq_tokens = benchmark_sequential(None, None, num_requests=n)
    print(f"{'Sequential':>15} {seq_tokens:>10} {seq_time:>10.2f} {seq_tps:>10.1f}")

    speedup = batched_tps / seq_tps
    print(f"\nSpeedup from batching: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()