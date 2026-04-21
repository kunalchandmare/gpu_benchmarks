import time
import torch
import pandas as pd

def benchmark(device_name="cpu", n=4096, iters=10):
    device = torch.device(device_name)

    try:
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
    except RuntimeError as e:
        print(f"Could not allocate on {device_name}: {e}")
        return None

    _ = a @ b  # warmup

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iters
    return avg_ms

def main():
    results = []

    cpu_ms = benchmark("cpu")
    if cpu_ms is not None:
        results.append({"device": "cpu", "avg_ms": cpu_ms})
        print(f"CPU avg time: {cpu_ms:.4f} ms")

    if torch.cuda.is_available():
        gpu_ms = benchmark("cuda")
        if gpu_ms is not None:
            results.append({"device": "cuda", "avg_ms": gpu_ms})
            print(f"GPU avg time: {gpu_ms:.4f} ms")
    else:
        print("CUDA GPU not available.")

    if len(results) == 2:
        speedup = results[0]["avg_ms"] / results[1]["avg_ms"]
        print(f"Speedup (CPU/GPU): {speedup:.2f}x")

    df = pd.DataFrame(results)
    df.to_csv("cpu_gpu_benchmark.csv", index=False)
    print("\nSaved results to cpu_gpu_benchmark.csv")

if __name__ == "__main__":
    main()