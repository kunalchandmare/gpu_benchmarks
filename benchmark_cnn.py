import subprocess
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def gpu_info():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}")
            print("  Name:", props.name)
            print("  Total memory (GB):", round(props.total_memory / 1024**3, 3))
            print("  Compute capability:", f"{props.major}.{props.minor}")

def query_nvidia_smi():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits"
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            rows.append({
                "index": int(parts[0]),
                "name": parts[1],
                "util_gpu": float(parts[2]),
                "mem_used_mb": float(parts[3]),
                "mem_total_mb": float(parts[4]),
                "temp_c": float(parts[5]),
            })
    return rows

class GPUMonitor:
    def __init__(self, gpu_index=0, interval=1.0):
        self.gpu_index = gpu_index
        self.interval = interval
        self.running = False
        self.thread = None
        self.samples = []

    def _loop(self):
        while self.running:
            try:
                rows = query_nvidia_smi()
                row = next((r for r in rows if r["index"] == self.gpu_index), None)
                if row is not None:
                    row["time"] = time.time()
                    self.samples.append(row)
                    print(
                        f"GPU {row['index']} | util={row['util_gpu']:.0f}% | "
                        f"mem={row['mem_used_mb']:.0f}/{row['mem_total_mb']:.0f} MB | "
                        f"temp={row['temp_c']:.0f}C"
                    )
            except Exception as e:
                print(f"[GPU monitor stopped] {e}")
                break

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_loader(batch_size=128, max_samples=20000):
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    ds = Subset(ds, range(max_samples))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

def train_once(device, epochs=3, batch_size=128, max_samples=20000):
    loader = get_loader(batch_size=batch_size, max_samples=max_samples)
    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    model.train()

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    result = {
        "device": device.type,
        "train_seconds": elapsed,
        "allocated_mb": None,
        "reserved_mb": None,
        "peak_allocated_mb": None,
    }

    if device.type == "cuda":
        result["allocated_mb"] = torch.cuda.memory_allocated(device) / 1024**2
        result["reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
        result["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    return result

def main():
    gpu_info()
    results = []

    cpu_result = train_once(torch.device("cpu"))
    results.append(cpu_result)
    print("\nCPU result:", cpu_result)

    if torch.cuda.is_available():
        monitor = GPUMonitor(gpu_index=0, interval=1.0)
        monitor.start()
        try:
            gpu_result = train_once(torch.device("cuda"))
        finally:
            monitor.stop()

        results.append(gpu_result)
        print("\nGPU result:", gpu_result)
        print(f"\nSpeedup: {cpu_result['train_seconds'] / gpu_result['train_seconds']:.2f}x")

        if monitor.samples:
            avg_util = sum(s["util_gpu"] for s in monitor.samples) / len(monitor.samples)
            print(f"Average GPU utilization during training: {avg_util:.1f}%")
    else:
        print("\nCUDA GPU not available.")

    pd.DataFrame(results).to_csv("cnn_benchmark_gpu_monitor.csv", index=False)
    print("\nSaved cnn_benchmark_gpu_monitor.csv")

if __name__ == "__main__":
    main()