import time
import threading
import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMonitor:
    def __init__(self, gpu_index=0, interval=1.0):
        self.gpu_index = gpu_index
        self.interval = interval
        self.running = False
        self.samples = []
        self.thread = None

        if not PYNVML_AVAILABLE:
            raise ImportError("Install pynvml first: pip install nvidia-ml-py3")

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def _loop(self):
        while self.running:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            sample = {
                "time": time.time(),
                "gpu_util_percent": util.gpu,
                "mem_used_mb": mem.used / 1024**2,
                "mem_total_mb": mem.total / 1024**2,
                "temp_c": temp,
            }
            self.samples.append(sample)

            print(
                f"GPU {self.gpu_index} | "
                f"util={sample['gpu_util_percent']}% | "
                f"mem={sample['mem_used_mb']:.0f}/{sample['mem_total_mb']:.0f} MB | "
                f"temp={sample['temp_c']}C"
            )

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        pynvml.nvmlShutdown()


def train_dummy_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 10),
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(100):
        x = torch.randn(256, 1024, device="cuda")
        y = torch.randint(0, 10, (256,), device="cuda")

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step}, loss={loss.item():.4f}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        monitor = GPUMonitor(gpu_index=0, interval=1.0)
        monitor.start()

        try:
            train_dummy_model()
        finally:
            monitor.stop()

        print("\nCollected samples:", len(monitor.samples))
    else:
        print("CUDA GPU not available.")