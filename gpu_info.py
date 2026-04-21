import torch

def gpu_info():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print("Current GPU index:", idx)
        print("GPU name:", torch.cuda.get_device_name(idx))
        print("PyTorch CUDA version:", torch.version.cuda)
        print("Allocated memory (GB):", round(torch.cuda.memory_allocated(idx) / 1024**3, 3))
        print("Reserved memory (GB):", round(torch.cuda.memory_reserved(idx) / 1024**3, 3))

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}")
            print("  Name:", props.name)
            print("  Total memory (GB):", round(props.total_memory / 1024**3, 3))
            print("  Multi processor count:", props.multi_processor_count)
    else:
        print("No CUDA GPU detected.")

gpu_info()