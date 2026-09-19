import sys
import torch
import ultralytics
import cpuinfo

print("--- ENVIRONMENT AUDIT ---")
print("Python Version:", sys.version.replace('\n', ''))
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    print("CUDA Version: None")
    print("GPU Name: None")
    print("GPU VRAM: None")
print("Ultralytics Version:", ultralytics.__version__)
try:
    info = cpuinfo.get_cpu_info()
    print("CPU:", info['brand_raw'])
except Exception:
    print("CPU: Information unavailable (py-cpuinfo not installed)")
