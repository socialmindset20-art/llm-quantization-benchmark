import torch
import sys

def verify_gpu_hardware():
    print("=== [MLOps AI Infrastructure] Verificação de Hardware de Inferência ===")
    print(f"Versão do Python: {sys.version}")
    print(f"Versão do PyTorch: {torch.__version__}")

    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        print("\n[CRITICAL ERROR] GPU NVIDIA não detectada! Por favor, ative a GPU T4.")
        return False

    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)
    compute_cap = torch.cuda.get_device_capability(device_id)
    vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
    vram_gb = vram_bytes / (1024 ** 3)

    print(f"GPU Ativa: {gpu_name}")
    print(f"CUDA Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"CUDA Runtime Version: {torch.version.cuda}")
    print(f"VRAM Total Disponível: {vram_gb:.2f} GB")
    return True
