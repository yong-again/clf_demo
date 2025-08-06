import torch

def get_device():
    """
    GPU, MPS(Apple Silicon), CPU 감지
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (no GPU detected)")
    return device

def get_amp_dtye(device):
    """
    AMP(automatic mixed precision을 위한 dtype 선택
    """
    if device.type == 'cuda':
        return torch.float16 # Gpu에서 Float16
    elif device.type == 'mps':
        return torch.float32
    else:
        return torch.float32 # Cpu