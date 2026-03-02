"""Quick GPU Check Script"""
import torch

print("="*60)
print("GPU/CUDA Configuration Check")
print("="*60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
    
    # Test GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y
    print(f"✓ GPU Test Successful - Tensor on: {z.device}")
else:
    print("⚠ No GPU available - will use CPU")
print("="*60)
