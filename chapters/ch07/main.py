import os
import time

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

major, minor = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}+PTX"


cuda_extension = load(
    name="extension_ch07",
    sources=["./bindings.cpp", "./functions_torch.cu", "./kernels.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)


@torch.inference_mode()
def main():
    radius = 3
    height = 4096
    width = 4096
    k = 2 * radius + 1

    in_cuda = torch.randn(height, width, device="cuda", dtype=torch.float32)
    filter_cuda = torch.randn(k, k, device="cuda", dtype=torch.float32)

    # Torch reference conv expects NCHW and OIHW for weight
    in_torch = in_cuda.unsqueeze(0).unsqueeze(0)
    filter_torch = filter_cuda.unsqueeze(0).unsqueeze(0)

    start_time = time.time()
    torch_output = F.conv2d(in_torch, filter_torch, padding=radius).squeeze(0).squeeze(0)
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    print(f"PyTorch convolution completed in {torch_time:.4f} seconds")

    start_time = time.time()
    cuda_output = cuda_extension.conv2d(in_cuda, filter_cuda, radius)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    print(f"CUDA convolution completed in {cuda_time:.4f} seconds")

    max_diff = torch.max(torch.abs(cuda_output - torch_output)).item()
    print(f"Maximum difference between CUDA and Torch output: {max_diff}")
    print(f"Speedup: {torch_time / cuda_time:.2f}x")

if __name__ == "__main__":
    main()
