import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from utils.py_utils import load_cuda_extension


@dataclass
class Constants:
    radius = 3
    height = 4096
    width = 4096
    k = 2 * radius + 1
    rtol = 1e-6
    atol = 1e-6


@torch.inference_mode()
def main():
    cuda_extension = load_cuda_extension(
        sources=("bindings.cpp", "functions_torch.cu", "kernels.cu"),
        verbose=True
    )

    in_cuda = torch.randn(
        Constants.height, Constants.width, device="cuda", dtype=torch.float32
    )
    filter_cuda = torch.randn(
        Constants.k, Constants.k, device="cuda", dtype=torch.float32
    )

    # Torch reference conv expects NCHW and OIHW for weight
    in_torch = in_cuda.unsqueeze(0).unsqueeze(0)
    filter_torch = filter_cuda.unsqueeze(0).unsqueeze(0)

    start_time = time.time()
    torch_output = (
        F.conv2d(in_torch, filter_torch, padding=Constants.radius).squeeze(0).squeeze(0)
    )
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    print(f"PyTorch time: {torch_time:.4f}s")

    start_time = time.time()
    basic_cuda_output = cuda_extension.conv2d(in_cuda, filter_cuda, Constants.radius)
    torch.cuda.synchronize()
    basic_cuda_time = time.time() - start_time

    assert_close(
        basic_cuda_output,
        torch_output,
        rtol=Constants.rtol,
        atol=Constants.atol,
        msg="Basic CUDA output does not match PyTorch reference.",
    )
    print(f"\nBasic CUDA time: {basic_cuda_time:.4f}s")
    print(f"Speedup basic: {torch_time / basic_cuda_time:.2f}x")

    start_time = time.time()
    const_mem_cuda_output = cuda_extension.conv2dConstMem(
        in_cuda, filter_cuda, Constants.radius
    )
    torch.cuda.synchronize()
    const_cuda_time = time.time() - start_time
    
    assert_close(
        const_mem_cuda_output,
        torch_output,
        rtol=Constants.rtol,
        atol=Constants.atol,
        msg="Constant-memory CUDA output does not match PyTorch.",
    )
    print(f"\nConstant-memory CUDA time: {const_cuda_time:.4f}s")
    print(f"Speedup constant-memory: {torch_time / const_cuda_time:.2f}x")


if __name__ == "__main__":
    main()
