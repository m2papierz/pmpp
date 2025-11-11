import time
from dataclasses import dataclass
from typing import Callable, Tuple

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


def time_torch_reference(
    input_base: torch.Tensor, filter_base: torch.Tensor, radius: int
) -> Tuple[torch.Tensor, float]:
    """Compute the PyTorch conv2d reference and measure time."""
    start = time.time()
    out = (
        F.conv2d(
            input_base.unsqueeze(0).unsqueeze(0),
            filter_base.unsqueeze(0).unsqueeze(0),
            padding=radius,
        )
        .squeeze(0)
        .squeeze(0)
    )
    torch.cuda.synchronize()
    return out, time.time() - start


def time_cuda_kernel(
    kernel_fn: Callable,
    input_base: torch.Tensor,
    filter_base: torch.Tensor,
    radius: int,
) -> Tuple[torch.Tensor, float]:
    """
    Run a CUDA kernel and measure time.
    IMPORTANT: Clone inputs because kernels may modify arrays in-place.
    """
    start = time.time()
    out = kernel_fn(input_base.clone(), filter_base.clone(), radius)
    torch.cuda.synchronize()
    return out, time.time() - start


def print_result(label: str, elapsed: float, torch_elapsed: float):
    print(f"\n{label} time: {elapsed:.4f}s")
    print(f"Speedup {label.lower()}: {torch_elapsed / elapsed:.2f}x")


@torch.inference_mode()
def main():
    cuda_extension = load_cuda_extension(
        sources=("bindings.cpp", "functions_torch.cu", "kernels.cu"), verbose=True
    )

    input_base = torch.randn(
        Constants.height, Constants.width, device="cuda", dtype=torch.float32
    )
    filter_base = torch.randn(
        Constants.k, Constants.k, device="cuda", dtype=torch.float32
    )

    torch_out, torch_time = time_torch_reference(
        input_base=input_base,
        filter_base=filter_base,
        radius=Constants.radius,
    )
    print(f"PyTorch time: {torch_time:.4f}s")

    # Basic conv2d kernel
    basic_out, basic_t = time_cuda_kernel(
        kernel_fn=cuda_extension.conv2d,
        input_base=input_base,
        filter_base=filter_base,
        radius=Constants.radius,
    )
    assert_close(
        basic_out,
        torch_out,
        rtol=Constants.rtol,
        atol=Constants.atol,
        msg="Basic CUDA output does not match PyTorch reference.",
    )
    print_result("Basic CUDA", basic_t, torch_time)

    # Constant memory conv2d kernel
    const_out, const_t = time_cuda_kernel(
        kernel_fn=cuda_extension.conv2dConstMem,
        input_base=input_base,
        filter_base=filter_base,
        radius=Constants.radius,
    )
    assert_close(
        const_out,
        torch_out,
        rtol=Constants.rtol,
        atol=Constants.atol,
        msg="Constant-memory CUDA output does not match PyTorch.",
    )
    print_result("Constant-memory CUDA", const_t, torch_time)

    # Tiled-in conv2d kernel
    tiled_in_out, tiled_in_t = time_cuda_kernel(
        kernel_fn=cuda_extension.conv2dTiledIn,
        input_base=input_base,
        filter_base=filter_base,
        radius=Constants.radius,
    )
    assert_close(
        tiled_in_out,
        torch_out,
        rtol=Constants.rtol,
        atol=Constants.atol,
        msg="Tiled-in CUDA output does not match PyTorch.",
    )
    print_result("Tiled-in CUDA", tiled_in_t, torch_time)

    # Tiled-out conv2d kernel
    tiled_out_out, tiled_out_t = time_cuda_kernel(
        kernel_fn=cuda_extension.conv2dTiledOut,
        input_base=input_base,
        filter_base=filter_base,
        radius=Constants.radius,
    )
    assert_close(
        tiled_out_out,
        torch_out,
        rtol=Constants.rtol,
        atol=Constants.atol,
        msg="Tiled-out CUDA output does not match PyTorch.",
    )
    print_result("Tiled-out CUDA", tiled_out_t, torch_time)


if __name__ == "__main__":
    main()
