import time
from dataclasses import dataclass
from typing import Callable, Tuple, List

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from utils.py_utils import load_cuda_extension


@dataclass
class KernelSpec:
    label: str      # Pretty label for printing
    attr_name: str  # Attribute name on the loaded CUDA extension


@dataclass
class Constants:
    radius = 3
    height = 4096
    width = 4096
    k = 2 * radius + 1
    rtol = 1e-6
    atol = 1e-6


def time_torch_reference(
    input_base: torch.Tensor,
    filter_base: torch.Tensor,
    radius: int,
) -> Tuple[torch.Tensor, float]:
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

def run_and_check(
    label: str,
    kernel_fn: Callable,
    input_base: torch.Tensor,
    filter_base: torch.Tensor,
    radius: int,
    torch_out: torch.Tensor,
    torch_time: float,
    rtol: float,
    atol: float,
):
    """Time one kernel, check correctness, and print results."""
    out, elapsed = time_cuda_kernel(
        kernel_fn=kernel_fn,
        input_base=input_base.contiguous(),
        filter_base=filter_base.contiguous(),
        radius=radius,
    )

    assert_close(
        out,
        torch_out,
        rtol=rtol,
        atol=atol,
        msg=f"{label} output does not match PyTorch reference.",
    )

    print(f"\n{label} time: {elapsed:.4f}s")
    print(f"Speedup {label.lower()}: {torch_time / elapsed:.2f}x")


@torch.inference_mode()
def main():
    cuda_extension = load_cuda_extension(
        sources=(
            "bindings.cpp",
            "functions_torch.cu",
            "kernels/kernels2d.cu",
            "kernels/kernels3d.cu",
        ),
        verbose=True,
    )

    # ---------------- 2D ----------------
    input2d = torch.randn(Constants.height, Constants.width, device="cuda", dtype=torch.float32)
    filt2d = torch.randn(Constants.k, Constants.k, device="cuda", dtype=torch.float32)

    torch_out, torch_time = time_torch_reference(
        input_base=input2d,
        filter_base=filt2d,
        radius=Constants.radius,
    )
    print(f"PyTorch 2D time: {torch_time:.4f}s")

    # Kernels to run
    cases2d: List[KernelSpec] = [
        KernelSpec("Basic CUDA 2D", "conv2d"),
        KernelSpec("Constant-memory CUDA 2D", "conv2dConstMem"),
        KernelSpec("Tiled-in CUDA 2D", "conv2dTiledIn"),
        KernelSpec("Tiled-out CUDA 2D", "conv2dTiledOut"),
        KernelSpec("Tiled-cached CUDA 2D", "conv2dTiledCached"),
    ]

    # Run all kernels with the same procedure
    for spec in cases2d:
        run_and_check(
            label=spec.label,
            kernel_fn=getattr(cuda_extension, spec.attr_name),
            input_base=input2d,
            filter_base=filt2d,
            radius=Constants.radius,
            torch_out=torch_out,
            torch_time=torch_time,
            rtol=Constants.rtol,
            atol=Constants.atol,
        )


if __name__ == "__main__":
    main()
