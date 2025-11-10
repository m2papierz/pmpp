#include "kernels.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

torch::Tensor conv2d(
    const torch::Tensor& inArray,
    const torch::Tensor& filter,
    int radius
) {
    TORCH_CHECK(inArray.is_cuda(), "input must be CUDA");
    TORCH_CHECK(filter.is_cuda(), "filter must be CUDA");
    TORCH_CHECK(inArray.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(filter.dtype() == torch::kFloat32, "filter must be float32");
    TORCH_CHECK(inArray.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(filter.is_contiguous(), "filter must be contiguous");

    const int height { static_cast<int>(inArray.size(0)) };
    const int width { static_cast<int>(inArray.size(1)) };
    auto outArray { torch::empty_like(inArray) };

    const float* inArrayPtr { inArray.data_ptr<float>() };
    const float* filterPtr { filter.data_ptr<float>() };
    float* outArrayPtr { outArray.data_ptr<float>() };

    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(
        static_cast<int>((width + BLOCK_SIZE - 1) / BLOCK_SIZE),
        static_cast<int>((height + BLOCK_SIZE - 1) / BLOCK_SIZE),
        1
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2dKernel<<<gridSize, blockSize, 0 , stream>>>(inArrayPtr, filterPtr, outArrayPtr, radius, height, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return outArray;
}

torch::Tensor conv2dConstMem(
    const torch::Tensor& inArray,
    const torch::Tensor& filter,
    int radius
) {
    TORCH_CHECK(inArray.is_cuda(), "input must be CUDA");
    TORCH_CHECK(filter.is_cuda(), "filter must be CUDA");
    TORCH_CHECK(inArray.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(filter.dtype() == torch::kFloat32, "filter must be float32");
    TORCH_CHECK(inArray.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(filter.is_contiguous(), "filter must be contiguous");

    const int height { static_cast<int>(inArray.size(0)) };
    const int width { static_cast<int>(inArray.size(1)) };
    auto outArray { torch::empty_like(inArray) };

    const float* inArrayPtr { inArray.data_ptr<float>() };
    const float* filterPtr { filter.data_ptr<float>() };
    float* outArrayPtr { outArray.data_ptr<float>() };

    // Initialize constant mememory
    std::size_t filterSize {
        static_cast<std::size_t>((2*FILTER_RADIUS + 1)*(2*FILTER_RADIUS + 1)*sizeof(float))
    };
    C10_CUDA_CHECK(uploadConstFilter(filterPtr, filterSize));

    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(
        static_cast<int>((width + BLOCK_SIZE - 1) / BLOCK_SIZE),
        static_cast<int>((height + BLOCK_SIZE - 1) / BLOCK_SIZE),
        1
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2dKernelConstMem<<<gridSize, blockSize, 0 , stream>>>(inArrayPtr, outArrayPtr, radius, height, width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return outArray;
}
