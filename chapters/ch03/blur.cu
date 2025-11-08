#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

#define CHANNELS 3               // bgr format (opencv)
#define BLUR_SIZE 3              // 1 pixel on each side giving a 3x3 box
#define THREADS_PER_BLOCK 32

__host__ __device__ __forceinline__ unsigned char clamp_u8(float x) {
    x = fminf(fmaxf(x, 0.f), 255.f);
    return (unsigned char)(x + 0.5f);
}

void blurImageC(
    const unsigned char* inputImage_h,
    unsigned char* outputImage_h,
    int width,
    int height
) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int pixels = 0;
            int pixValR = 0, pixValG = 0, pixValB = 0;

            // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
            for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                    int currentRow = row + blurRow;
                    int currentCol = col + blurCol;
                    // Verify we have a valid image pixel
                    if (currentRow>=0 && currentRow<height && currentCol>=0 && currentCol<width) {
                        std::size_t rgbOffset = ((std::size_t)currentRow * width + currentCol) * CHANNELS;
                        pixValB += inputImage_h[rgbOffset];
                        pixValG += inputImage_h[rgbOffset + 1];
                        pixValR += inputImage_h[rgbOffset + 2];
                        pixels++; // Keep track of the number of pixels in the average
                    }
                }
            }
            // Write new pixels values
            std::size_t outRGBOffset = ((std::size_t)row * width + col) * CHANNELS;
            outputImage_h[outRGBOffset] = clamp_u8(pixValB/pixels);
            outputImage_h[outRGBOffset + 1] = clamp_u8(pixValG/pixels);
            outputImage_h[outRGBOffset + 2] = clamp_u8(pixValR/pixels);
        }
    }
}

__global__ void blurKernel(
    const unsigned char* inputImage,
    unsigned char* outputImage,
    int width,
    int height
) {
    unsigned int row { blockIdx.y*blockDim.y + threadIdx.y };
    unsigned int col { blockIdx.x*blockDim.x + threadIdx.x };
    if (row < height && col < width) {
        int pixels { 0 };
        int pixValR { 0 }, pixValG { 0 }, pixValB { 0 };

        // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                unsigned int currentRow { row + blurRow };
                unsigned int currentCol { col + blurCol };
                // Verify we have a valid image pixel
                if (currentRow<height && currentCol<width) {
                    size_t rgbOffset { ((size_t)currentRow * width + currentCol) * CHANNELS };
                    pixValB += inputImage[rgbOffset];
                    pixValG += inputImage[rgbOffset + 1];
                    pixValR += inputImage[rgbOffset + 2];
                    pixels++; // Keep track of the number of pixels in the average
                }
            }
        }
        // Write new pixels values
        std::size_t outRGBOffset = ((std::size_t)row * width + col) * CHANNELS;
        outputImage[outRGBOffset] = clamp_u8(pixValB/pixels);
        outputImage[outRGBOffset + 1] = clamp_u8(pixValG/pixels);
        outputImage[outRGBOffset + 2] = clamp_u8(pixValR/pixels);
    }
}

void blurImageCUDA(
    const unsigned char* inputImage_h,
    unsigned char* outputImage_h,
    int width,
    int height
) {
    // Allocate device memory
    std::size_t size { (std::size_t)(width * height * CHANNELS) };
    unsigned char *inputImage_d { nullptr }, *outputImage_d { nullptr };
    CUDA_CHECK(cudaMalloc((void**)&inputImage_d, size));
    CUDA_CHECK(cudaMalloc((void**)&outputImage_d, size));

    // Copy image to the device memory
    CUDA_CHECK(cudaMemcpy(inputImage_d, inputImage_h, size, cudaMemcpyHostToDevice));

    // Call the kernel
    dim3 gridSize(ceil(width / THREADS_PER_BLOCK), ceil(height / THREADS_PER_BLOCK));
    dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    blurKernel<<<gridSize, blockSize>>>(inputImage_d, outputImage_d, width, height);

    // Check launch/runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output image from device memory
    CUDA_CHECK(cudaMemcpy(outputImage_h, outputImage_d, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(inputImage_d));
    CUDA_CHECK(cudaFree(outputImage_d));
}

int main() {
    const char* inputFilePath { "./chapters/ch03/images/perfect_martini_rgb.jpg" };
    const char* outputFilePathC { "./chapters/ch03/images/perfect_martini_blur_C.jpg" };
    const char* outputFilePathCUDA { "./chapters/ch03/images/perfect_martini_blur_CUDA.jpg" };

    // Load image
    cv::Mat inputImage { cv::imread(inputFilePath, cv::IMREAD_COLOR) };
    if(inputImage.empty()) {
        std::cerr << "Could not load image: " << inputFilePath << '\n';
        return 1;
    }
    const int width { inputImage.cols };
    const int height { inputImage.rows };
    const unsigned char* inputPtr { inputImage.data };

    // Host buffers for output images
    std::vector<unsigned char> outputImageC(width * height * CHANNELS);
    std::vector<unsigned char> outputImageCUDA(width * height * CHANNELS);

    // Image blurring using CPU, time it and save
    double secondsCPU {
        utils::executeAndTimeFunction([&]{
            blurImageC(inputPtr, outputImageC.data(), width, height);
        })
    };
    std::cout << "CPU version elapsed time: " << secondsCPU << "seconds\n";

    // Save CPU output image
    cv::Mat outputCMat(height, width, CV_8UC3, outputImageC.data());
    cv::imwrite(outputFilePathC, outputCMat);

    // Image blurring using GPU, time it and save
    double secondsGPU {
        utils::cudaExecuteAndTimeFunction([&]{
            blurImageCUDA(inputPtr, outputImageCUDA.data(), width, height);
        })
    };
    std::cout << "GPU version elapsed time: " << secondsGPU << "seconds\n";

    // Save GPU output image
    cv::Mat outputCUDAMat(height, width, CV_8UC3, outputImageCUDA.data());
    cv::imwrite(outputFilePathCUDA, outputCUDAMat);

    return 0;
}
