#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

#define CHANNELS 3              // bgr format (opencv)
#define THREADS_PER_BLOCK 32

__host__ __device__ float computeGrayscale(
    unsigned char rValue,
    unsigned char gValue,
    unsigned char bValue
) {
    return 0.21f*rValue + 0.71f*gValue + 0.07f*bValue;
}

void convertToGrayscaleC(
    const unsigned char* inputImage_h,
    unsigned char* outputImage_h,
    int width,
    int height
) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            // Get 1D offset for grayscale image and rgb image
            int grayOffset { row * width + col };
            int rgbOffset { grayOffset * CHANNELS };
            
            // Read rgb values 
            unsigned char b { inputImage_h[rgbOffset] };
            unsigned char g { inputImage_h[rgbOffset + 1] };
            unsigned char r { inputImage_h[rgbOffset + 2] };

            // Perform grayscale conversion
            outputImage_h[grayOffset] = computeGrayscale(r, g, b);
        }
    }
}

__global__ void grayscaleKernel(
    const unsigned char* inputImage,
    unsigned char* outputImage,
    int width,
    int height
) {
    unsigned int col { blockIdx.x * blockDim.x + threadIdx.x };
    unsigned int row { blockIdx.y * blockDim.y + threadIdx.y };

    if (col < width && row < height) {
        // Get 1D offset for grayscale image and rgb image
        unsigned int grayOffset { row * width + col };
        unsigned int rgbOffset { grayOffset * CHANNELS };
        
        // Read rgb values 
        unsigned char b { inputImage[rgbOffset] };
        unsigned char g { inputImage[rgbOffset + 1] };
        unsigned char r { inputImage[rgbOffset + 2] };

        // Perform grayscale conversion
        outputImage[grayOffset] = computeGrayscale(r, g, b);
    }
}

void convertToGrayscaleCUDA(
    const unsigned char* inputImage_h,
    unsigned char* outputImage_h,
    int width,
    int height
) {
    std::size_t sizeRGB { (std::size_t)(width * height * CHANNELS) };
    std::size_t sizeGray { (std::size_t)(width * height) };

    // Allocate device memory
    unsigned char *inputImage_d { nullptr }, *outputImage_d { nullptr };
    CUDA_CHECK(cudaMalloc((void**)&inputImage_d, sizeRGB));
    CUDA_CHECK(cudaMalloc((void**)&outputImage_d, sizeGray));

    // Copy image to device memory
    CUDA_CHECK(cudaMemcpy(inputImage_d, inputImage_h, sizeRGB, cudaMemcpyHostToDevice));

    // Call the kernel
    dim3 gridSize(ceil(width / THREADS_PER_BLOCK), ceil(height / THREADS_PER_BLOCK));
    dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    grayscaleKernel<<<gridSize, blockSize>>>(inputImage_d, outputImage_d, width, height);

    // Check launch/runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output image from device memory
    CUDA_CHECK(cudaMemcpy(outputImage_h, outputImage_d, sizeGray, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(inputImage_d));
    CUDA_CHECK(cudaFree(outputImage_d));
}


int main() {
    const std::string inputFilePath { "./chapters/ch03/images/perfect_martini_rgb.jpg" };
    const std::string outputFilePathC { "./chapters/ch03/images/perfect_martini_gray_C.jpg" };
    const std::string outputFilePathCUDA { "./chapters/ch03/images/perfect_martini_gray_CUDA.jpg" };

    // Load image
    cv::Mat inputImage { cv::imread(inputFilePath, cv::IMREAD_COLOR) };
    if(inputImage.empty()) {
        std::cerr << "Could not load image: " << inputFilePath << '\n';
        return 1;
    }
    const int width { inputImage.cols };
    const int height { inputImage.rows };
    const unsigned char* inputPtr { inputImage.data };

    // Host buffers for grayscale
    std::vector<unsigned char> outputImageC(width * height);
    std::vector<unsigned char> outputImageCUDA(width * height);

    // Convert to graysacle using C, time it and save
    double secondsCPU {
        utils::executeAndTimeFunction([&]{
            convertToGrayscaleC(inputPtr, outputImageC.data(), width, height);
        })
    };
    std::cout << "CPU version elapsed time: " << secondsCPU << "seconds\n";

    // Save C grayscale image
    cv::Mat outputCMat(height, width, CV_8UC1, outputImageC.data());
    cv::imwrite(outputFilePathC, outputCMat);

    // Convert to graycscale using CUDA and time it
    float secondsGPU { 
        utils::cudaExecuteAndTimeFunction([&]{
            convertToGrayscaleCUDA(inputPtr, outputImageCUDA.data(), width, height);
        })
    };
    std::cout << "GPU version elapsed time: " << secondsGPU << "seconds\n";

    // Save CUDA grayscale image
    cv::Mat outputCUDAMat(height, width, CV_8UC1, outputImageCUDA.data());
    cv::imwrite(outputFilePathCUDA, outputCUDAMat);

    return 0;
}
