#include "utils.hpp"
#include "functions.cuh"

#include <iostream>

namespace Constants {
    constexpr int height { 4096 };
    constexpr int width { 4096 };
    constexpr int radius { 3 }; 
}

int main() {
    std::vector<float> inArray(Constants::height * Constants::width);
    std::vector<float> filter(2 * Constants::radius + 1);
    std::vector<float> outArray(Constants::height * Constants::width);

    for (int i { 0 }; i < Constants::height * Constants::width; ++i) {
        inArray[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i { 0 }; i < 2 * Constants::radius + 1; ++i) {
        filter[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float secondsGPU {
        utils::cudaExecuteAndTimeFunction([&]{
            conv2d(
                inArray.data(), filter.data(), outArray.data(),
                Constants::radius, Constants::height, Constants::width
            );
        })
    };
    std::cout << "GPU version elapsed time: " << secondsGPU << "seconds\n";

    return 0;
}
