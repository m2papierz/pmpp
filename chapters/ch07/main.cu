#include "utils.hpp"
#include "kernels.cuh"
#include "functions.cuh"

#include <iostream>

namespace Constants {
    constexpr int height { 4096 };
    constexpr int width { 4096 };
}

int main() {
    std::vector<float> inArray(Constants::height * Constants::width);
    std::vector<float> filter(2 * FILTER_RADIUS + 1);
    std::vector<float> outArray(Constants::height * Constants::width);

    for (int i { 0 }; i < Constants::height * Constants::width; ++i) {
        inArray[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i { 0 }; i < 2 * FILTER_RADIUS + 1; ++i) {
        filter[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float secondsBasic {
        utils::cudaExecuteAndTimeFunction([&]{
            conv2d(
                inArray.data(), filter.data(), outArray.data(),
                FILTER_RADIUS, Constants::height, Constants::width
            );
        })
    };
    std::cout << "Basic GPU version elapsed time: " << secondsBasic << "seconds\n";

    float secondsConstMem {
        utils::cudaExecuteAndTimeFunction([&]{
            conv2dConstMem(
                inArray.data(), filter.data(), outArray.data(),
                FILTER_RADIUS, Constants::height, Constants::width
            );
        })
    };
    std::cout << "Constant memory GPU version elapsed time: " << secondsConstMem << "seconds\n";

    return 0;
}
