#pragma once

#include <chrono>
#include <random>

namespace Random {
    // Returns a seeded Mersenne Twister
    inline std::mt19937 generate() {
        std::random_device rd{};

        // Create seed_seq with clock and 7 random numbers from std::random_device
        std::seed_seq ss{
            static_cast<std::seed_seq::result_type>(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
            rd(), rd(), rd(), rd(), rd(), rd(), rd()
        };

        return std::mt19937{ ss };
    }

    inline std::mt19937 mt{ generate() };

    // Generate a random value between [min, max] (inclusive)
    template <typename T>
	T get(T min, T max) {
		return std::uniform_int_distribution<T>{min, max}(mt);
	}
}
