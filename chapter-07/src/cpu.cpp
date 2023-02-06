#include <cstdint>

#include <chrono>

#include "./cpu.h"

using hclk = std::chrono::high_resolution_clock;

double cpuNaive(uint8_t *output, const uint8_t *input, uint64_t height, uint64_t width, const float *weights, uint64_t radius){
  auto start = hclk::now();

  const int64_t radius_s = radius;
  const uint64_t sideLength = 2 * radius + 1;

  for (uint64_t y = 0; y < height; ++y){
    for (uint64_t x = 0; x < width; ++x){
      const uint64_t outputBase = 3 * (y * width + x);
      float rgb[3] = {};
      
      for (int64_t dy = -radius; dy <= radius_s; ++dy){
        for (int64_t dx = -radius; dx <= radius_s; ++dx){
          const int64_t fy = y + dy, fx = x + dx;
          if (0 <= fy && fy < height && 0 <= fx && fx < width){
            const uint64_t inputBase = 3 * (fy * width + fx);
            const uint64_t filterIdx = (dy + radius) * sideLength + (dx + radius);
            for (uint64_t i = 0; i < 3; ++i){
              rgb[i] += static_cast<float>(input[inputBase+i]) * weights[filterIdx];
            }
          }
        }
      }

      for (uint64_t i = 0; i < 3; ++i){
        output[outputBase + i] = static_cast<uint8_t>(rgb[i]);
      }
    }
  }

  auto stop = hclk::now();
  return std::chrono::duration<double>(stop-start).count();
}