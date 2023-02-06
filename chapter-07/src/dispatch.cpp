#include <cstdint>

extern "C" double dispatch(const char *kernelName, uint8_t *output, const uint8_t *input, uint64_t height, uint64_t width, const float *weights, uint64_t radius){
  uint64_t sideLength = 2 * radius + 1;
  
  double sum = 0.0;
  for (uint64_t i = 0; i < (sideLength * sideLength); ++i){
    sum += weights[i];
  }

  return sum;
}