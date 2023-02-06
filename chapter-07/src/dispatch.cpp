#include <cstdint>
#include <cstring>

#include "./cpu.h"

extern "C" double dispatch(const char *kernelName, uint8_t *output, const uint8_t *input, uint64_t height, uint64_t width, const float *weights, uint64_t radius){

  if (std::strcmp(kernelName, "cpu") == 0){
    return cpuNaive(output, input, height, width, weights, radius);
  } else {
    return -1;
  }
}