#include <cstdint>
#include <cstring>

#include "./cpu.h"
#include "./gpu.h"

extern "C" double dispatch(const char *kernelName, uint8_t *output, const uint8_t *input, uint64_t height, uint64_t width, const float *weights, uint64_t radius){
  uint64_t sideLength = 2 * radius + 1;

  if (sideLength * sideLength * 4 >= 1024){
    return -1.0;
  }


  if (std::strcmp(kernelName, "cpu") == 0){
    return cpuNaive(output, input, height, width, weights, radius);
  } else if (std::strcmp(kernelName, "naive") == 0){
    return gpuNaive(output, input, height, width, weights, radius);
  } else if (std::strcmp(kernelName, "const_cache") == 0){
    return gpuConstCache(output, input, height, width, weights, radius);
  } else if (std::strcmp(kernelName, "shared") == 0){
    return gpuSharedMem(output, input, height, width, weights, radius);
  }else {
    return -1;
  }
}