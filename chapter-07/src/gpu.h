#include <cstdint>

extern "C" {
  double gpuNaive(uint8_t *output_h, const uint8_t *input_h, uint64_t height, uint64_t width, const float *weights_h, uint64_t radius);
}
