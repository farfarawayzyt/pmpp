#include <cstdint>
#include <cstring>

#include <check.cuh>

#include "./gpu.h"

__constant__ float weights_ccache[1024];

__global__ void naiveKernel(uint8_t *output, const uint8_t *input, uint32_t height, uint32_t width, const float *weights, uint32_t radius){
  const uint32_t oy = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t ox = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t sideLength = 2 * radius + 1;

  if (oy < height && ox < width){
    float rgb[3] = {};
    for (int32_t dy = -radius; dy <= (int32_t)radius; ++dy){
      for (int32_t dx = -radius; dx <= (int32_t)radius; ++dx){
        const int32_t iy = oy + dy, ix = ox + dx;
        if (0 <= iy && iy < height && 0 <= ix && ix < width){
          const uint32_t inputIdxBase = 3 * (iy * width + ix);
          const uint32_t filterParamIdx = (dy + radius) * sideLength + (dx + radius);

          for (uint32_t i = 0; i < 3; ++i){
            rgb[i] += static_cast<float>(input[inputIdxBase+i]) * weights[filterParamIdx];
          }
        }
      }
    }

    const uint32_t outputIdxBase = 3 * (oy * width + ox);
    for (uint32_t i = 0; i < 3; ++i){
      output[outputIdxBase + i] = static_cast<uint8_t>(rgb[i]);
    }
  }
}

__global__ void constCacheKernel(uint8_t *output, const uint8_t *input, uint32_t height, uint32_t width, uint32_t radius){
  const uint32_t oy = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t ox = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t sideLength = 2 * radius + 1;

  if (oy < height && ox < width){
    float rgb[3] = {};
    for (int32_t dy = -radius; dy <= (int32_t)radius; ++dy){
      for (int32_t dx = -radius; dx <= (int32_t)radius; ++dx){
        const int32_t iy = oy + dy, ix = ox + dx;
        if (0 <= iy && iy < height && 0 <= ix && ix < width){
          const uint32_t inputIdxBase = 3 * (iy * width + ix);
          const uint32_t filterParamIdx = (dy + radius) * sideLength + (dx + radius);

          for (uint32_t i = 0; i < 3; ++i){
            rgb[i] += static_cast<float>(input[inputIdxBase+i]) * weights_ccache[filterParamIdx];
          }
        }
      }
    }

    const uint32_t outputIdxBase = 3 * (oy * width + ox);
    for (uint32_t i = 0; i < 3; ++i){
      output[outputIdxBase + i] = static_cast<uint8_t>(rgb[i]);
    }
  }
}

enum class KernelType {
  Naive,
  ConstCache
};

template<bool CacheWeights, KernelType ktype>
double gpuCommon(uint8_t *output_h, const uint8_t *input_h, uint64_t height, uint64_t width, const float *weights_h, uint64_t radius){
  const uint64_t pixelsNum = height * width * 3;
  const uint64_t sideLength = 2 * radius + 1;
  const uint64_t weightsNbyte = sideLength * sideLength * sizeof(float);
  
  uint8_t *output_d, *input_d;
  float *weights_d;

  CHECK_CUDA_ERROR(cudaMalloc(&output_d, pixelsNum));
  CHECK_CUDA_ERROR(cudaMalloc(&input_d, pixelsNum));
  CHECK_CUDA_ERROR(cudaMemcpy(input_d, input_h, pixelsNum, cudaMemcpyHostToDevice));

  if constexpr (CacheWeights){
    CHECK_CUDA_ERROR(cudaMalloc(&weights_d, weightsNbyte));
    CHECK_CUDA_ERROR(cudaMemcpy(weights_d, weights_h, weightsNbyte, cudaMemcpyHostToDevice));
  } else {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(weights_ccache, weights_h, weightsNbyte));
  }

  bool invalid_kernel = false;
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start));

  if constexpr (ktype == KernelType::Naive){
    dim3 BlockSize(16, 16);
    dim3 GridSize((width + 15) / 16, (height + 15) / 16);
    naiveKernel<<<GridSize, BlockSize>>>(output_d, input_d, height, width, weights_d, radius);
  } else if constexpr(ktype == KernelType::ConstCache){
    dim3 BlockSize(16, 16);
    dim3 GridSize((width + 15) / 16, (height + 15) / 16);
    constCacheKernel<<<GridSize, BlockSize>>>(output_d, input_d, height, width, radius);
  }

  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float elasped_time = -1.0f;
  if (!invalid_kernel){
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elasped_time, start, stop));
  }

  CHECK_CUDA_ERROR(cudaMemcpy(output_h, output_d, pixelsNum, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  if constexpr (CacheWeights){
    CHECK_CUDA_ERROR(cudaFree(weights_d));
  }
  CHECK_CUDA_ERROR(cudaFree(input_d));
  CHECK_CUDA_ERROR(cudaFree(output_d));

  return elasped_time;
}

extern "C" {
  double gpuNaive(uint8_t *output_h, const uint8_t *input_h, uint64_t height, uint64_t width, const float *weights_h, uint64_t radius){
    return gpuCommon<true, KernelType::Naive>(output_h, input_h, height, width, weights_h, radius) / 1000.0;
  }

  double gpuConstCache(uint8_t *output_h, const uint8_t *input_h, uint64_t height, uint64_t width, const float *weights_h, uint64_t radius){
    return gpuCommon<false, KernelType::ConstCache>(output_h, input_h, height, width, weights_h, radius) / 1000.0;
  } 
}