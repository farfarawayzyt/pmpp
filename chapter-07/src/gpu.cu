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

__global__ void sharedMemKernel(uint8_t *output, const uint8_t *input, uint32_t height, uint32_t width, uint32_t radius){
  extern __shared__ uint8_t inputTile[];

  const int32_t copyBeginY = blockIdx.y * blockDim.y - radius;
  const int32_t copyBeginX = blockIdx.x * blockDim.x - radius;

  const uint32_t inputTileHeight = 2 * radius + blockDim.y;
  const uint32_t inputTileWidth = 2 * radius + blockDim.x;
  const uint32_t inputTileArea = inputTileHeight * inputTileWidth;
  const uint32_t stride = blockDim.x * blockDim.y;

  for (uint32_t i = 0; i < inputTileArea; i += stride){
    const uint32_t copyLinearIdx = i + threadIdx.y * blockDim.x + threadIdx.x;

    if (copyLinearIdx < inputTileArea){
      const uint32_t copyDeltaY = copyLinearIdx / inputTileWidth;
      const uint32_t copyDeltaX = copyLinearIdx % inputTileWidth;

      const int32_t cy = copyBeginY + copyDeltaY;
      const int32_t cx = copyBeginX + copyDeltaX;

      if (0 <= cy && cy < height && 0 <= cx && cx < width){
        const uint32_t cidx = 3 * (cy * width + cx);
        inputTile[3 * copyLinearIdx] = input[cidx];
        inputTile[3 * copyLinearIdx + 1] = input[cidx + 1];
        inputTile[3 * copyLinearIdx + 2] = input[cidx + 2]; 
      } else {
        inputTile[3 * copyLinearIdx] = 0.0f;
        inputTile[3 * copyLinearIdx + 1] = 0.0f;
        inputTile[3 * copyLinearIdx + 2] = 0.0f;
      }
    }
  }

  __syncthreads();

  const uint32_t oy = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t ox = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t oidx = 3 * (oy * width + ox);

  // auto fprint = [oy, ox](int32_t dy, int32_t dx, uint32_t fy, uint32_t fx, uint32_t ty, uint32_t tx, uint32_t idx, float i){
  //   if (oy == 1200 && ox == 500){
  //     std::printf("%d, %d, %u, %u, %u, %u, %u, %f\n", dy, dx, fy, fx, ty, tx, idx, i);
  //   }
  // };

  if (oy < height && ox < width){
    float rgb[3] = {};
    for (int32_t dy = -radius; dy <= (int32_t)radius; ++dy){
      for (int32_t dx = -radius; dx <= (int32_t)radius; ++dx){
        const uint32_t fy = dy + radius;
        const uint32_t fx = dx + radius;
        const uint32_t fidx = fy * (2 * radius + 1) + fx; 

        const uint32_t ty = fy + threadIdx.y;
        const uint32_t tx = fx + threadIdx.x;
        const uint32_t tidx = 3 * (ty * inputTileWidth + tx);

        for (uint32_t i = 0; i < 3; ++i){
          rgb[i] += static_cast<float>(inputTile[tidx + i]) * weights_ccache[fidx];
          // fprint(dy, dx, fy, fx, ty, tx, i, inputTile[tidx + i]);
        }
      }
    }
    
    for (uint32_t i = 0; i < 3; ++i){
      output[oidx + i] = static_cast<uint8_t>(rgb[i]);
    }
  }
}

enum class KernelType {
  Naive,
  ConstCache,
  SharedMem
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
  } else if constexpr(ktype == KernelType::SharedMem){
    dim3 BlockSize(16, 16);
    dim3 GridSize((width + 15) / 16, (height + 15) / 16);
    const uint32_t sharedMemoryNbytes = 3 * (16 + 2 * radius) * (16 + 2 * radius);
    sharedMemKernel<<<GridSize, BlockSize, sharedMemoryNbytes>>>(output_d, input_d, height, width, radius);
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

  double gpuSharedMem(uint8_t *output_h, const uint8_t *input_h, uint64_t height, uint64_t width, const float *weights_h, uint64_t radius){
    return gpuCommon<false, KernelType::SharedMem>(output_h, input_h, height, width, weights_h, radius) / 1000.0;
  }
}