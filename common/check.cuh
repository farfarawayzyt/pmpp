#include <cstdio>

template<typename T>
static inline void _checkCudaError(T result, const char *const func, const char *const file, const int line){
  if (result){
    std::printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<int>(result), cudaGetErrorName(result), func);
  }
}

#define CHECK_CUDA_ERROR(val) _checkCudaError((val), #val, __FILE__, __LINE__)