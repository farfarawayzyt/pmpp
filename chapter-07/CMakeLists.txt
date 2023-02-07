add_library(gaussian-blur SHARED)

target_sources(
  gaussian-blur PRIVATE
  src/dispatch.cpp
  src/cpu.cpp
  src/gpu.cu
)

target_link_libraries(gaussian-blur PRIVATE common_config)