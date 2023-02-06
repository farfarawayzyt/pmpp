add_library(gaussian-blur SHARED)
target_sources(
  gaussian-blur PRIVATE
  src/dispatch.cpp
)