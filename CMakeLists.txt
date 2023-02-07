cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(PMPP CXX CUDA)

add_library(common_config INTERFACE)
target_include_directories(common_config INTERFACE common)

add_subdirectory(chapter-07)