cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_library(cub INTERFACE)
target_include_directories(cub INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
