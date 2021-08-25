cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

add_library(indicators INTERFACE)
target_include_directories(indicators INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
