cmake_minimum_required (VERSION 3.16)

project(GLAD LANGUAGES C)
add_library (glad src/glad.c)
target_include_directories(glad PUBLIC include)
target_compile_features(glad PUBLIC cxx_std_17)
