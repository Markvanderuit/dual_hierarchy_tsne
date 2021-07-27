cmake_minimum_required (VERSION 3.16)

# Define project leveraging C++ and CUDA
project(DualHierarchytSNE LANGUAGES CXX CUDA)

# Add third party libraries
add_subdirectory(third_party/resource_embed)
add_subdirectory(third_party/glad)

# Add tsne libraries
add_library(util 
  include/util/timer.hpp
  include/util/enum.hpp
  include/util/types.hpp
  include/util/gl/timer.hpp
  src/util/gl/timer.cpp
)
target_link_libraries(util glad)
target_include_directories(util PUBLIC include)
target_compile_features(util PUBLIC cxx_std_17)

# Add test executable
add_executable(foo src/app/foo.cpp)
target_compile_features(foo PRIVATE cxx_std_17)
target_link_libraries(foo ResourceEmbed util)
rsrc_embed_dir(foo ${CMAKE_SOURCE_DIR}/resources/shaders)