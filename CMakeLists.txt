cmake_minimum_required (VERSION 3.21)

# Specify vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE 
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

# Define project leveraging C++ and CUDA
project(DualHierarchytSNE LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/cmake/Modules/ ${CMAKE_MODULE_PATH})

# Add included third party libraries
add_subdirectory(third_party/resource_embed)
add_subdirectory(third_party/glad)
add_subdirectory(third_party/glm)
add_subdirectory(third_party/glfw)
add_subdirectory(third_party/cub)
add_subdirectory(third_party/faiss)

# Function to add a library while recursively globbing its files
function(add_library_recurse target dir)
  file(GLOB_RECURSE srcs src/${dir}/*.cpp  src/${dir}/*.cu)
  file(GLOB_RECURSE includes include/${dir}/*.hpp  include/${dir}/*.cuh)
  add_library(${target} ${includes} ${srcs})
endfunction()

# Specify util library
add_library_recurse(util util)
target_compile_features(util PRIVATE cxx_std_17)
target_link_libraries(util PRIVATE glad glm glfw)
target_include_directories(util PUBLIC include)

# Specify sne library
add_library_recurse(sne sne)
target_compile_features(sne PRIVATE cxx_std_17)
target_link_libraries(sne PRIVATE glad glm util cub ResourceEmbed gpufaiss)
target_include_directories(sne PUBLIC include)
rsrcs_embed(sne ${CMAKE_SOURCE_DIR}/resources/shaders)

# Add test executable
add_executable(foo src/app/foo.cpp)
target_compile_features(foo PRIVATE cxx_std_17)
target_link_libraries(foo ResourceEmbed glad glm glfw util sne)
rsrcs_embed(foo ${CMAKE_SOURCE_DIR}/resources/shaders)