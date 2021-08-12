cmake_minimum_required (VERSION 3.21)

# Specify vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE 
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

# Define C++/CUDA project
project(DualHierarchytSNE LANGUAGES CXX CUDA)

# Include third party libraries from vcpkg
find_package(date CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(LAPACK CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

# Include third party libraries built from source
add_subdirectory(third_party/resource_embed)
add_subdirectory(third_party/glad)
add_subdirectory(third_party/cub)
add_subdirectory(third_party/faiss)

# Define function to add a library while recursively globbing its files
function(add_library_recurse target dir)
  file(GLOB_RECURSE srcs src/${dir}/*.cpp  src/${dir}/*.cu)
  file(GLOB_RECURSE includes include/${dir}/*.hpp  include/${dir}/*.cuh)
  add_library(${target} ${includes} ${srcs})
endfunction()

# Gather shader resources 
rsrcs_embed(shaders ${CMAKE_SOURCE_DIR}/resources/shaders)

# Specify util library
add_library_recurse(util util)
target_compile_features(util PRIVATE cxx_std_17)
target_include_directories(util PUBLIC include)
target_link_libraries(util PUBLIC 
  cub 
  glad
  glfw
  glm::glm
  date::date
)

# Specify sne library
add_library_recurse(sne sne)
target_compile_features(sne PRIVATE cxx_std_17)
target_include_directories(sne PUBLIC include)
target_sources(sne PRIVATE ${shaders})
target_link_libraries(sne PRIVATE 
  util 
  gpufaiss 
  ResourceEmbed
)

# Specify vis library
add_library_recurse(vis vis)
target_compile_features(vis PRIVATE cxx_std_17)
target_include_directories(vis PUBLIC include)
target_sources(vis PRIVATE ${shaders})
target_link_libraries(vis PRIVATE
  util 
  sne 
  ResourceEmbed
)

# Add sne_cmd demo executable
add_executable(sne_cmd src/app/sne_cmd.cpp)
target_compile_features(sne_cmd PRIVATE cxx_std_17)
target_link_libraries(sne_cmd PRIVATE
  util
  sne
  vis
)