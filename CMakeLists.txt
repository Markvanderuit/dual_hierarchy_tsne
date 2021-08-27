cmake_minimum_required (VERSION 3.21)

# Specify vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE 
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

# Define C++/CUDA project
project(DualHierarchytSNE LANGUAGES CXX CUDA)

# Include third party libraries from vcpkg
find_package(cxxopts CONFIG REQUIRED)
find_package(date CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(LAPACK CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# Include third party libraries built from source (or added as header only interfaces)
add_subdirectory(third_party/cub)
add_subdirectory(third_party/faiss)
add_subdirectory(third_party/glad)
add_subdirectory(third_party/indicators)
add_subdirectory(third_party/resource_embed)

# Define function to add a library while recursively globbing its files from a specified directory
function(add_library_recurse target srcDir inclDir)
  file(GLOB_RECURSE srcs ${srcDir}/*.cpp  ${srcDir}/*.cu)
  file(GLOB_RECURSE includes ${inclDir}/*.hpp ${inclDir}/*.cuh)
  add_library(${target} ${includes} ${srcs})
endfunction()

# Gather shader resources 
rsrcs_embed(shaders ${CMAKE_SOURCE_DIR}/resources/shaders)

# Specify util library
add_library_recurse(util ${CMAKE_SOURCE_DIR}/src/util ${CMAKE_SOURCE_DIR}/include/dh/util)
target_compile_features(util PRIVATE cxx_std_17)
target_include_directories(util PUBLIC include)
target_link_libraries(util PUBLIC 
  cub 
  glad
  glfw
  glm::glm
  indicators
  date::date
)

# Specify sne library
add_library_recurse(sne ${CMAKE_SOURCE_DIR}/src/sne ${CMAKE_SOURCE_DIR}/include/dh/sne)
target_compile_features(sne PRIVATE cxx_std_17)
target_include_directories(sne PUBLIC include)
target_sources(sne PRIVATE ${shaders})
target_link_libraries(sne PRIVATE 
  util 
  gpufaiss 
  ResourceEmbed
)

# Specify vis library
add_library_recurse(vis ${CMAKE_SOURCE_DIR}/src/vis ${CMAKE_SOURCE_DIR}/include/dh/vis)
target_compile_features(vis PRIVATE cxx_std_17)
target_include_directories(vis PUBLIC include)
target_sources(vis PRIVATE ${shaders})
target_link_libraries(vis PRIVATE
  imgui::imgui
  util 
  sne 
  ResourceEmbed
)

# Add sne_cmd demo executable
add_executable(sne_cmd ${CMAKE_SOURCE_DIR}/src/app/sne_cmd.cpp)
target_compile_features(sne_cmd PRIVATE cxx_std_17)
target_link_libraries(sne_cmd PRIVATE
  cxxopts::cxxopts
  util
  sne
  vis
)