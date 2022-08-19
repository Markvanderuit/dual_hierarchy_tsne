cmake_minimum_required (VERSION 3.21)

# Specify project options
option(BUILD_VIS "Build visualization library if ON" ON)
option(BUILD_DEMO "Build demo application if ON; requires BUILD_VIS" ON)

# Tell vcpkg to include required features depending on project options
if(BUILD_VIS OR BUILD_DEMO)
  list(APPEND VCPKG_MANIFEST_FEATURES "vis")
endif()
if(BUILD_DEMO)
  list(APPEND VCPKG_MANIFEST_FEATURES "demo")
endif()

# Specify vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE 
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

# Specify most likely cuda architectures (so not future proof)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "72;75;86")
endif()

# Specify C++/CUDA project
project(DH-SNE LANGUAGES CXX CUDA)

# Include cuda toolkit for linking against CUBlas
find_package(CUDAToolkit REQUIRED)

# Include third party libraries provided through vcpkg
find_package(date CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(faiss CONFIG REQUIRED)
find_package(indicators CONFIG REQUIRED)

# Include optional third party libraries provided through vcpkg
if(BUILD_VIS OR BUILD_DEMO)
  find_package(imgui CONFIG REQUIRED)
endif()
if(BUILD_DEMO)
  find_package(cxxopts CONFIG REQUIRED)
endif()

# Include third party libraries built from source (or added as header only interfaces)
add_subdirectory(third_party/cub)
add_subdirectory(third_party/resource_embed)

# Define function to add a library while recursively globbing its files from a specified directory
function(add_library_recurse target srcDir inclDir)
  file(GLOB_RECURSE srcs ${srcDir}/*.cpp  ${srcDir}/*.cu)
  message("Added library ${target} using sources ${srcs}")
  add_library(${target} ${srcs})
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_include_directories(${target} PUBLIC include)
endfunction()

# Gather shader resources 
rsrcs_embed(shaders ${CMAKE_SOURCE_DIR}/resources/shaders)

# Specify util library
add_library_recurse(util ${CMAKE_SOURCE_DIR}/src/util ${CMAKE_SOURCE_DIR}/include/dh/util)
target_link_libraries(util PUBLIC cub glad::glad glfw glm::glm indicators::indicators date::date faiss)

# Specify sne library
add_library_recurse(sne ${CMAKE_SOURCE_DIR}/src/sne ${CMAKE_SOURCE_DIR}/include/dh/sne)
target_link_libraries(sne PRIVATE util ResourceEmbed indicators::indicators date::date)
target_sources(sne PRIVATE ${shaders}) # Add shaders to sne library

# Specify optional vis library
if(BUILD_VIS OR BUILD_DEMO)
  add_library_recurse(vis ${CMAKE_SOURCE_DIR}/src/vis ${CMAKE_SOURCE_DIR}/include/dh/vis)
  target_link_libraries(vis PRIVATE imgui::imgui util ResourceEmbed indicators::indicators date::date)
  target_sources(vis PRIVATE ${shaders}) # Add shaders to vis library
  target_link_libraries(sne PUBLIC vis) # Link vis library against sne as it is available
endif()

# Specify optional sne_cmd demo application
if(BUILD_VIS AND BUILD_DEMO)
  add_executable(sne_cmd ${CMAKE_SOURCE_DIR}/src/app/sne_cmd.cpp)
  target_link_libraries(sne_cmd PRIVATE cxxopts::cxxopts util sne vis)
  target_compile_features(sne_cmd PRIVATE cxx_std_20)
endif()