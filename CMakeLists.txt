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

# Ensure OpenMP is available
find_package(OpenMP REQUIRED)

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
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_include_directories(${target} PUBLIC include)
endfunction()

# Gather shader resources 
rsrcs_embed(shaders ${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders)

# Specify util library
add_library_recurse(dh_util ${CMAKE_CURRENT_SOURCE_DIR}/src/util ${CMAKE_CURRENT_SOURCE_DIR}/include/dh/util)
target_link_libraries(dh_util PUBLIC cub glad::glad glfw glm::glm indicators::indicators date::date faiss)

# Specify sne library
add_library_recurse(dh_sne ${CMAKE_CURRENT_SOURCE_DIR}/src/sne ${CMAKE_CURRENT_SOURCE_DIR}/include/dh/sne)
target_link_libraries(dh_sne PRIVATE dh_util ResourceEmbed indicators::indicators date::date)
target_sources(dh_sne PRIVATE ${shaders}) # Add shaders to sne library

# Specify optional vis library
if(BUILD_VIS OR BUILD_DEMO)
  add_library_recurse(dh_vis ${CMAKE_CURRENT_SOURCE_DIR}/src/vis ${CMAKE_CURRENT_SOURCE_DIR}/include/dh/vis)
  target_link_libraries(dh_vis PRIVATE imgui::imgui dh_util ResourceEmbed indicators::indicators date::date)
  target_sources(dh_vis PRIVATE ${shaders}) # Add shaders to vis library
  target_link_libraries(dh_sne PUBLIC dh_vis) # Link vis library against sne as it is available
endif()

# Specify optional dh_sne_cmd demo application
if(BUILD_VIS AND BUILD_DEMO)
  add_executable(dh_sne_cmd ${CMAKE_CURRENT_SOURCE_DIR}/src/app/sne_cmd.cpp)
  target_link_libraries(dh_sne_cmd PRIVATE cxxopts::cxxopts dh_util dh_sne dh_vis)
  target_compile_features(dh_sne_cmd PRIVATE cxx_std_17)
endif()

# Specify optional sne_example demo application
if (BUILD_DEMO)
  add_executable(dh_sne_example ${CMAKE_CURRENT_SOURCE_DIR}/src/app/sne_example.cpp)
  target_link_libraries(dh_sne_example PRIVATE dh_util dh_sne)
  target_compile_features(dh_sne_example PRIVATE cxx_std_20)
endif()

# Configure OpenMP
if(OpenMP_CXX_FOUND)
  target_link_libraries(dh_vis  PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(dh_sne  PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(dh_util PUBLIC OpenMP::OpenMP_CXX)
endif()