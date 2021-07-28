cmake_minimum_required (VERSION 3.16)

# Define project leveraging C++ and CUDA
project(DualHierarchytSNE LANGUAGES CXX CUDA)

# Add third party libraries
add_subdirectory(third_party/resource_embed)
add_subdirectory(third_party/glad)
add_subdirectory(third_party/glm)
add_subdirectory(third_party/glfw)

# Function to add a library while recursively globbing its files
function(add_library_recurse target dir)
  file(GLOB_RECURSE sources
    src/${dir}/*.cpp
  )
  file(GLOB_RECURSE headers
    include/${dir}/*.hpp
  )
  add_library(${target} ${headers} ${sources})
  target_compile_features(${target} PUBLIC cxx_std_17)
  target_include_directories(${target} PUBLIC include)
endfunction()

# Add and link project libraries
add_library_recurse(util util)
target_link_libraries(util glad glm glfw)

# Add test executable
add_executable(foo src/app/foo.cpp)
target_compile_features(foo PRIVATE cxx_std_17)
target_link_libraries(foo ResourceEmbed util)
rsrc_embed_dir(foo ${CMAKE_SOURCE_DIR}/resources/shaders)