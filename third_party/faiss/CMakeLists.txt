cmake_minimum_required(VERSION 3.21)

project(faiss CXX CUDA)

# Ensure the CUDA toolkit is available for FAISS
find_package(CUDAToolkit REQUIRED)

if(NOT MSVC)
	message("adding CXX flags -std=c++14 -fPIC -m64	-Wall -g -O3  -msse4 -mpopcnt -fopenmp -Wno-sign-compare -fpermissive")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -m64 -Wall -g -O3  -msse4 -mpopcnt -fopenmp -Wno-sign-compare -fpermissive")
else(NOT MSVC)
	# set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	add_definitions(-D__AVX__ -D__SSE__)
endif(NOT MSVC)
add_definitions(-DFINTEGER=int)


# Specify FAISS source files
file(GLOB faiss_sources
  src/*.cpp
  src/impl/*.cpp 
  src/utils/*.cpp
)

# Specify GPUFAISS source files
file(GLOB gpufaiss_sources
  src/gpu/*.cpp
  src/gpu/*.cu
  src/gpu/impl/*.cpp 
  src/gpu/impl/*.cu 
  src/gpu/utils/*.cpp
  src/gpu/utils/*.cu
  src/gpu/utils/blockselect/*.cpp
  src/gpu/utils/blockselect/*.cu
  src/gpu/utils/nvidia/*.cpp
  src/gpu/utils/nvidia/*.cu
  src/gpu/utils/warpselect/*.cpp
  src/gpu/utils/warpselect/*.cu
)

# Specify FAISS library
add_library(faiss STATIC ${faiss_sources})
target_link_libraries(faiss OpenBLAS::OpenBLAS LAPACK::LAPACK)
target_compile_features(faiss PRIVATE cxx_std_14)
target_include_directories(faiss PUBLIC include)

# Specify gpuFAISS library
add_library(gpufaiss STATIC ${gpufaiss_sources})
target_link_libraries(gpufaiss faiss CUDA::cudart CUDA::cublas)
target_compile_features(gpufaiss PRIVATE cuda_std_14)
target_include_directories(gpufaiss PUBLIC include)

# Thanks for not updating OpenMP, microsoft
if (NOT MSVC)
  find_package(OpenMP REQUIRED)
  target_link_libraries(faiss OpenMP::OpenMP_CXX)
  target_link_libraries(gpufaiss OpenMP::OpenMP_CXX)
endif(NOT MSVC)