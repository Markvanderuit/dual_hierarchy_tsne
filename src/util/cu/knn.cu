/*
 * MIT License
 *
 * Copyright (c) 2021 Mark van de Ruit (Delft University of Technology)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include "dh/util/cu/knn.cuh"
#include "dh/util/cu/error.cuh"

namespace dh::util {
  // Tuning parameters for FAISS
  constexpr uint nProbe = 12;
  constexpr uint nListMult = 4;
  constexpr size_t addBatchSize = 32768;
  constexpr size_t searchBatchSize = 16384;

  // Downcast kernel to move from int64_t to int32_t data
  // Used because FAISS sticks to returning 64 bit indices
  // even for the GPU, which hates 64 bit anyways...
  __global__
  void kernDownCast(uint n, const int64_t * input, int32_t * output) {
    for (uint i = blockIdx.x * blockDim.x + threadIdx.x; 
      i < n; 
      i += blockDim.x * gridDim.x) 
    {
      output[i] = static_cast<int32_t>(input[i]);
    }
  }

  KNN::KNN() 
  : _isInit(false), _n(0), _k(0), _d(0), _dataPtr(nullptr) {
    // ...
  }

  KNN::KNN(const float * dataPtr, GLuint distancesBuffer, GLuint indicesBuffer, uint n, uint k, uint d)
  : _isInit(false), _n(n), _k(k), _d(d), _dataPtr(dataPtr) {
    
    // Set up OpenGL-CUDA interoperability
    _interopBuffers(BufferType::eDistances) = CUGLInteropBuffer(distancesBuffer, CUGLInteropType::eNone);
    _interopBuffers(BufferType::eIndices) = CUGLInteropBuffer(indicesBuffer, CUGLInteropType::eNone);

    _isInit = true;
  }

  KNN::~KNN() {
    if (_isInit) {
      // ...
    }
  }

  KNN::KNN(KNN&& other) noexcept {
    swap(*this, other);
  }

  KNN& KNN::operator=(KNN&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void KNN::comp() {
    // Map interop buffers for access on CUDA side
    for (auto& buffer : _interopBuffers) {
      buffer.map();
    }

    // Nr. of inverted lists used by FAISS IVL.
    // x * O(sqrt(n)) | x := 4, is apparently reasonable?
    // src: https://github.com/facebookresearch/faiss/issues/112
    const uint nLists = nListMult * static_cast<uint>(std::sqrt(_n)); 

    // Use a single GPU device. For now, just grab device 0 and pray
    faiss::gpu::StandardGpuResources faissResources;
    faiss::gpu::GpuIndexIVFFlatConfig faissConfig;
    faissConfig.device = 0;
    faissConfig.indicesOptions = faiss::gpu::INDICES_32_BIT;
    faissConfig.flatConfig.useFloat16 = true;
    faissConfig.interleavedLayout = false; // memory impact; 3M point dataset can't fit in 8GB? Bah.

    // Construct search index
    // Inverted file flat list gives accurate results at significant memory overhead.
    faiss::gpu::GpuIndexIVFFlat faissIndex(
      &faissResources,
      _d, 
      nLists,
      faiss::METRIC_L2, 
      faissConfig
    );
    faissIndex.setNumProbes(nProbe);
    faissIndex.train(_n, _dataPtr);

    // Add data in batches
    for (size_t i = 0; i < ceilDiv((size_t) _n, addBatchSize); ++i) {
      const size_t offset = i * addBatchSize;
      const size_t size = std::min(addBatchSize, _n - offset);
      faissIndex.add(size, _dataPtr + (_d * offset));
    }

    // Create temporary space for storing 64 bit faiss indices
    void * tempIndicesHandle;
    cudaMalloc(&tempIndicesHandle, _n * _k * sizeof(faiss::Index::idx_t));

    // Perform search in batches   
    for (size_t i = 0; i < ceilDiv((size_t) _n, searchBatchSize); ++i) {
      const size_t offset = i * searchBatchSize;
      const size_t size = std::min(searchBatchSize, _n - offset);
      faissIndex.search(
        size,
        _dataPtr + (_d * offset),
        _k,
        ((float *) _interopBuffers(BufferType::eDistances).cuHandle()) + (_k * offset),
        ((faiss::Index::idx_t *) tempIndicesHandle) + (_k * offset)
      );
    }
    
    // Tell FAISS to bugger off
    faissIndex.reset();
    faissIndex.reclaimMemory();

    // Free 64-bit temporary indices, after downcasting to 32 bit in the interop buffer
    kernDownCast<<<1024, 256>>>(_n * _k, (int64_t *) tempIndicesHandle, (int32_t *) _interopBuffers(BufferType::eIndices).cuHandle());
    cudaDeviceSynchronize();
    cudaFree(tempIndicesHandle);

    // Unmap interop buffers
    for (auto& buffer : _interopBuffers) {
      buffer.unmap();
    }
  }
} // dh::util