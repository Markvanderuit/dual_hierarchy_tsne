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

#include <cuda_runtime.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include "dh/util/cu/knn.cuh"
#include "dh/util/cu/error.cuh"

namespace dh::util {
  // Tuning parameters for FAISS
  constexpr uint nProbe = 12;
  constexpr uint nListMult = 1;

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

  void swap(KNN& a, KNN& b) noexcept {
    using std::swap;
    swap(a._isInit, b._isInit);
    swap(a._n, b._n);
    swap(a._k, b._k);
    swap(a._d, b._d);
    swap(a._dataPtr, b._dataPtr);
    swap(a._interopBuffers, b._interopBuffers);
  }

  void KNN::comp() {
    // Map interop buffers for access on CUDA side
    for (auto& buffer : _interopBuffers) {
      buffer.map();
    }

    // Create temporary space for 64 bit faiss indices
    void * tempIndicesHandle;
    cudaMalloc(&tempIndicesHandle, _n * _k * sizeof(faiss::Index::idx_t));

    // Nr. of inverted lists used by FAISS IVL.
    // O(sqrt(n)) is apparently reasonable
    // src: https://github.com/facebookresearch/faiss/issues/112
    const uint nLists = nListMult * static_cast<uint>(std::sqrt(_n)); 

    // Use a single GPU device. For now, just grab device 0 and pray
    faiss::gpu::StandardGpuResources faissResources;
    faiss::gpu::GpuIndexIVFFlatConfig faissConfig;
    faissConfig.device = 0;
    faissConfig.indicesOptions = faiss::gpu::INDICES_32_BIT;
    faissConfig.flatConfig.useFloat16 = true;
    faissConfig.interleavedLayout = false;

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
    faissIndex.add(_n, _dataPtr);

    // Perform actual search
    // Store results device cide in cuKnnSquaredDistances, cuKnnIndices, as the
    // rest of construction is performed on device as well.
    faissIndex.search(
      _n,
      _dataPtr,
      _k,
      (float *) _interopBuffers(BufferType::eDistances).cuHandle(),
      (faiss::Index::idx_t *) tempIndicesHandle
    );
    
    // Tell FAISS to bugger off
    faissIndex.reset();
    faissIndex.reclaimMemory();

    // Free temporary indices memory, after writing it to 32 bit interoperability buffer
    kernDownCast<<<1024, 256>>>(_n * _k, (int64_t *) tempIndicesHandle, (int32_t *) _interopBuffers(BufferType::eIndices).cuHandle());
    cudaDeviceSynchronize();
    cudaFree(tempIndicesHandle);

    // Unmap interop buffers
    for (auto& buffer : _interopBuffers) {
      buffer.unmap();
    }
  }
} // dh::util