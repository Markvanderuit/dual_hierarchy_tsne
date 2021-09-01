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
#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <cub/cub.cuh>
#include "dh/util/cu/error.cuh"
#include "dh/util/cu/inclusive_scan.cuh"

namespace dh::util {
  InclusiveScan::InclusiveScan()
  : _isInit(false), _n(0), _tempHandle(nullptr), _tempSize(0) {
    // ...
  }

  InclusiveScan::InclusiveScan(GLuint inputBuffer, GLuint outputBuffer, uint n)
  : _isInit(false), _n(n), _tempHandle(nullptr), _tempSize(0) {      
    // Set up temp memory
    cub::DeviceScan::InclusiveSum<uint *, uint *>(nullptr, _tempSize, nullptr, nullptr, _n);
    cuAssert(cudaMalloc(&_tempHandle, _tempSize));

    // Set up OpenGL-CUDA interoperability
    _interopBuffers(BufferType::eInputBuffer) = CUGLInteropBuffer(inputBuffer, CUGLInteropType::eReadOnly);
    _interopBuffers(BufferType::eOutputBuffer) = CUGLInteropBuffer(outputBuffer, CUGLInteropType::eWriteDiscard);

    _isInit = true;
  }

  InclusiveScan::~InclusiveScan() {
    if (_isInit) {
      cudaFree(_tempHandle);
    }
  }
  
  InclusiveScan::InclusiveScan(InclusiveScan&& other) noexcept {
    swap(*this, other);
  }

  InclusiveScan& InclusiveScan::operator=(InclusiveScan&& other) noexcept {
    swap(*this, other);
    return *this;
  }
  
  void swap(InclusiveScan& a, InclusiveScan& b) noexcept {
    using std::swap;
    swap(a._isInit, b._isInit);
    swap(a._n, b._n);
    swap(a._tempHandle, b._tempHandle);
    swap(a._tempSize, b._tempSize);
    swap(a._interopBuffers, b._interopBuffers);
  }
  
  void InclusiveScan::comp() {
    // Map interop buffers for access on CUDA side
    for (auto& buffer : _interopBuffers) {
      buffer.map();
    }
    
    // Perform inclusive scan
    cub::DeviceScan::InclusiveSum<uint *, uint *>(
      (void *) _tempHandle,
      _tempSize,
      (uint *) _interopBuffers(BufferType::eInputBuffer).cuHandle(),
      (uint *) _interopBuffers(BufferType::eOutputBuffer).cuHandle(),
      _n
    );

    // Unmap interop buffers
    for (auto& buffer : _interopBuffers) {
      buffer.unmap();
    }
  }
} // dh::util