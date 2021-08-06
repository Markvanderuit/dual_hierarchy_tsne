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

#include <utility>
#include <cuda_runtime.h>
#include "util/cu/error.cuh"
#include "util/cu/interop.cuh"
#include <cuda_gl_interop.h>

namespace dh::util {
  CUGLInteropBuffer::CUGLInteropBuffer()
  : _isInit(false), _isMapped(false) {
    // ...
  }

  CUGLInteropBuffer::CUGLInteropBuffer(GLuint handle, CUGLInteropType type)
  : _isInit(false), _isMapped(false), _glHandle(handle) {
    // Register buffer for interoperability
    uint flag;
    switch (type) {
      case CUGLInteropType::eNone:
        flag = cudaGraphicsRegisterFlagsNone;  
        break;
      case CUGLInteropType::eReadOnly:
        flag = cudaGraphicsRegisterFlagsReadOnly;  
        break;
      case CUGLInteropType::eWriteDiscard:
        flag = cudaGraphicsRegisterFlagsWriteDiscard;  
        break;
    };
    cuAssert(cudaGraphicsGLRegisterBuffer(&_cuResource, _glHandle, flag));
    
    _isInit = true;
  }

  CUGLInteropBuffer::~CUGLInteropBuffer() {
    if (_isInit) {
      if (_isMapped) {
        unmap();
      }
      cuAssert(cudaGraphicsUnregisterResource(_cuResource));
    }
  }

  CUGLInteropBuffer::CUGLInteropBuffer(CUGLInteropBuffer&& other) noexcept {
    swap(*this, other);
  }

  CUGLInteropBuffer& CUGLInteropBuffer::operator=(CUGLInteropBuffer&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void swap(CUGLInteropBuffer& a, CUGLInteropBuffer& b) noexcept {
    using std::swap;
    swap(a._isInit, b._isInit);
    swap(a._isMapped, b._isMapped);
    swap(a._cuResource, b._cuResource);
    swap(a._glHandle, b._glHandle);
    swap(a._cuHandle, b._cuHandle);
  }
  
  void CUGLInteropBuffer::map() {
    if (_isMapped) {
      return;
    }
    cuAssert(cudaGraphicsMapResources(1, &_cuResource));
    cuAssert(cudaGraphicsResourceGetMappedPointer(&_cuHandle, nullptr, _cuResource));
    _isMapped = true;
  }

  void CUGLInteropBuffer::unmap() {
    if (!_isMapped) {
      return;
    }
    cuAssert(cudaGraphicsUnmapResources(1, &_cuResource));
    _isMapped = false;
  }
  
  void* CUGLInteropBuffer::cuHandle() const {
    runtimeAssert(_isMapped, "CUGLInteropBuffer was unmapped when trying to access CUDA handle");
    return _cuHandle;
  }

  GLuint CUGLInteropBuffer::glHandle() const {
    runtimeAssert(!_isMapped, "CUGLInteropBuffer was mapped when trying to access OpenGL handle");
    return _glHandle;
  }
} // dh::util