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

#pragma once

#include <glad/glad.h>
#include "types.hpp"

namespace dh::util {
  enum class CUGLInteropType {
    eNone,
    eReadOnly,
    eWriteDiscard
  };
 
  class CUGLInteropBuffer {
  public:
    CUGLInteropBuffer();
    CUGLInteropBuffer(GLuint handle, CUGLInteropType type = CUGLInteropType::eNone);
    ~CUGLInteropBuffer();

    // Copy constr/assignment is explicitly deleted (no copying handles)
    CUGLInteropBuffer(const CUGLInteropBuffer&) = delete;
    CUGLInteropBuffer& operator=(const CUGLInteropBuffer&) = delete;

    // Move constr/operator moves handles
    CUGLInteropBuffer(CUGLInteropBuffer&&) noexcept;
    CUGLInteropBuffer& operator=(CUGLInteropBuffer&&) noexcept;

    // Swap internals with another object
    friend void swap(CUGLInteropBuffer& a, CUGLInteropBuffer& b) noexcept;
    
    // Map/unmap CUDA handle for access on CUDA side
    void map();
    void unmap();
    
    // Get pointer to interoperable memory on either side
    void * cuHandle() const;  // Must be mapped before access
    GLuint glHandle() const;  // Must be unmapped before access

    bool isInit() const { return _isInit; }
    bool isMapped() const { return _isMapped; }

  private:
    bool _isInit;
    bool _isMapped;
    struct cudaGraphicsResource *_cuResource;
    GLuint _glHandle;
    void* _cuHandle;
  };

  /* class CUInterop {
  public:
    CUInterop();
    ~CUInterop();

    void map();
    void unmap();

    bool isInit() const { return _isInit; }
    bool isMapped() const { return _isMapped; }

  private:
    bool _isInit;
    bool _isMapped;
  };

  class CUPairSorter : public CUInterop {
  public:
    void init();
  };

  class CUInclusiveScanner : public CUInterop {
  public:
    void init();
  }; */
} // dh::util