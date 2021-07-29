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

#include "types.hpp"
#include "util/enum.hpp"
#include "util/cu/interop.cuh"

namespace dh::util {
  class InclusiveScan {
  public:
    InclusiveScan();
    InclusiveScan(GLuint inputBuffer, GLuint outputBuffer, uint n);
    ~InclusiveScan();

    // Copy constr/assignment is explicitly deleted (no copying handles)
    InclusiveScan(const InclusiveScan&) = delete;
    InclusiveScan& operator=(const InclusiveScan&) = delete;

    // Move constr/operator moves handles
    InclusiveScan(InclusiveScan&&) noexcept;
    InclusiveScan& operator=(InclusiveScan&&) noexcept;

    // Swap internals with another object
    friend void swap(InclusiveScan& a, InclusiveScan& b) noexcept;

    // Perform inclusive scan over input buffer, store in output buffer
    void comp();

    bool isInit() const { return _isInit; }

  private:
    enum class BufferType { 
      eInputBuffer,
      eOutputBuffer,

      Length
    };

    bool _isInit;
    uint _n;
    void * _tempHandle;
    size_t _tempSize;
    EnumArray<BufferType, CUGLInteropBuffer> _interopBuffers;
  };
} // dh::util