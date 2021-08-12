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

#include <numeric>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <cub/cub.cuh>
#include "dh/util/cu/error.cuh"
#include "dh/util/cu/key_sort.cuh"

namespace dh::util {
  KeySort::KeySort()
  : _isInit(false), _n(0), _bits(0), _orderHandle(nullptr), _tempHandle(nullptr), _tempSize(0) {
    // ...
  }

  KeySort::KeySort(GLuint inputBuffer, GLuint outputBuffer, GLuint outputOrderBuffer, uint n, uint bits)
  : _isInit(false), _n(n), _bits(bits), _orderHandle(nullptr), _tempHandle(nullptr), _tempSize(0) {
    // Set up temp memory
    const int msb = 30;
    const int lsb = msb - _bits;
    cub::DeviceRadixSort::SortPairs<uint, uint>(nullptr, _tempSize, nullptr, nullptr, nullptr, nullptr, _n, lsb, msb);
    cuAssert(cudaMalloc(&_tempHandle, _tempSize));

    // Set up order memory to get a mapping of the sort
    std::vector<uint> order(_n);
    std::iota(order.begin(), order.end(), 0u);
    cuAssert(cudaMalloc(&_orderHandle, order.size() * sizeof(uint)));
    cudaMemcpy(_orderHandle, order.data(), order.size() * sizeof(uint), cudaMemcpyHostToDevice);

    // Set up OpenGL-CUDA interoperability
    _interopBuffers(BufferType::eInputBuffer) = CUGLInteropBuffer(inputBuffer, CUGLInteropType::eReadOnly);
    _interopBuffers(BufferType::eOutputBuffer) = CUGLInteropBuffer(outputBuffer, CUGLInteropType::eWriteDiscard);
    _interopBuffers(BufferType::eOutputOrderBuffer) = CUGLInteropBuffer(outputOrderBuffer, CUGLInteropType::eWriteDiscard);

    _isInit = true;
  }

  KeySort::~KeySort() {
    if (_isInit) {
      cudaFree(_orderHandle);
      cudaFree(_tempHandle);
    }
  }

  KeySort::KeySort(KeySort&& other) noexcept {
    swap(*this, other);
  }

  KeySort& KeySort::operator=(KeySort&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void swap(KeySort& a, KeySort& b) noexcept {
    using std::swap;
    swap(a._isInit, b._isInit);
    swap(a._n, b._n);
    swap(a._bits, b._bits);
    swap(a._orderHandle, b._orderHandle);
    swap(a._tempHandle, b._tempHandle);
    swap(a._tempSize, b._tempSize);
    swap(a._interopBuffers, b._interopBuffers);
  }

  void KeySort::sort() {
    // Map interop buffers for access on CUDA side
    for (auto& buffer : _interopBuffers) {
      buffer.map();
    }

    // Perform radix sort
    const int msb = 30;
    const int lsb = msb - _bits;
    cub::DeviceRadixSort::SortPairs<uint, uint>(
      (void *) _tempHandle,
      _tempSize,
      (uint *) _interopBuffers(BufferType::eInputBuffer).cuHandle(),
      (uint *) _interopBuffers(BufferType::eOutputBuffer).cuHandle(),
      (uint *) _orderHandle,
      (uint *) _interopBuffers(BufferType::eOutputOrderBuffer).cuHandle(),
      (int) _n,
      lsb, msb
    );

    // Unmap interop buffers
    for (auto& buffer : _interopBuffers) {
      buffer.unmap();
    }
  }
} // dh::util