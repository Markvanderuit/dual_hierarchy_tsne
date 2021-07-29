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
#include "util/cu/timer.cuh"

namespace dh::util {
  CUTimer::CUTimer() : Timer() {
    cudaEventCreate((cudaEvent_t *) &_startHandle);
    cudaEventCreate((cudaEvent_t *) &_stopHandle);
  }

  CUTimer::~CUTimer() {
    cudaEventDestroy((cudaEvent_t) _startHandle);
    cudaEventDestroy((cudaEvent_t) _stopHandle);
  }

  CUTimer::CUTimer(CUTimer&& other) noexcept {
    swap(*this, other);
  }

  CUTimer& CUTimer::operator=(CUTimer&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void swap(CUTimer& a, CUTimer& b) noexcept {
    using std::swap;
    swap(a._values, b._values);
    swap(a._iterations, b._iterations);
    swap(a._startHandle, b._startHandle);
    swap(a._stopHandle, b._stopHandle);
  }

  void CUTimer::tick() {
    cudaEventRecord((cudaEvent_t) _startHandle);
  }

  void CUTimer::tock() {
    cudaEventRecord((cudaEvent_t) _stopHandle);
  }

  void CUTimer::poll() {
    float fElapsed;
    
    // Query elapsed time (maximum microsecond resolution)
    cudaEventSynchronize((cudaEvent_t) _stopHandle);
    cudaEventElapsedTime(&fElapsed, (cudaEvent_t) _startHandle, (cudaEvent_t) _stopHandle);
    long long elapsed = static_cast<long long>(1000000.f * fElapsed);

    // Update last, total, average times
    _values(TimerValue::eLast) = std::chrono::nanoseconds(elapsed);
    _values(TimerValue::eTotal) += _values(TimerValue::eLast);
    _values(TimerValue::eAverage) = _values(TimerValue::eAverage)
      + (_values(TimerValue::eLast) - _values(TimerValue::eAverage)) / (++_iterations);
  }
}