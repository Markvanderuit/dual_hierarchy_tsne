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

#include <utility>
#include <cuda_runtime.h>
#include "util/timer.hpp"

namespace dh::util {
  class CUTimer : public Timer {
  public:
    CUTimer();
    ~CUTimer();

    // Copy constr/assignment is explicitly deleted (no copying handles)
    CUTimer(const CUTimer&) = delete;
    CUTimer& operator=(const CUTimer&) = delete;

    // Move constr/operator moves resource handles
    CUTimer(CUTimer&&) noexcept;
    CUTimer& operator=(CUTimer&&) noexcept;

    // Swap internals with another timer object
    void swap(CUTimer& other) noexcept;

    // Override and implement for this specific timer
    void tick() override;
    void tock() override;
    void poll() override;

  private:
    void *_startHandle;
    void *_stopHandle;
  }
} // dh::util