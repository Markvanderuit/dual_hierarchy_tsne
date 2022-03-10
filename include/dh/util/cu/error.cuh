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

#include <cuda_runtime.h>
#include "dh/constants.hpp"
#include "dh/util/error.hpp"

namespace dh::util {
  namespace detail {
    inline 
    void cuAssertImpl(cudaError_t err, const char *file, int line) {
      if (err != cudaSuccess) {
        RuntimeError error("CUDA assertion failed");
        error.logs["code"] = cudaGetErrorString(err);
        error.logs["file"] = file;
        error.logs["line"] = line;
        throw error;
      }
    }
  }
} // dh::util

// Simple CUDA assert with err code, file name and line nr. attached. Throws RuntimeError
#ifdef DH_ENABLE_ASSERT
#define cuAssert(statement) dh::util::detail::cuAssertImpl(statement, __FILE__, __LINE__);
#else
#define cuAssert(statement) statement;
#endif