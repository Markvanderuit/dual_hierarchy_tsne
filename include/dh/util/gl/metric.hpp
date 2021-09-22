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

#include <iostream>
#include <glad/glad.h>
#include "dh/util/enum.hpp"

namespace dh::util {
  // Query a buffer's memory size in bytes
  inline
  GLuint glGetBufferSize(GLuint handle) {
    GLint size = 0;
    glGetNamedBufferParameteriv(handle, GL_BUFFER_SIZE, &size);
    return static_cast<GLuint>(size);
  }

  inline 
  GLuint glGetBuffersSize(GLsizei n, GLuint* handles) {
    GLuint size = 0;
    for (GLsizei i = 0; i < n; ++i) {
      size += glGetBufferSize(handles[i]);
    }
    return size;
  }

  inline
  GLuint glGetTextureSize(GLuint handle) {
    GLint size = 0;

    // Determine resolution
    {
      GLint w, h, d;
      glGetTextureLevelParameteriv(handle, 0, GL_TEXTURE_WIDTH, &w);
      glGetTextureLevelParameteriv(handle, 0, GL_TEXTURE_HEIGHT, &h);
      glGetTextureLevelParameteriv(handle, 0, GL_TEXTURE_DEPTH, &d);
      size = w * h * d;
    }

    // Determine component size
    if (size != 0) {
      GLint r, g, b, a, d;
      glGetTextureLevelParameteriv(handle, 0, GL_TEXTURE_RED_SIZE, &r);
      glGetTextureLevelParameteriv(handle, 0, GL_TEXTURE_GREEN_SIZE, &g);
      glGetTextureLevelParameteriv(handle, 0, GL_TEXTURE_BLUE_SIZE, &b);
      glGetTextureLevelParameteriv(handle, 0, GL_TEXTURE_ALPHA_SIZE, &a);
      glGetTextureLevelParameteriv(handle, 0, GL_TEXTURE_DEPTH_SIZE, &d);
      size *= (r + g + b + a + d);
    }

    return size;
  }

  inline
  GLuint glGetTexturesSize(GLsizei n, GLuint* handles) {
    GLuint size = 0;
    for (GLsizei i = 0; i < n; ++i) {
      size += glGetTextureSize(handles[i]);
    }
    return size;
  }

  /**
   * NVIDIA-only memory queries using the GL_NVX_gpu_memory_info extension. Given that we
   * depend on CUDA anyways, there's no point in adapting to other systems for now.
   * 
   * Note: these are guesstimates, and not representative of actual current memory status,
   * as storage allocations tend to lag behind until they're actually necessary. Bummer.
   */
  #define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
  #define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

  inline
  GLuint glGetTotalMemory() {
    GLint r;
    glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &r);
    return (GLuint) r;
  }

  inline
  GLuint glGetAvailableMemory() {
    GLint r;
    glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &r);
    return (GLuint) r;
  }

  inline
  GLuint glGetReservedMemory() {
    return glGetTotalMemory() - glGetAvailableMemory();
  }
} // dh::util