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
#include "util/enum.hpp"
#include "util/gl/program.hpp"
#include "sne/sne_params.hpp"
#include "sne/buffers/sne_minimization_buffers.hpp"
#include "vis/render_queue.hpp"

namespace dh::vis {
  template <uint D>
  class EmbeddingRenderTask : public RenderTask {
  public:
    EmbeddingRenderTask();
    EmbeddingRenderTask(sne::SNEMinimizationBuffers minimization, sne::SNEParams params, int priority);
    ~EmbeddingRenderTask();

    // Copy constr/assignment is explicitly deleted
    EmbeddingRenderTask(const EmbeddingRenderTask&) = delete;
    EmbeddingRenderTask& operator=(const EmbeddingRenderTask&) = delete;

    // Move constr/operator moves handles
    EmbeddingRenderTask(EmbeddingRenderTask&&) noexcept;
    EmbeddingRenderTask& operator=(EmbeddingRenderTask&&) noexcept;

    void render(glm::mat4 transform, GLuint labelsHandle) override;

  private:
    enum class BufferType {
      eQuadPositions,
      eQuadElements,

      Length
    };

    // State
    bool _isInit;
    sne::SNEMinimizationBuffers _minimization;
    sne::SNEParams _params;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::GLProgram _program;
    GLuint _vaoHandle;

  public:
    bool isInit() const { return _isInit; }
    
    friend void swap(EmbeddingRenderTask& a, EmbeddingRenderTask& b) noexcept {
      using std::swap;
      swap(static_cast<RenderTask&>(a), static_cast<RenderTask&>(b));
      swap(a._isInit, b._isInit);
      swap(a._minimization, b._minimization);
      swap(a._params, b._params);
      swap(a._buffers, b._buffers);
      swap(a._program, b._program);
      swap(a._vaoHandle, b._vaoHandle);
    }
  };
} // dh::vis