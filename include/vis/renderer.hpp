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

#include <memory>
#include <utility>
#include <vector>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "types.hpp"
#include "util/gl/window.hpp"
#include "sne/params.hpp"
#include "vis/trackball_input_task.hpp"

namespace dh::vis {
  template <uint D>
  class Renderer {
  public:
    // Constr/destr
    Renderer();
    Renderer(const util::GLWindow& window, sne::Params params, const std::vector<uint>& labels = {});
    ~Renderer();

    // Copy constr/assignment is explicitly deleted (no copying OpenGL handles)
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Move constr/operator moves handles
    Renderer(Renderer&&) noexcept;
    Renderer& operator=(Renderer&&) noexcept;

    // Tell renderer to process render queue
    void render();

  private:
    // State
    bool _isInit;
    sne::Params _params;
    const util::GLWindow * _windowHandle;
    glm::ivec2 _fboSize;

    // OpenGL object handles
    GLuint _labelsHandle;
    GLuint _fboHandle;
    GLuint _fboColorTextureHandle;
    GLuint _fboDepthTextureHandle;

    // Subcomponents
    std::shared_ptr<vis::TrackballInputTask> _trackballInputTask;

  public:
    bool isInit() { return _isInit; }

    // std::swap impl
    friend void swap(Renderer& a, Renderer& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._windowHandle, b._windowHandle);
      swap(a._fboSize, b._fboSize);
      swap(a._labelsHandle, b._labelsHandle);
      swap(a._fboHandle, b._fboHandle);
      swap(a._fboColorTextureHandle, b._fboColorTextureHandle);
      swap(a._fboDepthTextureHandle, b._fboDepthTextureHandle);
      swap(a._trackballInputTask, b._trackballInputTask);
    }
  };
} // dh::vis