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

#include "dh/types.hpp"
#include "dh/util/aligned.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/gl/program.hpp"
#include "dh/sne/params.hpp"
#include "dh/sne/components/buffers.hpp"
#include "dh/vis/render_queue.hpp"

namespace dh::vis {
  template <uint D>
  class FieldHierarchyRenderTask : public RenderTask {
    using vec = util::AlignedVec<D, float>;

  public:
    FieldHierarchyRenderTask();
    FieldHierarchyRenderTask(sne::FieldHierarchyBuffers fieldHierarchy, sne::Params params, int priority);
    ~FieldHierarchyRenderTask();

    // Copy constr/assignment is explicitly deleted
    FieldHierarchyRenderTask(const FieldHierarchyRenderTask&) = delete;
    FieldHierarchyRenderTask& operator=(const FieldHierarchyRenderTask&) = delete;

    // Move constr/operator moves handles
    FieldHierarchyRenderTask(FieldHierarchyRenderTask&&) noexcept;
    FieldHierarchyRenderTask& operator=(FieldHierarchyRenderTask&&) noexcept;

    void render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle) override;
    void drawImGuiComponent() override;

  private:
    enum class BufferType {
      ePositions,
      eElements,

      Length
    };

    // State
    bool _isInit;
    sne::Params _params;
    sne::MinimizationBuffers _minimization;
    sne::FieldBuffers _field;
    sne::FieldHierarchyBuffers _fieldHierarchy;

    // ImGui state
    float _lineWidth;
    float _lineOpacity;
    bool _selectLvl;
    uint _selectedLvl;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::GLProgram _program;
    GLuint _vaoHandle;

  public:
    bool isInit() const { return _isInit; }
    
    // std::swap impl
    friend void swap(FieldHierarchyRenderTask& a, FieldHierarchyRenderTask& b) noexcept {
      using std::swap;
      swap(static_cast<RenderTask&>(a), static_cast<RenderTask&>(b));
      swap(a._isInit, b._isInit);
      swap(a._fieldHierarchy, b._fieldHierarchy);
      swap(a._lineWidth, b._lineWidth);
      swap(a._lineOpacity, b._lineOpacity);
      swap(a._selectLvl, b._selectLvl);
      swap(a._selectedLvl, b._selectedLvl);
      swap(a._params, b._params);
      swap(a._buffers, b._buffers);
      swap(a._program, b._program);
      swap(a._vaoHandle, b._vaoHandle);
    }
  };
} // dh::vis