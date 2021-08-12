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

#include <array>
#include <resource_embed/resource_embed.hpp>
#include <glad/glad.h>
#include "dh/util/gl/error.hpp"
#include "dh/vis/components/field_hierarchy_render_task.hpp"

namespace dh::vis {
  // Quad vertex position data
  constexpr std::array<glm::vec2, 4> quadPositions = {
    glm::vec2(-0.5, -0.5),  // 0
    glm::vec2(0.5, -0.5),   // 1
    glm::vec2(0.5, 0.5),    // 2
    glm::vec2(-0.5, 0.5)    // 3
  };

  // Quad element index data (for line draw)
  constexpr std::array<uint, 8> quadElements = {
    0, 1,  1, 2,  2, 3,  3, 0
  };
  
  // Cube vertex position data
  constexpr std::array<glm::vec4, 8> cubePositions = {
    glm::vec4(-0.5, -0.5, -0.5, 0.f),  // 0
    glm::vec4(0.5, -0.5, -0.5, 0.f),   // 1
    glm::vec4(0.5, 0.5, -0.5, 0.f),    // 2
    glm::vec4(-0.5, 0.5, -0.5, 0.f),   // 3
    glm::vec4(-0.5, -0.5, 0.5, 0.f),   // 4
    glm::vec4(0.5, -0.5, 0.5, 0.f),    // 5
    glm::vec4(0.5, 0.5, 0.5, 0.f),     // 6
    glm::vec4(-0.5, 0.5, 0.5, 0.f)     // 7
  };

  // Cube element index data (for line draw)
  constexpr std::array<uint, 24> cubeElements = {
    0, 1,  1, 2,  2, 3,  3, 0,    // bottom
    0, 4,  1, 5,  2, 6,  3, 7,    // sides
    4, 5,  5, 6,  6, 7,  7, 4     // top
  };

  template <uint D>
  FieldHierarchyRenderTask<D>::FieldHierarchyRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  template <uint D>
  FieldHierarchyRenderTask<D>::FieldHierarchyRenderTask(sne::FieldHierarchyBuffers fieldHierarchy, sne::Params params, int priority)
  : RenderTask(priority), _isInit(false), _fieldHierarchy(fieldHierarchy), _params(params) {
    // Initialize shader program 
    {
      if constexpr (D == 2) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/field_hierarchy/2D/field_hierarchy.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/field_hierarchy/2D/field_hierarchy.frag"));
      } else if constexpr (D == 3) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/field_hierarchy/3D/field_hierarchy.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/field_hierarchy/3D/field_hierarchy.frag"));
      }
      _program.link();
      glAssert();
    }
    
    // Initialize buffer objects
    {
      glCreateBuffers(_buffers.size(), _buffers.data());
      if constexpr (D == 2) {
        glNamedBufferStorage(_buffers(BufferType::ePositions), quadPositions.size() * sizeof(vec), quadPositions.data(), 0);
        glNamedBufferStorage(_buffers(BufferType::eElements), quadElements.size() * sizeof(uint), quadElements.data(), 0);
      } else if constexpr (D == 3) {
        glNamedBufferStorage(_buffers(BufferType::ePositions), cubePositions.size() * sizeof(vec), cubePositions.data(), 0);
        glNamedBufferStorage(_buffers(BufferType::eElements), cubeElements.size() * sizeof(uint), cubeElements.data(), 0);
      }
      glAssert();
    }
    
    // Initialize vertex array object
    {
      glCreateVertexArrays(1, &_vaoHandle);

      // Specify vertex buffers and element buffer
      glVertexArrayVertexBuffer(_vaoHandle, 0, _buffers(BufferType::ePositions), 0, sizeof(vec));
      glVertexArrayVertexBuffer(_vaoHandle, 1, _fieldHierarchy.node, 0, sizeof(uint));
      glVertexArrayElementBuffer(_vaoHandle, _buffers(BufferType::eElements));

      // Field hierarchy properties advance once for the full set of vertices drawn
      glVertexArrayBindingDivisor(_vaoHandle, 0, 0);
      glVertexArrayBindingDivisor(_vaoHandle, 1, 1);

      // Specify vertex array data organization
      glVertexArrayAttribFormat(_vaoHandle, 0, (D == 2) ? 2 : 4, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribIFormat(_vaoHandle, 1, 1, GL_UNSIGNED_INT, 0);

      // Other VAO properties
      glEnableVertexArrayAttrib(_vaoHandle, 0);
      glEnableVertexArrayAttrib(_vaoHandle, 1);
      glVertexArrayAttribBinding(_vaoHandle, 0, 0);
      glVertexArrayAttribBinding(_vaoHandle, 1, 1);

      glAssert();
    }

    _isInit = true;
  }

  template <uint D>
  FieldHierarchyRenderTask<D>::~FieldHierarchyRenderTask() {
    if (_isInit) {
      glDeleteVertexArrays(1, &_vaoHandle);
      glDeleteBuffers(_buffers.size(), _buffers.data());
    }
  }

  template <uint D>
  FieldHierarchyRenderTask<D>::FieldHierarchyRenderTask(FieldHierarchyRenderTask&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  FieldHierarchyRenderTask<D>& FieldHierarchyRenderTask<D>::operator=(FieldHierarchyRenderTask<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void FieldHierarchyRenderTask<D>::render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle) {
    _program.bind();

    // Set uniforms
    _program.uniform<glm::mat4>("transform", proj * model_view);
    _program.uniform<float>("opacity", 1.0f);
    _program.uniform<bool>("selectLvl", false);
    _program.uniform<uint>("selectedLvl", 3);
    if constexpr (D == 2) {
      _program.uniform<bool>("sumLvls", false);
    }

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _fieldHierarchy.field);

    // Specify line width for GL_LINES draw
    glLineWidth(1.0f);

    // Obtain nr. of instances to draw based on buffer sizes
    int nNodes;
    glGetNamedBufferParameteriv(_fieldHierarchy.node, GL_BUFFER_SIZE, &nNodes);
    nNodes /= sizeof(uint);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElementsInstanced(GL_LINES, (D == 2) ? quadElements.size() : cubeElements.size(), GL_UNSIGNED_INT, nullptr, nNodes);
  }

  // Explicit template instantiations
  template class FieldHierarchyRenderTask<2>;
  template class FieldHierarchyRenderTask<3>;
}