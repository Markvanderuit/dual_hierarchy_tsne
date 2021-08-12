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
#include "dh/util/gl/error.hpp"
#include "dh/vis/embedding_render_task.hpp"

namespace dh::vis {
  // Quad vertex position data
  constexpr std::array<glm::vec2, 4> quadPositions = {
    glm::vec2(-1, -1),  // 0
    glm::vec2(1, -1),   // 1
    glm::vec2(1, 1),    // 2
    glm::vec2(-1, 1)    // 3
  };

  // Quad element index data
  constexpr std::array<uint, 6> quadElements = {
    0, 1, 2,  2, 3, 0
  };
  
  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask()
  : RenderTask(), _isInit(false) {
    // ...
  }

  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask(sne::MinimizationBuffers minimization, sne::Params params, int priority)
  : RenderTask(priority), _isInit(false), _minimization(minimization), _params(params) {
    // Initialize shader program
    {
      if constexpr (D == 2) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/embedding/2D/embedding.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/embedding/2D/embedding.frag"));
      } else if constexpr (D == 3) {
        _program.addShader(util::GLShaderType::eVertex, rsrc::get("vis/embedding/3D/embedding.vert"));
        _program.addShader(util::GLShaderType::eFragment, rsrc::get("vis/embedding/3D/embedding.frag"));
      }
      _program.link();
      glAssert();
    }

    // Initialize buffer objects
    {
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::ePositions), quadPositions.size() * sizeof(glm::vec2), quadPositions.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eElements), quadElements.size() * sizeof(uint), quadElements.data(), 0);
      glAssert();
    }

    // Initialize vertex array object
    {
      glCreateVertexArrays(1, &_vaoHandle);

      // Specify vertex buffers and element buffer
      constexpr uint embeddingStride = (D == 2) ? 2 : 4;
      glVertexArrayVertexBuffer(_vaoHandle, 0, _buffers(BufferType::ePositions), 0, sizeof(glm::vec2));   // Quad positions
      glVertexArrayVertexBuffer(_vaoHandle, 1, _minimization.embedding, 0, embeddingStride * sizeof(float));  // Embedding positions
      glVertexArrayElementBuffer(_vaoHandle, _buffers(BufferType::eElements));                            // Quad elements/indices

      // Embedding positions advance once for the full set of vertices drawn
      glVertexArrayBindingDivisor(_vaoHandle, 0, 0);
      glVertexArrayBindingDivisor(_vaoHandle, 1, 1);
      
      // Specify vertex array data organization
      constexpr uint embeddingSize = (D == 2) ? 2 : 3;
      glVertexArrayAttribFormat(_vaoHandle, 0, 2, GL_FLOAT, GL_FALSE, 0);
      glVertexArrayAttribFormat(_vaoHandle, 1, embeddingSize, GL_FLOAT, GL_FALSE, 0);

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
  EmbeddingRenderTask<D>::~EmbeddingRenderTask() {
    if (_isInit) {
      glDeleteVertexArrays(1, &_vaoHandle);
      glDeleteBuffers(_buffers.size(), _buffers.data());
    }
  }

  template <uint D>
  EmbeddingRenderTask<D>::EmbeddingRenderTask(EmbeddingRenderTask&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  EmbeddingRenderTask<D>& EmbeddingRenderTask<D>::operator=(EmbeddingRenderTask<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void EmbeddingRenderTask<D>::render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle) {
    _program.bind();

    // Set uniforms
    _program.uniform<glm::mat4>("model_view", model_view);
    _program.uniform<glm::mat4>("proj", proj);
    _program.uniform<float>("pointOpacity", 1.0f);
    _program.uniform<float>("pointRadius", 0.005f);
    _program.uniform<bool>("drawLabels", labelsHandle > 0);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.bounds);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, labelsHandle);

    // Perform draw
    glBindVertexArray(_vaoHandle);
    glDrawElementsInstanced(GL_TRIANGLES, quadElements.size(), GL_UNSIGNED_INT, nullptr, _params.n);
  }

  // Template instantiations for 2/3 dimensions
  template class EmbeddingRenderTask<2>;
  template class EmbeddingRenderTask<3>;
} // dh::vis