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

#include <algorithm>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>
#include <resource_embed/resource_embed.hpp>
#include "dh/sne/components/field.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Field]");

  // Constants
  constexpr uint knode = 4;
  constexpr uint logk = 2;
  constexpr util::AlignedVec<2, uint> fieldSizePrealloc(256);
  constexpr uint queueHighWater = DH_WORKQUEUE_HIGHWATER * 1024 * 1024;

  // Initial set of node pairs used for dual hierarchy traversal
  constexpr auto initPairs = []() {
    const uint fBegin = (0x2AAAAAAA >> (31u - logk * DH_HIER_INIT_LVL_2D));
    const uint fNodes = 1u << (logk * DH_HIER_INIT_LVL_2D);
    const uint eBegin = (0x2AAAAAAA >> (31u - logk * DH_HIER_INIT_LVL_2D));
    const uint eNodes = 1u << (logk * DH_HIER_INIT_LVL_2D);
    std::array<glm::uvec2, fNodes * eNodes> a {};
    for (uint i = 0; i < fNodes; ++i) {
      for (uint j = 0; j < eNodes; ++j) {
        a[i * eNodes + j] = glm::uvec2(fBegin + i, eBegin + j);
      }
    }
    return a;
  }();
  
  template <>
  Field<2>::Field(MinimizationBuffers minimization, Params params)
  : _isInit(false),
    _minimization(minimization),
    _params(params),
    _hierarchyRebuildIterations(0),
    _size(0),
    _useEmbeddingHierarchy(params.singleHierarchyTheta > 0.0f),
    _useFieldHierarchy(params.dualHierarchyTheta > 0.0f) {
    Logger::newt() << prefix << "Initializing...";
    
    // Initialize shader programs
    {      
      _programs(ProgramType::eDispatch).addShader(util::GLShaderType::eCompute, rsrc::get("sne/dispatch.comp"));
      _programs(ProgramType::eQueryFieldComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/queryField.comp"));
      _programs(ProgramType::eFullCompactDraw).addShader(util::GLShaderType::eVertex, rsrc::get("sne/field/2D/fullCompactStencil.vert"));
      _programs(ProgramType::eFullCompactDraw).addShader(util::GLShaderType::eFragment, rsrc::get("sne/field/2D/fullCompactStencil.frag"));
      _programs(ProgramType::eFullCompactComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/fullCompact.comp"));
      _programs(ProgramType::eFullFieldComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/fullField.comp"));
      _programs(ProgramType::eSingleHierarchyCompactComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/singleHierarchyCompact.comp"));
      _programs(ProgramType::eSingleHierarchyFieldComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/singleHierarchyField.comp"));
      _programs(ProgramType::eDualHierarchyFieldDualSubdivideComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/dualHierarchyFieldIterative.comp"));
      _programs(ProgramType::eDualHierarchyFieldSingleSubdivideComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/dualHierarchyFieldRest.comp"));
      _programs(ProgramType::eDualHierarchyFieldLeafComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/dualHierarchyFieldLeaf.comp"));
      _programs(ProgramType::eDualHierarchyFieldAccumulateComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/dualHierarchyFieldAccumulate.comp"));
      glAssert();
      
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();

      // Set these uniforms exactly once
      const auto theta12 = _params.singleHierarchyTheta * _params.singleHierarchyTheta;
      const auto theta22 = _params.dualHierarchyTheta * _params.dualHierarchyTheta;
      _programs(ProgramType::eQueryFieldComp).template                        uniform<uint>("nPoints", _params.n);
      _programs(ProgramType::eQueryFieldComp).template                        uniform<int>("fieldSampler", 0);
      _programs(ProgramType::eFullCompactComp).template                       uniform<int>("stencilSampler", 0);
      _programs(ProgramType::eFullFieldComp).template                         uniform<uint>("nPoints", _params.n);
      _programs(ProgramType::eSingleHierarchyFieldComp).template              uniform<float>("theta2", theta12);
      _programs(ProgramType::eSingleHierarchyFieldComp).template              uniform<bool>("doBhCrit", true);
      _programs(ProgramType::eDualHierarchyFieldDualSubdivideComp).template   uniform<float>("theta2", theta22);
      _programs(ProgramType::eDualHierarchyFieldDualSubdivideComp).template   uniform<bool>("doBhCrit", true);
      _programs(ProgramType::eDualHierarchyFieldSingleSubdivideComp).template uniform<float>("theta2", theta22);
      _programs(ProgramType::eDualHierarchyFieldSingleSubdivideComp).template uniform<bool>("doBhCrit", true);
      glAssert();
    }

    // Initialize buffer objects
    {
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::ePixelQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glAssert();

      // Dual hierarchy approximation used; initialize work queues
      if (_useFieldHierarchy) {
        _initQueueSize = initPairs.size() * sizeof(glm::uvec2);

        // Reserve significant memory for queues. Estimation of actual required sizes is difficult.
        const uint totalQueueSize = util::glGetAvailableMemory() - queueHighWater;
        const uint queueSize = totalQueueSize / 4u;
        
        glNamedBufferStorage(_buffers(BufferType::ePairsInitQueue), _initQueueSize, initPairs.data(), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsInputQueue), queueSize, nullptr, 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsOutputQueue), queueSize, nullptr, 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsRestQueue), queueSize, nullptr, 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsLeafQueue), queueSize, nullptr, 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsInitQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(initPairs.size(), 1, 1, 1)), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsInputQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsOutputQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsRestQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsLeafQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
        glAssert();
      }
    }

    // Initialize other components
    {
      // Create object handles
      glCreateTextures(GL_TEXTURE_2D, _textures.size(), _textures.data());
      glCreateFramebuffers(1, &_stencilFBOHandle);
      glCreateVertexArrays(1, &_stencilVAOHandle);
      glAssert();

      // Full computation used, specify FBO and VAO for stencil operation
      if (!_useEmbeddingHierarchy) {
        // FBO
        glNamedFramebufferTexture(_stencilFBOHandle, GL_COLOR_ATTACHMENT0, _textures(TextureType::eStencil), 0);
        glNamedFramebufferDrawBuffer(_stencilFBOHandle, GL_COLOR_ATTACHMENT0);
        glAssert();

        // VAO
        glVertexArrayVertexBuffer(_stencilVAOHandle, 0, _minimization.embedding, 0, 2 * sizeof(float));
        glEnableVertexArrayAttrib(_stencilVAOHandle, 0);
        glVertexArrayAttribFormat(_stencilVAOHandle, 0, 2, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayAttribBinding(_stencilVAOHandle, 0, 0);
        glAssert();
      }
    }

    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::rest() << prefix << "Initialized, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Embedding hierarchy used, initialize
    if (_useEmbeddingHierarchy) {
      const EmbeddingHierarchy<2>::Layout layout(_params.n);
      _embeddingHierarchy = EmbeddingHierarchy<2>(_minimization, layout, _params);
    }

    // Field hierarchy used, initialize
    if (_useFieldHierarchy) {
      const FieldHierarchy<2>::Layout layout(fieldSizePrealloc);
      _fieldHierarchy = FieldHierarchy<2>(buffers(), layout, _params);
    }

    _isInit = true;
  }

  template <>
  void Field<2>::resizeField(uvec size) {
    if (_size == size) {
      return;
    }

    _size = size;
    _hierarchyRebuildIterations = 0;

    // (re-)create field texture
    glDeleteTextures(1, &_textures(TextureType::eField));
    glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eField));
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureStorage2D(_textures(TextureType::eField), 1, GL_RGBA32F, _size.x, _size.y);
    glAssert();

    // Full computation used, (re-)create stencil texture
    if (!_useEmbeddingHierarchy) {
      glDeleteTextures(1, &_textures(TextureType::eStencil));
      glCreateTextures(GL_TEXTURE_2D, 1, &_textures(TextureType::eStencil));
      glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTextureStorage2D(_textures(TextureType::eStencil), 1, GL_R8UI, _size.x, _size.y);
      glNamedFramebufferTexture(_stencilFBOHandle, GL_COLOR_ATTACHMENT0, _textures(TextureType::eStencil), 0);
      glAssert();
    }

    // (re-)create work queue for which pixels to compute in field texture
    glDeleteBuffers(1, &_buffers(BufferType::ePixelQueue));
    glCreateBuffers(1, &_buffers(BufferType::ePixelQueue));
    glNamedBufferStorage(_buffers(BufferType::ePixelQueue), product(_size) * sizeof(uvec), nullptr, 0);
    glAssert();
  }
} // dh::sne