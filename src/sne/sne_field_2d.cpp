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
#include <glm/gtc/type_ptr.hpp>
#include <resource_embed/resource_embed.hpp>
#include "util/gl/error.hpp"
#include "util/gl/metric.hpp"
#include "sne/sne_minimization.hpp"
#include "sne/sne_field_2d.hpp"

namespace dh::sne {
  // Constants
  constexpr float stencilPointSize = 3.0f;
  constexpr uint hierarchyRebuildPadding = 50;
  constexpr uint hierarchyRebuildIterations = 4;
  constexpr uint inputQueueMinSize = 256 * 1024 * 1024;
  constexpr uint leafQueueMinSize = 128 * 1024 * 1024;
  constexpr AlignedVec<2, uint> fieldSizePrealloc(256);
  constexpr uint embeddingHierarchyInitLvl = 3;
  constexpr uint fieldHierarchyInitLvl = 3;
  constexpr uint k = 4;
  constexpr uint logk = 2;

  // Initial set of node pairs used for dual hierarchy traversal
  constexpr auto initPairs = []() {
    const uint fBegin = (0x2AAAAAAA >> (31u - logk * fieldHierarchyInitLvl));
    const uint fNodes = 1u << (logk * fieldHierarchyInitLvl);
    const uint eBegin = (0x2AAAAAAA >> (31u - logk * embeddingHierarchyInitLvl));
    const uint eNodes = 1u << (logk * embeddingHierarchyInitLvl);
    std::array<glm::uvec2, fNodes * eNodes> a {};
    for (uint i = 0; i < fNodes; ++i) {
      for (uint j = 0; j < eNodes; ++j) {
        a[i * eNodes + j] = glm::uvec2(fBegin + i, eBegin + j);
      }
    }
    return a;
  }();

  // Constr/destr
  SNEField2D::SNEField2D()
  : _isInit(false), _logger(nullptr) {
    // ...
  }

  SNEField2D::SNEField2D(SNEMinimizationBuffers minimization, SNEParams params, util::Logger* logger)
  : _isInit(false),
    _minimization(minimization),
    _params(params),
    _logger(logger),
    _hierarchyRebuildIterations(0),
    _size(0),
    _useEmbeddingHierarchy(params.singleHierarchyTheta > 0.0f),
    _useFieldHierarchy(params.dualHierarchyTheta > 0.0f) {
    util::log(_logger, "[SNEField2D] Initializing...");
      
    // Data size
    const uint n = _params.n;

    // Initialize shader programs
    {
      util::log(_logger, "[SNEField2D]   Creating shader programs");
      
      _programs(ProgramType::eDispatch).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/dispatch.glsl"));
      _programs(ProgramType::eQueryField).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/queryField.glsl"));
      _programs(ProgramType::eCompFullDrawStencil).addShader(util::GLShaderType::eVertex, rsrc::get("sne/field/2D/compFullDrawStencilVert.glsl"));
      _programs(ProgramType::eCompFullDrawStencil).addShader(util::GLShaderType::eFragment, rsrc::get("sne/field/2D/compFullDrawStencilFrag.glsl"));
      _programs(ProgramType::eCompFullCompact).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/compFullCompact.glsl"));
      _programs(ProgramType::eCompFullField).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/compFullField.glsl"));
      _programs(ProgramType::eCompSingleHierarchyCompact).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/compSingleHierarchyCompact.glsl"));
      _programs(ProgramType::eCompSingleHierarchyField).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/compSingleHierarchyField.glsl"));
      _programs(ProgramType::eCompDualHierarchyFieldIterative).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/compDualHierarchyFieldIterative.glsl"));
      _programs(ProgramType::eCompDualHierarchyFieldRest).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/compDualHierarchyFieldRest.glsl"));
      _programs(ProgramType::eCompDualHierarchyFieldLeaf).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/compDualHierarchyFieldLeaf.glsl"));
      _programs(ProgramType::eCompDualHierarchyFieldAccumulate).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field/2D/compDualHierarchyFieldAccumulate.glsl"));
      
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer objects
    {
      util::log(_logger, "[SNEField2D]   Creating buffer storage");
      
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::ePixelQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::ePixelQueueHeadReadback), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), GL_CLIENT_STORAGE_BIT); 

      // Dual hierarchy approximation used; initialize work queues
      if (_useFieldHierarchy) {
        // Linearly grow buffers in powers of two for finer approximations
        const uint iBufferSize = static_cast<uint>(std::pow(2, std::ceil(std::log2((
          std::pow(8, 2.f * std::max(0.0, 0.5 - _params.dualHierarchyTheta))
        ))))) * inputQueueMinSize;
        const uint lBufferSize = static_cast<uint>(std::pow(2, std::ceil(std::log2((
          std::pow(8, 2.f * std::max(0.0, 0.5 - _params.dualHierarchyTheta))
        ))))) * leafQueueMinSize;
        glNamedBufferStorage(_buffers(BufferType::ePairsInitQueue), initPairs.size() * sizeof(glm::uvec2), initPairs.data(), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsInputQueue), iBufferSize, nullptr, 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsOutputQueue), iBufferSize, nullptr, 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsRestQueue), iBufferSize, nullptr, 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsLeafQueue), lBufferSize, nullptr, 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsInputQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsOutputQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsRestQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsLeafQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
        glNamedBufferStorage(_buffers(BufferType::ePairsInitQueueHead), sizeof(glm::uvec3), glm::value_ptr(glm::uvec3(initPairs.size(), 1, 1)), 0);
      }
      glAssert();
      
      // Report buffer storage size
      const GLuint size = util::glGetBuffersSize(_buffers.size(), _buffers.data());
      util::logValue(_logger, "[SNEField2D]   Buffer storage (mb)", static_cast<float>(size) / 1'048'576.0f);
    }

    // Initialize other components
    {
      util::log(_logger, "[SNEField2D]   Creating other components");
      
      glCreateTextures(GL_TEXTURE_2D, _textures.size(), _textures.data());
      glCreateFramebuffers(1, &_stencilFBOHandle);
      glCreateVertexArrays(1, &_stencilVAOHandle);

      // Specify texture parameters
      glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTextureParameteri(_textures(TextureType::eStencil), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTextureParameteri(_textures(TextureType::eField), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

      // Full computation used, initialize FBO and VAO for stencil operation
      if (!_useEmbeddingHierarchy) {
        // FBO
        glNamedFramebufferTexture(_stencilFBOHandle, GL_COLOR_ATTACHMENT0, _textures(TextureType::eStencil), 0);
        glNamedFramebufferDrawBuffer(_stencilFBOHandle, GL_COLOR_ATTACHMENT0);

        // VAO
        glVertexArrayVertexBuffer(_stencilVAOHandle, 0, _minimization.embedding, 0, 2 * sizeof(float));
        glEnableVertexArrayAttrib(_stencilVAOHandle, 0);
        glVertexArrayAttribFormat(_stencilVAOHandle, 0, 2, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayAttribBinding(_stencilVAOHandle, 0, 0);
      }

      // Embedding hierarchy used, initialize
      if (_useEmbeddingHierarchy) {
        const EmbeddingHierarchy<2>::Layout layout(n);
        _embeddingHierarchy = EmbeddingHierarchy<2>(_minimization, layout, _params, _logger);
      }
    }

    _isInit = true;
    util::log(_logger, "[SNEField2D] Initialized");
  }

  SNEField2D::~SNEField2D() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
      glDeleteTextures(_textures.size(), _textures.data());
      glDeleteFramebuffers(1, &_stencilFBOHandle);
      glDeleteVertexArrays(1, &_stencilVAOHandle);
      _isInit = false;
    }
  }

  SNEField2D::SNEField2D(SNEField2D&& other) noexcept {
    swap(*this, other);
  }

  SNEField2D& SNEField2D::operator=(SNEField2D&& other) noexcept {
    swap(*this, other);
    return *this;
  }
  
  void SNEField2D::comp(uvec size, uint iteration) {
    // Resize field if necessary
    if (_size != size) {
      _size = size;
      compResize(iteration);
    }

    // Build embedding hierarchy if necessary
    if (_useEmbeddingHierarchy) {
      _embeddingHierarchy.comp(_hierarchyRebuildIterations == 0);
    }

    // Generate work queue with pixels in the field texture requiring computation
    compCompact(iteration);

    // Clear field texture
    // TODO is this necessary?
    glClearTexImage(_textures(TextureType::eField), 0, GL_RGBA, GL_FLOAT, nullptr);

    // Build field hierarchy if necessary
    // if (_useFieldHierarchy) {
      // ...
    // }

    // Perform field computation using one of the available techniques
    // if (_useFieldHierarchy) {
      // ...
    // }
    if (_useEmbeddingHierarchy) {
      compSingleHierarchyField(iteration);
    } else {
      compFullField(iteration);
    }

    // Update field buffer in SNEMinimization by querying the field texture
    queryField(iteration);

    // Update hierarchy rebuild countdown
    if (_useEmbeddingHierarchy) {
      if (iteration <= (_params.removeExaggerationIter + hierarchyRebuildPadding)
      || _hierarchyRebuildIterations >= hierarchyRebuildIterations) {
        _hierarchyRebuildIterations = 0;
      } else {
        _hierarchyRebuildIterations++;
      }
    }
  }

  void SNEField2D::compResize(uint iteration) {
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

    util::logString(_logger, "[SNEField2D]   Resized field", dh::to_string(_size));
  }


  void SNEField2D::compCompact(uint iteration) {
    auto &timer = _timers(TimerType::eCompCompact);
    timer.tick();

    // Reset queue head
    glClearNamedBufferSubData(_buffers(BufferType::ePixelQueueHead),
      GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

    if (_useEmbeddingHierarchy) {
      // Obtain hierarchy data
      const auto layout = _embeddingHierarchy.layout();
      const auto buffers = _embeddingHierarchy.buffers();
      
      // Determine which pixels to draw using the embedding hierarchy
      auto& program = _programs(ProgramType::eCompSingleHierarchyCompact);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nLvls", layout.nLvls);
      program.uniform<uvec>("textureSize", _size);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.minb);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _minimization.bounds);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::ePixelQueue));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePixelQueueHead));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_size.x, 16u), ceilDiv(_size.y, 16u), 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    } else {
      // Determine which pixels to draw by creating and querying a stencil texture
      // a. draw the stencil texture
      {
        auto& program = _programs(ProgramType::eCompFullDrawStencil);
        program.bind();

        // Specify and clear framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, _stencilFBOHandle);
        constexpr std::array<float, 4> clearColor = { 0.f, 0.f, 0.f, 0.f };
        constexpr float clearDepth = 1.f;
        glClearNamedFramebufferfv(_stencilFBOHandle, GL_COLOR, 0, clearColor.data());
        glClearNamedFramebufferfv(_stencilFBOHandle, GL_DEPTH, 0, &clearDepth);

        // Specify viewport and point size, bind bounds buffer
        glViewport(0, 0, _size.x, _size.y);
        glPointSize(stencilPointSize);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.bounds);
        
        // Draw a point at each embedding position
        glBindVertexArray(_stencilVAOHandle);
        glDrawArrays(GL_POINTS, 0, _params.n);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);        
      }
      // b. Compact stencil texture into BufferType::ePixelQueue
      {
        auto &program = _programs(ProgramType::eCompFullCompact);
        program.bind();

        // Set uniforms
        program.uniform<uvec>("textureSize", _size);
        program.uniform<int>("stencilSampler", 0);

        // Bind texture units
        glBindTextureUnit(0, _textures(TextureType::eStencil));

        // Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixelQueue));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::ePixelQueueHead));

        // Dispatch compute shader
        glDispatchCompute(ceilDiv(_size.x, 16u), ceilDiv(_size.y, 16u),  1u);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }
    }

    timer.tock();
    glAssert();
  }

  void SNEField2D::compFullField(uint iteration) {
    auto& timer = _timers(TimerType::eCompField);
    timer.tick();

    auto& program = _programs(ProgramType::eCompFullField);
    program.bind();

    // Set uniforms
    program.uniform<uvec>("textureSize", _size);
    program.uniform<uint>("nPoints", _params.n);

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.embedding);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimization.bounds);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePixelQueue));

    // Bind output texture
    glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    // Dispatch shader
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::ePixelQueueHead));
    glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
    glDispatchComputeIndirect(0);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    
    timer.tock();
    glAssert();
  }

  void SNEField2D::compSingleHierarchyField(uint iteration) {
    auto& timer = _timers(TimerType::eCompField);
    timer.tick();

    // Obtain hierarchy data
    const auto layout = _embeddingHierarchy.layout();
    const auto buffers = _embeddingHierarchy.buffers();

    // Divide pixel buffer head by workgroup size
    {
      auto &program = _programs(ProgramType::eDispatch);
      program.bind();
      program.uniform<uint>("div", 256);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixelQueueHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));
      glDispatchCompute(1, 1, 1);
    }

    auto& program = _programs(ProgramType::eCompSingleHierarchyField);
    program.bind();

    // Set uniforms
    program.uniform<uint>("nLvls", layout.nLvls);
    program.uniform<float>("theta2", _params.singleHierarchyTheta * _params.singleHierarchyTheta);
    program.uniform<uvec>("textureSize", _size);
    program.uniform<uint>("doBhCrit", 1);
    
    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.node1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers.minb);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _minimization.bounds);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePixelQueue));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::ePixelQueueHead));

    // Bind output texture
    glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    // Dispatch compute shader based on Buffertype::eDispatch
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
    glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
    glDispatchComputeIndirect(0);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    timer.tock();
    glAssert();
  }

  void SNEField2D::compDualHierarchyField(uint iteration) {
    // ...
  }

  void SNEField2D::queryField(uint iteration) {
    auto& timer = _timers(TimerType::eQueryField);
    timer.tick();

    auto& program = _programs(ProgramType::eQueryField);
    program.bind();

    // Set uniforms, bind texture unit
    program.uniform<uint>("num_points", _params.n);
    program.uniform<int>("fields_texture", 0);
    glBindTextureUnit(0, _textures(TextureType::eField));

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.embedding);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimization.field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _minimization.bounds);

    // Dispatch compute shader
    glDispatchCompute(ceilDiv(_params.n, 128u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    timer.tock();
    glAssert();
  }
} // dh::sne
