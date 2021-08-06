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

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <resource_embed/resource_embed.hpp>
#include "aligned.hpp"
#include "util/gl/error.hpp"
#include "util/gl/metric.hpp"
#include "sne/hierarchy/embedding_hierarchy.hpp"

namespace dh::sne {
  template <uint D>
  EmbeddingHierarchy<D>::EmbeddingHierarchy()
  : _isInit(false), _nRebuilds(0), _logger(nullptr) {
    // ...
  }

  template <uint D>
  EmbeddingHierarchy<D>::EmbeddingHierarchy(SNEMinimizationBuffers minimization, Layout layout, SNEParams params, util::Logger* logger)
  : _isInit(false), _nRebuilds(0), _minimization(minimization), _layout(layout), _params(params), _logger(logger) {
    // Constants
    constexpr uint nodek = (D == 2) ? 4 : 8;
    constexpr uint logk = (D == 2) ? 2 : 3;
    
    util::log(_logger, "[EmbeddingHierarchy] Initializing...");

    // Initialize shader programs
    {
      util::log(_logger, "[EmbeddingHierarchy]   Creating shader programs");

      _programs(ProgramType::eDispatch).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/dispatch.glsl"));
      if constexpr (D == 2) {
        _programs(ProgramType::eCompMortonUnsorted).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/compMortonUnsorted.glsl"));
        _programs(ProgramType::eCompEmbeddingSorted).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/compEmbeddingSorted.glsl"));
        _programs(ProgramType::eCompSubdivision).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/compSubdivision.glsl"));
        _programs(ProgramType::eCompLeaves).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/compLeaves.glsl"));
        _programs(ProgramType::eCompNodes).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/compNodes.glsl"));
      } else if constexpr (D == 3) {
      //   _programs(ProgramType::eCompMortonUnsorted).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/compMortonUnsorted.glsl"));
      //   _programs(ProgramType::eCompEmbeddingSorted).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/compEmbeddingSorted.glsl"));
      //   _programs(ProgramType::eCompSubdivision).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/compSubdivision.glsl"));
      //   _programs(ProgramType::eCompLeaves).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/compLeaves.glsl"));
      //   _programs(ProgramType::eCompNodes).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/compNodes.glsl"));
      }

      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }
    
    // Initialize buffer objects
    {
      util::log(_logger, "[EmbeddingHierarchy]   Creating buffer storage");

      // Root node data set at initialization
      std::vector<glm::vec4> nodeData(_layout.nNodes, glm::vec4(0));
      nodeData[0].w = _layout.nPos;
      glm::uvec4 head(0);

      using vec = AlignedVec<D, float>;
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::eMortonUnsorted), _layout.nPos * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eMortonSorted), _layout.nPos * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eIndicesSorted), _layout.nPos * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eEmbeddingSorted), _layout.nPos * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafQueue), _layout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::eNode0), _layout.nNodes * sizeof(glm::vec4), nodeData.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eNode1), _layout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eMinB), _layout.nNodes * sizeof(vec), nullptr, 0);
      glAssert();
      
      // Report buffer storage size
      const GLuint size = util::glGetBuffersSize(_buffers.size(), _buffers.data());
      util::logValue(_logger, "[EmbeddingHierarchy]   Buffer storage (mb)", static_cast<float>(size) / 1'048'576.0f);
    }

    // Initialize other components
    {
      util::log(_logger, "[EmbeddingHierarchy]   Creating other components");
      keySort = util::KeySort(
        _buffers(BufferType::eMortonUnsorted),
        _buffers(BufferType::eMortonSorted),
        _buffers(BufferType::eIndicesSorted),
        _layout.nPos,
        _layout.nLvls * logk
      );
    }

    _isInit = true;
    util::log(_logger, "[EmbeddingHierarchy] Initialized");
  }

  template <uint D>
  EmbeddingHierarchy<D>::~EmbeddingHierarchy() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
    }
  }
  
  template <uint D>
  EmbeddingHierarchy<D>::EmbeddingHierarchy(EmbeddingHierarchy<D>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  EmbeddingHierarchy<D>& EmbeddingHierarchy<D>::operator=(EmbeddingHierarchy<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void EmbeddingHierarchy<D>::comp(bool rebuild) {
    // Constants
    constexpr uint nodek = (D == 2) ? 4 : 8;
    constexpr uint logk = (D == 2) ? 2 : 3;

    if (rebuild) {
      _nRebuilds++;
    }

    // 1.
    // Sort embedding positions along a Morton order
    {
      auto& timer = _timers(TimerType::eCompSort);
      timer.tick();
      
      // a. Generate morton codes over unsorted embedding
      // Skip this step on a refit
      if (rebuild) {
        auto& program = _programs(ProgramType::eCompMortonUnsorted);
        program.bind();

        // Set uniforms
        program.uniform<uint>("nPoints", _layout.nPos);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.embedding);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimization.bounds);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eMortonUnsorted));

        // Dispatch shader
        glDispatchCompute(ceilDiv(_layout.nPos, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glAssert();
      }

      // b. Perform radix sort, creating a morton order and a mapping from unsorted to sorted list
      // Skip this step on a refit
      if (rebuild) {
        keySort.sort();
      }

      // c. Generate sorted embedding positions based on the mapping
      {
        auto& program = _programs(ProgramType::eCompEmbeddingSorted);
        program.bind();

        // Set uniforms
        program.uniform<uint>("nPoints", _layout.nPos);

        // Set buffer bindings
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eIndicesSorted));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimization.embedding);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eEmbeddingSorted));

        // Dispatch shader
        glDispatchCompute(ceilDiv(_layout.nPos, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glAssert();
      }

      timer.tock();
      glAssert();
    }

    // 2.
    // Perform subdivision, following the morton order
    // Skip this step on a refit
    if (rebuild) {
      auto& timer = _timers(TimerType::eCompSubdivision);
      timer.tick();

      auto& program = _programs(ProgramType::eCompSubdivision);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eMortonSorted));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eLeafQueue));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eLeafQueueHead));

      // Reset leaf queue head
      glClearNamedBufferData(_buffers(BufferType::eLeafQueueHead), 
        GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      // Iterate through tree levels from top to bottom
      uint begin = 0;
      for (uint lvl = 0; lvl < _layout.nLvls - 1; lvl++) {
        const uint end = begin + (1u << (logk * lvl)) - 1;
        const uint range = 1 + end - begin;

        // Set uniforms
        program.uniform<uint>("isBottom", lvl == _layout.nLvls - 2); // All further subdivided nodes are leaves
        program.uniform<uint>("rangeBegin", begin);
        program.uniform<uint>("rangeEnd", end);

        // Dispatch shader
        glDispatchCompute(ceilDiv(range, 256u / nodek), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        begin = end + 1;
      }

      timer.tock();
      glAssert();
    }

    // 3.
    // Compute leaf data
    {
      auto& timer = _timers(TimerType::eCompLeaves);
      timer.tick();
      
      // Divide contents of BufferType::eLeafHead by workgroup size
      if (rebuild) {
        auto &program = _programs(ProgramType::eDispatch);
        program.bind();
        program.uniform<uint>("div", 256);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLeafQueueHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));
        glDispatchCompute(1, 1, 1);
      }

      auto& program = _programs(ProgramType::eCompLeaves);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eMinB));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eEmbeddingSorted));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eLeafQueue));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eLeafQueueHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _minimization.bounds);

      // Dispatch shader
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
      glDispatchComputeIndirect(0);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
    // 4.
    // Compute node data
    {
      auto& timer = _timers(TimerType::eCompNodes);
      timer.tick();

      auto& program = _programs(ProgramType::eCompNodes);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode0));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eNode1));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eMinB));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _minimization.bounds);

      // Iterate through tree levels from bottom to top.
      uint end = _layout.nNodes - 1;
      for (int lvl = _layout.nLvls - 1; lvl > 0; lvl--) {
        const uint begin = 1 + end - (1u << (logk * lvl));
        const uint range = end - begin;
                  
        // Set uniforms
        program.uniform<uint>("rangeBegin", begin);
        program.uniform<uint>("rangeEnd", end);

        // Dispatch shader
        glDispatchCompute(ceilDiv(range, 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        end = begin - 1;
      }

      timer.tock();
      glAssert();
    }
  }

  // Explicit template instantiations
  template class EmbeddingHierarchy<2>;
  template class EmbeddingHierarchy<3>;
} // dh::sne