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

#include <resource_embed/resource_embed.hpp>
#include "dh/constants.hpp"
#include "dh/sne/components/hierarchy/embedding_hierarchy.hpp"
#include "dh/vis/components/embedding_hierarchy_render_task.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[EmbeddingHierarchy]");

  template <uint D>
  EmbeddingHierarchy<D>::EmbeddingHierarchy()
  : _isInit(false), _nRebuilds(0) {
    // ...
  }

  template <uint D>
  EmbeddingHierarchy<D>::EmbeddingHierarchy(MinimizationBuffers minimization, Layout layout, Params params)
  : _isInit(false), _nRebuilds(0), _minimization(minimization), _layout(layout), _params(params) {
    Logger::newt() << prefix << "Initializing...";

    // Constants
    constexpr uint nodk = (D == 2) ? 4 : 8;
    constexpr uint logk = (D == 2) ? 2 : 3;
    
    // Initialize shader programs
    {
      _programs(ProgramType::eDispatch).addShader(util::GLShaderType::eCompute, rsrc::get("sne/dispatch.comp"));
      if constexpr (D == 2) {
        _programs(ProgramType::eMortonUnsortedComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/mortonUnsorted.comp"));
        _programs(ProgramType::eEmbeddingSortedComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/embeddingSorted.comp"));
        _programs(ProgramType::eSubdivisionComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/subdivision.comp"));
        _programs(ProgramType::eLeavesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/leaves.comp"));
        _programs(ProgramType::eNodesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/2D/nodes.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eMortonUnsortedComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/mortonUnsorted.comp"));
        _programs(ProgramType::eEmbeddingSortedComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/embeddingSorted.comp"));
        _programs(ProgramType::eSubdivisionComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/subdivision.comp"));
        _programs(ProgramType::eLeavesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/leaves.comp"));
        _programs(ProgramType::eNodesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/embedding_hierarchy/3D/nodes.comp"));
      }
      glAssert();

      for (auto& program : _programs) {
        program.link();
      }
      glAssert();

      // Set these uniforms exactly once
      _programs(ProgramType::eMortonUnsortedComp).template  uniform<uint>("nPoints", _layout.nPos);
      _programs(ProgramType::eEmbeddingSortedComp).template uniform<uint>("nPoints", _layout.nPos);
      _programs(ProgramType::eDispatch).template uniform<uint>("div", 256);
    }
    
    // Initialize buffer objects
    {
      // Root node data set at initialization
      std::vector<glm::vec4> nodeData(_layout.nNodes, glm::vec4(0));
      nodeData[0].w = _layout.nPos;
      glm::uvec4 head(0);

      using vec = util::AlignedVec<D, float>;
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::eMortonUnsorted), _layout.nPos * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eMortonSorted), _layout.nPos * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eIndicesSorted), _layout.nPos * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eEmbeddingSorted), _layout.nPos * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafQueue), _layout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLeafQueueHead), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::eNode0), _layout.nNodes * sizeof(glm::vec4), nodeData.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eNode1), _layout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eMinB), _layout.nNodes * sizeof(vec), nullptr, 0);
      glAssert();      
    }

    // Initialize other components
    {
      keySort = util::KeySort(
        _buffers(BufferType::eMortonUnsorted),
        _buffers(BufferType::eMortonSorted),
        _buffers(BufferType::eIndicesSorted),
        _layout.nPos,
        _layout.nLvls * logk
      );
    }

#ifdef DH_ENABLE_VIS_EMBEDDING_HIERARCHY
    // Setup render task
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      queue.emplace(vis::EmbeddingHierarchyRenderTask<D>(_minimization, buffers(), _params, 1));
    }
#endif // DH_ENABLE_VIS_EMBEDDING_HIERARCHY

    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::rest() << prefix << "Initialized, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    _isInit = true;
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
    constexpr uint nodk = (D == 2) ? 4 : 8;
    constexpr uint logk = (D == 2) ? 2 : 3;

    if (rebuild) {
      _nRebuilds++;
    }

    // 1.
    // Sort embedding positions along a Morton order
    {
      auto& timer = _timers(TimerType::eSort);
      timer.tick();
      
      // a. Generate morton codes over unsorted embedding
      // Skip this step on a refit
      if (rebuild) {
        auto& program = _programs(ProgramType::eMortonUnsortedComp);
        program.bind();

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
        auto& program = _programs(ProgramType::eEmbeddingSortedComp);
        program.bind();

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
      auto& timer = _timers(TimerType::eSubdivisionComp);
      timer.tick();

      auto& program = _programs(ProgramType::eSubdivisionComp);
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
        program.template uniform<uint>("isBottom", lvl == _layout.nLvls - 2); // All further subdivided nodes are leaves
        program.template uniform<uint>("rangeBegin", begin);
        program.template uniform<uint>("rangeEnd", end);

        // Dispatch shader
        glDispatchCompute(ceilDiv(range, 256u / nodk), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        begin = end + 1;
      }

      timer.tock();
      glAssert();
    }

    // 3.
    // Compute leaf data
    {
      auto& timer = _timers(TimerType::eLeavesComp);
      timer.tick();
      
      // Divide contents of BufferType::eLeafHead by workgroup size
      if (rebuild) {
        auto &program = _programs(ProgramType::eDispatch);
        program.bind();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eLeafQueueHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));
        glDispatchCompute(1, 1, 1);
      }

      auto& program = _programs(ProgramType::eLeavesComp);
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
      auto& timer = _timers(TimerType::eNodesComp);
      timer.tick();

      auto& program = _programs(ProgramType::eNodesComp);
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
        program.template uniform<uint>("rangeBegin", begin);
        program.template uniform<uint>("rangeEnd", end);

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