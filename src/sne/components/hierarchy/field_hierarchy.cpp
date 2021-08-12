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
#include <resource_embed/resource_embed.hpp>
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/vis/components/field_hierarchy_render_task.hpp"
#include "dh/sne/components/hierarchy/field_hierarchy.hpp"

namespace dh::sne {
  template <uint D>
  FieldHierarchy<D>::FieldHierarchy()
  : _isInit(false), _nRebuilds(0), _logger(nullptr) {
    // ...
  }

  template <uint D>
  FieldHierarchy<D>::FieldHierarchy(FieldBuffers field, Layout constrLayout, Params params, util::Logger* logger)
  : _isInit(false), _nRebuilds(0), _field(field), _constrLayout(constrLayout), _params(params), _logger(logger) {
    util::log(_logger, "[FieldHierarchy] Initializing...");

    // Initialize shader programs
    {
      util::log(_logger, "[FieldHierarchy]   Creating shader programs");

      if constexpr (D == 2) {
        _programs(ProgramType::eLeavesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field_hierarchy/2D/leaves.comp"));
        _programs(ProgramType::eNodesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field_hierarchy/2D/nodes.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eLeavesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field_hierarchy/3D/leaves.comp"));
        _programs(ProgramType::eNodesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/field_hierarchy/3D/nodes.comp"));
      }

      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer objects
    {
      util::log(_logger, "[FieldHierarchy ]   Creating buffer storage");
      
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eNode), _constrLayout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _constrLayout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glAssert();

      // Report buffer storage size
      const GLuint size = util::glGetBuffersSize(_buffers.size(), _buffers.data());
      util::logValue(_logger, "[FieldHierarchy]   Buffer storage (mb)", static_cast<float>(size) / 1'048'576.0f);
    }

    // Setup render task
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      queue.emplace(vis::FieldHierarchyRenderTask<D>(buffers(), _params, 2));
    }

    _isInit = true;
    util::log(_logger, "[FieldHierarchy] Initialized");
  }
  
  template <uint D>
  FieldHierarchy<D>::~FieldHierarchy() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
    }
  }
  
  template <uint D>
  FieldHierarchy<D>::FieldHierarchy(FieldHierarchy<D>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  FieldHierarchy<D>& FieldHierarchy<D>::operator=(FieldHierarchy<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void FieldHierarchy<D>::comp(bool rebuild, Layout compLayout) {
    _compLayout = compLayout;

    if (rebuild) {
      _nRebuilds++;
    }

    // 1.
    // Ensure available memory accomodates the new layout; expand if the field has grown too large 
    if (rebuild && _constrLayout.nNodes < _compLayout.nNodes) {
      util::log(_logger, "[FieldHierarchy]   Expanding field hierarchy");

      // Compute new layout as nearest larger power of two
      const uvec newSize = uvec(dh::max(vec(glm::pow(vec(2), glm::ceil(glm::log(vec(_compLayout.size)) / vec(glm::log(2)))))));  
      _constrLayout = Layout(newSize);

      // Re-allocate buffers
      // TODO Look into optimizing this
      glDeleteBuffers(_buffers.size(), _buffers.data());
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eNode), _constrLayout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _constrLayout.nNodes * sizeof(glm::vec4), nullptr, 0);

      // Report buffer storage size
      const GLuint size = util::glGetBuffersSize(_buffers.size(), _buffers.data());
      util::log(_logger, "[FieldHierarchy]   Expanded field hierarchy");
      util::logValue(_logger, "[FieldHierarchy]   Buffer storage (mb)", static_cast<float>(size) / 1'048'576.0f);
    }

    // Clear hierarchy data
    if (rebuild) {
      glClearNamedBufferData(_buffers(BufferType::eNode), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    }
    glClearNamedBufferData(_buffers(BufferType::eField), GL_RGBA32F, GL_RGBA, GL_FLOAT, nullptr);

    // 2.
    // Compute leaf nodes. Essentially just fill in BufferType::eNode for used pixels
    if (rebuild) {
      auto& timer = _timers(TimerType::eLeavesComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eLeavesComp);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPixels", _compLayout.nPixels);
      program.uniform<uint>("lvl", _compLayout.nLvls - 1);
      
      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _field.pixelQueue);

      // Dispatch shader
      glDispatchCompute(ceilDiv(_compLayout.nPixels, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // 3.
    // Compute rest of nodes. Do a reduction over hierarchy levels, filling in BufferType::eNode
    if (rebuild) {
      // Constants
      constexpr uint nodek = (D == 2) ? 4 : 8;
      constexpr uint logk = (D == 2) ? 2 : 3;

      auto& timer = _timers(TimerType::eNodesComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eNodesComp);
      program.bind();
      
      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode));
      
      // Iterate through tree levels from bottom to top
      for (uint lvl = _compLayout.nLvls - 1; lvl > 0; lvl--) {
        // Set uniform
        program.uniform<uint>("lvl", lvl);
        
        // Dispatch shader
        glDispatchCompute(ceilDiv(1u << (logk * lvl), 256u), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      }

      timer.tock();
      glAssert();
    }
  }
  
  // Explicit template instantiations
  template class FieldHierarchy<2>;
  template class FieldHierarchy<3>;
} // dh::sne