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
#include "dh/sne/components/hierarchy/field_hierarchy.hpp"
#include "dh/vis/components/field_hierarchy_render_task.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[FieldHierarchy]");

  template <uint D>
  FieldHierarchy<D>::FieldHierarchy()
  : _isInit(false), _nRebuilds(0) {
    // ...
  }

  template <uint D>
  FieldHierarchy<D>::FieldHierarchy(FieldBuffers field, Layout constrLayout, Params params)
  : _isInit(false), _nRebuilds(0), _field(field), _constrLayout(constrLayout), _params(params) {
    Logger::newt() << prefix << "Initializing...";

    // Initialize shader programs
    {
      _programs(ProgramType::eDispatch).addShader(util::GLShaderType::eCompute, rsrc::get("sne/dispatch.comp"));
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
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::eNode), _constrLayout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _constrLayout.nNodes * sizeof(glm::vec4), nullptr, 0);
      glAssert();
    }

#ifdef DH_ENABLE_VIS_FIELD_HIERARCHY
    // Setup render task
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      queue.emplace(vis::FieldHierarchyRenderTask<D>(buffers(), _params, 2));
    }
#endif // DH_ENABLE_VIS_FIELD_HIERARCHY

    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::rest() << prefix << "Initialized, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    _isInit = true;
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
      Logger::newt() << prefix << "Expanding hierarchy";

      // Compute new layout as nearest larger power of two
      const uvec newSize = uvec(dh::util::max(vec(glm::pow(vec(2), glm::ceil(glm::log(vec(_compLayout.size)) / vec(glm::log(2)))))));  
      _constrLayout = Layout(newSize);

      // Re-allocate buffers
      // TODO Look into optimizing this
      glDeleteBuffers(_buffers.size(), _buffers.data());
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eDispatch), sizeof(glm::uvec4), glm::value_ptr(glm::uvec4(1)), 0);
      glNamedBufferStorage(_buffers(BufferType::eNode), _constrLayout.nNodes * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _constrLayout.nNodes * sizeof(glm::vec4), nullptr, 0);

      // Report buffer storage size
      const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
      Logger::rest() << prefix << "Expanded hierarchy, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";
      Logger::newl();
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
      
      // Divide pixel queue head by workgroup size for use as indirect dispatch buffer
      {
        auto &program = _programs(ProgramType::eDispatch);
        program.bind();
        program.template uniform<uint>("div", 256);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _field.pixelQueueHead);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

        glDispatchCompute(1, 1, 1);
      }

      auto& program = _programs(ProgramType::eLeavesComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("lvl", _compLayout.nLvls - 1);
      
      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eNode));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _field.pixelQueue);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _field.pixelQueueHead);

      // Dispatch compute shader based on Buffertype::eDispatch
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
      glDispatchComputeIndirect(0);
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
        program.template uniform<uint>("lvl", lvl);
        
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