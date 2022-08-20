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

#include "dh/sne/components/field.hpp"
#include "dh/util/gl/error.hpp"

namespace dh::sne {
  template <uint D>
  void Field<D>::compSingleHierarchyCompact() {
    auto& timer = _timers(TimerType::eCompact);
    timer.tick();

    auto& program = _programs(ProgramType::eSingleHierarchyCompactComp);
    program.bind();
    
    // Reset pixel queue buffer's head
    glClearNamedBufferSubData(_buffers(BufferType::ePixelQueueHead),
      GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

    // Obtain hierarchy data
    const auto layout = _embeddingHierarchy.layout();
    const auto buffers = _embeddingHierarchy.buffers();

    // Set uniforms
    program.template uniform<uint>("nLvls", layout.nLvls);
    program.template uniform<uint, D>("textureSize", _size);

    // Set buffer bindings
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.minb);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _minimization.bounds);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::ePixelQueue));
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePixelQueueHead));

    // Dispatch shader
    if constexpr (D == 3) {
      glDispatchCompute(ceilDiv(_size.x, 32u), ceilDiv(_size.y, 8u), _size.z);
    } else if constexpr (D == 2) {
      glDispatchCompute(ceilDiv(_size.x, 16u), ceilDiv(_size.y, 16u), 1);
    }
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    timer.tock();
    glAssert();
  }

  template <uint D>
  void Field<D>::compSingleHierarchyField() {
    auto& timer = _timers(TimerType::eField);
    timer.tick();

    // Obtain hierarchy data
    const auto layout = _embeddingHierarchy.layout();
    const auto buffers = _embeddingHierarchy.buffers();

    // Divide pixel queue head by workgroup size for use as indirect dispatch buffer
    {
      auto &program = _programs(ProgramType::eDispatch);
      program.bind();
      program.template uniform<uint>("div", 256);

      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixelQueueHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

      glDispatchCompute(1, 1, 1);
    }
    
    auto& program = _programs(ProgramType::eSingleHierarchyFieldComp);
    program.bind();

    // Set uniforms
    program.template uniform<uint>("nLvls", layout.nLvls);
    program.template uniform<uint, D>("textureSize", _size);
    
    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers.node0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers.node1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers.embeddingSorted);
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
  
  // Template instantiations for 2/3 dimensions
  template void Field<2>::compSingleHierarchyCompact();
  template void Field<2>::compSingleHierarchyField();
  template void Field<3>::compSingleHierarchyCompact();
  template void Field<3>::compSingleHierarchyField();
} // dh::sne