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
  // Constants
  constexpr float stencilPointSize = 3.0f; // Size of points splatted onto stencil texture
  constexpr std::array<float, 4> fboClearColor = { 0.f, 0.f, 0.f, 0.f };
  constexpr float fboClearDepth = 1.f;

  template <uint D>
  void Field<D>::compFullCompact() {
    auto& timer = _timers(TimerType::eCompact);
    timer.tick();

    // Reset pixel queue head
    glClearNamedBufferSubData(_buffers(BufferType::ePixelQueueHead),
      GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

    // 1.
    // Determine active pixels by splatting embedding points into a texture
    // * 2D approach: splat into a simple stencil texture
    // * 3D approach: splat into a compact voxel grid:
    //     "Single-Pass GPU Solid Voxelization for Real-Time Applications",
    //     Eisemann and Decoret, 2008. Src: https://dl.acm.org/doi/10.5555/1375714.1375728
    {
      auto& program = _programs(ProgramType::eFullCompactDraw);
      program.bind();
      
      // (3D only) Set required uniforms, bind cellmap to a texture unit, and enable OR logic operations to 
      // operate on the voxel grid
      if constexpr (D == 3) {
        program.template uniform<int>("cellMap", 0);
        program.template uniform<uint>("zDims", std::min(128u, _size.z));
        glBindTextureUnit(0, _textures(TextureType::eCellmap));
        glEnable(GL_COLOR_LOGIC_OP);
        glLogicOp(GL_OR);
      }

      // Bind and clear framebuffer
      glBindFramebuffer(GL_FRAMEBUFFER, _stencilFBOHandle);
      glClearNamedFramebufferfv(_stencilFBOHandle, GL_COLOR, 0, fboClearColor.data());
      glClearNamedFramebufferfv(_stencilFBOHandle, GL_DEPTH, 0, &fboClearDepth);

      // Specify viewport and point size, bind bounds buffer
      glViewport(0, 0, _size.x, _size.y);
      glPointSize(stencilPointSize);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.bounds);

      // Draw a point at each embedding position
      glBindVertexArray(_stencilVAOHandle);
      glDrawArrays(GL_POINTS, 0, _params.n);
      
      // (3D only) Disable OR logic operations
      if constexpr (D == 3) {
        glDisable(GL_COLOR_LOGIC_OP);
      }

      glBindFramebuffer(GL_FRAMEBUFFER, 0);   
    }

    // 2.
    // Query the splatted texture and record "active" pixels in the pixel queue
    {
      auto& program = _programs(ProgramType::eFullCompactComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint, D>("textureSize", _size);
      program.template uniform<int>("stencilSampler", 0);
      if constexpr (D == 3) {
        program.template uniform<uint>("gridDepth", std::min(128u, _size.z));
      }

      // Bind texture units (stencil texture is now input)
      glBindTextureUnit(0, _textures(TextureType::eStencil));

      // Bind buffers (pixel queue is output)
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixelQueue));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::ePixelQueueHead));

      // Dispatch compute shader
      if constexpr (D == 3) {
        glDispatchCompute(ceilDiv(_size.x, 8u), ceilDiv(_size.y, 4u),  ceilDiv(_size.z, 4u));
      } else if constexpr (D == 2) {
        glDispatchCompute(ceilDiv(_size.x, 16u), ceilDiv(_size.y, 16u),  1u);
      }
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    timer.tock();
    glAssert();
  }

  template <uint D>
  void Field<D>::compFullField() {
    auto& timer = _timers(TimerType::eField);
    timer.tick();

    auto& program = _programs(ProgramType::eFullFieldComp);
    program.bind();

    // Set uniforms
    program.template uniform<uint, D>("textureSize", _size);
    program.template uniform<uint>("nPoints", _params.n);

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
  
  // Template instantiations for 2/3 dimensions
  template void Field<2>::compFullCompact();
  template void Field<2>::compFullField();
  template void Field<3>::compFullCompact();
  template void Field<3>::compFullField();
} // dh::sne