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
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"

namespace dh::sne {
  template <uint D>
  void Field<D>::compDualHierarchyField() {
    // Constants
    constexpr uint knode = (D == 2) ? 4 : 8;
    constexpr uint logk = (D == 2) ? 2 : 3;
    constexpr uint hierarchyInitLvl = (D == 2) ? DH_HIER_INIT_LVL_2D : DH_HIER_INIT_LVL_3D;
    
    auto& timer = _timers(TimerType::eField);
    timer.tick();
    
    // Obtain hierarchy data (e = embedding, f = field)
    const auto eLayout = _embeddingHierarchy.layout();
    const auto eBuffers = _embeddingHierarchy.buffers();
    const auto fLayout = _fieldHierarchy.layout();
    const auto fBuffers = _fieldHierarchy.buffers();

    // Reset input queue by doing a copy of init queue
    glCopyNamedBufferSubData(_buffers(BufferType::ePairsInitQueue), _buffers(BufferType::ePairsInputQueue),
      0, 0, _initQueueSize);
    glCopyNamedBufferSubData(_buffers(BufferType::ePairsInitQueueHead), _buffers(BufferType::ePairsInputQueueHead),
      0, 0, sizeof(uint));
    glAssert();

    // Reset rest and leaf queues (set heads to 0)
    glClearNamedBufferSubData(_buffers(BufferType::ePairsRestQueueHead),
      GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    glClearNamedBufferSubData(_buffers(BufferType::ePairsLeafQueueHead),
      GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    glAssert();

    // 1. Perform dual-hierarchy traversal
    {
      // Set uniforms (dual-subdivision program)
      auto& dsProgram = _programs(ProgramType::eDualHierarchyFieldDualSubdivideComp);
      dsProgram.template uniform<uint>("eLvls", eLayout.nLvls);
      dsProgram.template uniform<uint>("fLvls", fLayout.nLvls);

      // Set uniforms (single-subdivision program)
      auto& ssProgram = _programs(ProgramType::eDualHierarchyFieldSingleSubdivideComp);
      ssProgram.template uniform<uint>("fLvls", fLayout.nLvls);

      // Set buffer bindings which are reused throughout traversal
      glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, _buffers(BufferType::eDispatch));
      // Buffers 0-1 are continuously rebound
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, eBuffers.embeddingSorted);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fBuffers.node);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, fBuffers.field);
      // Buffers 5-8 are continuously rebound
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _buffers(BufferType::ePairsLeafQueue));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, _buffers(BufferType::ePairsLeafQueueHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, _minimization.bounds);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, _buffers(BufferType::ePairsRestQueue));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, _buffers(BufferType::ePairsRestQueueHead));

      // Compute number of iterations necessary for the full traversal
      const uint minLvls = std::min(fLayout.nLvls, eLayout.nLvls);
      const uint maxLvls = std::max(fLayout.nLvls, eLayout.nLvls);
      const uint nrIters = ceilDiv(minLvls + maxLvls, 2u) + 2u;
      const uint dualSubdivideLastLvl = minLvls - 2u;

      // Track traversal state during iteration
      DualHierarchyState state = DualHierarchyState::eDualSubdivide;

      // Perform traversal level by level until leaves are reached
      for (uint i = hierarchyInitLvl; i <= nrIters; ++i) {
        // Update traversal state
        if (state == DualHierarchyState::eDualSubdivide && i == dualSubdivideLastLvl) {
          state = DualHierarchyState::eDualSubdivideLast;
        } else if (state == DualHierarchyState::eDualSubdivideLast) {
          state = DualHierarchyState::eSingleSubdivideFirst;
        } else if (state == DualHierarchyState::eSingleSubdivideFirst) {
          state = DualHierarchyState::eSingleSubdivide;
        }

        // Select the correct input queue depending on traversal state
        // * in general, use the current input queue
        // * when state is eSingleSubdivideFirst, use the accumulated rest queue instead
        GLuint iQueue = (state == DualHierarchyState::eSingleSubdivideFirst)
                      ? _buffers(BufferType::ePairsRestQueue) 
                      : _buffers(BufferType::ePairsInputQueue);
        GLuint iQueueHead = (state == DualHierarchyState::eSingleSubdivideFirst)
                      ? _buffers(BufferType::ePairsRestQueueHead) 
                      : _buffers(BufferType::ePairsInputQueueHead);

        // Reset output queue (set head to 0)
        if (state != DualHierarchyState::eSingleSubdivideFirst) {
          glClearNamedBufferSubData(_buffers(BufferType::ePairsOutputQueueHead),
            GL_R32UI, 0, sizeof(uint), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        }

        // Divide input queue head by workgroup size for use as indirect dispatch buffer
        {
          auto& program = _programs(ProgramType::eDispatch);
          program.bind();
          program.template uniform<uint>("div", 256 / knode); 
          
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, iQueueHead);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

          glDispatchCompute(1, 1, 1);
          glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
        }

        // Perform one step down the dual hierarchy
        {
          // Bind program either for dual subdivision, or single subdivision
          if (state == DualHierarchyState::eDualSubdivide || state == DualHierarchyState::eDualSubdivideLast) {
            dsProgram.bind();
            dsProgram.template uniform<uint>("dhLvl", i + 1);
          } else {
            ssProgram.bind();
            ssProgram.template uniform<uint>("isLastIter", i == nrIters);
          }

          // Set buffer bindings
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, eBuffers.node0);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, eBuffers.node1);
          // 2-4 remain the same throughout traversal
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, iQueue);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, iQueueHead);
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _buffers(BufferType::ePairsOutputQueue));
          glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _buffers(BufferType::ePairsOutputQueueHead));
          // 9-13 remain the same throughout traversal

          // Dispatch shader based on indirect dispatch buffer
          glDispatchComputeIndirect(0);
          glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        
       /*  {
          GLuint input, output, rest, leaf;
          glGetNamedBufferSubData(_buffers(BufferType::ePairsInputQueueHead), 0, sizeof(uint), &input);
          glGetNamedBufferSubData(_buffers(BufferType::ePairsOutputQueueHead), 0, sizeof(uint), &output);
          glGetNamedBufferSubData(_buffers(BufferType::ePairsRestQueueHead), 0, sizeof(uint), &rest);
          glGetNamedBufferSubData(_buffers(BufferType::ePairsLeafQueueHead), 0, sizeof(uint), &leaf);

          std::string ststr;
          switch (state)  {
          case DualHierarchyState::eDualSubdivide:
            ststr = "eDualSubdivide";
            break;          
          case DualHierarchyState::eDualSubdivideLast:
            ststr = "eDualSubdivideLast";
            break;        
          case DualHierarchyState::eSingleSubdivideFirst:
            ststr = "eSingleSubdivideFirst";
            break;   
          case DualHierarchyState::eSingleSubdivide:
            ststr = "eSingleSubdivide";
            break;  
          }

          const uint eNodes = 1u << (logk * (eLayout.nLvls - 1));
          const uint fNodes = 1u << (logk * (fLayout.nLvls - 1));

          util::Logger::newl() 
            << "iteration " << i << ", eLvls " << eLayout.nLvls << ", fLvls " << fLayout.nLvls << '\n'
            << ststr << ", eNodes " << eNodes << ", fNodes " << fNodes << "\n\t"
            << "input : " << input << ",    " 
            << "output : " << output << ",    " 
            << "rest : " << rest << ",    " 
            << "leaf : " << leaf << '\n'; 
        } */

        // Swap input and output queues (by swapping their handles)
        std::swap(_buffers(BufferType::ePairsInputQueue), _buffers(BufferType::ePairsOutputQueue));
        std::swap(_buffers(BufferType::ePairsInputQueueHead), _buffers(BufferType::ePairsOutputQueueHead));
      }
    }    

    // 2. Compute remaining large leaves of the dual hierarchy
    {
      // Divide leaf queue head by workgroup size for use as indirect dispatch buffer
      {
        auto& program = _programs(ProgramType::eDispatch);
        program.bind();
        program.template uniform<uint>("div", 256 / 4); 

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePairsLeafQueueHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
      }

      auto& program = _programs(ProgramType::eDualHierarchyFieldLeafComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("fLvls", fLayout.nLvls);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, eBuffers.node0);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, eBuffers.node1);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, eBuffers.embeddingSorted);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fBuffers.field);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::ePairsLeafQueue));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::ePairsLeafQueueHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _minimization.bounds);

      // Dispatch shader based on indirect dispatch buffer
      glDispatchComputeIndirect(0);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // 3. Accumulate sparsely computed values in the field hierarchy into a gpu texture
    {
      // Divide pixel queue head by workgroup size for use as indirect dispatch buffer
      {
        auto &program = _programs(ProgramType::eDispatch);
        program.bind();
        program.template uniform<uint>("div", 256);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixelQueueHead));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eDispatch));

        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT);
      }

      auto& program = _programs(ProgramType::eDualHierarchyFieldAccumulateComp);
      program.bind();

      // Set uniforms
      program.template uniform<uint>("fLvls", fLayout.nLvls);
      program.template uniform<uint>("startLvl", hierarchyInitLvl);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::ePixelQueue));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::ePixelQueueHead));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, fBuffers.field);

      // Bind output texture's image
      glBindImageTexture(0, _textures(TextureType::eField), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);

      // Dispatch shader based on indirect dispatch buffer
      glDispatchComputeIndirect(0);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    }
    
    timer.tock();
    glAssert();
  }
  
  // Template instantiations for 2/3 dimensions
  template void Field<2>::compDualHierarchyField();
  template void Field<3>::compDualHierarchyField();
} // dh::sne