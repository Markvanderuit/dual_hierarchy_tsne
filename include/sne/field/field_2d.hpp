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

#pragma once

#include <utility>
#include "aligned.hpp"
#include "types.hpp"
#include "util/enum.hpp"
#include "util/logger.hpp"
#include "util/gl/timer.hpp"
#include "util/gl/program.hpp"
#include "sne/sne_params.hpp"
#include "sne/hierarchy/embedding_hierarchy.hpp"
#include "sne/buffers/sne_minimization_buffers.hpp"

namespace dh::sne {
  class Field2D {
    // aligned types
    using Bounds = AlignedBounds<2>;
    using vec = AlignedVec<2, float>;
    using uvec = AlignedVec<2, uint>;

  public:
    // Constr/destr
    Field2D();
    Field2D(SNEMinimizationBuffers minimization, SNEParams params, util::Logger* logger = nullptr);
    ~Field2D();

    // Copy constr/assignment is explicitly deleted
    Field2D(const Field2D&) = delete;
    Field2D& operator=(const Field2D&) = delete;

    // Move constr/operator moves handles
    Field2D(Field2D&&) noexcept;
    Field2D& operator=(Field2D&&) noexcept;

    // Compute the field for a size (resolution) and iteration (determines technique)
    void comp(uvec size, uint iteration);

    bool isInit() const { return _isInit; }

  private:
    // Functions called by Field2D::comp(uint);
    void compResize(uint iteration);
    void compCompact(uint iteration);
    void compFullField(uint iteration);
    void compSingleHierarchyField(uint iteration);
    void compDualHierarchyField(uint iteration);
    void queryField(uint iteration);
    
    enum class BufferType {
      eDispatch,

      // Work queue for pixels in field texture requiring computation
      ePixelQueue,
      ePixelQueueHead,
      ePixelQueueHeadReadback,
      
      // Work queue of node pairs to start hierarchy traversal (precomputed)
      ePairsInitQueue,
      ePairsInitQueueHead,

      // Work queue of node pairs for iterative hierarchy traversal
      ePairsInputQueue,
      ePairsInputQueueHead,
      ePairsOutputQueue,
      ePairsOutputQueueHead,

      // Work queue of node pairs which fall outside iterative hierarchy traversal
      ePairsRestQueue,
      ePairsRestQueueHead,
      
      // Work queue of node pairs which form large leaves during hierarchy traversal
      ePairsLeafQueue,
      ePairsLeafQueueHead,

      Length
    };

    enum class ProgramType {
      // General programs
      eDispatch,
      eQueryField,

      // Programs for computation without hierarchy
      eCompFullDrawStencil,
      eCompFullCompact,
      eCompFullField,

      // Programs for computation with single hierarchy
      eCompSingleHierarchyCompact,
      eCompSingleHierarchyField,
      
      // Programs for computation with dual hierarchy
      eCompDualHierarchyFieldIterative,
      eCompDualHierarchyFieldRest,
      eCompDualHierarchyFieldLeaf,
      eCompDualHierarchyFieldAccumulate,

      Length
    };

    enum class TextureType {
      eStencil,
      eField,

      Length
    };

    enum class TimerType {
      eCompCompact,
      eCompField,
      eCompDualHierarchyFieldRest,
      eCompDualHierarchyFieldLeaf,
      eCompDualHierarchyFieldAccumulate,
      eQueryField,

      Length
    };

    // State
    bool _isInit;
    SNEMinimizationBuffers _minimization;
    SNEParams _params;
    util::Logger* _logger;
    GLuint _stencilVAOHandle;
    GLuint _stencilFBOHandle;
    uint _hierarchyRebuildIterations;
    uvec _size;
    bool _useEmbeddingHierarchy;
    bool _useFieldHierarchy;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TextureType, GLuint> _textures;
    util::EnumArray<TimerType, util::GLTimer> _timers;

    // Subcomponents
    EmbeddingHierarchy<2> _embeddingHierarchy;

  public:
    friend void swap(Field2D& a, Field2D& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._minimization, b._minimization);
      swap(a._params, b._params);
      swap(a._logger, b._logger);
      swap(a._stencilVAOHandle, b._stencilVAOHandle);
      swap(a._stencilFBOHandle, b._stencilFBOHandle);
      swap(a._hierarchyRebuildIterations, b._hierarchyRebuildIterations);
      swap(a._size, b._size);
      swap(a._useEmbeddingHierarchy, b._useEmbeddingHierarchy);
      swap(a._useFieldHierarchy, b._useFieldHierarchy);
      swap(a._buffers, b._buffers);
      swap(a._programs, b._programs);
      swap(a._textures, b._textures);
      swap(a._timers, b._timers);
      swap(a._embeddingHierarchy, b._embeddingHierarchy);
    }
  };
}