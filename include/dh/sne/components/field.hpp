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

#include "dh/types.hpp"
#include "dh/util/aligned.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/gl/timer.hpp"
#include "dh/util/gl/program.hpp"
#include "dh/sne/params.hpp"
#include "dh/sne/components/buffers.hpp"
#include "dh/sne/components/hierarchy/embedding_hierarchy.hpp"
#include "dh/sne/components/hierarchy/field_hierarchy.hpp"

namespace dh::sne {
  template <uint D>
  class Field {
    // aligned types
    using Bounds = util::AlignedBounds<D>;
    using vec = util::AlignedVec<D, float>;
    using uvec = util::AlignedVec<D, uint>;

  public:
    // Constr/destr
    Field();
    Field(MinimizationBuffers minimization, Params params);
    ~Field();

    // Copy constr/assignment is explicitly deleted
    Field(const Field&) = delete;
    Field& operator=(const Field&) = delete;

    // Move constr/operator moves handles
    Field(Field&&) noexcept;
    Field& operator=(Field&&) noexcept;

    // Compute the field for a size (resolution) and iteration (determines technique)
    void comp(uvec size, uint iteration);

  private:
    // Functions called by Field::comp(size, uint);
    // 1. Functions used by full computation
    void compFullCompact();
    void compFullField();
    // 2. Functions used by at least single hierarchy computation
    void compSingleHierarchyCompact();
    void compSingleHierarchyField();
    // 3. Functions used by dual hierarchy computation
    void compDualHierarchyField();
    // 4. Functions used by all computations
    void resizeField(uvec size);
    void queryField();
    
    enum class BufferType {
      eDispatch,

      // Work queue for pixels in field texture requiring computation
      ePixelQueue,
      
      // Work queue of node pairs to start hierarchy traversal (precomputed)
      ePairsInitQueue,

      // Work queue of node pairs for iterative hierarchy traversal
      ePairsInputQueue,
      ePairsOutputQueue,

      // Work queue of node pairs which fall outside dual-subdivide hierarchy traversal
      ePairsRestQueue,
      
      // Work queue of node pairs which form large leaves during hierarchy traversal
      ePairsLeafQueue,

      // Work queue write heads
      ePixelQueueHead,
      ePairsInitQueueHead,
      ePairsInputQueueHead,
      ePairsOutputQueueHead,
      ePairsRestQueueHead,
      ePairsLeafQueueHead,

      Length
    };

    enum class ProgramType {
      // General programs
      eDispatch,
      eQueryFieldComp,

      // Programs for computation without hierarchy
      eFullCompactDraw,
      eFullCompactComp,
      eFullFieldComp,

      // Programs for computation with single hierarchy
      eSingleHierarchyCompactComp,
      eSingleHierarchyFieldComp,
      
      // Programs for computation with dual hierarchy
      eDualHierarchyFieldDualSubdivideComp,
      eDualHierarchyFieldSingleSubdivideComp,
      eDualHierarchyFieldLeafComp,
      eDualHierarchyFieldAccumulateComp,

      Length
    };

    enum class TextureType {
      eCellmap,
      eStencil,
      eField,

      Length
    };

    enum class TimerType {
      eCompact,
      eField,
      eQueryFieldComp,

      Length
    };

    enum class DualHierarchyState {
      eDualSubdivide,
      eDualSubdivideLast,
      eSingleSubdivideFirst,
      eSingleSubdivide
    };

    // State
    bool _isInit;
    MinimizationBuffers _minimization;
    Params _params;
    GLuint _stencilVAOHandle;
    GLuint _stencilFBOHandle;
    uint _hierarchyRebuildIterations;
    uint _initQueueSize;
    uvec _size;
    bool _useEmbeddingHierarchy;
    bool _useFieldHierarchy;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TextureType, GLuint> _textures;
    util::EnumArray<TimerType, util::GLTimer> _timers;

    // Subcomponents
    EmbeddingHierarchy<D> _embeddingHierarchy;
    FieldHierarchy<D> _fieldHierarchy;

  public:
    // Getters
    FieldBuffers buffers() const {
      return {
        _buffers(BufferType::ePixelQueue),
        _buffers(BufferType::ePixelQueueHead),
      };
    }
    bool isInit() const { return _isInit; }
    uvec size() const { return _size; }
    size_t memSize() const;
    
    // std::swap impl
    friend void swap(Field& a, Field& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._minimization, b._minimization);
      swap(a._params, b._params);
      swap(a._stencilVAOHandle, b._stencilVAOHandle);
      swap(a._stencilFBOHandle, b._stencilFBOHandle);
      swap(a._hierarchyRebuildIterations, b._hierarchyRebuildIterations);
      swap(a._initQueueSize, b._initQueueSize);
      swap(a._size, b._size);
      swap(a._useEmbeddingHierarchy, b._useEmbeddingHierarchy);
      swap(a._useFieldHierarchy, b._useFieldHierarchy);
      swap(a._buffers, b._buffers);
      swap(a._programs, b._programs);
      swap(a._textures, b._textures);
      swap(a._timers, b._timers);
      swap(a._embeddingHierarchy, b._embeddingHierarchy);
      swap(a._fieldHierarchy, b._fieldHierarchy);
    }
  };
} // dh::sne