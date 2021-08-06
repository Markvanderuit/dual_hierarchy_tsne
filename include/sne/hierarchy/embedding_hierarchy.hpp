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
#include "types.hpp"
#include "util/enum.hpp"
#include "util/logger.hpp"
#include "util/key_sort.cuh"
#include "util/gl/program.hpp"
#include "util/gl/timer.hpp"
#include "sne/sne_params.hpp"
#include "sne/buffers/sne_minimization_buffers.hpp"
#include "sne/hierarchy/buffers/embedding_hierarchy_buffers.hpp"

namespace dh::sne {
  template <uint D>
  class EmbeddingHierarchy {
  public:
    // Wrapper class for hierarchy's layout data
    struct Layout {
      uint nPos;    // Nr. of embedding positions in hierarchy
      uint nNodes;  // Nr. of nodes of hierarchy
      uint nLvls;   // Nr. of levels of hierarchy
      
      Layout()
      : nPos(0), nNodes(0), nLvls(0) { }

      Layout(uint nPos)
      : nPos(nPos) {
        constexpr uint logk = (D == 2) ? 2 : 3;

        // Nr of levels in hierarchy, given a (binary) split between contained items
        nLvls = 1 + static_cast<uint>(std::ceil(std::log2(nPos) / logk));

        // Nr of nodes in hierarchy, given said levels
        nNodes = 0u;
        for (uint i = 0u; i < nLvls; i++) {
          nNodes |= 1u << (logk * i); 
        }
      }
    };

    // Constr/destr
    EmbeddingHierarchy();
    EmbeddingHierarchy(SNEMinimizationBuffers minimization, Layout layout, SNEParams params, util::Logger* logger = nullptr);
    ~EmbeddingHierarchy();

    // Copy constr/assignment is explicitly deleted
    EmbeddingHierarchy(const EmbeddingHierarchy&) = delete;
    EmbeddingHierarchy& operator=(const EmbeddingHierarchy&) = delete;

    // Move constr/operator moves handles
    EmbeddingHierarchy(EmbeddingHierarchy&&) noexcept;
    EmbeddingHierarchy& operator=(EmbeddingHierarchy&&) noexcept;

    // Compute hierarchy structure, either through rebuild or (alternatively) a faster refit
    void comp(bool rebuild);

  private:
    enum class BufferType {
      eDispatch,

      // Input and output buffers for dh::util::KeySort
      eMortonUnsorted,
      eMortonSorted,
      eIndicesSorted,

      // Sorted embedding
      eEmbeddingSorted,

      // Work queue of leaf nodes which require bbox computation
      eLeafQueue,
      eLeafQueueHead,

      // Hierarchy data (used in field computation)
      eNode0,
      eNode1,
      eMinB,

      Length
    };

    enum class ProgramType {
      eDispatch,
      eCompMortonUnsorted,
      eCompEmbeddingSorted,
      eCompSubdivision,
      eCompLeaves,
      eCompNodes,
      
      Length
    };

    enum class TimerType {
      eCompSort,
      eCompSubdivision,
      eCompLeaves,
      eCompNodes,

      Length
    };

    // State
    bool _isInit;
    uint _nRebuilds;
    SNEMinimizationBuffers _minimization;
    Layout _layout;
    SNEParams _params;
    util::Logger* _logger;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer> _timers;

    // Subcomponents
    util::KeySort keySort;
  
  public:
    // Getters
    EmbeddingHierarchyBuffers buffers() const {
      return {
        _buffers(BufferType::eNode0),
        _buffers(BufferType::eNode1),
        _buffers(BufferType::eMinB),
      };
    }
    bool isInit() const { return _isInit; }
    Layout layout() const { return _layout; }    

    // std::swap impl
    friend void swap(EmbeddingHierarchy& a, EmbeddingHierarchy& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._nRebuilds, b._nRebuilds);
      swap(a._minimization, b._minimization);
      swap(a._layout, b._layout);
      swap(a._params, b._params);
      swap(a._logger, b._logger);
      swap(a._buffers, b._buffers);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
      swap(a.keySort, b.keySort);
    }
  };
} // dh::sne