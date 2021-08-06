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
#include "util/gl/program.hpp"
#include "util/gl/timer.hpp"
#include "sne/sne_params.hpp"
#include "sne/buffers/sne_field_buffers.hpp"
#include "sne/hierarchy/buffers/field_hierarchy_buffers.hpp"

namespace dh::sne {
  template <uint D>
  class FieldHierarchy {
    // Aligned types
    using vec = AlignedVec<D, float>;
    using uvec = AlignedVec<D, uint>;

  public:
    // Wrapper class for hierarchy's layout data
    struct Layout {
      uint nPixels;   // Nr. of pixel positions in hierarchy
      uvec size;      // Field texture resolution
      uint nNodes;    // Nr. of nodes in hierarchy
      uint nLvls;     // Nr. of levels in hierarchy

      Layout()
      : nPixels(0), size(0), nNodes(0), nLvls(0) { }

      Layout(uvec size)
      : Layout(product(size), size) { }

      Layout(uint nPixels, uvec size)
      : nPixels(nPixels), size(size) {
        constexpr uint nodek = (D == 2) ? 4 : 8;
        constexpr uint logk = (D == 2) ? 2 : 3;

        // Nr of possible pixels in field texture
        size_t nPos = product(size);

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
    FieldHierarchy();
    FieldHierarchy(SNEFieldBuffers field, Layout layout, SNEParams params, util::Logger* logger = nullptr);
    ~FieldHierarchy();

    // Copy constr/assignment is explicitly deleted
    FieldHierarchy(const FieldHierarchy&) = delete;
    FieldHierarchy& operator=(const FieldHierarchy&) = delete;

    // Move constr/operator moves handles
    FieldHierarchy(FieldHierarchy&&) noexcept;
    FieldHierarchy& operator=(FieldHierarchy&&) noexcept;
    
    // Compute hierarchy structure, either through rebuild or (alternatively) a faster refit
    // Note that a new layout is provided every time as the underlying field keeps changing
    void comp(bool rebuild, Layout layout);

  private:
    enum class BufferType {
      eNode,  // Node type (2 lsb), leaf pointer to skip singleton chains (30 msb)
      eField, // Node data: sparse field density (x), sparse field gradient (yz/yzw)
      
      Length
    };

    enum class ProgramType {
      eCompLeaves,
      eCompNodes,

      Length
    };

    enum class TimerType {
      eCompLeaves,
      eCompNodes,

      Length
    };

    // State
    bool _isInit;
    uint _nRebuilds;
    SNEFieldBuffers _field;
    Layout _constrLayout;
    Layout _compLayout;
    SNEParams _params;
    util::Logger* _logger;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer> _timers;

  public:
    // Getters
    FieldHierarchyBuffers buffers() const {
      return {
        _buffers(BufferType::eNode),
        _buffers(BufferType::eField),
      };
    }
    bool isInit() const { return _isInit; }
    Layout layout() const { return _compLayout; }    
  
    // std::swap impl
    friend void swap(FieldHierarchy& a, FieldHierarchy& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._nRebuilds, b._nRebuilds);
      swap(a._field, b._field);
      swap(a._constrLayout, b._constrLayout);
      swap(a._compLayout, b._compLayout);
      swap(a._params, b._params);
      swap(a._logger, b._logger);
      swap(a._buffers, b._buffers);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
    }
  };
} // dh::sne