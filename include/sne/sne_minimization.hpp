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

#include "types.hpp"
#include "aligned.hpp"
#include "util/enum.hpp"
#include "util/logger.hpp"
#include "util/gl/timer.hpp"
#include "util/gl/program.hpp"
#include "sne/sne_params.hpp"
#include "sne/sne_similarities.hpp"
#include "sne/field/field_2d.hpp"
#include "sne/buffers/sne_minimization_buffers.hpp"
#include "sne/buffers/sne_similarities_buffers.hpp"

namespace dh::sne {
  template <uint D> // Dimension of produced embedding
  class SNEMinimization {
    // aligned types
    using Bounds = AlignedBounds<D>;
    using vec = AlignedVec<D, float>;
    using uvec = AlignedVec<D, uint>;

  public:
    // Constr/destr
    SNEMinimization();
    SNEMinimization(SNESimilaritiesBuffers similarities, SNEParams params, util::Logger* logger = nullptr);  
    ~SNEMinimization(); 

    // Copy constr/assignment is explicitly deleted
    SNEMinimization(const SNEMinimization&) = delete;
    SNEMinimization& operator=(const SNEMinimization&) = delete;

    // Move constr/operator moves handles
    SNEMinimization(SNEMinimization&&) noexcept;
    SNEMinimization& operator=(SNEMinimization&&) noexcept;

    // Compute a step of minimization
    void comp(uint iteration);

  private:
    enum class BufferType {
      eEmbedding,
      eBounds,
      eBoundsReduce,
      eZ,
      eZReduce,
      eField,
      eAttractive,
      eGradients,
      ePrevGradients,
      eGain,

      Length
    };

    enum class ProgramType {
      eCompBounds,
      eCompZ,
      eCompAttractive,
      eCompGradients,
      eUpdateEmbedding,
      eCenterEmbedding,

      Length
    };

    enum class TimerType {
      eCompBounds,
      eCompZ,
      eCompAttractive,
      eCompGradients,
      eUpdateEmbedding,
      eCenterEmbedding,

      Length
    };

    // State
    bool _isInit;
    SNEParams _params;
    util::Logger* _logger;
    SNESimilaritiesBuffers _similarities;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer> _timers;

    // Subcomponents
    Field2D _field2D;

  public:
    // Getters
    SNEMinimizationBuffers buffers() const {
      return {
        _buffers(BufferType::eEmbedding),
        _buffers(BufferType::eField),
        _buffers(BufferType::eBounds),
      };
    }
    bool isInit() const { return _isInit; }

    // std::swap impl
    friend void swap(SNEMinimization<D>& a, SNEMinimization<D>& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._logger, b._logger);
      swap(a._similarities, b._similarities);
      swap(a._buffers, b._buffers);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
      swap(a._field2D, b._field2D);
    }
  };
} // dh::sne
