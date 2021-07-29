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

#include <vector>
#include "types.hpp"
#include "util/enum.hpp"
#include "util/logger.hpp"
#include "util/gl/timer.hpp"
#include "util/gl/program.hpp"
#include "util/cu/timer.cuh"
#include "sne/sne_params.hpp"

namespace dh::sne {
  template <uint D> // Dimension of produced embedding
  class SNESimilarities {
  public:
    // Constr/destr
    SNESimilarities();
    SNESimilarities(const std::vector<float>& data, SNEParams params, util::Logger* logger = nullptr);
    ~SNESimilarities();

    // Copy constr/assignment is explicitly deleted
    SNESimilarities(const SNESimilarities&) = delete;
    SNESimilarities& operator=(const SNESimilarities&) = delete;

    // Move constr/operator moves handles
    SNESimilarities(SNESimilarities&&) noexcept;
    SNESimilarities& operator=(SNESimilarities&&) noexcept;

    // Compute similarities
    void comp();

    bool isInit() const { return _isInit; }

  public:
    enum class BufferType {
      eSimilarities,
      eLayout,
      eNeighbors,
      
      Length
    };

    GLuint buffer(BufferType type) const {
      return _buffers(type);
    }

  private:
    enum class ProgramType {
      eSimilarities,
      eExpand,
      eLayout,
      eNeighbors,
      
      Length
    };

    enum class TimerType {
      eSimilarities,
      eExpand,
      eLayout,
      eNeighbors,
      
      Length
    };

    // State
    bool _isInit;
    SNEParams _params;
    const float* _dataPtr;
    util::Logger* _logger;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer> _timers;
    util::CUTimer _knnTimer;

  public:
    friend void swap(SNESimilarities<D>& a, SNESimilarities<D>& b) noexcept {
      using std::swap;
      swap(a._buffers, b._buffers);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
      swap(a._knnTimer, b._knnTimer);
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._dataPtr, b._dataPtr);
    }
  };
} // dh::sne