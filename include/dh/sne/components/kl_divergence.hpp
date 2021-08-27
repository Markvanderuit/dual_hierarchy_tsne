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

namespace dh::sne {
  class KLDivergence {
  public:
    KLDivergence();
    KLDivergence(Params params, SimilaritiesBuffers similarities, MinimizationBuffers minimization);
    ~KLDivergence();

    // Copy constr/assignment is explicitly deleted
    KLDivergence(const KLDivergence&) = delete;
    KLDivergence& operator=(const KLDivergence&) = delete;

    // Move constr/operator moves handles
    KLDivergence(KLDivergence&&) noexcept;
    KLDivergence& operator=(KLDivergence&&) noexcept;

    // Compute KL-divergence
    float comp();

  private:
    enum class BufferType {
      eQijSum,
      eKLDSum,
      eReduce,
      eReduceFinal,

      Length
    };

    enum class ProgramType {
      eQijSumComp,
      eKLDSumComp,
      eReduceComp,

      Length
    };

    enum class TimerType {
      eQijSumComp,
      eQijSumReduce,
      eKLDSumComp,
      eKLDSumReduce,

      Length
    };

    // State
    bool _isInit;
    Params _params;
    SimilaritiesBuffers _similarities;
    MinimizationBuffers _minimization;

    // Objects
    util::EnumArray<BufferType, GLuint> _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer> _timers;

  public:
    // Getters
    bool isInit() const { return _isInit; }

    // std::swap impl
    friend void swap(KLDivergence& a, KLDivergence& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._similarities, b._similarities);
      swap(a._minimization, b._minimization);
      swap(a._buffers, b._buffers);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
    }
  };
} // dh::sne
