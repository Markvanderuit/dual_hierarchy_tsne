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
#include "dh/types.hpp"
#include "dh/util/io.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/gl/timer.hpp"
#include "dh/util/gl/program.hpp"
#include "dh/util/cu/timer.cuh"
#include "dh/sne/params.hpp"
#include "dh/sne/components/buffers.hpp"

namespace dh::sne {
  class Similarities {
  public:
    using InputSimilrs = std::vector<util::NXBlock>;
    using InputVectors = std::vector<float>;

    // Constr/destr
    Similarities();
    Similarities(const InputSimilrs& inputSimilarities, Params params);
    Similarities(const InputVectors& inputVectors, Params params);
    ~Similarities();

    // Compute similarities
    void comp();

  private:
    // Private delegating constructor handles components shared between the two data constructors
    Similarities(Params params); 
    void comp_full();
    void comp_part();

    enum class BufferType {
      eSimilarities,
      eLayout,
      eNeighbors,
      
      Length
    };

    enum class ProgramType {
      eSimilaritiesComp,
      eExpandComp,
      eLayoutComp,
      eNeighborsComp,
      
      Length
    };

    enum class TimerType {
      eSimilaritiesComp,
      eExpandComp,
      eLayoutComp,
      eNeighborsComp,
      
      Length
    };

    // State
    bool                 _isInit;
    Params               _params;
    const float         *_dataPtr;
    const util::NXBlock *_blockPtr;
    
    // Objects
    util::EnumArray<BufferType, GLuint>           _buffers;
    util::EnumArray<ProgramType, util::GLProgram> _programs;
    util::EnumArray<TimerType, util::GLTimer>     _timers;
    util::CUTimer                                 _knnTimer;
  
  public:
    bool isInit() const { return _isInit; }

    SimilaritiesBuffers buffers() const {
      return {
        _buffers(BufferType::eSimilarities),
        _buffers(BufferType::eLayout),
        _buffers(BufferType::eNeighbors),
      };
    }

    friend void swap(Similarities& a, Similarities& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._dataPtr, b._dataPtr);
      swap(a._blockPtr, b._blockPtr);
      swap(a._buffers, b._buffers);
      swap(a._programs, b._programs);
      swap(a._timers, b._timers);
      swap(a._knnTimer, b._knnTimer);
    }

    dh_declare_noncopyable(Similarities);
  };
} // dh::sne