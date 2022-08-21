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

#include <variant>
#include <vector>
#include "dh/types.hpp"
#include "dh/util/io.hpp"
#include "dh/util/timer.hpp"
#include "dh/sne/params.hpp"
#include "dh/sne/components/similarities.hpp"
#include "dh/sne/components/minimization.hpp"
#include "dh/sne/components/kl_divergence.hpp"

namespace dh::sne {
  /**
   * ResultFlags
   * 
   * Return object configuration. Allows you to request only specific results for 
   * sne::SNE::getResult(...) and sne::run(...).
   */
  enum class ResultFlags {
    eNone         = 0x000,
    eKLDivergence = 0x001,
    eTimings      = 0x002,
    eEmbedding    = 0x004,
    eAll          = ResultFlags::eKLDivergence | ResultFlags::eTimings | ResultFlags::eEmbedding
  };
  dh_declare_bitflag(ResultFlags);
  
  /**
   * Result
   * 
   * Return object. Returned by sne::SNE::getResult(...) and sne::run(...); not all data may be 
   * set, depending on which parameters were passed.
   */
  struct Result {
    using Embedding = std::vector<float>;
    using ms        = std::chrono::milliseconds;

    Embedding embedding        = { };
    float     klDivergence     = 0.f;
    ms        similaritiesTime = ms::zero();
    ms        minimizationTime = ms::zero();
  };

  /**
   * SNE
   * 
   * Main object. Holds most state (buffers, shader programs, textures) in its subcomponents
   * and is responsible for the underlying computation of the tSNE algorithm.
   */
  class SNE {
  public:
    using InputSimilrs = std::vector<util::NXBlock>;
    using InputVectors = std::vector<float>;
    using Embedding    = std::vector<float>;

    // Constr/destr
    SNE();
    SNE(const InputSimilrs& inputSimilarities, Params params);
    SNE(const InputVectors& inputVectors, Params params);
    ~SNE();

  private:
    // sne::Minimization<D> uses argument D to specify embedding dimensions but is identical in 
    // structure (on the CPU side). Define both in the same place, use std::visit for polymorphism 
    using Minimization = std::variant<sne::Minimization<2>, sne::Minimization<3>>;
    using ms           = std::chrono::milliseconds;

    // State
    bool              _isInit = false;
    Params            _params = { };
    util::ChronoTimer _similaritiesTimer;
    util::ChronoTimer _minimizationTimer;

    // Subcomponents
    Similarities _similarities;
    Minimization _minimization;
    KLDivergence _klDivergence;
    
  public:
    // Main computation functions
    void run();                  // Compute similarities and then perform minimization
    void runSimilarities();      // Only compute similarities
    void runMinimization();      // Only perform minimization
    void runMinimizationStep();  // Only perform a single step of minimization

    // Getters
    // Don't call most of these *while* minimizing; poor runtime performance
    bool      isInit() const { return _isInit; }
    Result    getResult(ResultFlags flags = ResultFlags::eNone);
    Embedding getEmbedding() const;
    float     getKLDivergence();
    ms        getSimilaritiesTime() const;
    ms        getMinimizationTime() const;

    // Swap internals with another object
    friend void swap(SNE& a, SNE& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._params, b._params);
      swap(a._similaritiesTimer, b._similaritiesTimer);
      swap(a._minimizationTimer, b._minimizationTimer);
      swap(a._similarities, b._similarities);
      swap(a._minimization, b._minimization);
      swap(a._klDivergence, b._klDivergence);
    }

    dh_declare_noncopyable(SNE);
  };

  inline
  Result run(const SNE::InputVectors& data, 
             Params                   params, 
             ResultFlags              flags = ResultFlags::eAll) {
    SNE sne(data, params);
    sne.run();
    return sne.getResult(flags);
  }

  inline
  Result run(const SNE::InputSimilrs& data, 
             Params                   params, 
             ResultFlags              flags = ResultFlags::eAll) {
    SNE sne(data, params);
    sne.run();
    return sne.getResult(flags);
  }
} // dh::sne