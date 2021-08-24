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
#include "dh/util/logger.hpp"
#include "dh/sne/params.hpp"
#include "dh/sne/components/similarities.hpp"
#include "dh/sne/components/minimization.hpp"
#include "dh/sne/components/kl_divergence.hpp"

namespace dh::sne {
  class SNE {
  private:
    using Minimization = std::variant<Minimization<2>, Minimization<3>>;
    
  public:
    // Constr/destr
    SNE();
    SNE(const std::vector<float>& data, Params params, util::Logger* logger = nullptr);
    ~SNE();

    // Copy constr/assignment is explicitly deleted (no copying underlying handles)
    SNE(const SNE&) = delete;
    SNE& operator=(const SNE&) = delete;

    // Move constr/operator moves handles
    SNE(SNE&&) noexcept;
    SNE& operator=(SNE&&) noexcept;
    
    // Main computation functions
    void comp();                  // Compute similarities and then perform minimization
    void compSimilarities();      // Only compute similarities
    void compMinimization();      // Only perform minimization
    void compMinimizationStep();  // Only perform a single step of minimization
    
    // Compute and return KL-Divergence of current embedding
    // (Don't do this while minimizing unless you don't care about performance)
    float klDivergence();

    // Return raw data of current embedding
    // ( Don't do this while minimizing unless you don't care about performance)
    std::vector<float> embedding() const;

  private:
    // State
    bool _isInit;
    uint _iteration;
    Params _params;
    util::Logger* _logger;

    // Subcomponents
    Similarities _sneSimilarities;
    Minimization _minimization;
    KLDivergence _sneKLDivergence;

  public:
    // Getters
    bool isInit() const { return _isInit; }

    // Swap internals with another object
    friend void swap(SNE& a, SNE& b) noexcept {
      using std::swap;
      swap(a._isInit, b._isInit);
      swap(a._iteration, b._iteration);
      swap(a._params, b._params);
      swap(a._logger, b._logger);
      swap(a._sneSimilarities, b._sneSimilarities);
      swap(a._minimization, b._minimization);
      swap(a._sneKLDivergence, b._sneKLDivergence);
    }
  };
} // dh::sne