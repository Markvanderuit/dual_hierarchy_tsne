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

#include "util/logger.hpp"
#include "util/timer.hpp"
#include "util/gl/error.hpp"
#include "sne/sne.hpp"

namespace dh::sne {
  template <uint D>
  SNE<D>::SNE() 
  : _isInit(false), _iteration(0), _logger(nullptr) {
    // ...
  }

  template <uint D>
  SNE<D>::SNE(const std::vector<float>& data, Params params, util::Logger* logger)
  : _isInit(false), _iteration(0), _params(params), _logger(logger) {
    util::log(_logger, "[SNE] Initializing...");

    _sneSimilarities = Similarities<D>(data, params, logger);

    _isInit = true;
    util::log(_logger, "[SNE] Initialized");
  }

  template <uint D>
  SNE<D>::~SNE() {
    // ...
  }

  template <uint D>
  SNE<D>::SNE(SNE<D>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  SNE<D>& SNE<D>::operator=(SNE<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void SNE<D>::comp() {
    runtimeAssert(_isInit, "SNE<D>::comp() called without proper initialization");
    compSimilarities();
    compMinimization();
  }

  template <uint D>
  void SNE<D>::compSimilarities() {
    _sneSimilarities.comp();
  }

  template <uint D>
  void SNE<D>::compMinimization() {
    util::ChronoTimer timer;
    timer.tick();
    for (uint i = 0; i < _params.iterations; ++i) {
      compMinimizationStep();
    }
    timer.tock();
    timer.poll();
    util::logTime(_logger, "[SNE] Minimization time", timer.get<util::TimerValue::eLast>());
  }

  template <uint D>
  void SNE<D>::compMinimizationStep() {
    if (!_sneMinimization.isInit()) {
      _sneMinimization = Minimization<D>(_sneSimilarities.buffers(), _params, _logger);
    }

    if (_iteration == _params.momentumSwitchIter) {
      util::log(_logger, "[SNE]  Switching to final momemtum...");
    }

    if (_iteration == _params.removeExaggerationIter) {
      util::log(_logger, "[SNE]  Removing exaggeration...");
    }
    
    _sneMinimization.comp(_iteration);
    _iteration++;
  }

  template <uint D>
  float SNE<D>::klDivergence() {
    if (!_sneKLDivergence.isInit()) {
      _sneKLDivergence = KLDivergence<D>(_params, _sneSimilarities.buffers(), _sneMinimization.buffers(), _logger);
    }

    return _sneKLDivergence.comp();
  }

  template <uint D>
  std::vector<float> SNE<D>::embedding() const {
    runtimeAssert(_sneKLDivergence.isInit(), "SNE<D>::embedding() called before minimization was started!");
    std::vector<float> embedding(_params.n);
    // TODO Implement with(out) padding
    return embedding;
  }

  // Template instantiations for 2/3 dimensions
  template class SNE<2>;
  template class SNE<3>;
} // dh::sne