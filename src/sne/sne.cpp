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

#include "dh/util/error.hpp"
#include "dh/util/timer.hpp"
#include "dh/sne/sne.hpp"

namespace dh::sne {
  SNE::SNE() 
  : _isInit(false), _iteration(0), _logger(nullptr) {
    // ...
  }

  SNE::SNE(const std::vector<float>& data, Params params, util::Logger* logger)
  : _isInit(false), _iteration(0), _params(params), _logger(logger) {
    util::log(_logger, "[SNE] Initializing...");

    _sneSimilarities = Similarities(data, params, logger);

    _isInit = true;
    util::log(_logger, "[SNE] Initialized");
  }

  SNE::~SNE() {
    // ...
  }

  SNE::SNE(SNE&& other) noexcept {
    swap(*this, other);
  }

  SNE& SNE::operator=(SNE&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void SNE::comp() {
    runtimeAssert(_isInit, "SNE::comp() called before initialization");

    compSimilarities();
    compMinimization();
  }

  void SNE::compSimilarities() {
    runtimeAssert(_isInit, "SNE::compSimilarities() called before initialization");

    _sneSimilarities.comp();
    if (_params.nLowDims == 2) {
      _minimization = sne::Minimization<2>(_sneSimilarities.buffers(), _params, _logger);
    } else if (_params.nLowDims == 3) {
      _minimization = sne::Minimization<3>(_sneSimilarities.buffers(), _params, _logger);
    }
  }

  void SNE::compMinimization() {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::compMinimization() called before initialization");
    runtimeAssert(mIsInit, "SNE::compMinimization() called before SNE::compSimilarities()");

    util::ChronoTimer timer;
    timer.tick();
    for (uint i = 0; i < _params.iterations; ++i) {
      compMinimizationStep();
    }
    timer.tock();
    timer.poll();
    util::logTime(_logger, "[SNE] Minimization time", timer.get<util::TimerValue::eLast>());
  }

  void SNE::compMinimizationStep() {
    if (_iteration == _params.momentumSwitchIter) {
      util::log(_logger, "[SNE]  Switching to final momemtum...");
    }

    if (_iteration == _params.removeExaggerationIter) {
      util::log(_logger, "[SNE]  Removing exaggeration...");
    }

    std::visit([&](auto& m) { m.comp(_iteration); }, _minimization);
    _iteration++;
  }

  float SNE::klDivergence() {
    runtimeAssert(_isInit, "SNE::klDivergence() called before initialization");
    
    if (!_sneKLDivergence.isInit()) {
      const auto buffers = std::visit([](const auto& m) { return m.buffers(); }, _minimization);
      _sneKLDivergence = KLDivergence(_params, _sneSimilarities.buffers(), buffers, _logger);
    }

    return _sneKLDivergence.comp();
  }

  std::vector<float> SNE::embedding() const {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::embedding() called before initialization");
    runtimeAssert(mIsInit, "SNE::embedding() called before minimization");

    std::vector<float> embedding(_params.n);
    return embedding; // TODO Implement with(out) padding
  }
} // dh::sne