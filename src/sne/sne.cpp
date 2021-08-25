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

#include <glad/glad.h>
#include "dh/util/aligned.hpp"
#include "dh/util/error.hpp"
#include "dh/sne/sne.hpp"

namespace dh::sne {
  SNE::SNE() 
  : _isInit(false), _iteration(0), _logger(nullptr) {
    // ...
  }

  SNE::SNE(const std::vector<float>& data, Params params, util::Logger* logger)
  : _isInit(false), _iteration(0), _params(params), _logger(logger) {
    util::log(_logger, "[SNE] Initializing...");

    _similarities = Similarities(data, params, logger);

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

    _similarities.comp();
    if (_params.nLowDims == 2) {
      _minimization = sne::Minimization<2>(_similarities.buffers(), _params, _logger);
    } else if (_params.nLowDims == 3) {
      _minimization = sne::Minimization<3>(_similarities.buffers(), _params, _logger);
    }
  }

  void SNE::compMinimization() {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::compMinimization() called before initialization");
    runtimeAssert(mIsInit, "SNE::compMinimization() called before SNE::compSimilarities()");

    for (uint i = 0; i < _params.iterations; ++i) {
      compMinimizationStep();
    }
  }

  void SNE::compMinimizationStep() {
    if (_iteration == _params.momentumSwitchIter) {
      util::log(_logger, "[SNE]  Switching to final momemtum...");
    }

    if (_iteration == _params.removeExaggerationIter) {
      util::log(_logger, "[SNE]  Removing exaggeration...");
    }

    _timer.tick();
    std::visit([&](auto& m) { m.comp(_iteration++); }, _minimization);
    _timer.tock();
    _timer.poll();
  }

  std::chrono::milliseconds SNE::minimizationTime() const {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::minimizationTime() called before initialization");
    runtimeAssert(mIsInit, "SNE::minimizationTime() called before minimization");

    return _timer.get<util::TimerValue::eTotal, std::chrono::milliseconds>();
  }

  float SNE::klDivergence() {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::klDivergence() called before initialization");
    runtimeAssert(mIsInit, "SNE::klDivergence() called before minimization");
    
    if (!_klDivergence.isInit()) {
      const auto buffers = std::visit([](const auto& m) { return m.buffers(); }, _minimization);
      _klDivergence = KLDivergence(_params, _similarities.buffers(), buffers, _logger);
    }

    return _klDivergence.comp();
  }

  std::vector<float> SNE::embedding() const {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::embedding() called before initialization");
    runtimeAssert(mIsInit, "SNE::embedding() called before minimization");

    const auto buffers = std::visit([](const auto& m) { return m.buffers(); }, _minimization);
    
    if (_params.nLowDims == 2) {
      // Copy embedding data over
      std::vector<float> buffer(_params.n * 2);
      glGetNamedBufferSubData(buffers.embedding, 0, buffer.size() * sizeof(float), buffer.data());
      return buffer;
    } else if (_params.nLowDims == 3) {
      // Copy embedding data over to a padded type (technically 4 floats)
      std::vector<dh::util::AlignedVec<3, float>> _buffer(_params.n);
      glGetNamedBufferSubData(buffers.embedding, 0, _buffer.size() * sizeof(dh::util::AlignedVec<3, float>), _buffer.data());

      // Copy embedding data over to unpadded type (3 floats)
      std::vector<glm::vec<3, float, glm::highp>> buffer(_buffer.begin(), _buffer.end());
      
      // Copy embedding over to floats only
      std::vector<float> embedding(_params.n * 3);
      std::memcpy(embedding.data(), buffer.data(), embedding.size() * sizeof(float));
      return embedding;
    }

    return {};
  }
} // dh::sne