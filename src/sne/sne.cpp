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

#include "dh/sne/sne.hpp"
#include "dh/util/aligned.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"

namespace dh::sne {
  SNE::SNE() 
  : _isInit(false) {
    // ...
  }

  SNE::SNE(const std::vector<float>& data, Params params)
  : _params(params),
    _similarities(data.data(), params),
    _isInit(true) {
    // ...
  }

  SNE::SNE(const float * data, Params params)
  : _params(params),
    _similarities(data, params),
    _isInit(true) {
    // ...
  }

  SNE::SNE(const std::vector<dh::util::NXBlock>& data, Params params)
  : _params(params),
    _similarities(data.data(), params),
    _isInit(true) {
    // ...
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

    // Run timer to track full similarities computation
    _similaritiesTimer.tick();
    _similarities.comp();
    _similaritiesTimer.tock();
    _similaritiesTimer.poll();

    // After similarities are available, initialize minimization subcomponent
    if (_params.nLowDims == 2) {
      _minimization = sne::Minimization<2>(_similarities.buffers(), _params);
    } else if (_params.nLowDims == 3) {
      _minimization = sne::Minimization<3>(_similarities.buffers(), _params);
    }

    // After similarities are available, initialize KL-divergence subcomponent
    const auto buffers = std::visit([](const auto& m) { return m.buffers(); }, _minimization);
    _klDivergence = KLDivergence(_params, _similarities.buffers(), buffers);
  }

  void SNE::compMinimization() {
    const bool mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::compMinimization() called before initialization");
    runtimeAssert(mIsInit, "SNE::compMinimization() called before SNE::compSimilarities()");

    // Run timer to track full minimization computation
    _minimizationTimer.tick();
    std::visit([&](auto& m) { m.comp(); }, _minimization);
    _minimizationTimer.tock();
    _minimizationTimer.poll();
  }

  void SNE::compMinimizationStep() {
    // Run timer to track full minimization computation
    _minimizationTimer.tick();
    std::visit([&](auto& m) { m.compIteration(); }, _minimization);
    _minimizationTimer.tock();
    _minimizationTimer.poll();
  }

  std::chrono::milliseconds SNE::similaritiesTime() const {
    runtimeAssert(_isInit, "SNE::similaritiesTime() called before initialization");

    return _similaritiesTimer.get<util::TimerValue::eTotal, std::chrono::milliseconds>();
  }

  std::chrono::milliseconds SNE::minimizationTime() const {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::minimizationTime() called before initialization");
    runtimeAssert(mIsInit, "SNE::minimizationTime() called before minimization");

    return _minimizationTimer.get<util::TimerValue::eTotal, std::chrono::milliseconds>();
  }

  float SNE::klDivergence() {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::klDivergence() called before initialization");
    runtimeAssert(mIsInit, "SNE::klDivergence() called before minimization");

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
      glAssert();

      return buffer;
    } else if (_params.nLowDims == 3) {
      // Copy embedding data over to a padded type (technically 4 floats)
      std::vector<dh::util::AlignedVec<3, float>> _buffer(_params.n);
      glGetNamedBufferSubData(buffers.embedding, 0, _buffer.size() * sizeof(dh::util::AlignedVec<3, float>), _buffer.data());
      glAssert();
      
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