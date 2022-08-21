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
  : _isInit(false) { }

  SNE::SNE(const InputVectors& inputVectors, Params params)
  : _params(params),
    _similarities(inputVectors, params),
    _isInit(true) { }

  SNE::SNE(const InputSimilrs& inputSimilarities, Params params)
  : _params(params),
    _similarities(inputSimilarities, params),
    _isInit(true) { }

  SNE::~SNE() { }

  void SNE::run() {
    runtimeAssert(_isInit, "SNE::run() called before initialization");
    runSimilarities();
    runMinimization();
  }

  void SNE::runSimilarities() {
    runtimeAssert(_isInit, "SNE::runSimilarities() called before initialization");

    // Run timer to track full similarities computation
    _similaritiesTimer.tick();
    _similarities.comp();
    _similaritiesTimer.tock();
    _similaritiesTimer.poll();

    // After similarities are available, initialize other subcomponents
    if (_params.nLowDims == 2) {
      _minimization = sne::Minimization<2>(_similarities.buffers(), _params);
    } else if (_params.nLowDims == 3) {
      _minimization = sne::Minimization<3>(_similarities.buffers(), _params);
    }
    constexpr auto visit_buffers = [](const auto& m) { return m.buffers(); };
    _klDivergence = KLDivergence(_params, _similarities.buffers(), std::visit(visit_buffers, _minimization));
  }

  void SNE::runMinimization() {
    const bool mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::runMinimization() called before initialization");
    runtimeAssert(mIsInit, "SNE::runMinimization() called before SNE::runSimilarities()");
    constexpr auto visit_comp = [](auto& m) { m.comp(); };

    // Run timer to track full minimization computation
    _minimizationTimer.tick();
    std::visit(visit_comp, _minimization);
    _minimizationTimer.tock();
    _minimizationTimer.poll();
  }

  void SNE::runMinimizationStep() {
    constexpr auto visit_comp = [](auto& m) { m.compIteration(); };
    
    // Run timer to track full minimization computation
    _minimizationTimer.tick();
    std::visit(visit_comp, _minimization);
    _minimizationTimer.tock();
    _minimizationTimer.poll();
  }

  SNE::ms SNE::getSimilaritiesTime() const {
    runtimeAssert(_isInit, "SNE::getSimilaritiesTime() called before initialization");
    return _similaritiesTimer.get<util::TimerValue::eTotal, std::chrono::milliseconds>();
  }

  SNE::ms SNE::getMinimizationTime() const {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::getMinimizationTime() called before initialization");
    runtimeAssert(mIsInit, "SNE::getMinimizationTime() called before minimization");
    return _minimizationTimer.get<util::TimerValue::eTotal, std::chrono::milliseconds>();
  }

  Result SNE::getResult(ResultFlags flags) {
    if (!_isInit || flags == ResultFlags::eNone) { 
      return { };
    }

    Result result;
    if (has_flag(flags, ResultFlags::eKLDivergence)) {
      result.klDivergence = getKLDivergence();
    }
    if (has_flag(flags, ResultFlags::eEmbedding)) {
      result.embedding = getEmbedding();
    }
    if (has_flag(flags, ResultFlags::eTimings)) {
      result.similaritiesTime = getSimilaritiesTime();
      result.minimizationTime = getMinimizationTime();
      result.totalTime = result.similaritiesTime + result.minimizationTime;
    }
    return result;
  }

  float SNE::getKLDivergence() {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::getKLDivergence() called before initialization");
    runtimeAssert(mIsInit, "SNE::getKLDivergence() called before minimization");
    return _klDivergence.comp();
  }

  std::vector<float> SNE::getEmbedding() const {
    const auto mIsInit = std::visit([](const auto& m) { return m.isInit(); }, _minimization);
    runtimeAssert(_isInit, "SNE::getEmbedding() called before initialization");
    runtimeAssert(mIsInit, "SNE::getEmbedding() called before minimization");

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
    
    return { };
  }
} // dh::sne