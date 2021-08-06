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

#include <algorithm>
#include <random>
#include <vector>
#include <glad/glad.h>
#include <resource_embed/resource_embed.hpp>
#include "util/gl/error.hpp"
#include "util/gl/metric.hpp"
#include "sne/sne_minimization.hpp"

namespace dh::sne {
  constexpr uint fieldMinSize = 5;

  template <uint D>
  SNEMinimization<D>::SNEMinimization()
  : _isInit(false), _logger(nullptr) {
    // ...
  }

  template <uint D>
  SNEMinimization<D>::SNEMinimization(SNESimilaritiesBuffers similarities, SNEParams params, util::Logger* logger)
  : _isInit(false), _similarities(similarities), _params(params), _logger(logger) {
    util::log(_logger, "[SNEMinimization] Initializing...");

    // Data size
    const uint n = _params.n;

    // Initialize shader programs
    {
      util::log(_logger, "[SNEMinimization]   Creating shader programs");
      
      if constexpr (D == 2) {
        _programs(ProgramType::eCompBounds).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/compBounds.glsl"));
        _programs(ProgramType::eCompZ).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/compZ.glsl"));
        _programs(ProgramType::eCompAttractive).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/compAttractive.glsl"));
        _programs(ProgramType::eCompGradients).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/compGradients.glsl"));
        _programs(ProgramType::eUpdateEmbedding).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/updateEmbedding.glsl"));
        _programs(ProgramType::eCenterEmbedding).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/centerEmbedding.glsl"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eCompBounds).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/compBounds.glsl"));
        _programs(ProgramType::eCompZ).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/compZ.glsl"));
        _programs(ProgramType::eCompAttractive).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/compAttractive.glsl"));
        _programs(ProgramType::eCompGradients).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/compGradients.glsl"));
        _programs(ProgramType::eUpdateEmbedding).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/updateEmbedding.glsl"));
        _programs(ProgramType::eCenterEmbedding).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/centerEmbedding.glsl"));
      }
      
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer objects
    {
      util::log(_logger, "[SNEMinimization]   Creating buffer storage");
      
      const std::vector<vec> zeroes(n, vec(0));
      const std::vector<vec> ones(n, vec(1));

      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eEmbedding), n * sizeof(vec), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBounds), 4 * sizeof(vec), ones.data(), GL_DYNAMIC_STORAGE_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBoundsReduce), 256 * sizeof(vec), ones.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eZ), 2 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eZReduce), 128 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), n * 4 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eAttractive), n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eGradients), n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePrevGradients), n * sizeof(vec), zeroes.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eGain), n * sizeof(vec), ones.data(), 0);
      glAssert();

      // Report buffer storage size
      const GLuint size = util::glGetBuffersSize(_buffers.size(), _buffers.data());
      util::logValue(_logger, "[SNEMinimization]   Buffer storage (mb)", static_cast<float>(size) / 1'048'576.0f);
    }

    // Generate randomized embedding data
    // Basically a copy of what BH-SNE used
    // TODO: look at CUDA-tSNE, which has several options available for initialization
    {
      util::log(_logger, "[SNEMinimization]   Creating randomized embedding");

      // Seed the (bad) rng
      std::srand(_params.seed);
      
      // Generate n random D-dimensional vectors
      std::vector<vec> embedding(n, vec(0.f));
      for (uint i = 0; i < n; ++i) {
        vec v;
        float r;

        do {
          r = 0.f;
          for (uint j = 0; j < D; ++j) {
            v[j] = 2.f * (static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) + 1.f)) - 1.f;
          }
          r = dot(v, v);
        } while (r > 1.f || r == 0.f);

        r = std::sqrt(-2.f * std::log(r) / r);
        embedding[i] = v * r * _params.rngRange;
      }

      // Copy to buffer
      glNamedBufferSubData(_buffers(BufferType::eEmbedding), 0, n * sizeof(vec), embedding.data());
      glAssert();
    }

    // Setup field subcomponent
    if constexpr (D == 2) {
      _field2D = SNEField2D(buffers(), _params, _logger);
    } else if constexpr (D == 3) {
      // ...
    }

    // Setup render task
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      queue.insert(vis::EmbeddingRenderTask<D>(buffers(), _params, 0));
    }

    _isInit = true;
    util::log(_logger, "[SNEMinimization] Initialized");
  }

  template <uint D>
  SNEMinimization<D>::~SNEMinimization() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
      _isInit = false;
    }
  }

  template <uint D>
  SNEMinimization<D>::SNEMinimization(SNEMinimization<D>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  SNEMinimization<D>& SNEMinimization<D>::operator=(SNEMinimization<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void SNEMinimization<D>::comp(uint iteration) {
    // Data size is used a lot
    const uint n = _params.n;

    // 1.
    // Compute embedding bounds
    {
      auto& timer = _timers(TimerType::eCompBounds);
      timer.tick();

      auto& program = _programs(ProgramType::eCompBounds);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", n);
      program.uniform<float>("padding", 0.0f);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eBoundsReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eBounds));

      // Dispatch shader
      program.uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Copy bounds back to host (hey look: an expensive thing I shouldn't be doing)
    Bounds bounds;
    glGetNamedBufferSubData(_buffers(BufferType::eBounds), 0, sizeof(Bounds),  &bounds);

    // 2.
    // Perform field approximation in subcomponent
    {
      // Determine field texture size
      const vec range = bounds.range();
      const float ratio = (D == 2) ? _params.fieldScaling2D : _params.fieldScaling3D;
      uvec size = dh::max(uvec(range * ratio), uvec(fieldMinSize));
      size = uvec(glm::pow(2, glm::ceil(glm::log(static_cast<float>(size.x)) / glm::log(2.f))));

      // Delegate to subclass
      if constexpr (D == 2) {
        _field2D.comp(size, iteration);
      } else if constexpr (D == 3) {
        // ...
      }
    }

    // 3.
    // Compute Z, ergo a sum over q_{ij}
    {
      auto& timer = _timers(TimerType::eCompZ);
      timer.tick();

      auto& program = _programs(ProgramType::eCompZ);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eZReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZ));

      // Dispatch shader
      program.uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 4.
    // Compute attractive forces
    { 
      auto& timer = _timers(TimerType::eCompAttractive);
      timer.tick();

      auto& program = _programs(ProgramType::eCompAttractive);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPos", n);
      program.uniform<float>("invPos", 1.f / static_cast<float>(n));

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _similarities.layout);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _similarities.neighbors);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similarities.similarities);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eAttractive));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n, 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Compute exaggeration factor
    // TODO COMPARE TO CUDA-SNE for simplification
    float exaggeration = 1.0f;
    if (iteration <= _params.removeExaggerationIter) {
      exaggeration = _params.exaggerationFactor;
    } else if (iteration <= _params.removeExaggerationIter + _params.exponentialDecayIter) {
      float decay = 1.0f
                  - static_cast<float>(iteration - _params.removeExaggerationIter)
                  / static_cast<float>(_params.exponentialDecayIter);
      exaggeration = 1.0f + (_params.exaggerationFactor - 1.0f) * decay;
    }

    // 5.
    // Compute gradient
    {
      auto& timer = _timers(TimerType::eCompGradients);
      timer.tick();

      auto& program = _programs(ProgramType::eCompGradients);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", n);
      program.uniform<float>("exaggeration", exaggeration);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eAttractive));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZ));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGradients));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // Precompute instead of doing it in shader N times
    const float iterMult = (static_cast<double>(iteration) < _params.momentumSwitchIter) 
                         ? _params.momentum 
                         : _params.finalMomentum;

    // 6.
    // Update embedding
    {
      auto& timer = _timers(TimerType::eUpdateEmbedding);
      timer.tick();

      auto& program = _programs(ProgramType::eUpdateEmbedding);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", n);
      program.uniform<float>("eta", _params.eta);
      program.uniform<float>("minGain", _params.minimumGain);
      program.uniform<float>("mult", 1.0);
      program.uniform<float>("iterMult", iterMult);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePrevGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGain));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 7.
    // Re-center embedding
    {
      const vec boundsCenter = bounds.center();
      const vec boundsRange = bounds.range();
      float scaling = 1.0f;
      if (exaggeration > 1.2f && boundsRange.y < 0.1f) {
        scaling = 0.1f / boundsRange.y; // TODO update for 3D
      }
      
      auto& timer = _timers(TimerType::eCenterEmbedding);
      timer.tick();

      auto& program = _programs(ProgramType::eCenterEmbedding);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", n);
      program.uniform<float>("scaling", scaling);
      program.uniform<vec>("center", boundsCenter);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eBounds));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }
  }

  // Template instantiations for 2/3 dimensions
  template class SNEMinimization<2>;
  template class SNEMinimization<3>;
} // dh::sne