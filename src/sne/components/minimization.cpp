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
#include <resource_embed/resource_embed.hpp>
#include "dh/constants.hpp"
#include "dh/sne/components/minimization.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/vis/components/embedding_render_task.hpp"

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Minimization]");

  // Params for field size
  constexpr uint fieldMinSize = 5;

  template <uint D>
  Minimization<D>::Minimization()
  : _isInit(false) {
    // ...
  }

  template <uint D>
  Minimization<D>::Minimization(SimilaritiesBuffers similarities, Params params)
  : _isInit(false), _similarities(similarities), _params(params), _iteration(0) {
    Logger::newt() << prefix << "Initializing...";

    // Generate randomized embedding data
    // TODO: look at CUDA-tSNE's approach, they have several options available for initialization
    std::vector<vec> embedding(_params.n, vec(0.f));
    {
      // Seed the (bad) rng
      std::srand(_params.seed);
      
      // Generate n random D-dimensional vectors
      for (uint i = 0; i < _params.n; ++i) {
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
    }

    // Initialize shader programs
    {      
      if constexpr (D == 2) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/gradients.comp"));
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/updateEmbedding.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/2D/centerEmbedding.comp"));
      } else if constexpr (D == 3) {
        _programs(ProgramType::eBoundsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/bounds.comp"));
        _programs(ProgramType::eZComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/Z.comp"));
        _programs(ProgramType::eAttractiveComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/attractive.comp"));
        _programs(ProgramType::eGradientsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/gradients.comp"));
        _programs(ProgramType::eUpdateEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/updateEmbedding.comp"));
        _programs(ProgramType::eCenterEmbeddingComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/minimization/3D/centerEmbedding.comp"));
      }
      glAssert();
      
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();

      // Set these uniform values only once
      _programs(ProgramType::eBoundsComp).template          uniform<uint>("nPoints", _params.n);
      _programs(ProgramType::eBoundsComp).template          uniform<float>("padding", 0.0f);
      _programs(ProgramType::eZComp).template               uniform<uint>("nPoints", _params.n);
      _programs(ProgramType::eZComp).template               uniform<uint>("nPoints", _params.n);
      _programs(ProgramType::eAttractiveComp).template      uniform<uint>("nPoints", _params.n);
      _programs(ProgramType::eAttractiveComp).template      uniform<float>("invPoints", 1.f / static_cast<float>(_params.n));
      _programs(ProgramType::eGradientsComp).template       uniform<uint>("nPoints", _params.n);
      _programs(ProgramType::eUpdateEmbeddingComp).template uniform<uint>("nPoints", _params.n);
      _programs(ProgramType::eUpdateEmbeddingComp).template uniform<float>("eta", _params.eta);
      _programs(ProgramType::eUpdateEmbeddingComp).template uniform<float>("minGain", _params.minimumGain);
      _programs(ProgramType::eUpdateEmbeddingComp).template uniform<float>("mult", 1.0);
      _programs(ProgramType::eCenterEmbeddingComp).template uniform<uint>("nPoints", _params.n);
    }

    // Initialize buffer objects
    {
      const std::vector<vec> zeroes(_params.n, vec(0));
      const std::vector<vec> ones(_params.n, vec(1));

      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eEmbedding), _params.n * sizeof(vec), embedding.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eBoundsReduce), 256 * sizeof(vec), ones.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eBoundsMapped), sizeof(Bounds), ones.data(), GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT);
      glNamedBufferStorage(_buffers(BufferType::eBounds), 2 * sizeof(Bounds), ones.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eZ), 2 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eZReduce), 128 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eField), _params.n * 4 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eAttractive), _params.n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eGradients), _params.n * sizeof(vec), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::ePrevGradients), _params.n * sizeof(vec), zeroes.data(), 0);
      glNamedBufferStorage(_buffers(BufferType::eGain), _params.n * sizeof(vec), ones.data(), 0);
      glAssert();
    }

    // Setup map to bounds object
    _bounds = (Bounds *) glMapNamedBufferRange(_buffers(BufferType::eBoundsMapped), 0, sizeof(Bounds), GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT);
    glAssert();

    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::rest() << prefix << "Initialized, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Setup field subcomponent
    _field = Field<D>(buffers(), _params);

#ifdef DH_ENABLE_VIS_EMBEDDING
    // Setup render task
    if (auto& queue = vis::RenderQueue::instance(); queue.isInit()) {
      queue.emplace(vis::EmbeddingRenderTask<D>(buffers(), _params, 0));
    }
#endif // DH_ENABLE_VIS_EMBEDDING

    _isInit = true;
  }

  template <uint D>
  Minimization<D>::~Minimization() {
    if (_isInit) {
      glUnmapNamedBuffer(_buffers(BufferType::eBoundsMapped));
      glDeleteBuffers(_buffers.size(), _buffers.data());
      _isInit = false;
    }
  }

  template <uint D>
  Minimization<D>::Minimization(Minimization<D>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  Minimization<D>& Minimization<D>::operator=(Minimization<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void Minimization<D>::comp() {
    while (_iteration < _params.iterations) {
      compIteration();
    }
  }

  template <uint D>
  void Minimization<D>::compIteration() {
    // 1.
    // Compute embedding bounds
    {
      auto& timer = _timers(TimerType::eBoundsComp);
      timer.tick();

      auto& program = _programs(ProgramType::eBoundsComp);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eBoundsReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eBounds));

      // Dispatch shader, wide iteration
      program.template uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      // Dispatch shader, last iteration
      program.template uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      // Copy results to mapped buffer
      glCopyNamedBufferSubData(_buffers(BufferType::eBounds), _buffers(BufferType::eBoundsMapped),
        0, 0, sizeof(Bounds));
      glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Insert fence object to wait until writes to Bounds have completed
    GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1'000'000); // 1 ms

    // 2.
    // Perform field approximation in subcomponent
    {
      // Determine field texture size by scaling bounds
      const vec range = _bounds->range();
      const float ratio = (D == 2) ? _params.fieldScaling2D : _params.fieldScaling3D;
      uvec size = dh::util::max(uvec(range * ratio), uvec(fieldMinSize));

      // Size becomes nearest larger power of two for field hierarchy
      size = uvec(glm::pow(2, glm::ceil(glm::log(static_cast<float>(size.x)) / glm::log(2.f))));

      // Delegate to subclass
      _field.comp(size, _iteration);
    }

    // 3.
    // Compute Z, ergo a reduction over q_{ij}
    {
      auto& timer = _timers(TimerType::eZComp);
      timer.tick();

      auto& program = _programs(ProgramType::eZComp);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eZReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZ));

      // Dispatch shader
      program.template uniform<uint>("iter", 0);
      glDispatchCompute(128, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.template uniform<uint>("iter", 1);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 4.
    // Compute attractive forces
    { 
      auto& timer = _timers(TimerType::eAttractiveComp);
      timer.tick();

      auto& program = _programs(ProgramType::eAttractiveComp);
      program.bind();

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _similarities.layout);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _similarities.neighbors);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similarities.similarities);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eAttractive));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u / 32u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Compute exaggeration factor
    float exaggeration = 1.0f;
    if (_iteration <= _params.removeExaggerationIter) {
      exaggeration = _params.exaggerationFactor;
    } else if (_iteration <= _params.removeExaggerationIter + _params.exponentialDecayIter) {
      float decay = 1.0f - static_cast<float>(_iteration - _params.removeExaggerationIter)
                         / static_cast<float>(_params.exponentialDecayIter);
      exaggeration = 1.0f + (_params.exaggerationFactor - 1.0f) * decay;
    }

    // 5.
    // Compute gradients
    {
      auto& timer = _timers(TimerType::eGradientsComp);
      timer.tick();

      auto& program = _programs(ProgramType::eGradientsComp);
      program.bind();

      // Set uniforms
      program.template uniform<float>("exaggeration", exaggeration);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eAttractive));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eField));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eZ));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGradients));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 6.
    // Update embedding
    {
      auto& timer = _timers(TimerType::eUpdateEmbeddingComp);
      timer.tick();

      auto& program = _programs(ProgramType::eUpdateEmbeddingComp);
      program.bind();

      // Set once and switch later
      if (_iteration == 0) {
        program.template uniform<float>("iterMult", static_cast<float>(_params.momentum));
      } else if (_iteration == _params.momentumSwitchIter) {
        program.template uniform<float>("iterMult", static_cast<float>(_params.finalMomentum));
      }

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::ePrevGradients));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eGain));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // 7.
    // Re-center embedding
    {
      auto& timer = _timers(TimerType::eCenterEmbeddingComp);
      timer.tick();

      auto& program = _programs(ProgramType::eCenterEmbeddingComp);
      program.bind();

      // Set uniforms
      program.template uniform<float>("exaggeration", exaggeration);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eEmbedding));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eBounds));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // Log progress; spawn progressbar on the current (new on first iter) line
    // reporting current iteration and size of field texture
    if (_iteration == 0) {
      Logger::newl();
    }
    if ((++_iteration % 100) == 0) {
      // Assemble string to print field's dimensions and memory usage
      std::stringstream fieldStr;
      {
        const uvec fieldSize = _field.size();
        fieldStr << fieldSize[0] << "x" << fieldSize[1];
        if constexpr (D == 3) {
          fieldStr << "x" << fieldSize[2];
        }
        fieldStr << " (" << (static_cast<float>(_field.memSize()) / 1'048'576.0f) << " mb)";
      }
      
      const std::string postfix = (_iteration < _params.iterations)
                                ? "iter: " + std::to_string(_iteration) + ", field: " + fieldStr.str()
                                : "Done!";
      util::ProgressBar progressBar(prefix + "Computing...", postfix);
      progressBar.setProgress(static_cast<float>(_iteration) / static_cast<float>(_params.iterations));
    }
  }

  // Template instantiations for 2/3 dimensions
  template class Minimization<2>;
  template class Minimization<3>;
} // dh::sne