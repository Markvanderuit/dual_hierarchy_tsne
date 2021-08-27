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

#include <resource_embed/resource_embed.hpp>
#include "dh/sne/components/kl_divergence.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[KLDivergence]");

  KLDivergence::KLDivergence()
  : _isInit(false) {
    // ...
  }

  KLDivergence::KLDivergence(Params params, SimilaritiesBuffers similarities, MinimizationBuffers minimization)
  : _isInit(false), _params(params), _similarities(similarities), _minimization(minimization) {
    Logger::newt() << prefix << "Initializing...";
    
    // Initialize shader programs
    {
      if (_params.nLowDims == 2) {
        _programs(ProgramType::eQijSumComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/kl_divergence/2D/qijSum.comp"));
        _programs(ProgramType::eKLDSumComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/kl_divergence/2D/KLDSum.comp"));
      } else if (_params.nLowDims == 3) {
        _programs(ProgramType::eQijSumComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/kl_divergence/3D/qijSum.comp"));
        _programs(ProgramType::eKLDSumComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/kl_divergence/3D/KLDSum.comp"));
      }
      _programs(ProgramType::eReduceComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/kl_divergence/reduce.comp"));

      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer objects
    {
      glCreateBuffers(_buffers.size(), _buffers.data());
      glNamedBufferStorage(_buffers(BufferType::eQijSum), _params.n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eKLDSum), _params.n * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eReduce), 256 * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eReduceFinal), sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);
      glAssert();
    }
    
    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::rest() << prefix << "Initialized";
    Logger::newt() << prefix << "Allocated buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    _isInit = true;
  }

  KLDivergence::~KLDivergence() {
    if (_isInit) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
      _isInit = false;
    }
  }
  
  KLDivergence::KLDivergence(KLDivergence&& other) noexcept {
    swap(*this, other);
  }

  KLDivergence& KLDivergence::operator=(KLDivergence&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  float KLDivergence::comp() {
    // 1.
    // Compute q_{ij} over all i and j in O(n^2) time.
    {
      auto& timer = _timers(TimerType::eQijSumComp);
      timer.tick();

      auto& program = _programs(ProgramType::eQijSumComp);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.embedding);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eQijSum));
      
      // In steps of 512, perforn sums over all j
      const uint step = 512;
      const uint end = _params.n;
      for (int begin = 0; begin < end; begin += step) {
        // Dispatch shader for a limited range
        program.uniform<uint>("begin", begin);
        glDispatchCompute(std::min(step, end - begin), 1, 1);
      }
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // 2.
    // Compute Z, i.e. do a parallel reduction over q_{ij}
    {
      auto& timer = _timers(TimerType::eQijSumReduce);
      timer.tick();

      auto& program = _programs(ProgramType::eReduceComp);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eQijSum));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eReduceFinal));

      // Dispatch shader
      program.uniform<uint>("iter", 0u);
      glDispatchCompute(256, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.uniform<uint>("iter", 1u);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // 3.
    // Compute inner sum of KLD: for each i, sum over all j
    // the values of p_{ij} ln (p_{ij} / q_{ij})
    {
      auto& timer = _timers(TimerType::eKLDSumComp);
      timer.tick();

      auto& program = _programs(ProgramType::eKLDSumComp);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.embedding);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eReduceFinal));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _similarities.layout);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _similarities.neighbors);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _similarities.similarities);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eKLDSum));

      // In steps of 512, perforn sums over all j
      const uint step = 512;
      const uint end = _params.n;
      for (int begin = 0; begin < end; begin += step) {
        // Dispatch shader for a limited range
        program.uniform<uint>("begin", begin);
        glDispatchCompute(std::min(step, end - begin), 1, 1);
      }
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // 4.
    // Compute KLD, i.e. do a parallel reduction over the inner sums of step 3.
    {
      auto& timer = _timers(TimerType::eKLDSumReduce);
      timer.tick();

      auto& program = _programs(ProgramType::eReduceComp);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _buffers(BufferType::eKLDSum));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eReduce));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _buffers(BufferType::eReduceFinal));

      // Dispatch shader
      program.uniform<uint>("iter", 0u);
      glDispatchCompute(256, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      program.uniform<uint>("iter", 1u);
      glDispatchCompute(1, 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Return result
    float kld = 0.0;
    glGetNamedBufferSubData(_buffers(BufferType::eReduceFinal), 0, sizeof(float), &kld);
    return kld;
  }
} // dh::sne