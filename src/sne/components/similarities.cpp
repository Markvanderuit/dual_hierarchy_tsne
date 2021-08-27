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

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <resource_embed/resource_embed.hpp>
#include "dh/sne/components/similarities.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/util/cu/inclusive_scan.cuh"

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Similarities]");

  // Params for FAISS
  constexpr uint kMax = 192;      // Don't exceeed this value for big vector datasets unless you have a lot of coffee
  constexpr uint nProbe = 12;
  constexpr uint nListMult = 1;
  
  Similarities::Similarities()
  : _isInit(false), _dataPtr(nullptr) {
    // ...
  }

  Similarities::Similarities(const std::vector<float>& data, Params params)
  : _isInit(false), _dataPtr(data.data()), _params(params) {
    Logger::newt() << prefix << "Initializing...";

    // Initialize shader programs
    {
      _programs(ProgramType::eSimilaritiesComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/similarities.comp"));
      _programs(ProgramType::eExpandComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/expand.comp"));
      _programs(ProgramType::eLayoutComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/layout.comp"));
      _programs(ProgramType::eNeighborsComp).addShader(util::GLShaderType::eCompute, rsrc::get("sne/similarities/neighbors.comp"));
      
      for (auto& program : _programs) {
        program.link();
      }
      glAssert();
    }

    // Initialize buffer object handles
    // Allocation is performed in Similarities::comp() as the required memory size is not yet known
    glCreateBuffers(_buffers.size(), _buffers.data());
    glAssert();

    _isInit = true;
    Logger::rest() << prefix << "Initialized";
  }

  Similarities::~Similarities() {
    if (isInit()) {
      glDeleteBuffers(_buffers.size(), _buffers.data());
    }
  }

  Similarities::Similarities(Similarities&& other) noexcept {
    swap(*this, other);
  }

  Similarities& Similarities::operator=(Similarities&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void Similarities::comp() {
    runtimeAssert(isInit(), "Similarities::comp() called without proper initialization");

    // Data size, dimensionality, requested nearest neighboor size constants
    const uint n = _params.n;
    const uint d = _params.nHighDims;
    const uint k = std::min(kMax, 3 * static_cast<uint>(_params.perplexity) + 1);

    // CPU-side temporary memory for KNN output
    // Though FAISS is CUDA-based, we move to CPU and back when switching to OpenGL
    // instead of using interopability, which requires double the memory
    std::vector<float> knnSquareDistances(n * k);
    std::vector<faiss::Index::idx_t> knnIndices(n * k);
    
    // Progress bar for logging.... well, progress of the similarity computation
    Logger::newl();
    util::ProgressBar progressBar(prefix + "Computing...");
    progressBar.setPostfix("Performing KNN search");
    progressBar.setProgress(0.0f);

    // 1.
    // Compute approximate KNN of each point using FAISS
    // Produce a fixed number (perplexity * 3 + 1) of neighbors
    {
      // Nr. of inverted lists used by FAISS. O(sqrt(n)) is apparently reasonable
      // src: https://github.com/facebookresearch/faiss/issues/112
      uint nLists = nListMult * static_cast<uint>(std::sqrt(n)); 

      // Use a single GPU device. For now, just grab device 0 and pray
      faiss::gpu::StandardGpuResources faissResources;
      faiss::gpu::GpuIndexIVFFlatConfig faissConfig;
      faissResources.setCudaMallocWarning(false); // please.... stop
      faissConfig.device = 0;
      faissConfig.indicesOptions = faiss::gpu::INDICES_32_BIT;
      faissConfig.flatConfig.useFloat16 = true;
        
      // Construct search index
      // Inverted file flat list gives accurate results at significant memory overhead.
      faiss::gpu::GpuIndexIVFFlat faissIndex(
        &faissResources,
        d, 
        nLists,
        faiss::METRIC_L2, 
        faissConfig
      );
      faissIndex.setNumProbes(nProbe);
      faissIndex.train(n,  _dataPtr);
      faissIndex.add(n,  _dataPtr);

      // Perform actual search
      // Store results device cide in cuKnnSquaredDistances, cuKnnIndices, as the
      // rest of construction is performed on device as well.
      faissIndex.search(
        n,
        _dataPtr,
        k,
        knnSquareDistances.data(),
        knnIndices.data()
      );
      
      // Tell FAISS to bugger off
      faissIndex.reset();
      faissIndex.reclaimMemory();
    }
    
    // Define temporary buffer object handles
    enum class TBufferType {
      eDistances,
      eNeighbors,
      eSimilarities,
      eSizes,
      eScan,

      Length
    };
    util::EnumArray<TBufferType, GLuint> tempBuffers;

    // Initialize temporary buffer objects
    {
      const std::vector<uint> zeroes(n * k, 0);
      const std::vector<uint> indices(knnIndices.begin(), knnIndices.end());

      glCreateBuffers(tempBuffers.size(), tempBuffers.data());
      glNamedBufferStorage(tempBuffers(TBufferType::eDistances), n * k * sizeof(float), knnSquareDistances.data(), 0);
      glNamedBufferStorage(tempBuffers(TBufferType::eNeighbors), n * k * sizeof(uint), indices.data(), 0);
      glNamedBufferStorage(tempBuffers(TBufferType::eSimilarities), n * k * sizeof(float), zeroes.data(), 0);
      glNamedBufferStorage(tempBuffers(TBufferType::eSizes), n * sizeof(uint), zeroes.data(), 0);
      glNamedBufferStorage(tempBuffers(TBufferType::eScan), n * sizeof(uint), nullptr, 0);
      glAssert();
    }

    // Update progress bar
    progressBar.setPostfix("Performing similarity computation");
    progressBar.setProgress(1.0f / 6.0f);

    // 2.
    // Compute similarities over generated KNN. This is pretty much a direct copy of the formulation
    // used in BH-SNE, and seems to also be used in CUDA-tSNE.
    {
      auto& timer = _timers(TimerType::eSimilaritiesComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eSimilaritiesComp);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", n);
      program.uniform<uint>("kNeighbours", k);
      program.uniform<float>("perplexity", _params.perplexity);
      program.uniform<uint>("nIters", 200);
      program.uniform<float>("epsilon", 1e-4);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempBuffers(TBufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempBuffers(TBufferType::eDistances));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tempBuffers(TBufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      
      timer.tock();
      glAssert();
    }

    // Update progress bar
    progressBar.setPostfix("Symmetrizing KNN data");
    progressBar.setProgress(2.0f / 6.0f);

    // 3.
    // Expand KNN data so it becomes symmetric. That is, every neigbor referred by a point
    // itself refers to that point as a neighbor.
    {
      auto& timer = _timers(TimerType::eExpandComp);
      timer.tick();
      
      auto& program = _programs(ProgramType::eExpandComp);
      program.bind();
      
      // Set uniforms
      program.uniform<uint>("nPoints", n);
      program.uniform<uint>("kNeighbours", k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempBuffers(TBufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempBuffers(TBufferType::eSizes));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }

    // Update progress bar
    progressBar.setPostfix("Allocating buffers");
    progressBar.setProgress(3.0f / 6.0f);

    // 4.
    // Determine sizes of expanded neighborhoods in memory through prefix sum
    // Leverages CUDA CUB library underneath
    uint symmetricSize;
    {
      util::InclusiveScan scan(tempBuffers(TBufferType::eSizes), tempBuffers(TBufferType::eScan), n);
      scan.comp();
      glGetNamedBufferSubData(tempBuffers(TBufferType::eScan), (n - 1) * sizeof(uint), sizeof(uint), &symmetricSize);
    }

    // Initialize permanent buffer objects
    {
      glNamedBufferStorage(_buffers(BufferType::eSimilarities), symmetricSize * sizeof(float), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eLayout), n * 2 * sizeof(uint), nullptr, 0);
      glNamedBufferStorage(_buffers(BufferType::eNeighbors), symmetricSize * sizeof(uint), nullptr, 0);
      // TODO are these necessary? Think not
      glClearNamedBufferData(_buffers(BufferType::eSimilarities), GL_R32F, GL_RED, GL_FLOAT, nullptr);
      glClearNamedBufferData(_buffers(BufferType::eLayout), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glClearNamedBufferData(_buffers(BufferType::eNeighbors), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glAssert();    
    }

    // Update progress bar
    progressBar.setPostfix("Computing layout");
    progressBar.setProgress(4.0f / 6.0f);

    // 5.
    // Fill layout buffer
    {
      auto& timer = _timers(TimerType::eLayoutComp);
      timer.tick();

      auto& program = _programs(ProgramType::eLayoutComp);
      program.bind();

      // Set uniforms
      program.uniform<uint>("nPoints", n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempBuffers(TBufferType::eScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
    // Update progress bar
    progressBar.setPostfix("Symmetrizing similarities");
    progressBar.setProgress(5.0f / 6.0f);

    // 6.
    // Generate expanded similarities and neighbor buffers, symmetrized and ready for
    // use during the minimization
    {
      auto& timer = _timers(TimerType::eNeighborsComp);
      timer.tick();

      // Clear sizes buffer, we recycle it as an atomic counter
      glClearNamedBufferData(tempBuffers(TBufferType::eSizes), GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

      auto &program = _programs(ProgramType::eNeighborsComp);
      program.bind();

      program.uniform<uint>("nPoints", n);
      program.uniform<uint>("kNeighbours", k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempBuffers(TBufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempBuffers(TBufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tempBuffers(TBufferType::eSizes));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(n * k, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
    // Update progress bar
    progressBar.setPostfix("Done!");
    progressBar.setProgress(1.0f);

    // Output memory use of OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::curt() << prefix << "Allocated buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Delete temporary buffers
    glDeleteBuffers(tempBuffers.size(), tempBuffers.data());
    glAssert();

    // Poll twice so front/back timers are swapped
    glPollTimers(_timers.size(), _timers.data());
    glPollTimers(_timers.size(), _timers.data());
  }
} // dh::sne