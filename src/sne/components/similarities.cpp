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
#include "dh/sne/components/similarities.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/util/cu/inclusive_scan.cuh"
#include "dh/util/cu/knn.cuh"
#include <algorithm>
#include <execution>
#include <numeric>
#include <span>

namespace dh::sne {
  // Logging shorthands
  using util::Logger;
  const std::string prefix = util::genLoggerPrefix("[Similarities]");

  // Constants
  constexpr uint kMax = 192; // Don't exceeed this value for big vector datasets unless you have a lot of coffee and memopry
  
  Similarities::Similarities()
  : _isInit(false), _dataPtr(nullptr) {
    // ...
  }

  Similarities::Similarities(const float * dataPtr, Params params)
  : _isInit(false), _dataPtr(dataPtr), _blockPtr(nullptr), _params(params) {
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

  Similarities::Similarities(const util::NXBlock * dataPtr, Params params)
  : _isInit(false), _dataPtr(nullptr), _blockPtr(dataPtr), _params(params) {
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
    if (_dataPtr) {
      comp_full();
    } else if (_blockPtr) {
      comp_part();
    }
  }
  
  void Similarities::comp_full() {
    runtimeAssert(isInit(), "Similarities::comp_full() called without proper initialization");
    runtimeAssert(_dataPtr, "Simiarities::comp_full() called without proper input data");

    // Actual k for KNN is limited to kMax, and is otherwise (3 * perplexity + 1)
    const uint k = std::min(kMax, 3 * static_cast<uint>(_params.perplexity) + 1);

    // Define temporary buffer object handles
    enum class TBufferType { eDistances, eNeighbors, eSimilarities, eSizes, eScan, Length };
    util::EnumArray<TBufferType, GLuint> tempBuffers;

    // Initialize temporary buffer objects
    {
      const std::vector<uint> zeroes(_params.n * k, 0);
      glCreateBuffers(tempBuffers.size(), tempBuffers.data());
      glNamedBufferStorage(tempBuffers(TBufferType::eDistances),    _params.n * k * sizeof(float), nullptr,       0);
      glNamedBufferStorage(tempBuffers(TBufferType::eNeighbors),    _params.n * k * sizeof(uint), nullptr,        0);
      glNamedBufferStorage(tempBuffers(TBufferType::eSimilarities), _params.n * k * sizeof(float), zeroes.data(), 0);
      glNamedBufferStorage(tempBuffers(TBufferType::eSizes),        _params.n * sizeof(uint), zeroes.data(),      0);
      glNamedBufferStorage(tempBuffers(TBufferType::eScan),         _params.n * sizeof(uint), nullptr,            0);
      glAssert();
    }
    
    // Progress bar for logging steps of the similarity computation
    Logger::newl();
    util::ProgressBar progressBar(prefix + "Computing...");
    progressBar.setPostfix("Performing KNN search");
    progressBar.setProgress(0.0f);

    // 1.
    // Compute approximate KNN of each point, delegated to FAISS
    // Produces a fixed number of neighbors
    {
      util::KNN knn(
        _dataPtr,
        tempBuffers(TBufferType::eDistances),
        tempBuffers(TBufferType::eNeighbors),
        _params.n, k, _params.nHighDims);
      knn.comp();
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
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbours", k);
      program.template uniform<float>("perplexity", _params.perplexity);
      program.template uniform<uint>("nIters", 200);
      program.template uniform<float>("epsilon", 1e-4);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempBuffers(TBufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempBuffers(TBufferType::eDistances));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tempBuffers(TBufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
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
      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbours", k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempBuffers(TBufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempBuffers(TBufferType::eSizes));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
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
      util::InclusiveScan scan(tempBuffers(TBufferType::eSizes), tempBuffers(TBufferType::eScan), _params.n);
      scan.comp();
      glGetNamedBufferSubData(tempBuffers(TBufferType::eScan), (_params.n - 1) * sizeof(uint), sizeof(uint), &symmetricSize);
    }

    // Initialize permanent buffer objects
    glNamedBufferStorage(_buffers(BufferType::eSimilarities), symmetricSize * sizeof(float), nullptr, 0);
    glNamedBufferStorage(_buffers(BufferType::eLayout), _params.n * 2 * sizeof(uint), nullptr, 0);
    glNamedBufferStorage(_buffers(BufferType::eNeighbors), symmetricSize * sizeof(uint), nullptr, 0);
    glAssert();    

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
      program.template uniform<uint>("nPoints", _params.n);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempBuffers(TBufferType::eScan));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _buffers(BufferType::eLayout));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n, 256u), 1, 1);
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

      program.template uniform<uint>("nPoints", _params.n);
      program.template uniform<uint>("kNeighbours", k);

      // Set buffer bindings
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tempBuffers(TBufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, tempBuffers(TBufferType::eSimilarities));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, tempBuffers(TBufferType::eSizes));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _buffers(BufferType::eLayout));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _buffers(BufferType::eNeighbors));
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _buffers(BufferType::eSimilarities));

      // Dispatch shader
      glDispatchCompute(ceilDiv(_params.n * k, 256u), 1, 1);
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

      timer.tock();
      glAssert();
    }
    
    // Update progress bar
    progressBar.setPostfix("Done!");
    progressBar.setProgress(1.0f);

    // Delete temporary buffers
    glDeleteBuffers(tempBuffers.size(), tempBuffers.data());
    glAssert();

    // Output memory use of persistent OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::curt() << prefix << "Completed, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Poll twice so front/back timers are swapped
    glPollTimers(_timers.size(), _timers.data());
    glPollTimers(_timers.size(), _timers.data());
  }

  template <typename T>
  std::span<T> buffer_map_sp(GLuint i, uint flags) {
    GLint params;
    glGetNamedBufferParameteriv(i, GL_BUFFER_SIZE, &params);
    return std::span<T>((T *) glMapNamedBuffer(i, flags), 
                        static_cast<size_t>(params) / sizeof(T));
  }

  void Similarities::comp_part() {
    runtimeAssert(isInit(), "Similarities::comp_part() called without proper initialization");
    runtimeAssert(_blockPtr, "Simiarities::comp_part() called without proper input data");

    // Actual k for KNN is limited to kMax, and is otherwise (3 * perplexity + 1)
    const uint k = std::min(kMax, 3 * static_cast<uint>(_params.perplexity) + 1);

    // Span over input data
    auto block_sp = std::span(_blockPtr, static_cast<size_t>(_params.n));
    
    // Placeholder type and buffer to describe layout block data
    struct LayoutType  {
      uint offset;
      uint size;
    };
    
    // Progress bar for logging steps of the similarity computation
    Logger::newl();
    util::ProgressBar progressBar(prefix + "Computing...");

    // Set progress bar value
    progressBar.setPostfix("Copying matrix layout");
    progressBar.setProgress(0.0f);

    // Transfer symmetric layout size into temporary buffer
    std::vector<LayoutType> blockLayout(block_sp.size());
    std::transform(std::execution::par_unseq,
      block_sp.begin(), block_sp.end(), blockLayout.begin(), 
      [](const util::NXBlock &b) { return LayoutType { 0u, static_cast<uint>(b.size()) }; });

    // Compute symmetric layout offsets using an out-of-place prefix sum
    std::vector<uint> tempBlockOffs(block_sp.size());
    std::transform(std::execution::par_unseq,
      block_sp.begin(), block_sp.end(), tempBlockOffs.begin(), 
      [](const auto &b) { return static_cast<uint>(b.size()); });
    std::exclusive_scan(std::execution::par_unseq,
      tempBlockOffs.begin(), tempBlockOffs.end(), tempBlockOffs.begin(), 0);
    std::transform(std::execution::par_unseq,
      tempBlockOffs.begin(), tempBlockOffs.end(), blockLayout.begin(), blockLayout.begin(),
      [](uint offs, const auto &l) { return LayoutType { offs, l.size }; });
    tempBlockOffs.clear();

    // Set progress bar value
    progressBar.setPostfix("Allocating buffers");
    progressBar.setProgress(1.f / 3.f);

    // Initialize temporary, mappable buffer objects for symmetrized data
    const uint symmetricSize = blockLayout[_params.n - 1].offset + blockLayout[_params.n - 1].size;
    util::EnumArray<BufferType, GLuint> tempBuffers;
    glCreateBuffers(tempBuffers.size(), tempBuffers.data());
    glNamedBufferStorage(tempBuffers(BufferType::eNeighbors),    symmetricSize * sizeof(uint),   nullptr, GL_MAP_WRITE_BIT);
    glNamedBufferStorage(tempBuffers(BufferType::eSimilarities), symmetricSize * sizeof(float),  nullptr, GL_MAP_WRITE_BIT);
    glAssert(); 

    // Set progress bar value
    progressBar.setPostfix("Copying matrix values");
    progressBar.setProgress(2.f / 3.f);

    // Acquire mapped access to neighbour/similarity data
    auto neighbours_sp = buffer_map_sp<uint>(tempBuffers(BufferType::eNeighbors), GL_WRITE_ONLY);
    auto similarity_sp = buffer_map_sp<float>(tempBuffers(BufferType::eSimilarities), GL_WRITE_ONLY);
    glAssert();

    // Perform scatter of SOA similarity/neighbour buffers to acquired AOS buffer maps
    #pragma omp parallel for
    for (size_t i = 0; i < _params.n; ++i) {
      // Acquire block data
      auto &layout = blockLayout[i];
      auto &block  = block_sp[i];

      // Scatter operands
      constexpr auto scatter_1 = [](const auto &p) -> uint  { return p.first;  };
      constexpr auto scatter_2 = [](const auto &p) -> float { return p.second; };

      // Perform sequential scatter copy
      std::transform(block.begin(), block.end(), (neighbours_sp.begin() + layout.offset), scatter_1);
      std::transform(block.begin(), block.end(), (similarity_sp.begin() + layout.offset), scatter_2);
    }

    // Release mapped access
    glUnmapNamedBuffer(tempBuffers(BufferType::eNeighbors));
    glUnmapNamedBuffer(tempBuffers(BufferType::eSimilarities));
    glAssert();

    // Initialize persistent buffers
    glNamedBufferStorage(_buffers(BufferType::eLayout),       _params.n * sizeof(LayoutType), blockLayout.data(), 0);
    glNamedBufferStorage(_buffers(BufferType::eNeighbors),    symmetricSize * sizeof(uint),   nullptr,            0);
    glNamedBufferStorage(_buffers(BufferType::eSimilarities), symmetricSize * sizeof(float),  nullptr,            0);
    glAssert();
    
    // Copy data to the persistent unmappable buffers (which seem to have better access time),
    // and destroy mappable buffers after
    glCopyNamedBufferSubData(tempBuffers(BufferType::eNeighbors), _buffers(BufferType::eNeighbors), 
      0, 0, symmetricSize * sizeof(uint));
    glCopyNamedBufferSubData(tempBuffers(BufferType::eSimilarities), _buffers(BufferType::eSimilarities), 
      0, 0, symmetricSize * sizeof(uint));
    glDeleteBuffers(tempBuffers.size(), tempBuffers.data());
    glAssert();

    // Update progress bar
    progressBar.setPostfix("Done!");
    progressBar.setProgress(1.0f);

    // Output memory use of persistent OpenGL buffer objects
    const GLuint bufferSize = util::glGetBuffersSize(_buffers.size(), _buffers.data());
    Logger::curt() << prefix << "Completed, buffer storage : " << static_cast<float>(bufferSize) / 1'048'576.0f << " mb";

    // Poll twice so front/back timers are swapped
    glPollTimers(_timers.size(), _timers.data());
    glPollTimers(_timers.size(), _timers.data());
  }
} // dh::sne