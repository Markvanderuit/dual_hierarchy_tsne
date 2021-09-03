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

#include <iostream>
#include "tsne.hpp"
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include "dh/util/logger.hpp"
// #include "dh/util/cu/knn.cuh"
// #include "dh/sne/sne.hpp"

int testAdd(int i, int j) {
  std::cout << "Hello world!" << std::endl;
  return i + j;
}

TSNE::TSNE() {
  std::cout << "Constructor" << std::endl;

  // Ensure an OpenGL context is made available
  if (!dh::util::GLWindow::hasContext()) {
    _context = dh::util::GLWindow::Offscreen();
  }

  faiss::gpu::StandardGpuResources faissResources;

  // dh::util::KNN knn;
  // dh::sne::Minimization<2> _mini;
  // dh::sne::SNE sne;
  // std::cout << sne.isInit() << std::endl;
}

/* TSNE::TSNE(
  int n_components,
  float perplexity,
  float early_exaggeration,
  float learning_rate,
  int n_iter,
  int n_iter_without_progress,
  float min_grad_norm,
  int verbose,
  int random_state,
  float angle
) {
  std::cout << "Constructor" << std::endl;
  
  // Convert parameter arguments to internal format, discarding unsupported arguments
  _params.nLowDims = n_components;
  _params.perplexity = perplexity;
  _params.iterations = n_iter;
  _params.seed = random_state;
  _params.dualHierarchyTheta = angle;

  // Init logging to standard output stream on verbose mode
  if (verbose) {
    dh::util::Logger::init(&std::cout);
  }
} */

TSNE::~TSNE() {
  std::cout << "Destructor" << std::endl;
}

/* array_t TSNE::fit_transform(array_t X) {
  fit(X);
  return embedding;
} */

/* void TSNE::fit(array_t X) {
  py::buffer_info info = X.request();

  // Finalize parameters object
  _params.nHighDims = info.ndim;
  _params.n = info.size / info.ndim;
  
  // Perform computation
  _sne = dh::sne::SNE(static_cast<float *>(info.ptr), _params);
  _sne.comp();

  // Obtain results
  n_iter = _params.iterations;
  kl_divergence = _sne.klDivergence();

  // TODO Return and convert embedding data
} */

// py::dict TSNE::get_params(bool deep) {
  // ...
// }

/* void TSNE::set_params(py::kwargs params) {
  // ...
} */