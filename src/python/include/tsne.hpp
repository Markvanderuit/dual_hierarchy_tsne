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

#pragma once

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dh/util/gl/window.hpp"
#include "dh/sne/params.hpp"

namespace py = pybind11;
using array_t = py::array_t<float, py::array::c_style | py::array::forcecast>;

int testAdd(int i = 1, int j = 2);

/**
 * TSNE wrapper class with python bindings.
 * 
 * Tries to follow the Scikit.Learn t-SNE naming conventions. Not all parameters from
 * that method are supported.
 * See: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
 */
class TSNE {

public:
  TSNE();

  /* // Class constructor (parameter names follow sklearn.manifold.TSNE parameters)
  TSNE(
    int n_components = 2,
    float perplexity = 30.0f,
    float early_exaggeration = 12.0f,
    float learning_rate = 200.0f,
    int n_iter = 1000,
    int n_iter_without_progress = 300,
    float min_grad_norm = 1e-7,
    int verbose = 0,
    int random_state = 1,
    float angle = 0.25
  ); */

  ~TSNE();

  // Class attributes (these are only set after optimization completes)
  int n_iter;
  array_t embedding;
  float kl_divergence;

  // Class methods
  // array_t fit_transform(array_t X);
  // void fit(array_t X);
  // py::dict get_params(bool deep = true);
  // void set_params(py::kwargs params);

private:
  // Class state 
  dh::util::GLWindow _context;
  dh::sne::Params _params;
};