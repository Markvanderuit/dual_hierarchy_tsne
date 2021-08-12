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

#include "types.hpp"

namespace dh::sne {
  struct Params {
    // Input dataset params
    uint n = 0;
    uint nHighDims = 0;

    // Basic tSNE parameters
    uint iterations = 1000;
    float perplexity = 30.f;

    // Approximation parameters
    float singleHierarchyTheta = 0.5f;
    float dualHierarchyTheta = 0.25f;
    float fieldScaling2D = 2.0f;
    float fieldScaling3D = 1.2f;

    // Embedding initialization parameters
    int seed = 1;
    float rngRange = 0.1f;
    
    // Gradient descent iteration parameters
    // TODO COMPARE TO CUDA-SNE for simplification
    uint momentumSwitchIter = 250;
    uint removeExaggerationIter = 250;
    uint exponentialDecayIter = 150;

    // Gradient descent parameters
    float minimumGain = 0.1f;
    float eta = 200.f;
    // TODO COMPARE TO CUDA-SNE for simplification
    float momentum = 0.5f;
    float finalMomentum = 0.8f;
    float exaggerationFactor = 12.0f;
  };
} // dh::sne
