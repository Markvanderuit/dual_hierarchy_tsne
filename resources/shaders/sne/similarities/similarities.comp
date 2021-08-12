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

#version 460 core

#define FLT_MAX 3.402823466e+38F
#define FLT_MIN 1.175494351e-38F

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer Indi { uint neighboursBuffer[]; };
layout(binding = 1, std430) restrict readonly buffer Dist { float distancesBuffer[]; };
layout(binding = 2, std430) restrict buffer SimilaritiesM { float similaritiesBuffer[]; };

// Uniform values
layout(location = 0) uniform uint nPoints;
layout(location = 1) uniform uint kNeighbours;
layout(location = 2) uniform float perplexity;
layout(location = 3) uniform uint nIters;
layout(location = 4) uniform float epsilon;

void main() {
  const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
  if (i >= nPoints) {
    return;
  }

  // Use beta instead of sigma, as per BH-SNE code
  float beta = 1.f;
  float lowerBoundBeta = -FLT_MAX;
  float upperBoundBeta = FLT_MAX;
  float sum = FLT_MIN;
  
  // Perform a binary search for the best fit for Beta
  bool foundBeta = false;
  uint iter = 0;
  while (!foundBeta && iter < nIters) {
    // Compute P_ij with current value of Beta (Sigma, actually)
    // Ignore j = 0, as that is i itself in the local neighbourhood
    sum = FLT_MIN;
    for (uint j = 1; j < kNeighbours; j++) {
      const uint ij = i * kNeighbours + j;
      const float v = exp(-beta * distancesBuffer[ij]);
      similaritiesBuffer[ij] = v;
      sum += v;
    }

    // Compute entropy over the current gaussian's values
    // float negativeEntropy = 0.f;
    float entropy = 0.f;
    for (uint j = 1; j < kNeighbours; j++) {
      const uint ij = i * kNeighbours + j;
      const float v = similaritiesBuffer[ij];
      const float e = v * log(v);
      // negativeEntropy += (e != e) ? 0 : e;
      entropy += beta * distancesBuffer[ij] * similaritiesBuffer[ij];
    }
    // negativeEntropy *= -1.f;
    entropy = (entropy / sum) + log(sum);

    // Test if the difference falls below epsilon
    // float entropy = (negativeEntropy / sum) + log(sum);
    float entropyDiff = entropy - log(perplexity);
    foundBeta = entropyDiff < epsilon && -entropyDiff < epsilon;

    // Tighten bounds for binary search
    if (!foundBeta) {
      if (entropyDiff > 0) {
        lowerBoundBeta = beta;
        beta = (upperBoundBeta == FLT_MAX || upperBoundBeta == -FLT_MAX)
              ? 2.f * beta
              : 0.5f * (beta + upperBoundBeta);
      } else {
        upperBoundBeta = beta;
        beta = (lowerBoundBeta == FLT_MAX || lowerBoundBeta == -FLT_MAX)
              ? 0.5f * beta
              : 0.5f * (beta + lowerBoundBeta);
      }
    }
    
    iter++;
  }

  // Normalize kernel at the end
  if (!foundBeta) {
    const float v = 1.f / float(kNeighbours - 1);
    for (uint j = 1; j < kNeighbours; j++) {
      const uint ij = i * kNeighbours + j;
      similaritiesBuffer[ij] = v;
    }
  } else {
    const float div = 1.f / sum;
    for (uint j = 1; j < kNeighbours; j++) {
      const uint ij = i * kNeighbours + j;
      similaritiesBuffer[ij] *= div;
    }
  }
}