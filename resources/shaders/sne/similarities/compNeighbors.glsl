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

struct Layout {
  uint offset;
  uint size;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z  = 1) in;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer ANei { uint asymmetricNeighboursBuffer[]; };
layout(binding = 1, std430) restrict readonly buffer ASym { float asymmetricSimilaritiesBuffer[]; };
layout(binding = 2, std430) restrict coherent buffer Coun { uint countBuffer[]; };
layout(binding = 3, std430) restrict readonly buffer Layu { Layout layoutBuffer[]; };
layout(binding = 4, std430) restrict writeonly buffer Nei { uint symmetricNeighboursBuffer[]; };
layout(binding = 5, std430) restrict writeonly buffer Sym { float symmetricSimilaritiesBuffer[]; };

// Uniform values
layout(location = 0) uniform uint nPoints;
layout(location = 1) uniform uint kNeighbours;

uint searchk(uint j, uint i) {
  // Unfortunately the list is unsorted, so search is linear
  for (uint k = 1; k < kNeighbours; k++) {
    if (asymmetricNeighboursBuffer[j * kNeighbours + k] == i) {
      return k;
    }
  }
  return 0;
}

void main() {
  const uint i = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / kNeighbours;
  const uint k = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) % kNeighbours;

  if (i >= nPoints) {
    return;
  }

  // TODO Optimize this so the first index is just intrinsically not part of the computation
  if (k == 0) {
    return;
  }

  // Read original asymmetric neighbours
  const uint j = asymmetricNeighboursBuffer[i * kNeighbours + k];
  const float f = asymmetricSimilaritiesBuffer[i * kNeighbours + k];

  // Read offsets and sizes
  const Layout li = layoutBuffer[i];
  const Layout lj = layoutBuffer[j];
  
  // Copy over to own neighbour indices
  symmetricNeighboursBuffer[li.offset + k - 1] = j;

  // Potentially copy self over to neighbour's neighbour indices, if they don't already know it
  uint _k = searchk(j, i);
  if (_k == 0) {
    // Copy self to neighbour's indices, as they don't have ours yet
    _k = atomicAdd(countBuffer[j], 1);
    symmetricNeighboursBuffer[lj.offset + kNeighbours - 1 + _k] = i;

    // Symmetrize similarities
    const float v = 0.5 * f;
    symmetricSimilaritiesBuffer[li.offset + k - 1] = v;
    symmetricSimilaritiesBuffer[lj.offset + (kNeighbours - 1) + _k] = v;
  } else if (j > i) {
    // Symmetrize similarities
    const float v = 0.5 * (f + asymmetricSimilaritiesBuffer[j * kNeighbours + _k]);
    symmetricSimilaritiesBuffer[li.offset + k - 1] = v;
    symmetricSimilaritiesBuffer[lj.offset + _k - 1] = v;
  }
}