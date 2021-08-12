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

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer Neig { uint neighboursBuffer[]; };
layout(binding = 1, std430) restrict coherent buffer Layu { uint sizesBuffer[]; };

// Uniform values
layout(location = 0) uniform uint nPoints;
layout(location = 1) uniform uint kNeighbours;

bool containsk(uint j, uint i) {
  // Unfortunately the list is unsorted, so search time is linear
  for (uint k = 1; k < kNeighbours; k++) {
    if (neighboursBuffer[j * kNeighbours + k] == i) {
      return true;
    }
  }
  return false;
}

void main() {
  const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
  if (i >= nPoints) {
    return;
  }

  atomicAdd(sizesBuffer[i], kNeighbours - 1); // remove 1 to remove self from own indices

  for (uint k = 1; k < kNeighbours; k++) {
    // Get index of neighbour
    const uint j = neighboursBuffer[i * kNeighbours + k];
    
    // Find index of self in neighbour's list of neighbours
    // Self not found, add 1 to neighbour's size
    if (!containsk(j, i)) {
      atomicAdd(sizesBuffer[j], 1);
    }
  }
}