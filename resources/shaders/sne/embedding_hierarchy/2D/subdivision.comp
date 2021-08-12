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

// Constants
#define BVH_KNODE_2D 4
#define BVH_LOGK_2D 2
#define BVH_KLEAF_2D 4

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer Mort { uint mortonBuffer[]; };
layout(binding = 1, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
layout(binding = 2, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
layout(binding = 3, std430) restrict writeonly buffer Leaf { uint leafBuffer[]; };
layout(binding = 4, std430) restrict coherent buffer LHead { uint leafHead; };

// Uniforms
layout(location = 0) uniform bool isBottom;
layout(location = 1) uniform uint rangeBegin;
layout(location = 2) uniform uint rangeEnd;

uint findSplit(uint first, uint last) {
  uint firstCode = mortonBuffer[first];
  uint lastCode = mortonBuffer[last];
  uint commonPrefix = findMSB(firstCode ^ lastCode);

  // Initial guess for split position
  uint split = first;
  uint step = last - first;

  // Perform a binary search to find the split position
  do {
    step = (step + 1) >> 1; // Decrease step size
    uint _split = split + step; // Possible new split position

    if (_split < last) {
      uint splitCode = mortonBuffer[_split];
      uint splitPrefix = findMSB(firstCode ^ splitCode);

      // Accept newly proposed split for this iteration
      if (splitPrefix < commonPrefix) {
        split = _split;
      }
    }
  } while (step > 1);

  return split;
}

void main() {
  // Check if invoc exceeds range of child nodes we want to compute
  const uint i = rangeBegin 
                + (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) 
                / BVH_KNODE_2D;
  if (i > rangeEnd) {
    return;
  }
  const uint t = gl_LocalInvocationID.x % BVH_KNODE_2D;

  // Load parent range
  uint begin = uint(node1Buffer[i].w);
  uint mass = uint(node0Buffer[i].w);

  // Subdivide if the mass is large enough
  if (mass <= BVH_KLEAF_2D) {
    begin = 0;
    mass = 0;
  } else {
    // First, find split position based on morton codes
    // Then set node ranges for left and right child based on split
    // If a range is too small to split, it will be passed to the leftmost invocation only
    uint end = begin + mass - 1;
    for (uint j = BVH_KNODE_2D; j > 1; j /= 2) {
      bool isLeft = (t % j) < (j / 2);
      if (mass > BVH_KLEAF_2D) {
        // Node is large enough, split it
        uint split = findSplit(begin, end);
        begin = isLeft ? begin : split + 1;
        end = isLeft ? split : end;
        mass = 1 + end - begin;
      } else {
        // Node is small enough, hand range only to leftmost invocation
        if (!isLeft) {
          begin = 0;
          mass = 0;
          break;
        }
      }
    }
  }

  // Store node data (each invoc stores their own child node)
  uint j = i * BVH_KNODE_2D + 1 + t;
  node0Buffer[j] = vec4(0, 0, 0, mass);
  node1Buffer[j] = vec4(0, 0, 0, begin);

  // Yeet node id on leaf queue if... well if they are a leaf node
  if (mass > 0 && (isBottom || mass <= BVH_KLEAF_2D)) {
    leafBuffer[atomicAdd(leafHead, 1)] = j;
  }
}