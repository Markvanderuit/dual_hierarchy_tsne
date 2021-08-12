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

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer Pos { vec2 positions[]; };
layout(binding = 1, std430) restrict buffer BoundsReduce {
  vec2 minBoundsReduceAdd[128];
  vec2 maxBoundsReduceAdd[128];
};
layout(binding = 2, std430) restrict writeonly buffer Bounds { 
  vec2 minBounds;
  vec2 maxBounds;
  vec2 range;
  vec2 invRange;
};

// Uniform bindings
layout(location = 0) uniform uint nPoints;
layout(location = 1) uniform uint iter;
layout(location = 2) uniform float padding;

// Constants, shared memory
const uint halfGroupSize = gl_WorkGroupSize.x / 2;
shared vec2 min_reduction[halfGroupSize];
shared vec2 max_reduction[halfGroupSize];

void main() {
  const uint lid = gl_LocalInvocationIndex.x;
  vec2 min_local = vec2(1e38);
  vec2 max_local = vec2(-1e38);

  if (iter == 0) {
    // First iteration adds all values
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + lid;
        i < nPoints;
        i += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {
      const vec2 pos = positions[i];
      min_local = min(pos, min_local);
      max_local = max(pos, max_local);
    }
  } else if (iter == 1) {
    // Second iteration adds resulting 128 values
    min_local = minBoundsReduceAdd[lid];
    max_local = maxBoundsReduceAdd[lid];
  }

  // Perform reduce-add over all points
  if (lid >= halfGroupSize) {
    min_reduction[lid - halfGroupSize] = min_local;
    max_reduction[lid - halfGroupSize] = max_local;
  }
  barrier();
  if (lid < halfGroupSize) {
    min_reduction[lid] = min(min_local, min_reduction[lid]);
    max_reduction[lid] = max(max_local, max_reduction[lid]);
  }
  for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
    barrier();
    if (lid < i) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + i]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + i]);
    }
  }
  barrier();
  
  // perform store in starting unit
  if (lid < 1) {
    min_local = min(min_reduction[0], min_reduction[1]);
    max_local = max(max_reduction[0], max_reduction[1]);
    if (iter == 0) {
      minBoundsReduceAdd[gl_WorkGroupID.x] = min_local;
      maxBoundsReduceAdd[gl_WorkGroupID.x] = max_local;
    } else if (iter == 1) {
      vec2 padding = (max_local - min_local) * 0.5f * padding;
      vec2 _minBounds = min_local - padding;
      vec2 _maxBounds = max_local + padding;
      vec2 _range = (_maxBounds - _minBounds);

      maxBounds = _maxBounds;
      minBounds = _minBounds;
      range = _range;
      // Prevent div-by-0, set 0 to 1
      invRange = 1.f / (_range + vec2(equal(_range, vec2(0))));
    }
  }
}