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

// Return types for approx(...) function below
#define STOP 0 // Voxel encompasses node, stop traversal
#define DESC 1 // Node encompasses voxel, descend
#define ASCN 2 // Node and voxel dont overlap, ascend

// Wrapper structure for BoundsBuffer data
struct Bounds {
  vec2 min;
  vec2 max;
  vec2 range;
  vec2 invRange;
};

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Buffer, sampler and image bindings
layout(binding = 0, std430) restrict readonly buffer Node1 { vec4 node1Buffer[]; };
layout(binding = 1, std430) restrict readonly buffer MinbB { vec2 minbBuffer[]; };
layout(binding = 2, std430) restrict readonly buffer Bound { Bounds bounds; };
layout(binding = 3, std430) restrict writeonly buffer Queue { uvec2 queueBuffer[]; };
layout(binding = 4, std430) restrict coherent buffer QHead { uint queueHead; }; 

// Uniforms
layout(location = 0) uniform uint nLvls;     // Nr of tree levels
layout(location = 1) uniform uvec2 textureSize;

// Constants
#define BVH_KNODE_2D 4
#define BVH_LOGK_2D 2

// Fun stuff for fast modulo and division and stuff
const uint bitmask = ~((~0u) << BVH_LOGK_2D);

// Traversal data
uint lvl = 1u; // We start a level lower, as the root node will probably never approximate
uint loc = 1u;
uint stack = 1u | (bitmask << (BVH_LOGK_2D * lvl));

// Pixel size
const vec2 bbox = bounds.range / vec2(textureSize);
const float extl = length(bbox);

void descend() {
  // Move down tree
  lvl++;
  loc = loc * BVH_KNODE_2D + 1u;

  // Push unvisited locations on stack
  stack |= (bitmask << (BVH_LOGK_2D * lvl));
}

void ascend() {
  // Find distance to next level on stack
  uint nextLvl = findMSB(stack) / BVH_LOGK_2D;
  uint dist = lvl - nextLvl;

  // Move dist up to where next available stack position is
  // and one to the right
  if (dist == 0) {
    loc++;
  } else {
    loc >>= BVH_LOGK_2D * dist;
  }
  lvl = nextLvl;

  // Pop visited location from stack
  uint shift = BVH_LOGK_2D * lvl;
  uint b = (stack >> shift) - 1;
  stack &= ~(bitmask << shift);
  stack |= b << shift;
}

bool overlap1D(vec2 a, vec2 b) {
  return a.y >= b.x && b.y >= a.x;
}

bool overlap2D(vec2 mina, vec2 maxa, vec2 minb, vec2 maxb) {
  return overlap1D(vec2(mina.x, maxa.x), vec2(minb.x, maxb.x))
      && overlap1D(vec2(mina.y, maxa.y), vec2(minb.y, maxb.y));
}

uint approx(vec2 minb, vec2 maxb, vec2 center) {
  // Fetch node data
  const vec2 _diam = node1Buffer[loc].xy;
  const vec2 _minb = minbBuffer[loc];
  const vec2 _maxb = _minb + _diam;
  const vec2 _center = _minb + 0.5 * _diam; 

  // Test bounding box containment
  const bool isOverlap = overlap2D(minb, maxb, _minb, _maxb);
  const bool isMin = min(_minb, minb) == minb;
  const bool isMax = max(_maxb, maxb) == maxb;

  if (isOverlap) {
    if (lvl == nLvls - 1 || isMin && isMax) {
      return STOP;
    } else {
      return DESC;
    }
  } else {
    return ASCN;
  }
}

void main() {
  const uvec2 xy = gl_WorkGroupID.xy * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy;
  if (min(xy, textureSize - 1) != xy) {
    return;
  }

  // Compute information about this voxel
  vec2 center = (vec2(xy) + 0.5) / vec2(textureSize); // Map to [0, 1]
  center = bounds.min + bounds.range * center; // Map to embedding domain
  const vec2 minb = center - 2.f * bbox;
  const vec2 maxb = center + 2.f * bbox;

  // Traverse tree to find out if there is a closest or contained point
  do {
    const uint appr = approx(minb, maxb, center);
    if (appr == STOP) {
      // Store result and exit
      queueBuffer[atomicAdd(queueHead, 1)] = xy;
      return;
    } else if (appr == ASCN) {
      ascend();
    } else if (appr == DESC) {
      descend();
    } else {
      return;
    }
  } while (lvl > 0);
}