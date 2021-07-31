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

#extension GL_NV_shader_atomic_float : require        // atomicAdd(f32) support
#extension GL_KHR_shader_subgroup_clustered : require // subgroupClusteredAdd(...) support
#extension GL_KHR_shader_subgroup_shuffle : require   // subgroupShuffle(...) support

// Wrapper structure for pair queue data
struct Pair {
  uint f;       // Field hierarchy node index
  uint e;       // Embedding hierarchy node index
};

// Wrapper structure for bounding box data
struct Bounds {
  vec2 min;
  vec2 max;
  vec2 range;
  vec2 invRange;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer ENod0 { vec4 eNode0Buffer[]; };
layout(binding = 1, std430) restrict readonly buffer ENod1 { vec4 eNode1Buffer[]; };
layout(binding = 2, std430) restrict readonly buffer EPosi { vec2 ePosBuffer[]; };
layout(binding = 3, std430) restrict coherent buffer FFiel { float fFieldBuffer[]; }; // atomic rw
layout(binding = 4, std430) restrict readonly buffer LQueu { Pair lQueueBuffer[]; };
layout(binding = 5, std430) restrict readonly buffer LQHea { uint lQueueHead; }; // atomic rw 
layout(binding = 6, std430) restrict readonly buffer Bound { Bounds boundsBuffer; };

// Uniforms
layout(location = 0) uniform uint fLvls;    // Nr. of levels in embedding hierarchy

// Shared memory
shared Bounds bounds;

// Constants
#define BVH_KNODE_2D 4
#define BVH_LOGK_2D 2
const uint nThreads = 4; // Threads to handle all the buffer reads together
const uint thread = gl_SubgroupInvocationID % nThreads;
const uint base = gl_SubgroupInvocationID - thread;
const uint foffset = 0x2AAAAAAAu >> (31u - BVH_LOGK_2D * (fLvls - 1));
const vec2 fbbox = vec2(1.f / float(1u << (fLvls - 1)));

uint shrinkBits15(uint i) {
  i = i & 0x55555555u;
  i = (i | (i >> 1u)) & 0x33333333u;
  i = (i | (i >> 2u)) & 0x0F0F0F0Fu;
  i = (i | (i >> 4u)) & 0x00FF00FFu;
  i = (i | (i >> 8u)) & 0x0000FFFFu;
  return i;
}

uvec2 decode(uint i) {
  uint x = shrinkBits15(i);
  uint y = shrinkBits15(i >> 1);
  return uvec2(x, y);
}

void main() {
  // Check if invoc is within nr of items on leaf queue
  uint idx = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / nThreads;
  if (idx >= lQueueHead) {
    return;
  }

  // Load bounds buffer into shared memory
  if (gl_LocalInvocationID.x == 0) {
    bounds = boundsBuffer;
  }
  barrier();

  // Load node pair from queue using subgroup operations
  Pair pair;
  if (thread == 0) {
    pair = lQueueBuffer[idx];
  }
  pair.f = subgroupShuffle(pair.f, base);
  pair.e = subgroupShuffle(pair.e, base);
  
  // Read node data using subgroup operations
  const vec4 eNode0 = subgroupShuffle((thread == 0) ? eNode0Buffer[pair.e] : vec4(0), base);
  const vec4 eNode1 = subgroupShuffle((thread == 2) ? eNode1Buffer[pair.e] : vec4(0), base + 2);

  // Determine field node position
  const vec2 fpos = bounds.min + bounds.range * fbbox * (vec2(decode(pair.f - foffset)) + 0.5);

  // Range of embedding positions to iterate over
  const uint eBegin = uint(eNode1.w) + thread;
  const uint eEnd = uint(eNode1.w) + uint(eNode0.w);

  // Directly compute forces over embedding positions using subgroup of threads
  vec3 field = vec3(0);
  for (uint i = eBegin; i < eEnd; i += nThreads) {
    const vec2 t = fpos - ePosBuffer[i];
    const float tStud = 1.f / (1.f +  dot(t, t));
    field += vec3(tStud, t * (tStud * tStud));
  }
  field = subgroupClusteredAdd(field, nThreads);

  // Add computed forces to field leaf, using one thread per field component
  if (field != vec3(0) && thread < 3) {
    const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
    atomicAdd(fFieldBuffer[addr[thread]], field[thread]);
  }
}