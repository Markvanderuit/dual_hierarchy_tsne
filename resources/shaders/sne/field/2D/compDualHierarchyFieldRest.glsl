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

// Necessary extensions (nvidia-centric)
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
layout(binding = 3, std430) restrict readonly buffer FNode { uint fNodeBuffer[]; };
layout(binding = 4, std430) restrict coherent buffer FFiel { float fFieldBuffer[]; };  // Atomic
layout(binding = 5, std430) restrict readonly buffer IQueu { Pair iQueueBuffer[]; };   // Input
layout(binding = 6, std430) restrict readonly buffer IQHea { uint iQueueHead; };       // Atomic
layout(binding = 7, std430) restrict writeonly buffer OQue { Pair oQueueBuffer[]; };   // Output
layout(binding = 8, std430) restrict readonly buffer OQHea { uint oQueueHead; };       // Atomic
layout(binding = 9, std430) restrict writeonly buffer LQue { Pair lQueueBuffer[]; };   // Leaf
layout(binding = 10, std430) restrict coherent buffer LQHe { uint lQueueHead; };       // Atomic
layout(binding = 11, std430) restrict readonly buffer Boun { Bounds boundsBuffer; };

// Uniform values
layout(location = 0) uniform uint eLvls;    // Total nr. of levels in embedding hierarchy
layout(location = 1) uniform uint fLvls;    // Total nr. of levels in field hierarchy
layout(location = 2) uniform float theta2;  // Squared approximation param. for Barnes-Hut
layout(location = 3) uniform bool doBhCrit; // Use new Barnes-Hut criterion
layout(location = 4) uniform bool isLastIter; // No more subdivision, should any errors occur

// Shared memory
shared Bounds bounds;

// Constants
#define BVH_KNODE_2D 4
#define BVH_LOGK_2D 2
#define BVH_KLEAF_2D 4
#define BVH_LARGE_LEAF 16
const uint nThreads = BVH_KNODE_2D;
const uint thread = gl_SubgroupInvocationID % nThreads;
const uint base = gl_SubgroupInvocationID - thread;
const uint fLeafOffset = 0x2AAAAAAAu >> (31u - BVH_LOGK_2D * (fLvls - 1));
const uint eLeafOffset = 0x2AAAAAAAu >> (31u - BVH_LOGK_2D * (eLvls - 1));

// Compute dot product of vector with itself, ie. squared eucl. distance to vector from origin
float sdot(in vec2 v) {
  return dot(v, v);
}

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
  const uint idx = (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x) / BVH_KNODE_2D;
  if (idx >= iQueueHead) {
    return;
  }

  // Load bounds buffer into shared memory
  if (gl_LocalInvocationID.x == 0) {
    bounds = boundsBuffer;
  }
  barrier();

  // Read node pair indices once, share with subgroup operations
  Pair pair;
  if (thread == 0) {
    pair = iQueueBuffer[idx];
  }
  pair.f = subgroupShuffle(pair.f, base);
  pair.e = subgroupShuffle(pair.e, base);

  // Determine node pair's levels in hierarchies
  uint fLvl = 0;
  uint eLvl = 0;
  {
    uint f = pair.f;
    uint e = pair.e;
    do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_2D) > 0);
    do { eLvl++; } while ((e = (e - 1) >> BVH_LOGK_2D) > 0);
  }

  // Read node data for determining subdivision
  uint fNode; // loaded at a later point
  vec4 eNode0 = subgroupShuffle((thread == 0) ? eNode0Buffer[pair.e] : vec4(0), base);
  vec4 eNode1 = subgroupShuffle((thread == 2) ? eNode1Buffer[pair.e] : vec4(0), base + 2);

  // Determine whether we are dealing with leaf nodes
  // Note: we can only end up in this shader given the previous situation! 
  bool fIsLeaf = false; // (pair.f >= fLeafOffset);
  bool eIsLeaf = true;  //(pair.e >= eLeafOffset) || (eNode0.w <= BVH_KLEAF_2D);    

  // Subdivide one side. All invocs subdivide the same side!
  // TODO subdivide the larger projected diagonal node
  const bool eSubdiv = false; // fIsLeaf || (!eIsLeaf && eLvl < fLvl);
  if (eSubdiv) {
    // Descend hierarchy
    eLvl++;
    pair.e = pair.e * BVH_KNODE_2D + 1 + thread;
    
    // Load new node data (now we can load fNode)
    fNode = subgroupShuffle((thread == 0) ? fNodeBuffer[pair.f] : 0u, base);
    eNode0 = eNode0Buffer[pair.e];
    eNode1 = eNode1Buffer[pair.e];

    // Update leaf status
    eIsLeaf = (pair.e >= eLeafOffset) || (eNode0.w <= BVH_KLEAF_2D);
  } else {
    // Descend hierarchy
    fLvl++;
    pair.f = pair.f * BVH_KNODE_2D + 1 + thread;

    // Load new node data (other node already loaded)
    fNode = fNodeBuffer[pair.f];

    // Update leaf status
    fIsLeaf = (pair.f >= fLeafOffset) || ((fNode & 3u) == 1u);
  }

  // Skip singleton chain in field hierarchy if encountered
  if ((fNode & 3u) == 1u) {
    pair.f = fNode >> 2u;
    fLvl = fLvls - 1;
    fIsLeaf = true;
  }

  // Skip computation if dead node is encountered
  vec3 field = vec3(0);
  if (eNode0.w != 0 && fNode != 0) {
    // Determine fNode bbox and center implicitly
    const uvec2 px = decode(pair.f - (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * fLvl)));
    const vec2 fbbox = bounds.range / vec2(1u << fLvl);
    const vec2 fpos = bounds.min + fbbox * (vec2(px) + 0.5);

    // Compute vector dist. and squared eucl dist. between nodes
    const vec2 t = fpos - eNode0.xy;
    const float t2 = dot(t, t);

    // Barnes-Hut criterion
    float r2;
    if (doBhCrit) {
      // Reflect vector so it always approaches almost-orthogonal to the diameter
      const vec2 b = normalize(abs(t) * vec2(-1, 1));
      // Compute relative squared diams
      r2 = max(
        sdot(eNode1.xy - b * dot(eNode1.xy, b)),
        sdot(fbbox - b * dot(fbbox, b))
      );
    } else {
      // Old Barnes-Hut criterion
      r2 = max(sdot(eNode1.xy), sdot(fbbox));
    }

    // Barnes-Hut test
    if (r2 / t2 < theta2) {
      // Approximation passes, compute approximated value and truncate
      const float tStud = 1.f / (1.f + t2);
      field += eNode0.w * vec3(tStud, t * (tStud * tStud));
    } else if (!fIsLeaf && !isLastIter) {
      // Push pair on work queue for further subdivision
      oQueueBuffer[atomicAdd(oQueueHead, 1)] = pair;
    } else if (eNode0.w >= BVH_LARGE_LEAF) {
      // Optimize large leaf pairs in separate shader
      lQueueBuffer[atomicAdd(lQueueHead, 1)] = pair;
    } else {
      // Compute small leaves directly
      const uint begin = uint(eNode1.w);
      const uint end = begin + uint(eNode0.w);
      for (uint j = begin; j < end; ++j) {
        const vec2 t = fpos - ePosBuffer[j];
        const float tStud = 1.f / (1.f +  dot(t, t));
        field += vec3(tStud, t * (tStud * tStud));
      }
    }
  }
  
  if (eSubdiv) {
    field = subgroupClusteredAdd(field, nThreads);
  }

  if (field != vec3(0)) {
    const uvec4 addr = uvec4(4 * pair.f) + uvec4(0, 1, 2, 3);
    if (!eSubdiv) {
      atomicAdd(fFieldBuffer[addr.x], field.x);
      atomicAdd(fFieldBuffer[addr.y], field.y);
      atomicAdd(fFieldBuffer[addr.z], field.z);
    } else if (thread < 3) {
      // Long live subgroups!
      atomicAdd(fFieldBuffer[addr[thread]], field[thread]);
    }
  }
}