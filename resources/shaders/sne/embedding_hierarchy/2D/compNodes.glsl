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

// Wrapper structure for dode data
struct Node {
  vec4 node0; // center of mass (xy/z) and range size (w)
  vec4 node1; // bbox bounds extent (xy/z) and range begin (w)
  vec2 minb;  // bbox minimum bounds
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
layout(binding = 0, std430) restrict buffer Node0 { vec4 node0Buffer[]; };
layout(binding = 1, std430) restrict buffer Node1 { vec4 node1Buffer[]; };
layout(binding = 2, std430) restrict buffer MinBB { vec2 minbBuffer[]; };
layout(binding = 3, std430) restrict buffer Bound { Bounds bounds; };

// Uniforms
layout(location = 0) uniform uint rangeBegin;
layout(location = 1) uniform uint rangeEnd;

// Shared memory
const uint groupSize = gl_WorkGroupSize.x;
shared Node sharedNode[groupSize / 2]; // should be smaller for larger fanouts, but "eh"

Node read(uint i) {
  // If node has no mass, return early with empty node
  vec4 node0 = node0Buffer[i];
  if (node0.w == 0) { 
    return Node(vec4(0), vec4(0), vec2(0));
  }
  return Node(node0, node1Buffer[i], minbBuffer[i]);
}

void write(uint i, Node node) {
  node0Buffer[i] = node.node0;
  node1Buffer[i] = node.node1;
  minbBuffer[i] = node.minb;
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

// BVH version
Node reduce(Node l, Node r, uint idx) {
  if (r.node0.w == 0) {
    return l;
  } else if (l.node0.w == 0) {
    return r;
  }

  // Center of mass
  float mass = l.node0.w + r.node0.w;
  float begin = min(l.node1.w, r.node1.w);
  vec2 center = (l.node0.xy * l.node0.w + r.node0.xy * r.node0.w)
                / mass;

  // Bounding boxes
  vec2 minb = min(l.minb, r.minb);
  vec2 diam = max(l.node1.xy + l.minb, r.node1.xy + r.minb) - minb;

  return Node(vec4(center, 0, mass), vec4(diam, 0, begin), minb);
}

void main() {
  const uint i = rangeBegin 
                + (gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x);
  if (i > rangeEnd) {
    return;
  }
  const uint s = gl_LocalInvocationID.x / BVH_KNODE_2D;
  const uint t = gl_LocalInvocationID.x % BVH_KNODE_2D;

  // Read in node data per invoc, and let first invoc store in shared memory
  const Node node = read(i);
  if (t == 0) {
    sharedNode[s] = node;
  }
  barrier();

  // Reduce into shared memory over BVH_KNODE_2D invocs
  for (uint _t = 1; _t < BVH_KNODE_2D; _t++) {
    if (t == _t && node.node0.w != 0f) {
      sharedNode[s] = reduce(node, sharedNode[s], (i - 1) >> BVH_LOGK_2D);
    }
    barrier();
  }

  // Let first invocation store result
  if (t == 0 && sharedNode[s].node0.w > 0) {
    const uint j = i / BVH_KNODE_2D;
    write(j, sharedNode[s]);
  }
}