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
layout(binding = 3, std430) restrict readonly buffer Posi { vec2 positionsBuffer[]; };
layout(binding = 4, std430) restrict readonly buffer Leaf { uint leafBuffer[]; };
layout(binding = 5, std430) restrict readonly buffer Head { uint leafHead; };
layout(binding = 6, std430) restrict readonly buffer Bound { Bounds bounds; };

// Uniforms
layout(location = 0) uniform uint nNodes;

Node read(uint i) {
  // If node has no mass, return early with empty node
  vec4 node0 = node0Buffer[i];
  if (node0.w == 0) {
    return Node(vec4(0), vec4(0), vec2(0));
  }
  return Node(node0, node1Buffer[i], vec2(0));
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

void main() {
  // Check if invoc is within nr of items on leaf queue
  const uint j = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
  if (j >= leafHead) {
    return;
  }

  // Read node from queue
  const uint i = leafBuffer[j];
  Node node = read(i);

  const uint begin = uint(node.node1.w);
  const uint end = begin + uint(node.node0.w);

  // Grab data for first embedding point in range
  vec2 pos = positionsBuffer[begin];
  vec2 maxb = pos;
  node.minb = pos;
  node.node0.xy = pos;

  // Iterate over rest of embedding points in range
  for (uint i = begin + 1; i < end; i++) {
    pos = positionsBuffer[i];
    node.node0.xy += pos;
    node.minb = min(node.minb, pos);
    maxb = max(maxb, pos);
  }

  node.node0.xy /= node.node0.w; // Divide center of mass by nr. of embedding points
  node.node1.xy = maxb - node.minb; // Store extent of bounding box, not maximum

  write(i, node);
}