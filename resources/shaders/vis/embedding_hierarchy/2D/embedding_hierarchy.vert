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

// Wrapper structure for Bounds data
struct Bounds {
  vec2 min;
  vec2 max;
  vec2 range;
  vec2 invRange;
};

// Input attributes
layout(location = 0) in vec2 positionIn;
layout(location = 1) in vec2 minbIn;
layout(location = 2) in vec4 node0In;
layout(location = 3) in vec4 node1In;

// Output attributes
layout(location = 0) out vec4 colorOut;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

// Uniform locations
layout(location = 0) uniform mat4 transform;
layout(location = 1) uniform float opacity;
layout(location = 2) uniform bool selectLvl;
layout(location = 3) uniform uint selectedLvl;

void main() {
  // Calculate index range in which instance should fall given level selection
  const uint offset = (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * selectedLvl));
  const uint range = offset + (1u << (BVH_LOGK_2D * selectedLvl));

  // Obtain normalized [0, 1] boundary values
  const vec2 minb = (minbIn - bounds.min) * bounds.invRange;
  const vec2 diam = node1In.xy * bounds.invRange;

  // Embedding node leaves have zero area, account for this in visualization
  /* if (_diam == vec2(0.f)) {
    _diam = vec2(0.005f);
    _minb = _minb - 0.5f * _diam;
  } */
  
  // Generate position from instanced data
  vec2 pos = minb + diam * (positionIn + 0.5);
  pos.y = 1.0f - pos.y;
  gl_Position = transform * vec4(pos, 0, 1);


  // Generate output color
  if (node0In.w == 0 || (selectLvl && (gl_InstanceID < offset || gl_InstanceID >= range))) {
    colorOut = vec4(0);
  } else {
    colorOut = vec4(0.1, 0.4, 1.0, opacity); // color for header image
  }
  
  // vec2 pos = vec2(0);
  // if (node0.w == 0 || (selectLvl && (gl_InstanceID < offset || gl_InstanceID >= range))) {
  //   // Do not draw node
  //   pos = vec2(-999); 
  // } else {
  //   pos = minb + (vert + 0.5) * diam;
  //   pos.y = 1.f - pos.y;
  // }

  // // Output color
  // // colorOut = labels[doFlags ? 0 : gl_InstanceID % 10] / 255.f;
  // colorOut = vec3(0.1, 0.4, 1.0); // for header image

  // // Apply camera transformation to output data
  // gl_Position = transform * vec4(posOut, 0, 1);
}