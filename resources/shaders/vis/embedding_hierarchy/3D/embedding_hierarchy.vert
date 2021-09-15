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
#define BVH_KNODE_3D 8
#define BVH_LOGK_3D 3

// Wrapper structure for Bounds data
struct Bounds {
  vec3 min;
  vec3 max;
  vec3 range;
  vec3 invRange;
};

// Input attributes
layout(location = 0) in vec3 positionIn;
layout(location = 1) in vec3 minbIn;
layout(location = 2) in vec4 node0In;
layout(location = 3) in vec4 node1In;

// Output attributes
layout(location = 0) out vec4 colorOut;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };

// Uniform locations
layout(location = 0) uniform mat4 transform;
layout(location = 1) uniform vec4 color;
layout(location = 2) uniform bool selectLvl;
layout(location = 3) uniform uint selectedLvl;

void main() {
  // Calculate range in which instance should fall
  const uint offset = (0x24924924u >> (32u - BVH_LOGK_3D * selectedLvl));
  const uint range = offset + (1u << (BVH_LOGK_3D * selectedLvl));

  // Obtain normalized [0, 1] boundary values
  const vec3 minb = (minbIn - bounds.min) * bounds.invRange;
  const vec3 diam = node1In.xyz * bounds.invRange;
  
  // Generate position from instanced data
  vec3 pos = minb + diam * (positionIn + 0.5);
  gl_Position = transform * vec4(pos, 1);

  // Generate output color
  if (node0In.w == 0 || (selectLvl && (gl_InstanceID < offset || gl_InstanceID >= range))) {
    colorOut = vec4(0);
  } else {
    colorOut = color;
  }
}