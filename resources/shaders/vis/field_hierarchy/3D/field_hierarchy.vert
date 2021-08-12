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

// Input attributes
layout(location = 0) in vec3 positionIn;
layout(location = 1) in uint nodeIn;

// Output attributes
layout(location = 0) out vec4 colorOut;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer FieldBuffer { vec3 fields[]; };

// Uniform locations
layout(location = 0) uniform mat4 transform;
layout(location = 1) uniform float opacity;
layout(location = 2) uniform bool selectLvl;
layout(location = 3) uniform uint selectedLvl;

uint shrinkBits10(uint i) {
  i = i & 0x09249249;
  i = (i | (i >> 2u)) & 0x030C30C3;
  i = (i | (i >> 4u)) & 0x0300F00F;
  i = (i | (i >> 8u)) & 0x030000FF;
  i = (i | (i >> 16u)) & 0x000003FF;
  return i;
}

uvec3 decode(uint i) {
  uint x = shrinkBits10(i);
  uint y = shrinkBits10(i >> 1);
  uint z = shrinkBits10(i >> 2);
  return uvec3(x, y, z);
}

void main() {
  // Determine lvl of node
  uint fLvl = 0;
  {
    uint f = uint(gl_InstanceID);
    do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_3D) > 0);
  }

  // Calculate range in which instance should fall
  const uint offset = (0x24924924u >> (32u - BVH_LOGK_3D * fLvl));
  const uint range = offset + (1u << (BVH_LOGK_3D * fLvl));

  // Calculate bounding box implicitly
  const uvec3 px = decode(uint(gl_InstanceID) - offset);
  const vec3 fbbox = vec3(1) / vec3(1u << fLvl);
  const vec3 fpos = fbbox * (vec3(px) + 0.5);
  
  // Generate position from bounding box and instanced vertex position
  vec3 pos = fpos + positionIn * fbbox;
  gl_Position = transform * vec4(pos, 1);

  // Generate output color
  if ((nodeIn & 3) == 0 || (selectLvl && fLvl != selectedLvl)) {
    colorOut = vec4(0);
  } else {
    colorOut = vec4(1.0, 0.4, 0.1, opacity); // for header image
  }
}