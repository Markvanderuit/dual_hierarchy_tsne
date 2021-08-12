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

// Input attributes
layout(location = 0) in vec2 positionIn;
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
layout(location = 4) uniform bool sumLvls;

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
  // Determine lvl of node
  uint fLvl = 0;
  {
    uint f = uint(gl_InstanceID);
    do { fLvl++; } while ((f = (f - 1) >> BVH_LOGK_2D) > 0);
  }

  // Calculate range in which instance should fall
  const uint offset = (0x2AAAAAAAu >> (31u - BVH_LOGK_2D * fLvl));
  const uint range = offset + (1u << (BVH_LOGK_2D * fLvl));

  // Calculate bounding box implicitly
  const uvec2 px = decode(uint(gl_InstanceID) - offset);
  const vec2 fbbox = vec2(1) / vec2(1u << fLvl);
  const vec2 fpos = fbbox * (vec2(px) + 0.5);
  
  // Generate position from bounding box and instanced vertex position
  vec2 pos = fpos + positionIn * fbbox;
  pos.y = 1.0f - pos.y;
  gl_Position = transform * vec4(pos, 0, 1);

  vec3 field = fields[uint(gl_InstanceID)];
  if (sumLvls) {
    // Gather field values for this node
    const uint addr = (uint(gl_InstanceID) - 1) >> BVH_LOGK_2D;
    const uint stop = 0x2AAAAAAAu >> (31u - BVH_LOGK_2D * (2));
    for (uint k = addr; k >= stop; k = (k - 1) >> BVH_LOGK_2D) {
      field += fields[k];
    }
  }

  // Generate output color
  if ((nodeIn & 3) == 0 || (selectLvl && fLvl != selectedLvl)) {
    colorOut = vec4(0);
  } else {
    // if (field != vec3(0)) {
    //   colorOut = vec4(0.5 + 0.5 * normalize(field.yz), 0.5, opacity);
    // } else {
    //   colorOut = vec4(0.5, 0.5, 0.5, opacity);
    // }
    colorOut = vec4(1.0, 0.4, 0.1, opacity); // for header image
  }
}