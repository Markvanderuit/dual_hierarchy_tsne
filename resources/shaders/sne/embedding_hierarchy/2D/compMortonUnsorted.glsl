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

struct Bounds {
  vec2 min;
  vec2 max;
  vec2 range;
  vec2 invRange;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, std430) restrict readonly buffer Posi { vec2 positions[]; };
layout(binding = 1, std430) restrict readonly buffer Boun { Bounds bounds; };
layout(binding = 2, std430) restrict writeonly buffer Mor { uint morton[]; };

layout(location = 0) uniform uint nPoints;

uint expandBits15(uint i) {
  i = (i | (i << 8u)) & 0x00FF00FFu;
  i = (i | (i << 4u)) & 0x0F0F0F0Fu;
  i = (i | (i << 2u)) & 0x33333333u;
  i = (i | (i << 1u)) & 0x55555555u;
  return i;
}

uint mortonCode(vec2 v) {
  v = clamp(v * 32768.f, 0.f, 32767.f);
  uint x = expandBits15(uint(v.x));
  uint y = expandBits15(uint(v.y));
  return x | (y << 1);
}

void main() {
  uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
  if (i >= nPoints) {
    return;
  }

  // Normalize position to [0, 1]
  vec2 pos = (positions[i] - bounds.min) * bounds.invRange;
  
  // Compute and store morton code
  morton[i] = mortonCode(pos);
}