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

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0, std430) restrict buffer Pos { vec3 Positions[]; };
layout(binding = 1, std430) restrict readonly buffer Bounds { 
  vec3 minBounds;
  vec3 maxBounds;
  vec3 range;
  vec3 invRange;
};
layout(location = 0) uniform uint nPoints;
layout(location = 1) uniform vec3 center;
layout(location = 2) uniform float scaling;

void main() {
  uint i = gl_WorkGroupID.x 
          * gl_WorkGroupSize.x
          + gl_LocalInvocationIndex.x;
  if (i >= nPoints) {
    return;
  }
  
  Positions[i] = scaling * (Positions[i] - center); 
}