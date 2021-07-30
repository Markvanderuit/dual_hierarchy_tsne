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
layout(binding = 0, std430) restrict readonly buffer Pos { vec2 PositiveForces[]; };
layout(binding = 1, std430) restrict readonly buffer Field { vec3 Fields[]; };
layout(binding = 2, std430) restrict readonly buffer SumQ { 
  float sumQ;
  float invSumQ;
};
layout(binding = 3, std430) restrict writeonly buffer Grad { vec2 Gradients[]; };
layout(location = 0) uniform uint nPoints;
layout(location = 1) uniform float exaggeration;

void main() {
  uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
  if (i >= nPoints) {
    return;
  }

  vec2 positive = PositiveForces[i];
  vec2 negative = Fields[i].yz * invSumQ;
  Gradients[i] = 4.f * (exaggeration * positive - negative);
}