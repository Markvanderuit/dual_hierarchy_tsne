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

struct Layout {
  uint offset;
  uint size;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z  = 1) in;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer Scan { uint scanBuffer[]; };
layout(binding = 1, std430) restrict writeonly buffer Lay { Layout layoutBuffer[]; };

// Uniform values
layout(location = 0) uniform uint nPoints;

void main() {
  const uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
  if (i >= nPoints) {
    return;
  }

  // Update layout based on results of prefix sum
  Layout l;
  if (i == 0) {
    l = Layout(0, scanBuffer[0]);
  } else {
    l = Layout(scanBuffer[i - 1], scanBuffer[i] - scanBuffer[i - 1]);
  }
  layoutBuffer[i] = l;
}