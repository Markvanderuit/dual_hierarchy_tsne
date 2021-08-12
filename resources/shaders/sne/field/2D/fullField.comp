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

// Wrapper structure for Bounds data
struct Bounds {
  vec2 min;
  vec2 max;
  vec2 range;
  vec2 invRange;
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Buffer object bindings
layout(binding = 0, std430) restrict readonly buffer Posit { vec2 posBuffer[]; };
layout(binding = 1, std430) restrict readonly buffer Bound { Bounds bounds; };
layout(binding = 2, std430) restrict readonly buffer Queue { uvec2 queueBuffer[]; };
layout(binding = 0, rgba32f) restrict writeonly uniform image2D fieldImage;

// Uniform locations
layout(location = 0) uniform uint nPoints;
layout(location = 1) uniform uvec2 textureSize;

// Reduction components
const uint groupSize = gl_WorkGroupSize.x;
const uint halfGroupSize = groupSize / 2;
shared vec3 reductionArray[halfGroupSize];

// Shared memory for pixel positiovn
shared uvec2 px;

void main() {
  // Read pixel position from work queue. Make sure not to exceed queue head
  const uint i = gl_WorkGroupID.x;
  const uint li = gl_LocalInvocationID.x;
  if (li == 0) {
    px = queueBuffer[i];
  }
  barrier();

  // Compute pixel position in [0, 1]
  // Then map pixel position to domain bounds
  vec2 pos = (vec2(px) + 0.5) / vec2(textureSize);
  pos = pos * bounds.range + bounds.min;

    // Iterate over points to obtain density/gradient field values
  vec3 field = vec3(0);
  for (uint j = li; j < nPoints; j += groupSize) {
    // Field layout is: S, V.x, V.y,
    vec2 t = pos - posBuffer[j];
    float tStud = 1.f / (1.f + dot(t, t));
    field += vec3(tStud, t * (tStud * tStud));
  }
  
  // Perform reduce add over all computed points for this pixel
  if (li >= halfGroupSize) {
    reductionArray[li - halfGroupSize] = field;
  }
  barrier();
  if (li < halfGroupSize) {
    reductionArray[li] += field;
  }
  for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
    barrier();
    if (li < i) {
      reductionArray[li] += reductionArray[li + i];
    }
  }
  barrier();
  
  if (li == 0) {
    vec3 reducedArray = reductionArray[0] + reductionArray[1];
    imageStore(fieldImage, ivec2(px), vec4(reducedArray, 0));
  }
}