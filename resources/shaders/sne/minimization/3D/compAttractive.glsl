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

#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require

struct Layout {
  uint offset;
  uint size;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer Posi { vec3 positionsBuffer[]; };
layout(binding = 1, std430) restrict readonly buffer Layo { Layout layoutsBuffer[]; };
layout(binding = 2, std430) restrict readonly buffer Neig { uint neighboursBuffer[]; };
layout(binding = 3, std430) restrict readonly buffer Simi { float similaritiesBuffer[]; };
layout(binding = 4, std430) restrict writeonly buffer Att { vec3 attrForcesBuffer[]; };

// Uniform values
layout(location = 0) uniform uint nPos;
layout(location = 1) uniform float invPos;

// Shorthand subgroup constants
const uint thread = gl_SubgroupInvocationID;
const uint nThreads = gl_SubgroupSize;

void main() {
  const uint i = (gl_WorkGroupSize.x * gl_WorkGroupID.x + gl_LocalInvocationID.x) / nThreads;
  if (i >= nPos) {
    return;
  }

  // Load data for subgroup
  const vec3 position = subgroupBroadcastFirst(thread < 1 ? positionsBuffer[i] : vec3(0));

  // Sum attractive force over nearest neighbours
  Layout l = layoutsBuffer[i];
  vec3 attrForce = vec3(0);
  for (uint ij = l.offset + thread; ij < l.offset + l.size; ij += nThreads) {
    // Calculate difference between the two positions
    const vec3 diff = position - positionsBuffer[neighboursBuffer[ij]];

    // High/low dimensional similarity measures of i and j
    const float p_ij = similaritiesBuffer[ij];
    const float q_ij = 1.f / (1.f + dot(diff, diff));

    // Calculate the attractive force
    attrForce += p_ij * q_ij * diff;
  }
  attrForce = subgroupAdd(attrForce * invPos);

  // Store result
  if (thread < 1) {
    attrForcesBuffer[i] = attrForce;
  }
}