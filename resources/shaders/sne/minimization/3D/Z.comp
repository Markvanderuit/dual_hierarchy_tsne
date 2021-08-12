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

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0, std430) restrict readonly buffer Value { vec4 Values[]; };
layout(binding = 1, std430) restrict buffer SumReduce { float SumReduceAdd[128]; };
layout(binding = 2, std430) restrict writeonly buffer Sum { 
  float sumQ;
  float invSumQ;
};
layout(location = 0) uniform uint nPoints;
layout(location = 1) uniform uint iter;

const uint groupSize = gl_WorkGroupSize.x;
const uint halfGroupSize = groupSize / 2;
shared float reduction_array[halfGroupSize];

void main() {
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.f;
  if (iter == 0) {
    // First iteration adds all values
    for (uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + lid;
        i < nPoints;
        i += gl_WorkGroupSize.x * gl_NumWorkGroups.x) {
      sum += max(Values[i].x - 1.f, 0f);
    }
  } else if (iter == 1) {
    // Second iteration adds resulting 128 values
    sum = SumReduceAdd[lid];      
  }

  // Reduce add to a single value
  if (lid >= halfGroupSize) {
    reduction_array[lid - halfGroupSize] = sum;
  }
  barrier();
  if (lid < halfGroupSize) {
    reduction_array[lid] += sum;
  }
  for (uint i = halfGroupSize / 2; i > 1; i /= 2) {
    barrier();
    if (lid < i) {
      reduction_array[lid] += reduction_array[lid + i];
    }
  }
  barrier();
  if (lid < 1) {
    if (iter == 0) {
      SumReduceAdd[gl_WorkGroupID.x] = reduction_array[0] + reduction_array[1];
    } else if (iter == 1) {
      float _sumQ = reduction_array[0] + reduction_array[1];
      sumQ = _sumQ;
      invSumQ = 1.f / _sumQ;
    }
  }
}