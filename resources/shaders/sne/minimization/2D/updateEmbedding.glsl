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
layout(binding = 0, std430) restrict buffer Pos { vec2 Positions[]; };
layout(binding = 1, std430) restrict readonly buffer Grad { vec2 Gradients[]; };
layout(binding = 2, std430) restrict buffer PrevGrad { vec2 PrevGradients[]; };
layout(binding = 3, std430) restrict buffer Gains { vec2 Gain[]; };
layout(location = 0) uniform uint nPoints;
layout(location = 1) uniform float eta;
layout(location = 2) uniform float minGain;
layout(location = 3) uniform float mult;
layout(location = 4) uniform float iterMult;

void main() {
  // Invocation location
  uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationIndex.x;
  if (i >= nPoints) {
    return;
  }
  
  // Load values
  vec2 grad = Gradients[i];
  vec2 pgrad = PrevGradients[i];
  vec2 gain = Gain[i];

  // Compute gain, clamp at minGain
  vec2 gainDir = vec2(!equal(sign(grad), sign(pgrad)));
  gain = mix(gain * 0.8f, 
    gain + 0.2f, 
    gainDir);
  gain = max(gain, minGain);

  // Compute gradient
  vec2 etaGain = eta * gain;
  grad = fma(
    vec2(greaterThan(grad, vec2(0))), 
    vec2(2f), 
    vec2(-1f)
  ) * abs(grad * etaGain) / etaGain;

  // Compute previous gradient
  pgrad = fma(
    pgrad,
    vec2(iterMult),
    -etaGain * grad
  );

  // Store values again
  Gain[i] = gain;
  PrevGradients[i] = pgrad;
  Positions[i] += pgrad * mult;
}