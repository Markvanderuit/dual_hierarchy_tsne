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
  vec3 min;
  vec3 max;
  vec3 range;
  vec3 invRange;
};

// 10 distinguishable label colors
const vec3 colors[10] = vec3[10](
  vec3(16, 78, 139),
  vec3(139, 90, 43),
  vec3(138, 43, 226),
  vec3(0, 128, 0),
  vec3(255, 150, 0),
  vec3(204, 40, 40),
  vec3(131, 139, 131),
  vec3(0, 205, 0),
  // vec3(235, 235, 235),
  vec3(20, 20, 20),
  vec3(0, 150, 255)
);

// Input attributes
layout(location = 0) in vec2 positionIn;
layout(location = 1) in vec3 embeddingIn;

// Output attributes
layout(location = 0) out vec3 embeddingOut;
layout(location = 1) out vec3 fragEmbeddingOut;
layout(location = 2) out vec4 colorOut;

// Buffer bindings
layout(binding = 0, std430) restrict readonly buffer BoundsBuffer { Bounds bounds; };
layout(binding = 1, std430) restrict readonly buffer LabelsBuffer { uint labels[]; };

// Uniforms
layout(location = 0) uniform mat4 model_view;
layout(location = 1) uniform mat4 proj;
layout(location = 2) uniform float pointOpacity;
layout(location = 3) uniform float pointRadius;
layout(location = 4) uniform bool drawLabels;

void main() {
  // Calculate embedding position, fragment position
  embeddingOut = (embeddingIn - bounds.min) * bounds.invRange;
  fragEmbeddingOut = embeddingOut + vec3(positionIn, 0) * pointRadius;

  // Calculate vertex position
  gl_Position = proj * model_view * vec4(embeddingOut, 1) 
              + proj * vec4(positionIn, 0, 1) * pointRadius;

  // Calculate output color depending on label
  const uint label = drawLabels ? labels[gl_InstanceID] % 10 : 9;
  colorOut = vec4(colors[label] / 255.0f, pointOpacity);
}