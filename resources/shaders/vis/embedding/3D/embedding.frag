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

// Input attributes
layout(location = 0) in vec3 embeddingIn;
layout(location = 1) in vec3 fragEmbeddingIn;
layout(location = 2) in vec4 colorIn;

// Output attributes
layout(location = 0) out vec4 colorOut;

// Uniforms
layout(location = 0) uniform mat4 model_view;
layout(location = 1) uniform mat4 proj;
layout(location = 2) uniform float pointOpacity;
layout(location = 3) uniform float pointRadius;
layout(location = 4) uniform bool drawLabels;

void main() {
  // Discard fragments outside of circular point's radius
  const float t = distance(embeddingIn, fragEmbeddingIn);
  if (t > pointRadius) {
    discard;
  }

  colorOut = colorIn;
}