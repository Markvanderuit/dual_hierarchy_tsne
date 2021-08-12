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

layout(location = 0) out uvec4 outColor;

// Uniform locations
layout(location = 0) uniform usampler1D cellMap;
layout(location = 1) uniform uint zDims;

void main() {
  // Z value to look up cell values at
  const float zFixed = gl_FragCoord.z * zDims - 0.5f;
  
  // Fetch the two nearest cell values which cover this range
  const uvec4 lowerCell = texelFetch(cellMap, int(floor(zFixed)), 0);
  const uvec4 greaterCell = texelFetch(cellMap, int(ceil(zFixed)), 0);

  // Output logical or of these cells, flagging both their pixels
  outColor = lowerCell | greaterCell;
}