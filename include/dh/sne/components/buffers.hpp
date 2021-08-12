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

#pragma once

#include "dh/types.hpp"

namespace dh::sne {
  // Data class provided by dh::sne::Minimization<D>::buffers() for other components
  struct MinimizationBuffers {
    GLuint embedding;
    GLuint field;
    GLuint bounds;
  };

  // Data class provided by dh::sne::Similarities<D>::buffers() for other components
  struct SimilaritiesBuffers {
    GLuint similarities;
    GLuint layout;
    GLuint neighbors;
  };

  // Data class provided by dh::sne::Field<D>::buffers() for other components
  struct FieldBuffers {
    GLuint pixelQueue;
    GLuint pixelQueueHead;
  };
  
  // Data class provided by dh::sne::EmbeddingHierarchy<D>::buffers() for other components
  struct EmbeddingHierarchyBuffers {
    GLuint embeddingSorted;
    GLuint node0;
    GLuint node1;
    GLuint minb;
  };

  // Data class provided by dh::sne::FieldHierarchy<D>::buffers() for others components
  struct FieldHierarchyBuffers {
    GLuint node;
    GLuint field;
  };
} // dh::sne