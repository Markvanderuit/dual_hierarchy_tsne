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

#include <algorithm>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>
#include <resource_embed/resource_embed.hpp>
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/metric.hpp"
#include "dh/sne/components/field.hpp"

namespace dh::sne {
  // Constants
  constexpr uint hierarchyRebuildPadding = 50;
  constexpr uint hierarchyRebuildIterations = 4;
  constexpr int maxDualHierarchyLvlDiff = 4;
  
  template <uint D>
  Field<D>::Field()
  : _isInit(false), _logger(nullptr) {
    // ...
  }
  
  template <uint D>
  Field<D>::~Field() {
    // ...
  }

  template <uint D>
  Field<D>::Field(Field<D>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  Field<D>& Field<D>::operator=(Field<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void Field<D>::comp(uvec size, uint iteration) {
    // Resize field if necessary
    resize(size);

    // Build embedding hierarchy if necessary
    if (_useEmbeddingHierarchy) {
      _embeddingHierarchy.comp(_hierarchyRebuildIterations == 0);
    }

    // Generate work queue with pixels in the field texture requiring computation
    if (_useEmbeddingHierarchy) {
      compSingleHierarchyCompact();
    } else {
      compFullCompact();
    }

    // Copy nr of flagged pixels from device to host-side buffer for cheaper readback later
    // TODO is this necessary
    glCopyNamedBufferSubData(_buffers(BufferType::ePixelQueueHead), 
                             _buffers(BufferType::ePixelQueueHeadReadback), 
                             0, 0, sizeof(uint));

    // Clear field texture
    // TODO is this necessary?
    glClearTexImage(_textures(TextureType::eField), 0, GL_RGBA, GL_FLOAT, nullptr);

    // Read back flagged pixels from host-side buffer
    // TODO is this necessary
    uint nPixels;
    glGetNamedBufferSubData(_buffers(BufferType::ePixelQueueHeadReadback), 0, sizeof(uint), &nPixels);

    // Determine if field hierarchy should be actively used this iteration
    bool fieldHierarchyActive = false;
    const FieldHierarchy<D>::Layout fLayout(nPixels, _size);
    if (_useFieldHierarchy) {
      const int dualHierarchyLvlDiff = static_cast<int>(_embeddingHierarchy.layout().nLvls) - static_cast<int>(fLayout.nLvls);
      if (iteration >= _params.removeExaggerationIter && dualHierarchyLvlDiff < maxDualHierarchyLvlDiff) {
        fieldHierarchyActive = true;
      }
    }

    // Build field hierarchy if necessary
    if (fieldHierarchyActive) {
      _fieldHierarchy.comp(true, fLayout);
    }
    
    // Perform field computation using one of the available techniques
    if (fieldHierarchyActive) {
      compDualHierarchyField();
    } else if (_useEmbeddingHierarchy) {
      compSingleHierarchyField();
    } else {
      compFullField();
    }

    // Update field buffer in Minimization by querying the field texture
    queryField();

    // Update hierarchy rebuild countdown
    if (_useEmbeddingHierarchy) {
      if (iteration <= (_params.removeExaggerationIter + hierarchyRebuildPadding)
      || _hierarchyRebuildIterations >= hierarchyRebuildIterations) {
        _hierarchyRebuildIterations = 0;
      } else {
        _hierarchyRebuildIterations++;
      }
    }
  }

  template <uint D>
  void Field<D>::queryField() {
    auto& timer = _timers(TimerType::eQueryFieldComp);
    timer.tick();

    auto& program = _programs(ProgramType::eQueryFieldComp);
    program.bind();

    // Set uniforms, bind texture unit
    program.uniform<uint>("nPoints", _params.n);
    program.uniform<int>("fieldSampler", 0);
    glBindTextureUnit(0, _textures(TextureType::eField));

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _minimization.embedding);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _minimization.field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _minimization.bounds);

    // Dispatch compute shader
    glDispatchCompute(ceilDiv(_params.n, 128u), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    timer.tock();
    glAssert();
  }

  // Template instantiations for 2/3 dimensions
  template Field<2>::Field();
  template Field<2>::Field(Field<2>&& other) noexcept;
  template Field<2>::~Field();
  template Field<2>& Field<2>::operator=(Field<2>&& other) noexcept;
  template void Field<2>::comp(util::AlignedVec<2, uint> size, uint iteration);
  template void Field<2>::queryField();
  template Field<3>::Field();
  template Field<3>::Field(Field<3>&& other) noexcept;
  template Field<3>::~Field();
  template Field<3>& Field<3>::operator=(Field<3>&& other) noexcept;
  template void Field<3>::comp(util::AlignedVec<3, uint> size, uint iteration);
  template void Field<3>::queryField();
} // dh::sne