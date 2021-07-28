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

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace dh {
  // Alignment specifier to match glm::vec1/2/3/4 to 4/8/16/16 bytes for usage in OpenGL buffers
  namespace detail {
    constexpr unsigned std430_align(unsigned D) {
      return D == 4 ? 16
            : D == 3 ? 16
            : D == 2 ? 8
            : 4;
    }
  }

  // Simple aligned glm::vec<D,T>
  template <unsigned D, typename genType>
  struct alignas(detail::std430_align(D)) AlignedVec : public glm::vec<D, genType> {
    using glm::vec<D, genType>::vec;

    AlignedVec<D, genType>(glm::vec<D, genType> other) : glm::vec<D, genType>(other) { }
  };  

  // Simple aligned bounding box class
  template <unsigned D>
  class AlignedBounds {
  private:
    using vec = AlignedVec<D, float>;

  public:
    vec min, max;

    vec range() const {
      return max - min;
    }

    vec center() const {
      return 0.5f * (max + min);
    }
  };
  
  template <unsigned D, typename genType>
  genType max(AlignedVec<D, genType> x) {
    genType t = x[0];
    for (uint i = 1; i < D; i++) {
      t = glm::max(t, x[i]);
    }
    return t;
  }

  template <unsigned D, typename genType>
  genType min(AlignedVec<D, genType> x) {
    genType t = x[0];
    for (uint i = 1; i < D; i++) {
      t = glm::min(t, x[i]);
    }
    return t;
  }

  template <unsigned D, typename genType>
  AlignedVec<D, genType> max(AlignedVec<D, genType> x, AlignedVec<D, genType> y) {
    return glm::max(static_cast<glm::vec<D, genType>>(x), static_cast<glm::vec<D, genType>>(y));
  }

  template <unsigned D, typename genType>
  AlignedVec<D, genType> min(AlignedVec<D, genType> x, AlignedVec<D, genType> y) {
    return glm::min(static_cast<glm::vec<D, genType>>(x), static_cast<glm::vec<D, genType>>(y));
  }

  template <unsigned D, typename genType>
  std::string to_string(AlignedVec<D, genType> x) {
    return glm::to_string(static_cast<glm::vec<D, genType>>(x));
  }

  template <unsigned D, typename genType>
  genType product(AlignedVec<D, genType> x) {
    genType t = x[0];
    for (uint i = 1; i < D; i++) {
      t *= x[i];
    }
    return t;
  }

  template <unsigned D, typename genType>
  genType dot(AlignedVec<D, genType> x, AlignedVec<D, genType> y) {
    return glm::dot(static_cast<glm::vec<D, genType>>(x), static_cast<glm::vec<D, genType>>(y));
  }
} // dh