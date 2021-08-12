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

#include <array>
#include <type_traits>

namespace dh::util {
  namespace detail {
    // Cast enum value to underlying type of enum class (eg. int)
    template <typename ETy>
    constexpr inline
    typename std::underlying_type<ETy>::type underlying(ETy e) noexcept {
      return static_cast<typename std::underlying_type<ETy>::type>(e);
    }
  } // detail
    
  // Array class using Enum as indices
  // For a used enum E, E::Length must be a member of the enum
  template <typename ETy, typename Ty>
  class EnumArray : public std::array<Ty, detail::underlying(ETy::Length)> {
  public:
    constexpr inline
    const Ty& operator()(ETy e) const {
      return this->operator[](detail::underlying<ETy>(e));
    }

    constexpr inline
    Ty& operator()(ETy e) {
      return this->operator[](detail::underlying<ETy>(e));
    }
    
    friend void swap(EnumArray<ETy, Ty>& a, EnumArray<ETy, Ty>& b) noexcept {
      a.swap(b);
    }
  };
} // dh::util