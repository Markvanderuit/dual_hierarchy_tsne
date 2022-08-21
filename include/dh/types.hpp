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

namespace dh {
  using GLuint = unsigned int; // Matches GLAD, use to prevent unnecessary glad includes but retain notation for OpenGL handles etc.
  using GLint = int;           // Matches GLAD, use to prevent unnecessary glad includes but retain notation for OpenGL handles etc.
  using uint = unsigned int;   // Matches GLSL, use to retain notation for unsigned integers outside shader code
  
  // Rounded up division of some n by div
  template <typename genType> 
  inline
  genType ceilDiv(genType n, genType div) {
    return (n + div - 1) / div;
  }

  // Simple range-like syntactic sugar
  #define range_iter(c) c.begin(), c.end()

  // For enum class T, declare bitflag operators and has_flag(T, T) boolean operator
  #define dh_declare_bitflag(T)                                                \
    constexpr T operator~(T a) { return (T) (~ (uint) a); }                    \
    constexpr T operator|(T a, T b) { return (T) ((uint) a | (uint) b); }      \
    constexpr T operator&(T a, T b) { return (T) ((uint) a & (uint) b); }      \
    constexpr T operator^(T a, T b) { return (T) ((uint) a ^ (uint) b); }      \
    constexpr T& operator|=(T &a, T b) { return a = a | b; }                   \
    constexpr T& operator&=(T &a, T b) { return a = a & b; }                   \
    constexpr T& operator^=(T &a, T b) { return a = a ^ b; }                   \
    constexpr bool has_flag(T flags, T t) { return (uint) (flags & t) != 0u; }

  // For class T, declare swap-based move constr/operator
  // and delete copy constr/operators, making T non-copyable
  #define dh_declare_noncopyable(T)                                             \
    T(const T &) = delete;                                                      \
    T & operator= (const T &) = delete;                                         \
    T(T &&o) noexcept { swap(*this, o); }                                 \
    inline T & operator= (T &&o) noexcept { swap(*this, o); return *this; }
} // dh