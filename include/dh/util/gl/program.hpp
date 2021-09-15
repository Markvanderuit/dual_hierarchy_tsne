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

#include <any>
#include <string>
#include <unordered_map>
#include <vector>
#include "dh/types.hpp"
#include "dh/util/aligned.hpp"

namespace dh::util {
  // Used OpenGL shader types
  enum class GLShaderType {
    eVertex,
    eFragment,
    eGeometry,
    eCompute
  };

  // Simple wrapper around a OpenGL program object
  class GLProgram {
  public:
    GLProgram();
    ~GLProgram();

    // Copy constr/assignment is explicitly deleted (no copying OpenGL objects)
    GLProgram(const GLProgram&) = delete;
    GLProgram& operator=(const GLProgram&) = delete;

    // Move constr/operator moves resource handles
    GLProgram(GLProgram&&) noexcept;
    GLProgram& operator=(GLProgram&&) noexcept;

    // Swap internals with another progam object
    // friend void swap(GLProgram& a, GLProgram& b) noexcept;

    // Compile shader from source, then store for later linking
    void addShader(GLShaderType type, const std::string& src);

    // Once all shaders are compiled, link the program object
    void link();

    // Bind program object for execution or rendering state
    void bind();

    // Base type uniform function
    // Supports uint, int, float, bool
    template <typename GenType>
    void uniform(const std::string& s, GenType t) {
      if (!implUniformValue(s, t)) implUniform<GenType, 1, 1>(implUniformLocation(s), &t);
    }

    // Vector type uniform function
    // Supports GLM-based N-component vectors of type uint, int, float, bool
    template <typename GenType, int N, glm::qualifier Q,
              template <int, typename, glm::qualifier> class ImplType>
    void uniform(const std::string& s, ImplType<N, GenType, Q> t) {
      if (!implUniformValue(s, t)) implUniform<GenType, N, 1>(implUniformLocation(s), &(t.x));
    }
    
    // Matrix type uniform function
    // Supports GLM-based 4x4 matrices of type float (others are unimplemented due to lack of use)
    template <typename GenType, int M, uint N, glm::qualifier Q,
              template <int, int, typename, glm::qualifier> class ImplType>
    void uniform(const std::string& s, ImplType<M, N, GenType, Q> t) {
      if (!implUniformValue(s, t)) implUniform<GenType, M, N>(implUniformLocation(s), &(t[0].x));
    }

  private:
    // State
    GLuint _handle;
    std::vector<GLuint> _shaderHandles;
    std::unordered_map<std::string, int> _impUniformLocationCache;
    std::unordered_map<std::string, std::any> _implUniformValueCache;
  
    // Helper function to set uniform values
    template <typename GenType, int M, int N>
    void implUniform(GLint location, const GenType* t);
    
    // Helper function to cache uniform locations, preventing program queries
    GLint implUniformLocation(const std::string& s);

    // Helper function to cache uniform values, preventing unnecessary state changes
    template <typename T>
    bool implUniformValue(const std::string& s, const T& value) {
      auto f = _implUniformValueCache.find(s);

      // Value was not previously cached, or does not match cache
      if (f == _implUniformValueCache.end() || std::any_cast<T>(f->second) != value) {
        _implUniformValueCache[s] = value;
        return false;
      }

      return true;
    }

  public:
    // std::swap impl
    friend void swap(GLProgram& a, GLProgram& b) noexcept {
      using std::swap;
      swap(a._handle, b._handle);
      swap(a._shaderHandles, b._shaderHandles);
      swap(a._impUniformLocationCache, b._impUniformLocationCache);
      swap(a._implUniformValueCache, b._implUniformValueCache);
    }
  };
} // dh::util