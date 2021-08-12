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

#include <string>
#include <unordered_map>
#include <vector>
#include "types.hpp"

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
    // void swap(GLProgram& other) noexcept;
    friend void swap(GLProgram& a, GLProgram& b) noexcept;

    // Compile shader from source, then store for later linking
    void addShader(GLShaderType type, const std::string& src);

    // Once all shaders are compiled, link the program object
    void link();

    // Bind program object for execution or rendering state
    void bind();

    // Set uniform at location s to value t.
    // Backed by implementations for float/int/uint and glm::vec/mat types
    template <typename T>
    void uniform(const std::string& s, T t);
    
  private:
    // State
    GLuint _handle;
    std::vector<GLuint> _shaderHandles;
    std::unordered_map<std::string, int> _locations;
  
    // Cache uniform locations in an unordered map for lookup
    int location(const std::string& s);

  public:
    // std::swap impl
    friend void swap(GLProgram& a, GLProgram& b) noexcept {
      using std::swap;
      swap(a._handle, b._handle);
      swap(a._shaderHandles, b._shaderHandles);
      swap(a._locations, b._locations);
    }
  };
} // dh