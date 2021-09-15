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

#include <iomanip>
#include <sstream>
#include <glad/glad.h>
#include "dh/util/aligned.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/program.hpp"

namespace dh::util {
  GLenum getGlShaderType(GLShaderType type) {
    switch (type) {
      case GLShaderType::eVertex:
        return GL_VERTEX_SHADER;
      case GLShaderType::eFragment:
        return GL_FRAGMENT_SHADER;
      case GLShaderType::eGeometry:
        return GL_GEOMETRY_SHADER;
      case GLShaderType::eCompute:
        return GL_COMPUTE_SHADER;
      default:
        return GL_COMPUTE_SHADER;
    }
  }

  std::string to_string(GLShaderType type) {
    switch (type) {
      case GLShaderType::eVertex:
        return "Vertex";
      case GLShaderType::eFragment:
        return "Fragment";
      case GLShaderType::eGeometry:
        return "Geometry";
      case GLShaderType::eCompute:
        return "Compute";
      default:
        return "Compute";
    }
  }
  
  GLProgram::GLProgram() {
    _handle = glCreateProgram();
    glAssert();
  }

  GLProgram::~GLProgram() {
    for (auto& handle : _shaderHandles) {
      glDeleteShader(handle);
    }
    glDeleteProgram(_handle);
  }

  GLProgram::GLProgram(GLProgram&& other) noexcept {
    swap(*this, other);
  }

  GLProgram& GLProgram::operator=(GLProgram&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void GLProgram::addShader(GLShaderType type, const std::string& src) {
    // Obtain handle to shader object for specified type
    GLuint handle = glCreateShader(getGlShaderType(type));

    // Compile shader from provided source
    const char* pSrc = src.c_str();
    const GLint srcLen = static_cast<GLint>(src.length());
    glShaderSource(handle, 1, &pSrc, &srcLen);
    glCompileShader(handle);

    // Check compilation success
    GLint success = 0;
    glGetShaderiv(handle, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
      // Compilation failed, obtain error log
      GLint length = 0;
      glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &length);
      std::string info(length, ' ');
      glGetShaderInfoLog(handle, GLint(info.size()), nullptr, info.data());

      // Construct error message with attached error log
      RuntimeError error("Shader compilation failed");
      std::stringstream ss;
      std::stringstream infss(info);
      for (std::string line; std::getline(infss, line);) {
        if (!line.empty() && line.find_first_not_of(' ') != std::string::npos) {
          ss << '\n' << std::string(16, ' ') << line;
        }
      }
      error.log = ss.str();

      throw error;
    }

    // Store handle for linking
    _shaderHandles.push_back(handle);
  }

  void GLProgram::link() {
    // Attach shaders
    for (auto& handle : _shaderHandles) {
      glAttachShader(_handle, handle);
    }

    // Perform linking
    glLinkProgram(_handle);

    // Check linking success
    GLint success = 0;
    glGetProgramiv(_handle, GL_LINK_STATUS, &success);
    if (success == GL_FALSE) {
      // Linking failed, obtain error log
      GLint length = 0;
      glGetProgramiv(_handle, GL_INFO_LOG_LENGTH, &length);
      std::string info(length, ' ');
      glGetProgramInfoLog(_handle, GLint(info.size()), nullptr, info.data());

      // Construct error message with attached error log
      RuntimeError error("Program linking failed");
      std::stringstream ss;
      std::stringstream infss(info);
      for (std::string line; std::getline(infss, line);) {
        if (!line.empty() && line.find_first_not_of(' ') != std::string::npos) {
          ss << '\n' << std::string(16, ' ') << line;
        }
      }
      error.log = ss.str();

      throw error;
    }

    // Detach and delete shaders
    for (auto& handle : _shaderHandles) {
      glDetachShader(_handle,  handle);
      glDeleteShader(handle);
    }
    
    _shaderHandles.clear();
    
    glAssert();
  }

  GLint GLProgram::implUniformLocation(const std::string& s) {
    auto f = _impUniformLocationCache.find(s);

    // Location was not previously cached
    if (f == _impUniformLocationCache.end()) {
      GLint i = glGetUniformLocation(_handle, s.c_str());
      runtimeAssert(i != -1, "Uniform location " + s + " does not exist");
      _impUniformLocationCache[s] = i;
      return i;
    }

    return f->second;
  }

  void GLProgram::bind() {
    glUseProgram(_handle);
    glAssert();
  }

  // Template specializations for base types
  template <> void GLProgram::implUniform<bool, 1, 1>(GLint location, const bool* t) { glProgramUniform1ui(_handle, location, *t); }
  template <> void GLProgram::implUniform<uint, 1, 1>(GLint location, const uint* t) { glProgramUniform1ui(_handle, location, *t); }
  template <> void GLProgram::implUniform<int, 1, 1>(GLint location, const int* t) { glProgramUniform1i(_handle, location, *t); }
  template <> void GLProgram::implUniform<float, 1, 1>(GLint location, const float* t) { glProgramUniform1f(_handle, location, *t); }

  // Template specializations for 2-component types
  template <> void GLProgram::implUniform<bool, 2, 1>(GLint location, const bool* t) { glProgramUniform2ui(_handle, location, t[0], t[1]); }
  template <> void GLProgram::implUniform<uint, 2, 1>(GLint location, const uint* t) { glProgramUniform2ui(_handle, location, t[0], t[1]); }
  template <> void GLProgram::implUniform<int, 2, 1>(GLint location, const int* t) { glProgramUniform2i(_handle, location, t[0], t[1]); }
  template <> void GLProgram::implUniform<float, 2, 1>(GLint location, const float* t) { glProgramUniform2f(_handle, location, t[0], t[1]); }

  // Template specializations for 3-component types
  template <> void GLProgram::implUniform<bool, 3, 1>(GLint location, const bool* t) { glProgramUniform3ui(_handle, location, t[0], t[1], t[2]); }
  template <> void GLProgram::implUniform<uint, 3, 1>(GLint location, const uint* t) { glProgramUniform3ui(_handle, location, t[0], t[1], t[2]); }
  template <> void GLProgram::implUniform<int, 3, 1>(GLint location, const int* t) { glProgramUniform3i(_handle, location, t[0], t[1], t[2]); }
  template <> void GLProgram::implUniform<float, 3, 1>(GLint location, const float* t) { glProgramUniform3f(_handle, location, t[0], t[1], t[2]); }

  // Template specializations for 4-component types
  template <> void GLProgram::implUniform<bool, 4, 1>(GLint location, const bool* t) { glProgramUniform4ui(_handle, location, t[0], t[1], t[2], t[3]); }
  template <> void GLProgram::implUniform<uint, 4, 1>(GLint location, const uint* t) { glProgramUniform4ui(_handle, location, t[0], t[1], t[2], t[3]); }
  template <> void GLProgram::implUniform<int, 4, 1>(GLint location, const int* t) { glProgramUniform4i(_handle, location, t[0], t[1], t[2], t[3]); }
  template <> void GLProgram::implUniform<float, 4, 1>(GLint location, const float* t) { glProgramUniform4f(_handle, location, t[0], t[1], t[2], t[3]); }
  
  // Matrix specializations... well. We only use mat4x4 and rarely at that, so let's forget about these for now.
  template <> void GLProgram::implUniform<float, 4, 4>(GLint location, const float* t) { glProgramUniformMatrix4fv(_handle, location, 1, GL_FALSE, t); }
}