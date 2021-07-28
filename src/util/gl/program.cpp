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
#include <exception>
#include <sstream>
#include <glm/glm.hpp>
#include "util/gl/program.hpp"

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
        return "eVertex";
      case GLShaderType::eFragment:
        return "eFragment";
      case GLShaderType::eGeometry:
        return "eGeometry";
      case GLShaderType::eCompute:
        return "eCompute";
      default:
        return "eCompute";
    }
  }
  
  GLProgram::GLProgram() {
    _handle = glCreateProgram();
  }

  GLProgram::~GLProgram() {
    glDeleteProgram(_handle);
    for (auto& handle : _shaderHandles) {
      glDeleteShader(handle);
    }
  }

  GLProgram::GLProgram(GLProgram&& other) noexcept {
    swap(other);
  }

  GLProgram& GLProgram::operator=(GLProgram&& other) noexcept
  {
    swap(other);
    return *this;
  }

  void GLProgram::swap(GLProgram& other) noexcept {
    std::swap(_handle, other._handle);
    std::swap(_shaderHandles, other._shaderHandles);
    std::swap(_locations, other._locations);
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
    glGetShaderiv(_handle, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
      // Compilation failed, obtain error log
      GLint length = 0;
      glGetShaderiv(_handle, GL_INFO_LOG_LENGTH, &length);
      std::string info(length, ' ');
      glGetShaderInfoLog(_handle, GLint(info.size()), nullptr, info.data());

      // Construct detailed exception message with attached error log
      std::stringstream ss;
      ss << "\n  " << std::left << std::setw(12) << "Src"
         << "Shader compilation";
      ss << "\n  " << std::left << std::setw(12) << "Shader" << to_string(type) << "";
      ss << "\n  " << std::left << std::setw(12) << "Log";

      // Format error log
      {
        std::stringstream infss(info);
        for (std::string line; std::getline(infss, line);) {
          if (!line.empty() && line.find_first_not_of(' ') != std::string::npos) {
            ss << "\n    " << line;
          }
        }
      }

      throw std::runtime_error(ss.str());
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

      // Construct detailed exception message with attached error log
      std::stringstream ss;
      ss << "  " << std::left << std::setw(24) << "Exception:"
         << "Program linking failed\n";
      ss << "  " << std::left << std::setw(24) << "Linking log:";

      // Format error log
      {
        std::stringstream infss(info);
        for (std::string line; std::getline(infss, line);) {
          if (!line.empty() && line.find_first_not_of(' ') != std::string::npos) {
            ss << "\n    " << line;
          }
        }
      }

      throw std::runtime_error(ss.str());
    }

    // Detach and delete shaders
    for (auto& handle : _shaderHandles) {
      glDetachShader(_handle,  handle);
      glDeleteShader(handle);
    }
    _shaderHandles.clear();
  }

  int GLProgram::location(const std::string& s) {
    int i;
    auto f = _locations.find(s);
    if (f == _locations.end()) {
      i = glGetUniformLocation(_handle, s.data());

      // Assert that uniform does exist
      if (i == -1) {
        // Construct detailed exception message
        std::stringstream ss;
        ss << "  " << std::left << std::setw(24) << "Exception:"
           << "Uniform location not found\n";
        ss << "  " << std::left << std::setw(24) << "Uniform:" << s;
        throw std::runtime_error(ss.str());
      }

      _locations[s] = i;
    } else {
      i = (*f).second;
    }

    return i;
  }

  template <>
  void GLProgram::uniform<bool>(const std::string& s, bool v) {
    glProgramUniform1ui(_handle, location(s), v ? 1 : 0);
  }

  template <>
  void GLProgram::uniform<unsigned>(const std::string& s, unsigned v) {
    glProgramUniform1ui(_handle, location(s), v);
  }

  template <>
  void GLProgram::uniform<int>(const std::string& s, int v) {
    glProgramUniform1i(_handle, location(s), v);
  }

  template <>
  void GLProgram::uniform<float>(const std::string& s, float v) {
    glProgramUniform1f(_handle, location(s), v);
  }

  template <>
  void GLProgram::uniform<glm::uvec2>(const std::string& s, glm::uvec2 v) {
    glProgramUniform2ui(_handle, location(s), v.x, v.y);
  }

  template <>
  void GLProgram::uniform<glm::ivec2>(const std::string& s, glm::ivec2 v) {
    glProgramUniform2i(_handle, location(s), v.x, v.y);
  }

  template <>
  void GLProgram::uniform<glm::vec2>(const std::string& s, glm::vec2 v) {
    glProgramUniform2f(_handle, location(s), v.x, v.y);
  }

  template <>
  void GLProgram::uniform<glm::uvec3>(const std::string& s, glm::uvec3 v) {
    glProgramUniform3ui(_handle, location(s), v.x, v.y, v.z);
  }

  template <>
  void GLProgram::uniform<glm::ivec3>(const std::string& s, glm::ivec3 v) {
    glProgramUniform3i(_handle, location(s), v.x, v.y, v.z);
  }

  template <>
  void GLProgram::uniform<glm::vec3>(const std::string& s, glm::vec3 v) {
    glProgramUniform3f(_handle, location(s), v.x, v.y, v.z);
  }

  template <>
  void GLProgram::uniform<glm::uvec4>(const std::string& s, glm::uvec4 v) {
    glProgramUniform4ui(_handle, location(s), v.x, v.y, v.z, v.w);
  }

  template <>
  void GLProgram::uniform<glm::ivec4>(const std::string& s, glm::ivec4 v) {
    glProgramUniform4i(_handle, location(s), v.x, v.y, v.z, v.w);
  }

  template <>
  void GLProgram::uniform<glm::vec4>(const std::string& s, glm::vec4 v) {
    glProgramUniform4f(_handle, location(s), v.x, v.y, v.z, v.w);
  }

  template <>
  void GLProgram::uniform<glm::mat4>(const std::string& s, glm::mat4 v) {
    glProgramUniformMatrix4fv(_handle, location(s), 1, GL_FALSE, &v[0][0]);
  }
}