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
#include "aligned.hpp"
#include <glm/gtc/type_ptr.hpp>
#include "aligned.hpp"
#include "util/gl/error.hpp"
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

  int GLProgram::location(const std::string& s) {
    int i;
    auto f = _locations.find(s);
    if (f == _locations.end()) {
      i = glGetUniformLocation(_handle, s.c_str());
      glAssert();
      runtimeAssert(i != -1, "Uniform location " + s + " does not exist");
      

      _locations[s] = i;
    } else {
      i = (*f).second;
    }

    return i;
  }

  void GLProgram::bind() {
    glUseProgram(_handle);
    glAssert();
  }

  // Template specializations for base types
  template <> void GLProgram::uniform<bool>(const std::string& s, bool v) { glProgramUniform1ui(_handle, location(s), v ? 1 : 0); }
  template <> void GLProgram::uniform<uint>(const std::string& s, uint v) { glProgramUniform1ui(_handle, location(s), v); }
  template <> void GLProgram::uniform<int>(const std::string& s, int v) { glProgramUniform1i(_handle, location(s), v); }
  template <> void GLProgram::uniform<float>(const std::string& s, float v) {  glProgramUniform1f(_handle, location(s), v); }

  // Matrix specializations
  template <> void GLProgram::uniform<glm::mat4>(const std::string& s, glm::mat4 v) { glProgramUniformMatrix4fv(_handle, location(s), 1, GL_FALSE, glm::value_ptr(v)); }

  // Template specializations for glm::vec<2, *> types
  template <> void GLProgram::uniform<glm::bvec2>(const std::string& s, glm::bvec2 v) { glProgramUniform2ui(_handle, location(s), v.x, v.y); }
  template <> void GLProgram::uniform<glm::uvec2>(const std::string& s, glm::uvec2 v) { glProgramUniform2ui(_handle, location(s), v.x, v.y); }
  template <> void GLProgram::uniform<glm::ivec2>(const std::string& s, glm::ivec2 v) { glProgramUniform2i(_handle, location(s), v.x, v.y); }
  template <> void GLProgram::uniform<glm::vec2>(const std::string& s, glm::vec2 v) { glProgramUniform2f(_handle, location(s), v.x, v.y); }
  template <> void GLProgram::uniform<AlignedVec<2, bool>>(const std::string& s, AlignedVec<2, bool> v) { glProgramUniform2ui(_handle, location(s), v.x, v.y); }
  template <> void GLProgram::uniform<AlignedVec<2, uint>>(const std::string& s, AlignedVec<2, uint> v) { glProgramUniform2ui(_handle, location(s), v.x, v.y); }
  template <> void GLProgram::uniform<AlignedVec<2, int>>(const std::string& s, AlignedVec<2, int> v) { glProgramUniform2i(_handle, location(s), v.x, v.y); }
  template <> void GLProgram::uniform<AlignedVec<2, float>>(const std::string& s, AlignedVec<2, float> v) { glProgramUniform2f(_handle, location(s), v.x, v.y); }

  // Template specializations for glm::vec<3, *> types
  template <> void GLProgram::uniform<glm::bvec3>(const std::string& s, glm::bvec3 v) { glProgramUniform3ui(_handle, location(s), v.x, v.y, v.z); }
  template <> void GLProgram::uniform<glm::uvec3>(const std::string& s, glm::uvec3 v) { glProgramUniform3ui(_handle, location(s), v.x, v.y, v.z); }
  template <> void GLProgram::uniform<glm::ivec3>(const std::string& s, glm::ivec3 v) { glProgramUniform3i(_handle, location(s), v.x, v.y, v.z); }
  template <> void GLProgram::uniform<glm::vec3>(const std::string& s, glm::vec3 v) { glProgramUniform3f(_handle, location(s), v.x, v.y, v.z); }
  template <> void GLProgram::uniform<AlignedVec<3, bool>>(const std::string& s, AlignedVec<3, bool> v) { glProgramUniform3ui(_handle, location(s), v.x, v.y, v.z); }
  template <> void GLProgram::uniform<AlignedVec<3, uint>>(const std::string& s, AlignedVec<3, uint> v) { glProgramUniform3ui(_handle, location(s), v.x, v.y, v.z); }
  template <> void GLProgram::uniform<AlignedVec<3, int>>(const std::string& s, AlignedVec<3, int> v) { glProgramUniform3i(_handle, location(s), v.x, v.y, v.z); }
  template <> void GLProgram::uniform<AlignedVec<3, float>>(const std::string& s, AlignedVec<3, float> v) { glProgramUniform3f(_handle, location(s), v.x, v.y, v.z); }

  // Template specializations for glm::vec<4, *> types
  template <> void GLProgram::uniform<glm::bvec4>(const std::string& s, glm::bvec4 v) { glProgramUniform4ui(_handle, location(s), v.x, v.y, v.z, v.w); }
  template <> void GLProgram::uniform<glm::uvec4>(const std::string& s, glm::uvec4 v) { glProgramUniform4ui(_handle, location(s), v.x, v.y, v.z, v.w); }
  template <> void GLProgram::uniform<glm::ivec4>(const std::string& s, glm::ivec4 v) { glProgramUniform4i(_handle, location(s), v.x, v.y, v.z, v.w); }
  template <> void GLProgram::uniform<glm::vec4>(const std::string& s, glm::vec4 v) { glProgramUniform4f(_handle, location(s), v.x, v.y, v.z, v.w); }
  template <> void GLProgram::uniform<AlignedVec<4, bool>>(const std::string& s, AlignedVec<4, bool> v) { glProgramUniform4ui(_handle, location(s), v.x, v.y, v.z, v.w); }
  template <> void GLProgram::uniform<AlignedVec<4, uint>>(const std::string& s, AlignedVec<4, uint> v) { glProgramUniform4ui(_handle, location(s), v.x, v.y, v.z, v.w); }
  template <> void GLProgram::uniform<AlignedVec<4, int>>(const std::string& s, AlignedVec<4, int> v) { glProgramUniform4i(_handle, location(s), v.x, v.y, v.z, v.w); }
  template <> void GLProgram::uniform<AlignedVec<4, float>>(const std::string& s, AlignedVec<4, float> v) { glProgramUniform4f(_handle, location(s), v.x, v.y, v.z, v.w); }
}