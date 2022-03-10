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
#include <algorithm>
#include <iostream>
#include <glad/glad.h>
#include "dh/constants.hpp"
#include "dh/util/error.hpp"

namespace dh::util {
  inline
  std::string glReadableError(GLenum err) {
    switch (err) {
    case GL_INVALID_ENUM:
      return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:
      return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:
      return "GL_INVALID_OPERATION";
    case GL_STACK_OVERFLOW:
      return "GL_STACK_OVERFLOW";
    case GL_STACK_UNDERFLOW:
      return "GL_STACK_UNDERFLOW";
    case GL_OUT_OF_MEMORY:
      return "GL_OUT_OF_MEMORY";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_CONTEXT_LOST:
      return "GL_CONTEXT_LOST";
    case GL_NO_ERROR:
      return "GL_NO_ERROR";
    default:
      return "unknown";
    }
  }

  inline
  std::string glReadableDebugSrc(GLenum src) {
    switch (src) {
    case GL_DEBUG_SOURCE_API:
      return "GL_DEBUG_SOURCE_API";
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
      return "GL_DEBUG_SOURCE_WINDOW_SYSTEM";
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
      return "GL_DEBUG_SOURCE_SHADER_COMPILER";
    case GL_DEBUG_SOURCE_THIRD_PARTY:
      return "GL_DEBUG_SOURCE_THIRD_PARTY";
    case GL_DEBUG_SOURCE_APPLICATION:
      return "GL_DEBUG_SOURCE_APPLICATION"; 
    case GL_DEBUG_SOURCE_OTHER:
      return "GL_DEBUG_SOURCE_OTHER";      
    default:
      return "unknown";
    }
  }

  inline
  std::string glReadableDebugType(GLenum type) {
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
      return "GL_DEBUG_TYPE_ERROR";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
      return "GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
      return "GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR";
    case GL_DEBUG_TYPE_PORTABILITY:
      return "GL_DEBUG_TYPE_PORTABILITY";
    case GL_DEBUG_TYPE_PERFORMANCE:
      return "GL_DEBUG_TYPE_PERFORMANCE";
    case GL_DEBUG_TYPE_MARKER:
      return "GL_DEBUG_TYPE_MARKER";
    case GL_DEBUG_TYPE_PUSH_GROUP:
      return "GL_DEBUG_TYPE_PUSH_GROUP";
    case GL_DEBUG_TYPE_POP_GROUP:
      return "GL_DEBUG_TYPE_POP_GROUP";
    case GL_DEBUG_TYPE_OTHER:
      return "GL_DEBUG_TYPE_OTHER";
    default:
      return "unknown";
    }
  }

  inline
  std::string glReadableDebugSeverity(GLenum severity) {
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH: 
      return "GL_DEBUG_SEVERITY_HIGH";
    case GL_DEBUG_SEVERITY_MEDIUM: 
      return "GL_DEBUG_SEVERITY_MEDIUM";
    case GL_DEBUG_SEVERITY_LOW: 
      return "GL_DEBUG_SEVERITY_LOW";
    case GL_DEBUG_SEVERITY_NOTIFICATION: 
      return "GL_DEBUG_SEVERITY_NOTIFICATION";
    default:
      return "unknown";
    }
  }

  namespace detail {
    inline
    void glAssertImpl(const char *file, int line) {
      GLenum err;
      while ((err = glGetError()) != GL_NO_ERROR) {
        RuntimeError error("OpenGL assertion failed");
        error.logs["code"] = glReadableError(err);
        error.logs["file"] = std::string(file);
        error.logs["line"] = std::to_string(line);
        throw error;
      }
    }

    inline
    void APIENTRY glDebugImpl(GLenum src, GLenum type, uint err, GLenum severity, GLsizei length,
                              const char *msg, const void *userParam) {         
        // Filter out insignificant codes
        std::array<uint, 4> ignored = { 131169, 131185, 131218, 131204 };
        if (std::find(ignored.begin(), ignored.end(), err) != ignored.end()) {
          return;
        }

        RuntimeError error("OpenGL debug callback");
        error.logs["code"] = std::to_string(err);
        error.logs["type"] = glReadableDebugType(type);
        error.logs["svr"] = glReadableDebugSeverity(severity);
        error.logs["src"] = glReadableDebugSrc(src);
        error.logs["msg"] = std::string(msg);
        throw error;
    }
  }

  inline
  void glInitDebug() {
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(detail::glDebugImpl, nullptr);

    // Enable all SEVERITY_HIGH messages
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
    // glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
    // glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
    // glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
    // glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
    // glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_MARKER, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
    // glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PUSH_GROUP, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
    // glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_POP_GROUP, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
    // glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_OTHER, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);

    // Enable select SEVERITY_MEDIUM messages
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_MEDIUM, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DEBUG_SEVERITY_MEDIUM, 0, nullptr, GL_TRUE);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, GL_DEBUG_SEVERITY_MEDIUM, 0, nullptr, GL_TRUE);

    // Disable others
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
  }
} // dh::util

// Simple OpenGL assert with err code, file name and line nr. attached. Throws RuntimeError
#ifdef DH_ENABLE_ASSERT
#define glAssert() dh::util::detail::glAssertImpl(__FILE__, __LINE__);
#else
#define glAssert()
#endif