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
#include <utility>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include "dh/constants.hpp"
#include "dh/util/error.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/window.hpp"

namespace dh::util {
  unsigned GLWindow::_nrHandles = 0u;
  GLWindow * GLWindow::_currentWindow = nullptr;

  GLWindow::GLWindow() : _isInit(false) { }

  GLWindow::GLWindow(GLWindowInfo windowInfo, GLContextInfo contextInfo)
  : _isInit(false)  {
    // Initialize GLFW on first window creation
    runtimeAssert(_nrHandles != 0u || glfwInit(), "glfwInit() failed");

    // Set context creation hints based on bit flags and version
    unsigned prf = contextInfo.profile == GLProfileType::eCore
                     ? GLFW_OPENGL_CORE_PROFILE
                     : (contextInfo.profile == GLProfileType::eCompatibility
                          ? GLFW_OPENGL_COMPAT_PROFILE
                          : GLFW_OPENGL_ANY_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, contextInfo.versionMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, contextInfo.versionMinor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, prf);

    // Set framebuffer creation hints based on bit flags
    glfwWindowHint(GLFW_SRGB_CAPABLE, (windowInfo.flags & GLWindowInfo::bSRGB) != 0);

    // Set window creation hints based on bit flags
    glfwWindowHint(GLFW_VISIBLE, (windowInfo.flags & GLWindowInfo::bOffscreen) == 0);
    glfwWindowHint(GLFW_DECORATED, (windowInfo.flags & GLWindowInfo::bDecorated) != 0);
    glfwWindowHint(GLFW_FLOATING, (windowInfo.flags & GLWindowInfo::bFloating) != 0);
    glfwWindowHint(GLFW_MAXIMIZED, (windowInfo.flags & GLWindowInfo::bMaximized) != 0);
    glfwWindowHint(GLFW_FOCUSED, (windowInfo.flags & GLWindowInfo::bFocused) != 0);
    glfwWindowHint(GLFW_RESIZABLE, (windowInfo.flags & GLWindowInfo::bResizable) != 0);
    
    // Set debug context creation hint
#ifdef DH_ENABLE_DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
#endif

    if ((windowInfo.flags & GLWindowInfo::bFullscreen) != 0) {
      // Create fullscreen window
      GLFWmonitor* pMonitor = glfwGetPrimaryMonitor();
      const GLFWvidmode* pMode = glfwGetVideoMode(pMonitor);
      glfwWindowHint(GLFW_RED_BITS, pMode->redBits);
      glfwWindowHint(GLFW_GREEN_BITS, pMode->greenBits);
      glfwWindowHint(GLFW_BLUE_BITS, pMode->blueBits);
      glfwWindowHint(GLFW_REFRESH_RATE, pMode->refreshRate);
      _handle = (void*) glfwCreateWindow(
        pMode->width, pMode->height, windowInfo.title.c_str(), pMonitor, nullptr);
    } else {
      // Create normal window
      _handle = (void*) glfwCreateWindow(
        windowInfo.width, windowInfo.height, windowInfo.title.c_str(), nullptr, nullptr);
    }

    // Check if window creation was successful
    runtimeAssert(_handle, "glfwCreateWindow() failed");

    // Initialize GLAD
    // Requires new context to become current
    if (_nrHandles == 0u) {
      makeCurrent();
      runtimeAssert(gladLoadGL(), "gladLoadGL() failed");
    }

#ifdef DH_ENABLE_DEBUG
    util::glInitDebug();
#endif

    _nrHandles++;
    _isInit = true;
  }

  GLWindow::~GLWindow() {
    if (_isInit) {
      // Check if handle should be set to nullptr
      // before destroying
      if (isCurrent()) {
        _currentWindow = nullptr;
      }

      glfwDestroyWindow((GLFWwindow*) _handle);

      // Decrement static window counter
      // and terminate GLFW on last window termination
      _nrHandles--;
      if (_nrHandles == 0u) {
        glfwTerminate();
      }
    }
  }

  GLWindow::GLWindow(GLWindow&& other) noexcept
  {
    swap(*this, other);
  }

  GLWindow& GLWindow::operator=(GLWindow&& other) noexcept
  {
    swap(*this, other);
    return *this;
  }

  void swap(GLWindow& a, GLWindow& b) noexcept
  {
    using std::swap;
    swap(a._isInit, b._isInit);
    swap(a._handle, b._handle);
    swap(a._size, b._size);
    if (a._currentWindow == &b) {
      a._currentWindow = &a;
    }
  }

  void GLWindow::makeCurrent()
  {
    glfwMakeContextCurrent((GLFWwindow*) _handle);
    _currentWindow = this; // set to wrapper, not to window
  }

  void GLWindow::setTitle(const std::string &title) {
    glfwSetWindowTitle((GLFWwindow *) _handle, title.c_str());
  }

  bool GLWindow::isCurrent() const
  {
    return ((GLFWwindow*) _handle) == glfwGetCurrentContext();
  }

  bool GLWindow::canDisplay() const
  {
    return !glfwWindowShouldClose((GLFWwindow*) _handle);
  }

  bool GLWindow::isVisible() const {
    return glfwGetWindowAttrib((GLFWwindow*) _handle, GLFW_VISIBLE) == GLFW_TRUE;
  }

  void GLWindow::display()
  {
    glfwSwapBuffers((GLFWwindow*) _handle);
    glfwGetFramebufferSize((GLFWwindow*) _handle, &_size[0], &_size[1]);
  }

  void GLWindow::processEvents()
  {
    glfwPollEvents();
  }

  glm::ivec2 GLWindow::size() const
  {
    return _size;
  }

  void* GLWindow::handle() const 
  {
    return _handle;
  }

  void GLWindow::setVisible(bool enabled) {
    if (!isCurrent()) {
      makeCurrent();
    }
    if (enabled) {
      glfwShowWindow((GLFWwindow *) _handle);
    } else {
      glfwHideWindow((GLFWwindow *) _handle);
    }
  }

  void GLWindow::setVsync(bool enabled) {
    if (!isCurrent()) {
      makeCurrent();
    }
    glfwSwapInterval(enabled ? 1 : 0);
  }

  bool GLWindow::hasContext()
  {
    return glfwGetCurrentContext() != nullptr;
  }

  GLWindow* GLWindow::currentWindow() 
  {
    return _currentWindow;
  }

  GLWindow GLWindow::Offscreen() 
  {
    GLWindowInfo info;
    info.flags = GLWindowInfo::bOffscreen;
    return GLWindow(info);
  }

  GLWindow GLWindow::Decorated() 
  {
    GLWindowInfo info;
    info.flags = GLWindowInfo::bDecorated | GLWindowInfo::bFocused | GLWindowInfo::bSRGB;
    info.width = 1024;
    info.height = 768;
    return GLWindow(info);
  }

  GLWindow GLWindow::DecoratedResizable() 
  {
    GLWindowInfo info;
    info.flags = GLWindowInfo::bDecorated | GLWindowInfo::bFocused | GLWindowInfo::bSRGB | GLWindowInfo::bResizable;
    info.width = 1024;
    info.height = 768;
    return GLWindow(info);
  }

  GLWindow GLWindow::Undecorated() 
  {
    GLWindowInfo info;
    info.flags = GLWindowInfo::bFocused | GLWindowInfo::bSRGB;
    info.width = 1024;
    info.height = 768;
    return GLWindow(info);
  }

  GLWindow GLWindow::UndecoratedResizable() 
  {
    GLWindowInfo info;
    info.flags = GLWindowInfo::bFocused | GLWindowInfo::bSRGB | GLWindowInfo::bResizable;
    info.width = 1024;
    info.height = 768;
    return GLWindow(info);
  }

  GLWindow GLWindow::Fullscreen() 
  {
    GLWindowInfo info;
    info.flags = GLWindowInfo::bFullscreen | GLWindowInfo::bFocused;
    return GLWindow(info);
  }
} // dh