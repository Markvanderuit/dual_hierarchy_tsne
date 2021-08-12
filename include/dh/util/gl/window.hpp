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
#include "dh/aligned.hpp"

namespace dh::util {
  // Possible OpenGL profile requests
  enum class GLProfileType {
    eAny,          // Context does not request a specific profile
    eCore,         // Context requests a core OpenGL profile
    eCompatibility // Context requests a compatibility OpenGL profile
  };

  /**
   * Wrapper structure with necessary information for context creation.
   * Default version creates the highest available context.
   */
  struct GLContextInfo {
    GLProfileType profile = GLProfileType::eAny;
    unsigned versionMajor = 1;
    unsigned versionMinor = 0;
  };

  /**
   * Wrapper structure with necessary information for window creation.
   * Default version creates a simple 256x256px. window
   */
  struct GLWindowInfo {
    enum FlagBits {
      bOffscreen = 1,  // Window is invisible (for offscreen rendering, or to hide it initially)
      bDecorated = 2,  // Window is decorated with title bar
      bFloating = 4,   // Window floats above all other windows
      bFullscreen = 8, // Window becomes borderless full screen
      bMaximized = 16, // Window is maximized on creation
      bFocused = 32,   // Window is focused for input on creation
      bResizable = 64, // Window is resizable
      bSRGB = 128,     // Framebuffer is sRGB capable
    };

    // Default window settings
    unsigned flags = bDecorated | bFocused;
    unsigned width = 256;
    unsigned height = 256;
    std::string title = "";
  };

  /**
   * OpenGL/GLFW window wrapper, manages context, default framebuffer, 
   * swap chain, input handling and so on.
   */
  class GLWindow {
  public:
    GLWindow();
    GLWindow(GLWindowInfo windowInfo, GLContextInfo contextInfo = GLContextInfo());
    ~GLWindow();

    // Copy constr/assignment is explicitly deleted (no copying OpenGL objects)
    GLWindow(const GLWindow&) = delete;
    GLWindow& operator=(const GLWindow&) = delete;

    // Move constr/operator moves resource handle
    GLWindow(GLWindow&&) noexcept;
    GLWindow& operator=(GLWindow&&) noexcept;

    // Swap internals with another window object
    friend void swap(GLWindow& a, GLWindow& b) noexcept;

    // Render loop
    void makeCurrent();
    void processEvents();
    void display();

    // Window properties
    void setVisible(bool enabled);
    void setVsync(bool enabled);
    void setTitle(const std::string &title);

    // Window handling
    bool isCurrent() const;
    bool canDisplay() const;
    bool isVisible() const;
    void *handle() const;
    glm::ivec2 size() const;

    // Static access
    static GLWindow *currentWindow();

    // Template window constructors
    static GLWindow Offscreen();
    static GLWindow Decorated();
    static GLWindow DecoratedResizable();
    static GLWindow Undecorated();
    static GLWindow UndecoratedResizable();
    static GLWindow Fullscreen();

  private:
    bool _isInit;
    void *_handle;
    glm::ivec2 _size;

    static unsigned _nrHandles;
    static GLWindow* _currentWindow;
  };
} // dh