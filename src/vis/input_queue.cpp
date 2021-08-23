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

#include <imgui.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "dh/util/error.hpp"
#include "dh/vis/input_queue.hpp"

namespace dh::vis {
  static void glfwKeyCallback(GLFWwindow* window, int key, int scan_code, int action, int mods) {
    InputQueue* queue = (InputQueue *) glfwGetWindowUserPointer(window);
    queue->fwdKeyCallback(key, scan_code, action, mods);
  }

  static void glfwMousePosCallback(GLFWwindow* window, double xPos, double yPos) {
    InputQueue* queue = (InputQueue *) glfwGetWindowUserPointer(window);
    queue->fwdMousePosCallback(xPos, yPos);
  }

  static void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    InputQueue* queue = (InputQueue *) glfwGetWindowUserPointer(window);
    queue->fwdMouseButtonCallback(button, action, mods);
  }

  static void glfwMouseScrollCallback(GLFWwindow* window, double xScroll, double yScroll) {
    InputQueue* queue = (InputQueue *) glfwGetWindowUserPointer(window);
    queue->fwdMouseScrollCallback(xScroll, yScroll);
  }

  InputTask::InputTask(int priority)
  : _priority(priority) { }

  void InputTask::keyboardInput(int key, int action) {
    // Override and implement
  }
  
  void InputTask::mousePosInput(double xPos, double yPos) {
    // Override and implement
  }

  void InputTask::mouseButtonInput(int button, int action) {
    // Override and implement
  }

  void InputTask::mouseScrollInput(double xScroll, double yScroll) {
    // Override and implement
  }

  void InputQueue::init(const util::GLWindow& window) {
    if (_isInit) {
      return;
    }
    
    // Register input callbacks
    GLFWwindow *handle = (GLFWwindow *) window.handle();
    runtimeAssert(handle, "could not register GLFW input callbacks");
    glfwSetWindowUserPointer(handle, this);
    glfwSetKeyCallback(handle, glfwKeyCallback);
    glfwSetCursorPosCallback(handle, glfwMousePosCallback);
    glfwSetMouseButtonCallback(handle, glfwMouseButtonCallback);
    glfwSetScrollCallback(handle, glfwMouseScrollCallback);
    
    _queue = Queue(cmpInputTask);
    _windowHandle = &window;
    _isInit = true;
  }

  void InputQueue::dstr() {
    if (_isInit) {
      return;
    }

    // Deregister input callbacks
    GLFWwindow *handle = (GLFWwindow *) _windowHandle->handle();
    runtimeAssert(handle, "could not deregister GLFW input callbacks");
    glfwSetWindowUserPointer(handle, nullptr);
    glfwSetKeyCallback(handle, nullptr);
    glfwSetCursorPosCallback(handle, nullptr);
    glfwSetMouseButtonCallback(handle, nullptr);
    glfwSetScrollCallback(handle, nullptr);

    _queue.clear();
    _windowHandle = nullptr;
    _isInit = false;
  }

  InputQueue::InputQueue() : _isInit(false) { }

  InputQueue::~InputQueue() {
    if (_isInit) {
      dstr();
    }
  }
  
  void InputQueue::fwdKeyCallback(int key, int scancode, int action, int mods) {
    // Forward keyboard input to ImGui
    auto& io = ImGui::GetIO();
    io.KeysDown[key] = (action == GLFW_PRESS || action == GLFW_REPEAT);
    io.KeyCtrl = (mods & GLFW_MOD_CONTROL != 0);
    io.KeyAlt = (mods & GLFW_MOD_ALT != 0);
    io.KeyShift = (mods & GLFW_MOD_SHIFT != 0);
    io.KeySuper = (mods & GLFW_MOD_SUPER != 0);

    // Ignore, imgui takes keyboard callback
    if (io.WantCaptureKeyboard) {
      return;
    }

    // Forward keyboard callback to input queue 
    for (auto& ptr : _queue) {
      ptr->keyboardInput(key, action);
    }
  }

  void InputQueue::fwdMousePosCallback(double xPos, double yPos) {
    // Do forward despite possible ImGui capture
    // The alternative is **severely annoying**
    for (auto& ptr : _queue) {
      ptr->mousePosInput(xPos, yPos);
    }
  }

  void InputQueue::fwdMouseButtonCallback(int button, int action, int mods) {
    // Ignore, imgui captures button callback
    if (ImGui::GetIO().WantCaptureMouse) {
      return;
    }

    // Forward button callback to input queue 
    for (auto& ptr : _queue) {
      ptr->mouseButtonInput(button, action);
    }
  }

  void InputQueue::fwdMouseScrollCallback(double xScroll, double yScroll) {
    // Forward mouse scroll to ImGui (seems to fail by default?)
    auto& io = ImGui::GetIO();
    io.MouseWheel = yScroll;
    io.MouseWheelH = xScroll;

    // Test if imgui captures scroll callback
    if (io.WantCaptureMouse) {
      return;
    }
    
    // Forward scroll callback to input queue 
    for (auto& ptr : _queue) {
      ptr->mouseScrollInput(xScroll, yScroll);
    }
  }
} // dh::vis