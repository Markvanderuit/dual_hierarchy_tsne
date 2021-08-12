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

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "vis/input_queue.hpp"

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
    _windowHandle = &window;
    _queue = std::set<std::shared_ptr<InputTask>, decltype(&cmpInputTask)>(cmpInputTask);
    
    // Register input callbacks
    GLFWwindow *handle = (GLFWwindow *) _windowHandle->handle();
    glfwSetWindowUserPointer(handle, this);
    glfwSetKeyCallback(handle, glfwKeyCallback);
    glfwSetCursorPosCallback(handle, glfwMousePosCallback);
    glfwSetMouseButtonCallback(handle, glfwMouseButtonCallback);
    glfwSetScrollCallback(handle, glfwMouseScrollCallback);
    
    _isInit = true;
  }

  void InputQueue::dstr() {
    if (_isInit) {
      return;
    }

    // Deregister input callbacks
    GLFWwindow *handle = (GLFWwindow *) _windowHandle->handle();
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
    // Quick and dirty escape key implementation
    // TODO Remove and add to a task!
    if (key == GLFW_KEY_ESCAPE) {
      std::exit(0);
    }

    // // ImGui captures keyboard input, do not forward
    // if (ImGui::GetIO().WantCaptureKeyboard) {
    //   return;
    // }

    for (auto& ptr : _queue) {
      ptr->keyboardInput(key, action);
    }
  }

  void InputQueue::fwdMousePosCallback(double xPos, double yPos) {
    // Do forward despite possible ImGui capture
    // The alternative is annoying
    for (auto& ptr : _queue) {
      ptr->mousePosInput(xPos, yPos);
    }
  }

  void InputQueue::fwdMouseButtonCallback(int button, int action, int mods) {
    // // ImGui captures mouse input, do not forward
    // if (ImGui::GetIO().WantCaptureMouse) {
    //   return;
    // }
    for (auto& ptr : _queue) {
      ptr->mouseButtonInput(button, action);
    }
  }

  void InputQueue::fwdMouseScrollCallback(double xScroll, double yScroll) {
    // // ImGui captures mouse input, do not forward
    // if (ImGui::GetIO().WantCaptureMouse) {
    //   return;
    // }
    for (auto& ptr : _queue) {
      ptr->mouseScrollInput(xScroll, yScroll);
    }
  }
} // dh::vis