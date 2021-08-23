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

#include <iostream>

#include <cmath>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "dh/util/aligned.hpp"
#include "dh/vis/components/trackball_input_task.hpp"

namespace dh::vis {
  TrackballInputTask::TrackballInputTask()
  : InputTask(1),
    _mouseTrackState(false), 
    _mouseScrollState(3.0f), 
    _mousePosState(0.0f), 
    _mousePosStatePrev(0.0f),
    _lookatState(1),
    _matrix(1),
    _mouseScrollMult(0.5f),
    _mousePosMult(1.0f) {
    // ...
  }

  void TrackballInputTask::process() {
    // Compute extremely simplified plane rotation
    if (_mouseTrackState && _mousePosState != _mousePosStatePrev) {
      glm::vec2 v = _mousePosMult * (_mousePosState - _mousePosStatePrev);
      glm::mat4 rx = glm::rotate(v.x, glm::vec3(0, 1, 0));
      glm::mat4 ry = glm::rotate(v.y, glm::vec3(1, 0, 0));
      _lookatState = ry * rx * _lookatState;
    }

    // Apply mouse scroll for final result
    _matrix = glm::lookAt(
      glm::vec3(0, 0, 1) * _mouseScrollState,
      glm::vec3(0, 0, 0),
      glm::vec3(0, 1, 0)
    ) * _lookatState;
  }

  void TrackballInputTask::mousePosInput(double xPos, double yPos) {
    // Obtain current window handle for window size
    util::GLWindow* window = util::GLWindow::currentWindow();
    if (!window) {
      return;
    }

    // Record previous position as last recorded position
    _mousePosStatePrev = _mousePosState;

    // Recorcd current position in [-1, 1]
    _mousePosState = glm::vec2(xPos, yPos)
                   / glm::vec2(window->size());    
    _mousePosState = 2.0f * _mousePosState - 1.0f;
  }

  void TrackballInputTask::mouseButtonInput(int button, int action) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
      _mouseTrackState = true;
    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
      _mouseTrackState = false;
    }
  }

  void TrackballInputTask::mouseScrollInput(double xScroll, double yScroll) {
    _mouseScrollState = std::max(0.001f, _mouseScrollState - _mouseScrollMult * static_cast<float>(yScroll));
  }
} // dh::vis