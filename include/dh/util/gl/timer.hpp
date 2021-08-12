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

#include <glad/glad.h>
#include "dh/util/timer.hpp"

namespace dh::util {
  // Simple wrapper around a OpenGL timer query object
  class GLTimer : public Timer {
  public:
    GLTimer();
    ~GLTimer();

    // Copy constr/assignment is explicitly deleted (no copying OpenGL objects)
    GLTimer(const GLTimer&) = delete;
    GLTimer& operator=(const GLTimer&) = delete;

    // Move constr/operator moves resource handles
    GLTimer(GLTimer&&) noexcept;
    GLTimer& operator=(GLTimer&&) noexcept;

    // Swap internals with another timer object
    friend void swap(GLTimer& a, GLTimer& b) noexcept;

    // Override and implement for this specific timer
    void tick() override;
    void tock() override;
    void poll() override;

  private:
    // Pair of timer queries that can be swapped
    GLuint _frontQuery;
    GLuint _backQuery;
  };

  // Helper function to poll a number of timers
  inline
  void glPollTimers(GLsizei n, GLTimer *timers) {
    for (GLsizei i = 0; i < n; ++i) {
      timers[i].poll();
    }
  }
} // dh