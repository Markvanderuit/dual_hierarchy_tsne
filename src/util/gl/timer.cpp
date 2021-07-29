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

#include <utility>
#include "util/gl/timer.hpp"

namespace dh::util {
  GLTimer::GLTimer() : Timer() {
    glCreateQueries(GL_TIME_ELAPSED, 1, &_frontQuery);
    glCreateQueries(GL_TIME_ELAPSED, 1, &_backQuery);
  }

  GLTimer::~GLTimer() {
    glDeleteQueries(1, &_frontQuery);
    glDeleteQueries(1, &_backQuery);
  }

  GLTimer::GLTimer(GLTimer&& other) noexcept {
    swap(*this, other);
  }

  GLTimer& GLTimer::operator=(GLTimer&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void swap(GLTimer& a, GLTimer& b) noexcept {
    using std::swap;
    swap(a._values, b._values);
    swap(a._iterations, b._iterations);
    swap(a._frontQuery, b._frontQuery);
    swap(a._backQuery, b._backQuery);
  }

  void GLTimer::tick() {
    glBeginQuery(GL_TIME_ELAPSED, _frontQuery);
  }

  void GLTimer::tock() {
    glEndQuery(GL_TIME_ELAPSED);
  }

  void GLTimer::poll() {
    // Swap timer queries
    std::swap(_frontQuery, _backQuery);
    
    // Query result of previous timer which likely already finished
    GLint64 elapsed;
    glGetQueryObjecti64v(_frontQuery, GL_QUERY_RESULT, &elapsed);
    _values(TimerValue::eLast) = std::chrono::nanoseconds(elapsed);
    _values(TimerValue::eTotal) += _values(TimerValue::eLast);
    _values(TimerValue::eAverage) = _values(TimerValue::eAverage)
      + (_values(TimerValue::eLast) - _values(TimerValue::eAverage)) / (++_iterations);
  }
} // dh