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

#include <utility>>
#include <chrono>
#include "types.hpp"
#include "util/enum.hpp"

namespace dh::util {
  // Types of recorded time the timer stores
  enum class TimerValue {
    eLast,    // Last time recorded
    eAverage, // Average of times recorded over n iterations
    eTotal,   // Sum of times recorded over n iterations

    Length
  };
  
  class Timer {
  public:
    Timer() : _iterations(0) {
      _values.fill(std::chrono::nanoseconds(0));
    }

    // Start recording an interval of time
    virtual void tick() = 0;

    // Stop recording an interval of time
    virtual void tock() = 0;

    // Poll results (OpenGL/CUDA, blocking if handled poorly)
    virtual void poll() = 0;

    // Return a type of recorded time
    template <TimerValue value = TimerValue::eLast, typename Duration = std::chrono::milliseconds>
    Duration get() const {
      return std::chrono::duration_cast<Duration>(_values(value));
    }

    // Return number of recorded iterations
    uint iterations() const {
      return _iterations;
    }

  protected:
    EnumArray<TimerValue, std::chrono::nanoseconds> _values;
    uint _iterations;
  };
  
  // Simple std::chrono based timer for cpu side timings
  class CppTimer : public Timer {
  public:
    CppTimer() : Timer() { }
    ~CppTimer() { }

    void tick() override {
      _values(TimerValue::eLast) = std::chrono::nanoseconds(0);
      _start = std::chrono::system_clock::now();
    }

    void tock() override {
      _stop = std::chrono::system_clock::now();
      _values(TimerValue::eLast) = _stop - _start;
      _values(TimerValue::eTotal) += _values(TimerValue::eLast);
      _values(TimerValue::eAverage) = _values(TimerValue::eAverage)
        + (_values(TimerValue::eLast) - _values(TimerValue::eAverage)) / (++_iterations);
    }

    void poll() override {
      // not handled ...
    }

    friend void swap(CppTimer& a, CppTimer& b) noexcept {
      using std::swap;
      swap(a._values, b._values);
      swap(a._iterations, b._iterations);
      swap(a._start, b._start);
      swap(a._stop, b._stop);
    }
  private:
    using Time = std::chrono::time_point<std::chrono::system_clock>;
    Time _start;
    Time _stop;
  };
} // dh::util