#pragma once

#include <chrono>
#include "util/types.hpp"
#include "util/enum.hpp"

namespace tsne {
  class Timer {
  public:
    // Types of durations the timer stores and tracks
    enum class OutputValue {
      eLast,
      eAverage,
      eTotal,

      Length
    };

    Timer() : _isInit(false), _iterations(0) {
      _values.fill(std::chrono::nanoseconds(0));
    }
  
    virtual void init() = 0;
    virtual void dstr() = 0;
    virtual void tick() = 0;
    virtual void tock() = 0;
    virtual void record() = 0;

    template <OutputValue value, typename Duration = std::chrono::milliseconds>
    Duration get() const {
      return std::chrono::duration_cast<Duration>(_values(value));
    }

    bool isInit() const {
      return _isInit;
    }

    uint iterations() const {
      return _iterations;
    }

  protected:
    EnumArray<OutputValue, std::chrono::nanoseconds> _values;
    bool _isInit;
    uint _iterations;
  };

  class CppTimer : public Timer {
  public:
    CppTimer() : Timer() { }
    ~CppTimer() {
      if (_isInit) {
        dstr();
      }
    }

    void init() override {
      if (_isInit) {
        return;
      }
      // _elapsed = 0;
      _iterations = 0;
      _isInit = true;
    }

    void dstr() override {
      if (!_isInit) {
        return;
      }
      _iterations = 0;
      _isInit = false;
    }

    void tick() override {
      _values(OutputValue::eLast) = std::chrono::nanoseconds(0);
      _start = std::chrono::system_clock::now();
    }

    void tock() override {
      _stop = std::chrono::system_clock::now();
      _values(OutputValue::eLast) = _stop - _start;
    }

    void record() override {
      _values(OutputValue::eTotal) += _values(OutputValue::eLast);
      _values(OutputValue::eAverage) = _values(OutputValue::eAverage)
        + (_values(OutputValue::eLast) - _values(OutputValue::eAverage)) / (++_iterations);
    }

  private:
    using Time = std::chrono::time_point<std::chrono::system_clock>;
    Time _start;
    Time _stop;
  };
} // tsne