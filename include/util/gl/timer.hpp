#pragma once

#include <utility>
#include "glad/glad.h"
#include "util/timer.hpp"

namespace tsne {
  class GLTimer : public AbstractTimer {
  public:
    GLTimer() : AbstractTimer() { }
    ~GLTimer() {
      if (_isInit) {
        dstr();
      }
    }

    void init() override {
      if (_isInit) {
        return;
      }
      glCreateQueries(GL_TIME_ELAPSED, 1, &_front);
      glCreateQueries(GL_TIME_ELAPSED, 1, &_back);
      _iterations = 0;
      _isInit = true;
    }

    void dstr() override {
      if (!_isInit) {
        return;
      }
      glDeleteQueries(1, &_front);
      glDeleteQueries(1, &_back);
      _iterations = 0;
      _isInit = false;
    }

    void tick() override {
      glBeginQuery(GL_TIME_ELAPSED, _front);
    }

    void tock() override {
      glEndQuery(GL_TIME_ELAPSED);
    }

    void record() override {
      GLint64 elapsed;
      glGetQueryObjecti64v(_back, GL_QUERY_RESULT, &elapsed);
      _values(OutputValue::eLast) = std::chrono::nanoseconds(elapsed);
      _values(OutputValue::eTotal) += _values(OutputValue::eLast);
      _values(OutputValue::eAverage) = _values(OutputValue::eAverage)
        + (_values(OutputValue::eLast) - _values(OutputValue::eAverage)) / (++_iterations);

      std::swap(_front, _back);
    }
  private:
    GLuint _front;
    GLuint _back;
  };
} // tsne