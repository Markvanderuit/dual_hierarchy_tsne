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

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <date/date.h>

namespace dh::util {
  class Logger {
  public:
    virtual void writeLine(const std::string &str) const = 0;
    virtual void flush() const = 0;
  };

  class CoutLogger : public Logger {
  public:
    virtual void writeLine(const std::string &str) const override {
      std::cout << str << '\n';
    }
    
    virtual void flush() const override {
      std::cout << std::flush;
    }
  };

  class DateCoutLogger : public CoutLogger {
  public:
    virtual void writeLine(const std::string &str) const override {
      using namespace date;
      using namespace std::chrono;

      std::cout << floor<milliseconds>(system_clock::now())
                << " " << str << '\n';
    }
  };
  
  template <class Logger, bool flush = false>
  inline
  void log(Logger* ptr, const std::string& str) {
    if (ptr) {
      ptr->writeLine(str);
      if constexpr (flush) {
        ptr->flush();
      }
    }
  }

  template <class Logger, typename T, bool flush = false>
  inline
  void logValue(Logger* ptr, const std::string& str, T t) {
    if (ptr) {
      std::stringstream ss;
      ss << str << " : " << std::to_string(value);
      ptr->writeLine(ss.str());
      if constexpr (flush) {
        ptr->flush();
      }
    }
  }

  template <class Logger, typename Duration = std::chrono::milliseconds, bool flush = false>
  inline
  void logTime(Logger* ptr, const std::string& str, Duration duration) {
    using namespace date;
    using namespace std::chrono;

    if (ptr) {
      std::stringstream ss;
      ss << str << " : " << duration;
      ptr->writeLine(ss.str());
      if constexpr (flush) {
        ptr->flush();
      }
    }
  }
} // dh::util
