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

// #include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <date/date.h>
#include <indicators/progress_bar.hpp>
#include "dh/constants.hpp"

namespace dh::util {
  class Logger {
  private:
    // State
    bool _isFirstLine;
    std::ostream *_stream;
    
    // Constr, hidden for singleton
    Logger() : _isFirstLine(true), _stream(nullptr) { }
    Logger(std::ostream &stream) :  _isFirstLine(true), _stream(&stream) { }

  public:
    // Singleton direct accessor
    static Logger& instance();

    // Singleton setup/teardown functions
    static void init(std::ostream &stream) { instance() = Logger(stream); }
    static void dstr() { instance() = Logger(); }

    // Obtain pointer to registered stream
    static std::ostream* stream() { return instance()._stream; }

    // Start new line for writes
    static Logger& newl() {
      Logger& logger = Logger::instance();
      if (logger._isFirstLine) {
        logger._isFirstLine = false;
        return logger;
      }
      return logger << '\n';
    }

    // Use current line for writes
    static Logger& curl() {
      return instance();
    }

    // Reset and reuse current line for writes
    static Logger& resl() {
      return curl() << "\33[2K\r";
    }

    // Start new line for writes, and prepend timestamp
    static Logger& newt() {
      return newl() << Logger::time;
    }
    
    // Use current line for writes, and prepend timestamp;
    static Logger& curt() {
      return curl() << Logger::time;
    }

    // Reset and reuse current line for writes, and prepend timestamp
    static Logger& rest() {
      return resl() << Logger::time;
    }

  public:
    // Accept generic input and forward to stream
    template <typename T>
    Logger& operator<<(const T& t) {
      using namespace date;
      using namespace std::chrono;
      if (_stream) { (*_stream) << t; }
      return *this;
    }

    // Accept ostream function manipulators and forward
    using manip = std::ostream& (*)(std::ostream&);
    Logger& operator<<(manip fp) {
      if (_stream) { (*_stream) << fp; }
      return *this;
    }
    
    // Timestamp manipulator
    static
    std::ostream& time(std::ostream& ostr) {      
#ifdef DH_LOG_TIMESTAMPS
      // Obtain UTC timestamp in format hours:minutes:seconds.milliseconds
      using namespace date;
      using namespace std::chrono;
      const auto curr = make_time(floor<milliseconds>(system_clock::now()) - floor<days>(system_clock::now()));
      return ostr << '[' << curr << "] ";
#else
      return ostr;
#endif // DH_LOG_TIMESTAMPS
    }
  };

  class ProgressBar {
  public:
    ProgressBar(std::string_view prefix = "", std::string_view postfix = "")
    : _prefix(prefix), _postfix(postfix) { }

    void setProgress(float dec) {
      // Logger does not have an output stream set, do not output progress
      if (!Logger::stream()) {
        return;
      }

#ifdef DH_LOG_TIMESTAMPS
      // Obtain UTC timestamp in format hours:minutes:seconds.milliseconds
      using namespace date;
      using namespace std::chrono;
      const auto curr = make_time(floor<milliseconds>(system_clock::now()) - floor<days>(system_clock::now()));
      // Prepend timestamp to prefix
      std::stringstream ss;
      ss << '[' << curr << "] " << _prefix;
      std::string prefix = ss.str();
#else
      std::string prefix = _prefix;
#endif // DH_LOG_TIMESTAMPS

      // Construct progress bar on the current line
      indicators::ProgressBar bar {
        indicators::option::BarWidth{30},
        indicators::option::Start{" ["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"] "},
        indicators::option::ForegroundColor{indicators::Color::white},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::PrefixText{prefix},
        indicators::option::PostfixText{_postfix}
      };
      bar.set_progress(static_cast<size_t>(dec * 100.f));
    }

    void setPostfix(std::string_view str) {
      _postfix = str;
    }

    void setPrefix(std::string_view str) {
      _prefix = str;
    }
    
  private:
    std::string _prefix;
    std::string _postfix;
  };
  
  // Helper function for printing nice class names in the log output
  inline
  std::string genLoggerPrefix(std::string_view str) {
    std::stringstream ss;
    ss << std::left << std::setw(DH_LOG_PREFIX_WIDTH) << str << ' ';
    return ss.str();
  }
} // dh::util