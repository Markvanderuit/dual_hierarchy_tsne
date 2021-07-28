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
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <string>
#include "types.hpp"

namespace dh::util {
  struct RuntimeError : public std::exception {
    std::string message;  // Basic required message attached to error
    std::string code;     // Optional error code provided by APIs (OpenGL/CUDA)
    std::string log;      // Optional log provided by APIs (OpenGL)
    std::string file;     // Optional file name, available during preprocessing
    int line;             // Optional line nr, available during preprocessing

    // Constr
    RuntimeError(const std::string& message)
    : std::exception(), message(message), line(-1) { }

    // Impl. of std::exception::what(), assembles detailed error from provided data
    const char* what() const noexcept override {
      std::stringstream ss;
      ss << "Runtime error\n";
      ss << std::left << std::setw(16) << "  message:" << message << '\n';

      if (!file.empty()) {
        ss << std::left << std::setw(16) << "  file:" << file << '\n';
      }
      if (line != -1) {
        ss << std::left << std::setw(16) << "  line:" << line << '\n';
      }
      if (!code.empty()) {
        ss << std::left << std::setw(16) << "  code:" << code << '\n';
      }
      if (!log.empty()) {
        ss << "  log:" << log << '\n';
      }
      
      _what = ss.str();
      return _what.c_str();
    }

  private:
    mutable std::string _what;
  };

  namespace detail {
    inline
    void runtimeAssertImpl(bool statement, const char* message, const char *file, int line) {
      if (!statement) {
        RuntimeError error(message);
        error.file = file;
        error.line = line;
        throw error;
      }
    }
  }
} // dh::util

// Simple assert with message, file name and line nr. attached. Throws RuntimeError
#define runtimeAssert(statement, message) dh::util::detail::runtimeAssertImpl(statement, message, __FILE__, __LINE__);