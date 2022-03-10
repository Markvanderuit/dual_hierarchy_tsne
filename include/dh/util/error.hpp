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
#include <exception>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include "dh/constants.hpp"
#include "dh/types.hpp"

namespace dh::util {
  struct RuntimeError : public std::exception {
    std::string message;                      // Basic required message attached to error
    std::map<std::string, std::string> logs;  // Optional other attached data in key/value pairs

    // Constr
    RuntimeError(const std::string& message)
    : std::exception(), message(message) { }

    // Impl. of std::exception::what(), assembles detailed error from provided data
    const char* what() const noexcept override {
      std::stringstream ss;

      ss << "\nRuntime exception\n";
      ss << std::left << std::setw(16) << "  message:" << message << '\n';

      for (const auto& [key, code] : logs) {
        std::string head = "  " + key + ":";
        ss << std::left << std::setw(16) << head << code << '\n';
      }
      
      _what = ss.str();
      return _what.c_str();
    }

  private:
    mutable std::string _what;
  };

  namespace detail {
    inline
    void runtimeAssertImpl(bool statement, const std::string& message, const char *file, int line) {
      if (!statement) {
        RuntimeError error(message);
        error.logs["file"] = file;
        error.logs["line"] = line;
        throw error;
      }
    }
  }
} // dh::util

// Simple assert with message, file name and line nr. attached. Throws RuntimeError
#ifdef DH_ENABLE_ASSERT
#define runtimeAssert(statement, message) dh::util::detail::runtimeAssertImpl(statement, message, __FILE__, __LINE__);
#else
#define runtimeAssert(statement, message) statement;
#endif