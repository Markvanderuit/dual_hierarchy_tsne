#pragma once

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

namespace tsne {
  class Logger {
  public:
    Logger() : _isInit(false) { }

    virtual void init() = 0;
    virtual void dstr() = 0;
    virtual void log(const std::string &str) const = 0;
    virtual void logFlush(const std::string &str) const = 0;

    template <typename Value, bool Flush = true>
    void logValue(const std::string &str, Value value) const {
      std::stringstream ss;
      ss << str << "\t:\t" << std::to_string(value);
      if constexpr (Flush) {
        logFlush(ss.str());
      } else constexpr {
        log(ss.str());
      }
    }

    template <typename Duration = std::chrono::milliseconds, bool Flush = true>
    void logTime(const std::string &str, Duration duration) const {
      std::stringstream ss;
      ss << str << "\t:\t" << std::to_string(duration.count());
      if constexpr (Flush) {
        logFlush(ss.str());
      } else constexpr {
        log(ss.str());
      }
    }

    bool isInit() const {
      return _isInit;
    }

  protected:
    bool _isInit;
  };

  class CoutLogger : public Logger {
  public:
    CoutLogger() : Logger() { }

    ~CoutLogger() {
      if (_isInit) {
        dstr();
      }
    }

    void init() override {
      _isInit = true;
      // CoutLogger has no resources to manage, so nothing to do here
    }

    void dstr() override {
      _isInit = false;
      // CoutLogger has no resources to manage, so nothing to do here
    }
    
    void log(const std::string &str) const override {
      std::cout << str << '\n';
    }
    
    void logFlush(const std::string &str) const override {
      std::cout << str << std::endl;
    }
  };
} // tsne
