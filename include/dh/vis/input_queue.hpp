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

#include <memory>
#include <utility>
#include <set>
#include <type_traits>
#include "dh/util/gl/window.hpp"

namespace dh::vis {
  class InputTask {
  public:
    // Constr
    InputTask(int priority = -1);

    // Override and implement an input task to be handled by the renderer
    virtual void process() = 0;

    // Override and implement if necessary for input to be forwarded by the renderer
    virtual void keyboardInput(int key, int action);
    virtual void mousePosInput(double xPos, double yPos);
    virtual void mouseButtonInput(int button, int action);
    virtual void mouseScrollInput(double xScroll, double yScroll);

  private:
    int _priority;

  public:
    // Getters
    int priority() const { return _priority; }

    // Compare function for sortable insertion
    using Pointer = std::shared_ptr<InputTask>;
    friend bool cmpInputTask(const Pointer& a, const Pointer& b) {
      return a->priority() < b->priority() && a != b;
    }
    
    // std::swap impl
    friend void swap(InputTask& a, InputTask& b) noexcept {
      using std::swap;
      swap(a._priority, b._priority);
    }
  };

  class InputQueue {
  private:
    using Pointer = std::shared_ptr<InputTask>;
    using Queue = std::set<Pointer, decltype(&cmpInputTask)>;

  public:
    // Accessor; there is one InputQueue used by the vis library
    // Ergo, InputQueue implements a singleton pattern, but
    // with controllable initialization/destruction. It is not 
    // functional for the sne lib until InputQueue::instance().init() has been called.
    static InputQueue& instance() {
      static InputQueue instance;
      return instance;
    }

    // Setup/teardown functions
    void init(const util::GLWindow& window);
    void dstr();

    // Insert input task, return a shared pointer to provide later access to task
    template <class DerivedTask,
              typename = std::enable_if_t<std::is_base_of_v<InputTask, DerivedTask>>>
    std::shared_ptr<DerivedTask> emplace(DerivedTask&& task) {
      if (!_isInit) {
        return nullptr;
      }
      auto ptr = std::make_shared<DerivedTask>(std::move(task));
      insert(ptr);
      return ptr;
    }

    // Insert input task, given a pre-existing shared pointer
    template <class DerivedTask,
              typename = std::enable_if_t<std::is_base_of_v<InputTask, DerivedTask>>>
    void insert(std::shared_ptr<DerivedTask> ptr) {
      if (!_isInit) {
        return;
      }
      _queue.insert(ptr);
    }

    // Erase input task from queue, assuming a pointer for access is available
    void erase(Pointer ptr) {
      if (!_isInit) {
        return;
      }
      if (auto i = _queue.find(ptr); i != _queue.end()) {
        _queue.erase(i);
      }
    }

    // Getters
    bool isInit() const { return _isInit; }
    Queue queue() { return _queue; }

    // Forwards for GLFW input callback
    void fwdKeyCallback(int key, int scancode, int action, int mods);
    void fwdMousePosCallback(double xPos, double yPos);
    void fwdMouseButtonCallback(int button, int action, int mods);
    void fwdMouseScrollCallback(double xScroll, double yScroll);

  private:
    // Hidden constr/destr
    InputQueue();
    ~InputQueue();

    // State
    bool _isInit;
    Queue _queue;
    const util::GLWindow * _windowHandle;
  };
} // dh::vis