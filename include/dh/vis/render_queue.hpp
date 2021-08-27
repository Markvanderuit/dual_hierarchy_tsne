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

#include <algorithm>
#include <memory>
#include <utility>
#include <set>
#include <string>
#include <type_traits>
#include "dh/types.hpp"
#include "dh/util/aligned.hpp"

namespace dh::vis {
  class RenderTask {
  public:
    // Constr
    RenderTask();
    RenderTask(int priority, const std::string& name);

    // Override and implement a render task to be handled by the renderer
    virtual void render(glm::mat4 model_view, glm::mat4 proj, GLuint labelsHandle = 0) = 0;
    virtual void drawImGuiComponent() = 0;

    // Publ, enable/disable rendering through imgui
    bool enable;

  private:
    int _priority;
    std::string _name;

  public:
    // Getters
    int priority() const { return _priority; }
    std::string name() const { return _name; }

    // Compare function for sortable insertion
    friend bool cmpRenderTask(const std::shared_ptr<RenderTask>& a, const std::shared_ptr<RenderTask>& b);
    
    // std::swap impl
    friend void swap(RenderTask& a, RenderTask& b) noexcept {
      using std::swap;
      swap(a.enable, b.enable);
      swap(a._priority, b._priority);
      swap(a._name, b._name);
    }
  };

  inline
  bool cmpRenderTask(const std::shared_ptr<RenderTask>& a, const std::shared_ptr<RenderTask>& b) {
    return a->priority() < b->priority() && a != b;
  }

  class RenderQueue {
  private:
    using Pointer = std::shared_ptr<RenderTask>;
    using Queue = std::set<Pointer, decltype(&cmpRenderTask)>;

  public:
    // Accessor; there is one RenderQueue used by the vis library
    // Ergo, RenderQueue implements a singleton pattern, but
    // with controllable initialization/destruction. It is not 
    // functional for the sne lib until RenderQueue::instance().init() has been called.
    static RenderQueue& instance() {
      static RenderQueue instance;
      return instance;
    }

    // Setup/teardown functions
    void init();
    void dstr();

    // Insert render task, return a shared pointer to provide later access to task
    template <class DerivedTask,
              typename = std::enable_if_t<std::is_base_of_v<RenderTask, DerivedTask>>>
    std::shared_ptr<DerivedTask> emplace(DerivedTask&& task) {
      if (!_isInit) {
        return nullptr;
      }
      auto ptr = std::make_shared<DerivedTask>(std::move(task));
      insert(ptr);
      return ptr;
    }

    // Insert render task, given a pre-existing shared pointer
    template <class DerivedTask,
              typename = std::enable_if_t<std::is_base_of_v<RenderTask, DerivedTask>>>
    void insert(std::shared_ptr<DerivedTask> ptr) {
      if (!_isInit) {
        return;
      }
      _queue.insert(ptr);
    }

    // Find a render task for its given name, or return nullptr otherwise
    Pointer find(const std::string& name) {
      if (!_isInit) {
        return nullptr;
      }
      auto iter = std::find_if(_queue.begin(), _queue.end(), [name](auto& ptr) { return ptr->name() == name;});
      if (iter == _queue.end()) {
        return nullptr;
      }
      return *iter;
    }

    // Erase render task from queue, assuming a pointer for access is available
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

  private:
    // Hidden constr/destr
    RenderQueue();
    ~RenderQueue();
    
    // State
    bool _isInit;
    Queue _queue;
  };
} // dh::vis