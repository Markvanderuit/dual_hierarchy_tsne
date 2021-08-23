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

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glad/glad.h>
#include "dh/util/gl/error.hpp"
#include "dh/vis/render_queue.hpp"
#include "dh/vis/input_queue.hpp"
#include "dh/vis/renderer.hpp"
#include "dh/vis/components/esc_input_task.hpp"

namespace dh::vis {
  template <uint D>
  Renderer<D>::Renderer()
  : _isInit(false), _fboSize(0) { }

  template <uint D>
  Renderer<D>::Renderer(const util::GLWindow& window, sne::Params params, const std::vector<uint>& labels)
  : _isInit(false), _params(params), _windowHandle(&window), _labelsHandle(0), _fboSize(0) {
    
    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    // ImGui platform components
    ImGui_ImplGlfw_InitForOpenGL((GLFWwindow *) window.handle(), true);
    const char *version = "#version 460";
    ImGui_ImplOpenGL3_Init(version);

    // Init static render queue/input queue so tasks can be added by the sne lib
    InputQueue::instance().init(window);
    RenderQueue::instance().init();

    // Add input tasks; first to capture ESC key and shut down application,
    // second as 3D rendertasks need trackball input to generate transformation matrix
    InputQueue::instance().emplace(vis::EscInputTask());
    _trackballInputTask = InputQueue::instance().emplace(vis::TrackballInputTask());

    // Init OpenGL objects: framebuffer, color and depth textures, label buffer
    glCreateFramebuffers(1, &_fboHandle);
    glCreateTextures(GL_TEXTURE_2D, 1, &_fboColorTextureHandle);
    glCreateTextures(GL_TEXTURE_2D, 1, &_fboDepthTextureHandle);
    if (labels.size() > 0) {
      glCreateBuffers(1, &_labelsHandle);
      glNamedBufferStorage(_labelsHandle, labels.size() * sizeof(uint), labels.data(), 0);    
    }
    glAssert();
    
    _isInit = true;
  }

  template <uint D>
  Renderer<D>::~Renderer() {
    if (_isInit) {
      // Shut down imgui
      ImGui_ImplOpenGL3_Shutdown();
      ImGui_ImplGlfw_Shutdown();
      ImGui::DestroyContext();

      // Shut down static render queue/input queue
      RenderQueue::instance().dstr();      
      InputQueue::instance().dstr();      

      // Destroy OpenGL objects
      glDeleteTextures(1, &_fboColorTextureHandle);
      glDeleteTextures(1, &_fboDepthTextureHandle);
      glDeleteFramebuffers(1, &_fboHandle);
      glDeleteBuffers(1, &_labelsHandle);

      _isInit = false;
    }
  }

  template <uint D>
  Renderer<D>::Renderer(Renderer<D>&& other) noexcept {
    swap(*this, other);
  }

  template <uint D>
  Renderer<D>& Renderer<D>::operator=(Renderer<D>&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  template <uint D>
  void Renderer<D>::render() {
    // Recreate framebuffer if window size has changed
    if (glm::ivec2 fboSize = _windowHandle->size(); _fboSize != fboSize) {
      _fboSize = fboSize;
      if (glIsTexture(_fboColorTextureHandle)) {
        glDeleteTextures(1, &_fboColorTextureHandle);
        glDeleteTextures(1, &_fboDepthTextureHandle);
      }
      glCreateTextures(GL_TEXTURE_2D, 1, &_fboColorTextureHandle);
      glCreateTextures(GL_TEXTURE_2D, 1, &_fboDepthTextureHandle);
      glTextureStorage2D(_fboColorTextureHandle, 1, GL_RGBA32F, _fboSize.x, _fboSize.y);
      glTextureStorage2D(_fboDepthTextureHandle, 1, GL_DEPTH_COMPONENT24, _fboSize.x, _fboSize.y);
      glNamedFramebufferTexture(_fboHandle, GL_COLOR_ATTACHMENT0, _fboColorTextureHandle, 0);
      glNamedFramebufferTexture(_fboHandle, GL_DEPTH_ATTACHMENT, _fboDepthTextureHandle, 0);
      glAssert();
    }

    // Process all tasks in input queue
    for (auto& ptr : InputQueue::instance().queue()) {
      ptr->process();
    }

    // Start new frame for IMGUI
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Add ImGui components
    drawImGuiMenuBar();
    drawImGuiComponents();

    // Model/view/proj matrices. 3D view matrix is provided by trackball
    glm::mat4 proj, model_view;
    if constexpr (D == 3) {
      model_view = _trackballInputTask->matrix() * glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f));
      proj = glm::perspectiveFov(0.5f, (float) _fboSize.x, (float) _fboSize.y, 0.0001f, 1000.f);
    } else if constexpr (D == 2) {
      // Center on screen
      model_view = glm::translate(glm::vec3(-0.5f, -0.5f, -1.0f));
      proj = glm::infinitePerspective(1.0f, (float) _fboSize.x / (float) _fboSize.y, 0.0001f);
    }

    // Set viewport for render tasks
    glViewport(0, 0, _fboSize.x, _fboSize.y);

    // Clear framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, _fboHandle);
    constexpr glm::vec4 clearColor(1.0f, 1.0f, 1.0f, 0.0f);
    constexpr float clearDepth = 1.0f;
    glClearNamedFramebufferfv(_fboHandle, GL_COLOR, 0, glm::value_ptr(clearColor));
    glClearNamedFramebufferfv(_fboHandle, GL_DEPTH, 0, &clearDepth);

    // Specify depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    // Specify blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  
    
    for (auto& ptr : RenderQueue::instance().queue()) {
      ptr->render(model_view, proj, _labelsHandle);
    }

    // Blit to default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBlitNamedFramebuffer(_fboHandle, 0, 
      0, 0, _fboSize.x, _fboSize.y, 
      0, 0, _fboSize.x, _fboSize.y,
      GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST  
    );

    // Finalize and render ImGui components
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  }

  template <uint D>
  void Renderer<D>::drawImGuiMenuBar() {
    ImGui::BeginMainMenuBar();
    
    if (ImGui::BeginMenu("Application")) {
      if (ImGui::MenuItem("About", nullptr, false)) {
        // ...
      }
      if (ImGui::MenuItem("Quit", "Escape", false)) { std::exit(0); }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Components")) {
      if (ImGui::MenuItem("Camera", nullptr, false)) {
        _showTrackballInputTaskMenu = !_showTrackballInputTaskMenu;
      }
      if (ImGui::MenuItem("Embedding", nullptr, false)) {
        _showEmbeddingRenderTaskMenu = !_showEmbeddingRenderTaskMenu;
      }
      if (ImGui::MenuItem("Embedding hierarchy", nullptr, false)) {
        _showEmbeddingHierarchyRenderTaskMenu = !_showEmbeddingHierarchyRenderTaskMenu;
      }
      if (ImGui::MenuItem("Field hierarchy", nullptr, false)) {
        _showFieldHierarchyRenderTaskMenu = !_showFieldHierarchyRenderTaskMenu;
      }
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }

  template <uint D>
  void Renderer<D>::drawImGuiComponents() {
    auto& queue = vis::RenderQueue::instance();
    if (auto ptr = queue.find("EmbeddingRenderTask"); _showEmbeddingRenderTaskMenu && ptr) { ptr->drawImGuiComponent(); }
    if (auto ptr = queue.find("EmbeddingHierarchyRenderTask"); _showEmbeddingHierarchyRenderTaskMenu && ptr) { ptr->drawImGuiComponent(); }
    if (auto ptr = queue.find("FieldHierarchyRenderTask"); _showFieldHierarchyRenderTaskMenu && ptr) { ptr->drawImGuiComponent(); }
  }

  // Template instantiations for 2/3 dimensions
  template class Renderer<2>;
  template class Renderer<3>;
} // dh::vis