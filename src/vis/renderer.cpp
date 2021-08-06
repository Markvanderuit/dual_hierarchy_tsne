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

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include "util/gl/error.hpp"
#include "vis/render_queue.hpp"
#include "vis/renderer.hpp"

namespace dh::vis {
  Renderer::Renderer()
  : _isInit(false), _fboSize(0) { }

  Renderer::Renderer(const util::GLWindow& window, sne::SNEParams params)
  : _isInit(false), _params(params), _windowHandle(&window), _fboSize(0) {
    
    // Init static render queue so RenderTasks can be added by tSNE
    auto& queue = RenderQueue::instance();
    queue.init();

    // Init OpenGL objects: framebuffer, color and depth textures, label buffer
    glCreateFramebuffers(1, &_fboHandle);
    glCreateTextures(GL_TEXTURE_2D, 1, &_fboColorTextureHandle);
    glCreateTextures(GL_TEXTURE_2D, 1, &_fboDepthTextureHandle);
    glCreateBuffers(1, &_labelsHandle);
    glAssert();
    
    _isInit = true;
  }

  Renderer::~Renderer() {
    if (_isInit) {
      auto& queue = RenderQueue::instance();
      queue.dstr();      
      glDeleteTextures(1, &_fboColorTextureHandle);
      glDeleteTextures(1, &_fboDepthTextureHandle);
      glDeleteFramebuffers(1, &_fboHandle);
      glDeleteBuffers(1, &_labelsHandle);
      _isInit = false;
    }
  }

  Renderer::Renderer(Renderer&& other) noexcept {
    swap(*this, other);
  }

  Renderer& Renderer::operator=(Renderer&& other) noexcept {
    swap(*this, other);
    return *this;
  }

  void Renderer::render() {
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

    // Render transformation matrix; 3d depends on a trackball impl
    glm::mat4 transform;
    if (_params.nLowDims == 3) {
      // TODO ...
    } else if (_params.nLowDims == 2) {
      // TODO set to sensible values
      // Center on screen
      transform = glm::scale(glm::vec3(1.66, 1.66, 1))
                * glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f));
    }

    // Set viewport
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
    
    // Process all tasks in render queue
    for (auto& ptr : RenderQueue::instance().queue()) {
      ptr->render(transform, _labelsHandle);
    }

    // Blit to default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBlitNamedFramebuffer(_fboHandle, 0, 
      0, 0, _fboSize.x, _fboSize.y, 
      0, 0, _fboSize.x, _fboSize.y,
      GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST  
    );
  }
} // dh::vis