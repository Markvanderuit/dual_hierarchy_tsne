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

#include <exception>
#include <cstdlib>
#include <string>
#include <vector>
#include "util/io.hpp"
#include "util/logger.hpp"
#include "util/timer.hpp"
#include "util/gl/window.hpp"
#include "vis/renderer.hpp"
#include "sne/sne.hpp"

using uint = unsigned;

int main(int argc, char** argv) {
  
  try {
    dh::util::DateCoutLogger logger;

    // Set example parameters
    dh::sne::SNEParams params;
    params.n = 60000;
    params.nHighDims = 784;
    params.nLowDims = 2;
    params.singleHierarchyTheta = 0.5;
    params.dualHierarchyTheta = 0.0;

    const bool doLabels = true;

    // Load example dataset
    const std::string inputFileName = "C:/users/markv/documents/repositories/dual_hierarchy_tsne/resources/data/mnist_labeled_60k_784d.bin";
    std::vector<float> data;
    std::vector<uint> labels;
    dh::util::readBinFile(inputFileName, data, labels, params.n, params.nHighDims, doLabels);

    // Create OpenGL context
    dh::util::GLWindow window = dh::util::GLWindow::DecoratedResizable();
    window.makeCurrent();
    window.display();

    // Create renderer
    dh::vis::Renderer renderer(window, params);

    // Test renderer
    /* {
      auto& queue = dh::vis::RenderQueue::instance();
      queue.insert(dh::vis::TestRenderTask());

      while (window.canDisplay()) {
        window.processEvents();
        renderer.render();
        window.display();
      }
    } */

    // Set up and run tsne
    dh::sne::SNE<2> sne(data, params, &logger);
    {
      // Do similarity computation
      sne.compSimilarities();
      sne.prepMinimization();

      // Disable vsync
      window.enableVsync(false);
      
      // Perform minimization
      sne.compMinimization();
      // dh::util::CppTimer timer;
      // timer.tick();
      // for (uint i = 0; i < params.iterations; ++i) {
      //   sne.compIteration();
      // }
      // timer.tock();

      // Report timer
      // timer.poll();
      // dh::util::logTime(&logger, "Minimization time", timer.get<dh::util::TimerValue::eLast>());

      // Render embedding
      window.enableVsync(true);
      while (window.canDisplay()) {
        window.processEvents();
        renderer.render();
        window.display();
      }
    }

    dh::util::log(&logger, "Goodbye!");
  } catch (const std::exception& e) {
    std::cout << std::endl;
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}