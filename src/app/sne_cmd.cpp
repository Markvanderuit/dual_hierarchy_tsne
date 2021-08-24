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
#include "dh/util/io.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/timer.hpp"
#include "dh/util/gl/window.hpp"
#include "dh/vis/renderer.hpp"
#include "dh/sne/sne.hpp"

using uint = unsigned int;

int main(int argc, char** argv) {
  try {
    dh::util::DateCoutLogger logger;

    // Set example parameters
    dh::sne::Params params;
    params.n = 60000;
    params.nHighDims = 784;
    params.nLowDims = 2;
    params.perplexity = 30;
    params.singleHierarchyTheta = 0.5;
    params.dualHierarchyTheta = 0.4;

    // Load example dataset
    const std::string inputFileName = "C:/users/markv/documents/repositories/dual_hierarchy_tsne/resources/data/mnist_labeled_60k_784d.bin";
    std::vector<float> data;
    std::vector<uint> labels;
    dh::util::readBinFile(inputFileName, data, labels, params.n, params.nHighDims, true);

    // Create window (and OpenGL context)
    dh::util::GLWindowInfo info;
    {
      // if only c++20's 'using enum' was a thing already
      using namespace dh::util;
      info.flags = GLWindowInfo::bDecorated 
                 | GLWindowInfo::bFocused 
                 | GLWindowInfo::bSRGB 
                 | GLWindowInfo::bResizable
                 | GLWindowInfo::bOffscreen;
      info.width = 1536;
      info.height = 1024;
      info.title = "sne_cmd";
    }
    dh::util::GLWindow window(info);

    // Create renderer and tsne runner
    dh::vis::Renderer renderer(window, params, labels);    
    dh::sne::SNE sne(data, params, &logger);

    // Run minimization AND render result
    {
      // Do similarity computation
      sne.compSimilarities();

      // Prep window
      window.setVsync(false);
      window.setVisible(true);
      window.display();

      // Set up cpu-side timer
      dh::util::ChronoTimer timer;
      timer.tick();
      
      // Minimize step-by-step while also rendering
      for (uint i = 0; i < params.iterations; ++i) {
        window.processEvents();
        sne.compMinimizationStep();
        renderer.render();
        window.display();
      }

      // Record timer values
      timer.tock();
      timer.poll();

      // Output kl divergence and runtime
      dh::util::logValue(&logger, "[SNE] KL-Divergence", sne.klDivergence());
      dh::util::logTime(&logger, "[SNE] Minimization time", sne.minimizationTime());
    }

    /* // Run minimization, THEN render result
    {
      // Do similarity computation
      sne.comp();

      // Output kl divergence
      dh::util::logValue(&logger, "[SNE] KL-Divergence", sne.klDivergence());
      dh::util::logTime(&logger, "[SNE] Minimization time", sne.minimizationTime());
    } */

    // Render loop post-minimization
    window.setVsync(true);
    window.setVisible(true);
    window.display();
    while (window.canDisplay()) {
      window.processEvents();
      renderer.render();
      window.display();
    }

    dh::util::log(&logger, "Goodbye!");
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}