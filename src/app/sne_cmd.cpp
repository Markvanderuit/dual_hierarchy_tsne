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
#include <cxxopts.hpp>
#include "dh/util/io.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/window.hpp"
#include "dh/vis/renderer.hpp"
#include "dh/sne/sne.hpp"

using uint = unsigned int;

// Constants
const std::string progDescr = "Demo application demonstrating dual-hierarchy t-SNE minimization.";
const std::string windowTitle = "Dual-Hierarchy t-SNE Demo";
constexpr uint windowWidth = 1536;
constexpr uint windowHeight = 1024;

// I/O and SNE parameters, set by cli(...)
std::string iptFilename;
std::string optFilename;
dh::sne::Params params;

// Program parameters, set by cli(...)
bool progDoKlDivergence = false;
bool progDoLabels       = false;
bool progDoVisDuring    = false;
bool progDoVisAfter     = false;
bool progDoSimLoad      = false;

void cli(int argc, char** argv) {
  // Configure command line options
  cxxopts::Options options("sne_cmd", progDescr);
  options.add_options()
    // Required arguments
    ("iptFilename", "Input data file (required)", cxxopts::value<std::string>())
    ("nPoints", "number of data points (required)", cxxopts::value<uint>())
    ("nHighDims", "number of input dims (required)", cxxopts::value<uint>())
    ("nLowDims", "number of output dims (required)", cxxopts::value<uint>())
    
    // Optional minimization arguments
    ("o,optFilename", "Output data file (default: none)", cxxopts::value<std::string>())
    ("p,perplexity", "Perplexity parameter (default: 30)", cxxopts::value<float>())
    ("i,iterations", "Number of minimization steps (default: 1000)", cxxopts::value<uint>())

    // Optional approximations arguments
    ("theta1", "Emb.  hierarchy approximation parameter (default: 0.0)", cxxopts::value<float>())
    ("theta2", "Field hierarchy approximation parameter (default: 0.0)", cxxopts::value<float>())

    // Optional program arguments
    ("lbl", "Input data file contains label data", cxxopts::value<bool>())
    ("kld", "Compute KL-Divergence", cxxopts::value<bool>())
    ("visDuring", "Visualize embedding during/after minimization", cxxopts::value<bool>())
    ("visAfter", "Visualize embedding after minimization", cxxopts::value<bool>())
    ("sim", "Input file provides similarities", cxxopts::value<bool>())
    ("h,help", "Print this help message and exit");

  options.parse_positional({"iptFilename", "nPoints", "nHighDims", "nLowDims"});
  options.positional_help("<iptFilename> <n> <nHighDims> <nLowDims>");
  auto result = options.parse(argc, argv);

  // Output help message as requested
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }

  // Check for required arguments, output help message otherwise
  if (!result.count("iptFilename") || !result.count("nPoints") 
      || !result.count("nHighDims") || !result.count("nLowDims")) {
    std::cout << options.help() << std::endl;
    std::exit(0);
  }

  // Parse required arguments
  iptFilename      = result["iptFilename"].as<std::string>();
  params.n         = result["nPoints"].as<uint>();
  params.nHighDims = result["nHighDims"].as<uint>();
  params.nLowDims  = result["nLowDims"].as<uint>();

  // Check for and parse optional arguments
  if (result.count("optFilename")) { optFilename = result["optFilename"].as<std::string>(); }
  if (result.count("perplexity")) { params.perplexity = result["perplexity"].as<float>(); }
  if (result.count("iterations")) { params.iterations = result["iterations"].as<uint>(); }
  if (result.count("theta1")) { params.singleHierarchyTheta = result["theta1"].as<float>(); }
  if (result.count("theta2")) { params.dualHierarchyTheta = result["theta2"].as<float>(); }
  if (result.count("kld")) { progDoKlDivergence = true; }
  if (result.count("lbl")) { progDoLabels = true; }
  if (result.count("sim")) { progDoSimLoad = true; }
  if (result.count("visDuring")) { progDoVisDuring = true; }
  if (result.count("visAfter")) { progDoVisAfter = true; }
}


/*{ // Begin scope
    using namespace dh;

    // Optional components
    util::Logger::init(std::cout);          // Forward an output stream to DH-SNE's logger
    auto ctx = util::GLWindow::Offscreen(); // Establish OpenGL context if you don't have one

    // Load similarity data
    std::vector<util::NXBlock> data;
    util::readBinFileNX("<path/to/file.bin>", data);

    // Initialize t-SNE object with the right parameters;
    // See 'include/dh/sne/params.hpp' for the full list
    sne::SNE obj(data, {
      .n                    = 60000, // n
      .nHighDims            = 768,   // d
      .nLowDims             = 2,     // or 3; shaders statically dispatched to 2/3
      .iterations           = 1000,
      .perplexity           = 30.f,
      .singleHierarchyTheta = 0.5f,  // 0.5 is a good maximum, don't go over
      .dualHierarchyTheta   = 0.0f,  // overkill and too much error in 2d
    });

    // (1) Either run the entire t-SNE computation and get coffee
    obj.comp();
    // (2) Or set up your own render loop and do other stuff as well
    // sne.compSimilarities();
    // for (uint i = 0; i < 1000; ++i) {
    //   sne.compMinimizationStep();
    //   <other stuff here...?>
    // }

    // Obtain KL-divergence and embedding data once you're done
    // Note: if you want gpu embedding data during minimization,
    // that uhh... can be hacked in with some work
    float kld = sne.klDivergence();
    std::vector<float> embedding = sne.embedding();
  } //End scope */

void sne() {
   // Set up logger to use standard output stream for demo
  dh::util::Logger::init(std::cout);
  
  // Create OpenGL context (and accompanying invisible window/renderer)
  dh::util::GLWindowInfo info;
  {
    using namespace dh::util;
    info.title  = windowTitle;
    info.width  = windowWidth;
    info.height = windowHeight;
    info.flags  = GLWindowInfo::bDecorated | GLWindowInfo::bFocused 
                | GLWindowInfo::bSRGB | GLWindowInfo::bResizable
                | GLWindowInfo::bOffscreen;
  }
  dh::util::GLWindow window(info);

  // Runtime components
  dh::sne::SNE sne;
  dh::vis::Renderer renderer;

  // Data objects
  std::vector<uint> labels;
  std::vector<float> data;
  std::vector<dh::util::NXBlock> sim_data;

  if (progDoSimLoad) {
    // Load similarity dataset, set labels to 0
    dh::util::readBinFileNX(iptFilename, sim_data);
    labels = std::vector<uint>(params.n, 0);
    
    // Setup components
    sne      = dh::sne::SNE(sim_data, params);
    renderer = dh::vis::Renderer(window, params, labels);
  } else {
    // Load original dataset and labels
    dh::util::readBinFileND(iptFilename, data, labels, params.n, params.nHighDims, progDoLabels);

    // Setup components
    sne      = dh::sne::SNE(data, params);
    renderer = dh::vis::Renderer(window, params, labels);
  }

  // If visualization is requested, minimize and render at the same time
  if (progDoVisDuring) {
    sne.compSimilarities();

    // Spawn window, disabling vsync so the minimization is not locked to swap interval
    window.setVsync(false);
    window.setVisible(true);
    window.display();

    // Render loop, one minimization step between frames
    for (uint i = 0; i < params.iterations; ++i) {
      sne.compMinimizationStep();
      renderer.render();
      window.display();
      window.processEvents();
    }
  } else {
    sne.comp();
  }

  // If requested, compute and output KL-divergence (might take a while on large datasets)
  if (progDoKlDivergence) {
    dh::util::Logger::newl() << "KL-divergence : " << sne.klDivergence();
  }

  // Output timings
  dh::util::Logger::newl() << "Similarities runtime : " << sne.similaritiesTime();
  dh::util::Logger::newl() << "Minimization runtime : " << sne.minimizationTime();

  // If requested, output embedding to file 
  if (!optFilename.empty()) {
    dh::util::writeBinFileND(optFilename, sne.embedding(), labels, params.n, params.nLowDims, progDoLabels);
  }

  // If requested, run visualization after minimization is completed
  if (progDoVisDuring || progDoVisAfter) {
    // Spawn window
    window.setVsync(true);
    window.setVisible(true);
    window.display();

    // Simple render loop
    while (window.canDisplay()) {
      window.processEvents();
      renderer.render();
      window.display();
    }
  }
}

int main(int argc, char** argv) {
  try {
    cli(argc, argv);
    sne();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}