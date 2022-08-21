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
bool progDoEmbeddingOut = false;
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
  if (result.count("theta1")) { params.theta1 = result["theta1"].as<float>(); }
  if (result.count("theta2")) { params.theta2 = result["theta2"].as<float>(); }
  if (result.count("kld")) { progDoKlDivergence = true; }
  if (result.count("lbl")) { progDoLabels = true; }
  if (result.count("sim")) { progDoSimLoad = true; }
  if (result.count("visDuring")) { progDoVisDuring = true; }
  if (result.count("visAfter")) { progDoVisAfter = true; }
  progDoEmbeddingOut = !optFilename.empty();
}

void sne() {
  using namespace dh;

   // Set up logger to use standard output stream for demo
  util::Logger::init(std::cout);
  
  // Create OpenGL context (and accompanying invisible window/renderer)
  util::GLWindowInfo info;
  {
    using namespace util;
    info.title  = windowTitle;
    info.width  = windowWidth;
    info.height = windowHeight;
    info.flags  = GLWindowInfo::bDecorated | GLWindowInfo::bFocused 
                | GLWindowInfo::bSRGB | GLWindowInfo::bResizable
                | GLWindowInfo::bOffscreen;
  }
  util::GLWindow window(info);

  // Runtime components
  sne::SNE sne;
  vis::Renderer renderer;

  // Data objects
  std::vector<uint> labels;
  std::vector<float> data;
  std::vector<dh::util::NXBlock> sim_data;

  if (progDoSimLoad) {
    // Load similarity dataset, set labels to 0
    util::readBinFileNX(iptFilename, sim_data);
    labels = std::vector<uint>(params.n, 0);
    
    // Setup components
    sne      = sne::SNE(sim_data, params);
    renderer = vis::Renderer(window, params, labels);
  } else {
    // Load original dataset and labels
    util::readBinFileND(iptFilename, data, labels, params.n, params.nHighDims, progDoLabels);

    // Setup components
    sne      = sne::SNE(data, params);
    renderer = vis::Renderer(window, params, labels);
  }

  // If visualization is requested, minimize and render at the same time
  if (progDoVisDuring) {
    sne.runSimilarities();

    // Spawn window, disabling vsync so the minimization is not locked to swap interval
    window.setVsync(false);
    window.setVisible(true);
    window.display();

    // Render loop, one minimization step between frames
    for (uint i = 0; i < params.iterations; ++i) {
      sne.runMinimizationStep();
      renderer.render();
      window.display();
      window.processEvents();
    }
  } else {
    sne.run();
  }

  // Obtain required results (KL-divergence might be slow on large datasets)
  auto resultFlags = sne::ResultFlags::eTimings |
    (progDoKlDivergence ? sne::ResultFlags::eKLDivergence : sne::ResultFlags::eNone) |
    (progDoEmbeddingOut ? sne::ResultFlags::eEmbedding    : sne::ResultFlags::eNone);
  auto result = sne.getResult(resultFlags);

  // Output KL-divergence
  if (progDoKlDivergence) {
    util::Logger::newl() << "KL-divergence        : " << result.klDivergence;
  }

  // Output timings
  util::Logger::newl() << "Similarities runtime : " << result.similaritiesTime;
  util::Logger::newl() << "Minimization runtime : " << result.minimizationTime;
  util::Logger::newl() << "Total runtime        : " << result.totalTime;

  // If requested, output embedding to file 
  if (!optFilename.empty()) {
    util::writeBinFileND(optFilename, result.embedding, labels, params.n, params.nLowDims, progDoLabels);
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