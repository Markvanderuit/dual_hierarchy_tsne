#include <iostream>
#include <dh/util/io.hpp>
#include <dh/util/logger.hpp>
#include <dh/util/gl/window.hpp>
#include <dh/sne/params.hpp>
#include <dh/sne/sne.hpp>

int main(int argc, char** argv) {
  using namespace dh;

  // Set up the library's internal logger to use standard output stream
  // If you want the library to shut up, skip this
  util::Logger::init(std::cout);
  
  // Ensure an OpenGL context is available. If you already have a context available, skip this.
  auto ctx = util::GLWindow::Offscreen();

  // Load some input similarities. If you already have IO code or are rolling, skip this 
  auto data = util::readBinFileNX("symmetricSims.bin");

  // Set up the runtime parameters you'd prefer
  // For the full list, refer to <dh/sne/params.hpp>
  sne::Params params = {
    .n          = 60000,
    .nHighDims  = 768,   // Note: only matters if you are providing vectors, not similarities
    .nLowDims   = 2,    
    .iterations = 1000,
    .perplexity = 30.f,
    .theta1     = 0.5f, // 0.5 at all times and you'll be fine
    .theta2     = 0.0f  // 0.0 for most data, 0.25 for some 2d data, 0.4 for some 3d data
  };
  
  // This function handles the entire tSNE computation, and outputs results
  // in the return object. If you want to configure what data it returns, modify
  // the result flags. If you want to have more precise control over your computation,
  // e.g. integrate it into your render loop, look at <dh/sne/sne.hpp> and <src/app/sne_cmd.cpp>
  // to see how the SNE object is actually used.
  sne::Result result = sne::run(data, params, sne::ResultFlags::eAll);

  // Print some information and write the embedding to file
  std::cout << "Time : " << result.totalTime << '\n'
            << "KLD  : " << result.klDivergence << std::endl;
  util::writeBinFileND("embedding.bin", result.embedding, params.n, params.nLowDims);

  return 0;
}