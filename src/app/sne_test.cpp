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
#include <ranges>
#include <random>
#include <vector>
#include <cxxopts.hpp>
#include "dh/types.hpp"
#include "dh/constants.hpp"
#include "dh/util/aligned.hpp"
#include "dh/util/enum.hpp"
#include "dh/util/io.hpp"
#include "dh/util/logger.hpp"
#include "dh/util/gl/error.hpp"
#include "dh/util/gl/timer.hpp"
#include "dh/util/gl/window.hpp"
#include "dh/util/cu/key_sort.cuh"
#include "dh/vis/renderer.hpp"
#include "dh/sne/sne.hpp"

using uint = unsigned int;

void test() {
  using namespace dh;

   // Set up logger to use standard output stream for demo
  util::Logger::init(std::cout);
  
  // Create OpenGL context (and accompanying invisible window/renderer)
  util::GLWindowInfo info;
  {
    using namespace util;
    info.title  = "";
    info.flags  = GLWindowInfo::bOffscreen;
  }
  util::GLWindow window(info);

  {
    constexpr uint n = 60'000;
      
    // Randomg nr. generator
    std::random_device rd;
    std::mt19937 g(rd());
    
    // Prepare data blocks
    std::vector<uint> input_data(n, 1), output_data(n);
    std::uniform_int_distribution<> distr(1, 32);
    std::ranges::generate(input_data, [&]() { return distr(g); });

    // Generate comparative data
    std::vector<uint> output_comparison(input_data);
    std::sort(range_iter(output_comparison));
    
    // Prepare buffers
    GLuint input_buffer, output_buffer, attached_buffer;
    glCreateBuffers(1, &input_buffer);
    glCreateBuffers(1, &output_buffer);
    glCreateBuffers(1, &attached_buffer);
    glNamedBufferStorage(input_buffer,    sizeof(uint) * input_data.size(), input_data.data(), 0);
    glNamedBufferStorage(output_buffer,   sizeof(uint) * input_data.size(), nullptr,           0);
    glNamedBufferStorage(attached_buffer, sizeof(uint) * input_data.size(), nullptr,           0);
    glAssert();

    // Preparing sort
    util::KeySort keySort(
      input_buffer, output_buffer, attached_buffer,
      n, 30
    );
    glAssert();

    util::GLTimer timer;
    for (uint i = 0; i < 1024; ++i) {
      glClearNamedBufferSubData(output_buffer, GL_R32UI, 0, sizeof(uint) * input_data.size(), GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      timer.tick();
      keySort.sort();
      timer.tock();
      timer.poll();
      std::cout << timer.get<util::TimerValue::eLast, std::chrono::microseconds>() << '\n';
    }
    glAssert();

    glGetNamedBufferSubData(output_buffer, 0, sizeof(uint) * output_data.size(), output_data.data());
    glAssert();
    
    /* for (uint i = 0; i < 128; ++i) {
      std::cout << i << " : " << input_data[i] << '\t' << output_data[i] << "\t" << output_comparison[i] << '\n';
    } */

    std::cout 
      << "Equal: " << std::equal(range_iter(output_data), output_comparison.begin()) << '\n'
      << "Took "   << timer.get() << std::endl;
  }
}


int main(int argc, char** argv) {
  try {
    test();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}