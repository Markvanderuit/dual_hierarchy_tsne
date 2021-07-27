#include <cstdlib>
#include <iostream>
#include <resource_embed/resource_embed.hpp>
#include "util/timer.hpp"
#include <string>

int main(int argc, char** argv) {
  tsne::CppTimer timer;
  timer.init();
  timer.tick();
  std::cout << "Hello world" << std::endl;
  timer.tock();
  timer.record();

  std::cout << std::to_string(timer.get<tsne::Timer::OutputValue::eTotal, std::chrono::microseconds>().count()) << '\n';
  return EXIT_SUCCESS;
}