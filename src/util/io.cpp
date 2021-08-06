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

#include <iostream> 
#include <fstream>
#include <stdexcept>
#include "util/io.hpp"

namespace dh::util {
  void readBinFile(const std::string &fileName, 
                   std::vector<float> &data,
                   std::vector<uint> &labels, 
                   uint n,
                   uint d,
                   bool withLabels)
  {
    std::ifstream ifs(fileName, std::ios::in | std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("Input file cannot be accessed: " + fileName);
    }

    // Clear vectors and create space to store data in
    data = std::vector<float>(n * d);
    if (withLabels) {
      labels = std::vector<uint>(n);
    }

    // Read data, either in a single call (no labels) or intermittently (to extract labels)
    if (withLabels) {
      for (uint i = 0; i < n; ++i) {
        ifs.read((char *) &labels[i], sizeof(uint));
        ifs.read((char *) &data[d * i], d * sizeof(float));
      }
    } else {
      ifs.read((char *) data.data(), data.size() * sizeof(float));
    }
  }

  void writeBinFile(const std::string &fileName,
                    const std::vector<float> &data,
                    const std::vector<uint> &labels,
                    uint n,
                    uint d,
                    bool withLabels)
  {
    std::ofstream ofs(fileName, std::ios::out | std::ios::binary);
    if (!ofs) {
      throw std::runtime_error("Output file cannot be accessed: " + fileName);
    }

    if (withLabels) {
      for (uint i = 0; i < n; ++i) {
        ofs.write((char *) &labels[i], sizeof(uint));
        ofs.write((char *) &data[d * i], d * sizeof(float));
      }
    } else {
      ofs.write((char *) data.data(), data.size() * sizeof(float));
    }
  }

  
  void writeTextValuesFile(const std::string &fileName, const std::vector<std::string> &values)
  {
    std::ofstream ofs (fileName, std::ios::out);
    if (!ofs) {
      throw std::runtime_error("Output file cannot be accessed: " + fileName);
    }

    for (const auto &v : values) {
      ofs << v << '\n';
    }
  }
} // dh::util