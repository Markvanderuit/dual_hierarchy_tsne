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
#include "dh/util/io.hpp"
#include "dh/util/logger.hpp"

namespace dh::util {
  // Logging shorthands
  const std::string prefix = genLoggerPrefix("[KLDivergence]");

  void readBinFileND(const std::string &fileName, 
                     std::vector<float> &data,
                     std::vector<uint> &labels, 
                     uint n,
                     uint d,
                     bool withLabels) {
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

  void writeBinFileND(const std::string &fileName,
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

  void readBinFileNX(const std::string &fileName, std::vector<NXBlock> &out) {
    // Attempt to open file stream to provided file at file's end
    std::ifstream ifs(fileName, std::ios::in | std::ios::ate | std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("Input file cannot be accessed: " + fileName);
    }

    // Read file size, then set character position to begin
    size_t file_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);

    // Read nr. of vectors as first 8 bytes
    size_t n;
    ifs.read((char *) &n, sizeof(size_t));

    // Read remainder of file into temporary buffer and close file stream 
    // after as we now operate on this buffer only
    std::vector<std::byte> buffer(file_size - ifs.tellg());
    ifs.read((char *) buffer.data(), buffer.size());
    ifs.close();

    // Read block region layout into secondary buffers
    std::vector<size_t> blockSizeB(n, sizeof(NXPair)), 
                        blockOffsB(n, sizeof(size_t));
    for (size_t i = 0, b = 0; i < n && b < buffer.size(); ++i) {
      // Read vector length as first 8 bytes, i.e. the size of the block
      size_t d;
      std::memcpy(&d, buffer.data() + b, sizeof(size_t));

      // Store byte size, and build exclusive prefix sum to start of block
      blockSizeB[i] *= d;
      blockOffsB[i] += ((i == 0) ? 0 : (blockOffsB[i - 1] + blockSizeB[i - 1]));
      
      // advance to next block start
      b += sizeof(size_t) + blockSizeB[i]; 
    }

    // Perform multithreaded copy into output blocks
    out.resize(n);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      // Get current block
      size_t sizeb = blockSizeB[i];
      size_t offsb = blockOffsB[i];
      auto & block = out[i];
      
      // Copy appropriate region data from buffer into block
      block.resize(sizeb / sizeof(NXPair));
      std::memcpy(block.data(), buffer.data() + offsb, sizeb);
    }
  }

  void readBinFileNXOld(const std::string &fileName, std::vector<NXBlock> &out) {
    // Attempt to open file stream to provided file at file's end
    std::ifstream ifs(fileName, std::ios::in | std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("Input file cannot be accessed: " + fileName);
    }
    
    // Read input length as first unit
    size_t n;
    ifs.read((char*) &n, sizeof(decltype(n)));
    out.resize(n);

    for (int i = 0; i < n; i++) {
      size_t d;
      ifs.read((char*) &d, sizeof(decltype(d)));
      out[i].resize(d);

      for (int j = 0; j < d; j++) {
        uint32_t id;
        float transVal;
        ifs.read((char*)&id, sizeof(decltype(id)));
        ifs.read((char*)&transVal, sizeof(decltype(transVal)));
        out[i][j] = { id , transVal };
      }
    }

    ifs.close();
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