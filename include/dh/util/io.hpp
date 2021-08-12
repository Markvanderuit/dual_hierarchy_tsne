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

#pragma once

#include <string>
#include <vector>
#include "dh/types.hpp"

namespace dh::util {
 /**
   * readBinFile(...)
   * 
   * Read a binary data file and interpret it as N D-dimensional vectors. Should the data file
   * contain labels for each vector, these can be read assuming they are stored as 32 bit uints.
   */
  void readBinFile(const std::string &fileName, 
                   std::vector<float> &data,
                   std::vector<uint> &labels, 
                   uint n,
                   uint d,
                   bool withLabels = false);
  
  /**
   * writeBinFile(...)
   * 
   * Write a binary data file, storing N D-dimensional vectors. Should labels be available for
   * each vector, these can be interleaved with the data as 32 bit uints.
   */
  void writeBinFile(const std::string &fileName,
                    const std::vector<float> &data,
                    const std::vector<uint> &labels,
                    uint n,
                    uint d,
                    bool withLabels = false);

  /**
   * writeTextValuesFile
   * 
   * Write a text file, storing specified string values. Useful for outputting time/kl divergence.
   */
  void writeTextValuesFile(const std::string &fileName,
                           const std::vector<std::string> &values);
} // dh::util