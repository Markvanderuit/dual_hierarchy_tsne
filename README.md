# Dual-hierarchy t-SNE

![minimization](resources/misc/readme_header.png)

## Introduction

This repository contains a library and accompanying demo application implementing our dual-hierarchy acceleration of *t-distributed Stochastic Neighbor Embedding* ([t-SNE](https://lvdmaaten.github.io/tsne/)).

In short, our method accelerates the t-SNE minimization by generating a pair of spatial hierarchies; one over the embedding, and another over a discretization of the embedding's space.
We consider approximations of the interactions between these hierarchies, allowing us to significantly reduce the number of N-body computations performed.
Our method runs fully on the GPU using OpenGL/CUDA, and currently outperforms both the CUDA implementation of FIt-SNE as well as linear-complexity t-SNE
Finally, it scales to 3D embeddings as well.

For full details and performance comparisons, check out our recent [paper](...) "*An Efficient Dual-Hierarchy t-SNE Minimization*"!

## Compilation
First, ensure your system satisfies the following requirements:
* Compiler: C++17 support is required; we've tested with [MSVC](https://visualstudio.microsoft.com/) 19.6, [CLANG](https://clang.llvm.org/) 12, and [GCC](https://gcc.gnu.org/) 11.1 on Windows and Ubuntu.
* [CMake](https://cmake.org/): 3.21 or later is required.
* [CUDA](https://developer.nvidia.com/cuda-toolkit): 10.0 or later is required; other versions may work but are untested.
* [GLFW](https://www.glfw.org): while this is bundled through vcpkg, some Unix systems require development packages for it to work (e.g. `sudo apt install xorg-dev` for X11 Ubuntu). Please refer to their excellent [compilation page](https://www.glfw.org/docs/3.3/compile.html) or vcpkg's error messages if you run into issues!

Next, clone the repository and include the required submodules.

```bash
git clone --recurse-submodules https://github.com/Markvanderuit/dual_hierarchy_tsne
```

Finally, you should be able to generate a CMake project and compile it. For example, on an arbitrary Unix system:

```bash
  mkdir dual_hierarchy_tsne/build
  cd dual_hierarchy_tsne/build
  cmake ..
  make
```

During CMake configuration, [vcpkg](https://github.com/microsoft/vcpkg) pulls in a number of third-party dependencies. If you experience unexpected issues with any of these, please refer to their respective build instructions for troubleshooting.

## Usage

### Library
The CMake project provides three library build targets: *utils*, *vis*, and *sne*. The *utils* library contains utility and boilerplate code. The *vis* library contains rendering code for the demo application discussed below. The *sne* library contains the only parts that really matter.

Below is a minimal example showing its usage to minimize a small dataset:

```c++
#include <vector>
#include "dh/sne/params.hpp"
#include "dh/sne/sne.hpp"

int main() {
  using namespace dh::sne;

  // 1. Create Params object 
  //    For all parameters, refer to: include/dh/sne/params.hpp
  Params params;
  params.n = 60000;                   // Dataset size
  params.nHighDims = 784;             // Dataset dimensionality
  params.nLowDims = 2;                // Embedding dimensionality (2 or 3)
  params.iterations = 1000;           // Nr. minimization iterations
  params.perplexity = 30.0f;          // Perplexity parameter
  params.dualHierarchyTheta = 0.25f;  // Approximation parameter

  // 2. Create and load high-dimensional dataset
  //    I'm skipping dataset loading code for this example
  std::vector<float> dataset(params.n * params.nHighDims);

  // 3. Construct SNE object
  //    This manages all gpu objects, state, computation, etc.
  SNE sne(dataset, params);

  // 4. Perform similarities computation and minimization
  //    For expanded examples of using the SNE class, such as
  //    doing minimizations step-by-step, refer to:
  //    a. the SNE header: include/dh/sne/sne.hpp
  //    b. the demo application: src/app/sne_cmd.cpp
  sne.comp();

  // 5. Obtain KL-divergence and embedding data
  float kld = sne.klDivergence();
  std::vector<float> embedding = sne.embedding();

  return 0;
}
```

### Demo application
The demo application (build target: `sne_cmd`, file: `src/app/sne_cmd.cpp`) provides a command-line application which can run t-SNE on arbitrary datasets, if they are provided as raw binary data. It additionally allows for starting a tiny renderer (the `vis` library) that shows the embedding, minimization, and the used dual-hierarchies.


### Datasets
...

## Citation
Please cite the following paper if you have applied it in your research:

...

## License and third-party software
The source code in this repository is released under the MIT License. However, all used third-party software libraries are governed by their own respective licenes. Without the following libraries, this project would have been considerably harder:
* [cxxopts](https://github.com/jarro2783/cxxopts)
* [cub](https://github.com/NVIDIA/cub)
* [date](https://github.com/HowardHinnant/date)
* [faiss](https://github.com/facebookresearch/faiss)
* [glad](https://glad.dav1d.de/)
* [GLFW](https://www.glfw.org/)
* [GLM](https://glm.g-truc.net/0.9.9/)
* [indicators](https://github.com/p-ranav/indicators)
* [vcpkg](https://github.com/microsoft/vcpkg) 

## References
"...the **go to** statement should be abolished..." [[1]](#1).

<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.