# Dual-hierarchy t-SNE

![minimization](resources/misc/readme_header.png)

## Introduction

This repository contains a library and accompanying demo application implementing our dual-hierarchy acceleration of *t-distributed Stochastic Neighbor Embedding* ([t-SNE](https://lvdmaaten.github.io/tsne/)).

In short, our method accelerates the t-SNE minimization by generating a pair of spatial hierarchies; one over the embedding, and another over a discretization of the embedding's space.
We consider approximations of the interactions between these hierarchies, allowing us to significantly reduce the number of N-body computations performed.
Our method runs fully on the GPU using OpenGL/CUDA, and currently outperforms both the CUDA implementation of FIt-SNE as well as linear-complexity t-SNE
Finally, it scales to 3D embeddings as well.

For full details and performance comparisons, check out our recent [paper](...) "*An Efficient Dual-Hierarchy t-SNE Minimization*"!

## Prerequisites
Ensure your system satisfies the following requirements:
* A compiler with C++17 support. Tested with MSVC 19.6, Clang 12, GCC 11.1
* CMake 3.21 or later.
* CUDA 10 or later.

All other dependencies are bundled through vcpkg. Note that, for [GLFW](https://www.glfw.org), some Unix systems may require X11/Wayland development packages to be installed. Vcpkg should provide sufficient information for you to install these, but if you run into issues, refer to the [GLFW compilation page](https://www.glfw.org/docs/3.3/compile.html).

## Compilation
First, clone the repository and include the required submodules.

```bash
git clone --recurse-submodules https://github.com/Markvanderuit/dual_hierarchy_tsne
```

Next, you should be able to generate a project and compile it. For example, by invoking CMake:

```bash
  mkdir dual_hierarchy_tsne/build
  cd dual_hierarchy_tsne/build
  cmake ..
  make
```

## Usage

**Library**

The CMake project provides three library build targets: *utils*, *vis*, and *sne*. The *utils* library contains utility and boilerplate code. The optional *vis* library contains rendering code for the demo application discussed below. The *sne* library contains the only parts that really matter.

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

**Demo application**

The demo (build target: `sne_cmd`, file: `src/app/sne_cmd.cpp`) provides a command-line application which can run t-SNE on arbitrary datasets, if they are provided as raw binary data. It additionally allows for starting a tiny renderer (the `vis` library) that shows the embedding, minimization, and the used dual-hierarchies.

**Datasets**

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