# Dual-hierarchy t-SNE
This repository contains a small library and demo application demonstrating a t-SNE implementation for our paper "*An Efficient Dual-Hierarchy t-SNE Minimization*" ([journal](...), [preprint](...)).

Our method obtains significant performance improvements over state-of-the-art techniques such as FIt-SNE and linear-complexity t-SNE for targeting 2D embeddings, and further remains performant when targeting 3D embeddings. To achieve a speedup, we generate a pair of spatial hierarchies; one over the embedding and another over the embedding's space. Considering the interactions between these hierarhices allows us to significantly reduce the number of N-body interactions required for the minimization.

# Usage

## Compilation
First, clone the repository to your system, being sure to include submodules.

```bash
git clone --recurse-submodules https://github.com/Markvanderuit/dual_hierarchy_tsne
```

Next, ensure your system satisfies the following requirements:

* Compiler: C++17 support required; tested with [MSVC](https://visualstudio.microsoft.com/) 19.6 (Windows) and [GCC](https://gcc.gnu.org/) 11 (Ubuntu Linux).
* [CMake](https://cmake.org/): version 3.21 or later required.
* [CUDA](https://developer.nvidia.com/cuda-toolkit): version 10.0 or later; other versions may work but are untested.
* [OpenMP](https://www.openmp.org/): likely installed on your system or bundled with your compiler.

On some Unix-like systems, certain dependencies may need to be installed for the bundled GLFW library. For example, on Ubuntu/Debian Linux, X11 development packages are needed:

```bash
  sudo apt install xorg-dev
```

Finally, if you have satistied the requirements, you should be able to generate a CMake project and compile it. For example, on a Unix system:

```bash
  mkdir dual_hierarchy_tsne/build
  cd dual_hierarchy_tsne/build
  cmake ..
  make
```

## Library usage
The CMake project consists of three libraries: `utils`, `sne`, `vis`. The `utils` library provides utility functions and wrapper code. The `vis` library provides rendering code for the demo application. The `sne` library contains the parts that really matter. An example showing its usage:

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

  // 2. Create and load dataset.
  //    (skipping dataset loading code for this example)
  std::vector<float> dataset(n * params.nHighDims);

  // 3. Create SNE object
  //    This is responsible for the full computation.
  SNE sne(dataset, params);

  // 3. Perform similarities computation and minimization
  //    For better examples of using the SNE class, such as
  //    doing minimizations step-by-step, refer to:
  //    a. the demo application: src/app/sne_cmd.cpp
  //    b. the SNE header: include/dh/sne/sne.hpp
  sne.comp();

  // 4. Obtain KL-divergence and embedding data
  float kld = sne.klDivergence();
  std::vector<float> embedding = sne.embedding();

  return 0;
}
```

## Demo usage
The demo application (build target: `sne_cmd`, file: `src/app/sne_cmd.cpp`) provides a command-line application which can run t-SNE on arbitrary datasets, if they are provided as raw binary data. It additionally allows for starting a tiny renderer (the `vis` library) that shows the embedding, minimization, and the used dual-hierarchies.



## Datasets
...

# Citation
Please cite the following paper if you have applied it in your research:

...

# License and third-party software
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