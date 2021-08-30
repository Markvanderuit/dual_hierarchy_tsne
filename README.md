# Dual-hierarchy t-SNE
This repository contains a small library and demo application demonstrating a dual-hierarchy t-SNE implementation. For full details, see our paper "*An Efficient Dual-Hierarchy t-SNE Minimization*" ([journal](...), [preprint](...)).

In short, our method provides a faster minimization than FIt-SNE and linear-complexity t-SNE on 2D embeddings, and aditionally performs well on 3D embeddings. To achieve a speedup, we generate a pair of spatial hierarchies; one over the embedding, and another over the embedding's space. We consider approximations of the interactions between these hierarchies, allowing us to significantly reduce the number of N-body computations involved.

## Compilation
First, ensure your system satisfies the following requirements:
* Compiler: at least C++17 support is required; we've tested with [MSVC](https://visualstudio.microsoft.com/) 19.6 (Windows) and [GCC](https://gcc.gnu.org/) 11 (Ubuntu).
* [CMake](https://cmake.org/): version 3.21 or later is required.
* [CUDA](https://developer.nvidia.com/cuda-toolkit): version 10.0 or later is required; other versions may work but are untested.
* [OpenMP](https://www.openmp.org/): likely installed on your system or bundled with your compiler.
* [GLFW](https://www.glfw.org): while GLFW is bundled, some Unix systems require X11 development packages for it to work (e.g. `sudo apt install xorg-dev` on Ubuntu). If you have issues with compilation due to GLFW dependencies, please refer to their [compilation](https://www.glfw.org/docs/3.3/compile.html) page.

Next, clone the repository and make sure to include the required submodules.

```bash
git clone --recurse-submodules https://github.com/Markvanderuit/dual_hierarchy_tsne
```

Finally, you should be able to generate a CMake project and compile it. For example, on a Unix system:

```bash
  mkdir dual_hierarchy_tsne/build
  cd dual_hierarchy_tsne/build
  cmake ..
  make
```
During CMake configuration, [vcpkg](https://github.com/microsoft/vcpkg) is used to pull in a number of third-party dependencies. If you experience issues with compiling these, please refer to their respective build instructions for troubleshooting.

## Usage

### Library
The CMake project provides three library build targets: `utils`, `vis`, and `sne`. The `utils` library provides utility and boilerplate code. The `vis` library provides a renderer for the demo application. The `sne` library contains the only parts that really matter.

Below is an example showing its usage to minimize a small dataset:

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