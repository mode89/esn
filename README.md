# ESN library

ESN library implements simulation of [Echo State Networks].
* Echo State Network with non-spiking linear integrator neurons
* Online training
* Orthonormal weight matrix
* Uniformly distributed leaking rate
* Customizable connectivity between neurons
* Input/output scaling
* C/C++/Python
* Using [Eigen] library for linear algebra computation

[Echo State Networks]: <http://www.scholarpedia.org/article/Echo_state_network>

# Dependencies

* [CMake]. It's used for building the library.
* [Eigen]. By default, the build script relies on a version of Eigen
library installed to system. If you don't have Eigen library installed
on your system, you can disable CMake option `ESN_USE_SYSTEM_EIGEN`
and CMake will download Eigen library during build step.

[CMake]: <https://cmake.org/>
[Eigen]: <http://eigen.tuxfamily.org/>

# Installation

Create and proceed to a building directory
```sh
mkdir build
cd build
```
Configure
```sh
cmake <options> <path-to-esn-folder>
```
Available options:
* `ESN_USE_SYSTEM_EIGEN` :
if `ON` ESN library uses system installed version of Eigen library;
if `OFF` CMake download Eigen library during build step.

Build and install
```sh
cmake --build . --target install
```

# License

BSD 2-clause license
