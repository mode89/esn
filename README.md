# ESN library

ESN library implements simulation of [Echo State Networks].
* Echo State Network with non-spiking linear integrator neurons
* Online training
* Orthonormal weight matrix
* Uniformly distributed leaking rate
* Customizable connectivity between neurons
* Input/output scaling
* C/C++/Python
* Linux/Windows
* Using [Eigen] library for linear algebra computation

[Echo State Networks]: <http://www.scholarpedia.org/article/Echo_state_network>

# Dependencies

* [CMake] 3.3. It's used for building the library.
* [Eigen]. By default, the build script downloads Eigens library during
configuration step. This behavior can be overriden by CMake options.
* [Google Test]. It's used for testing ESN library. By default, the build
script downloads Google Test during configuration step. This behavior
can be overriden by CMake options.

[CMake]: <https://cmake.org/>
[Eigen]: <http://eigen.tuxfamily.org/>
[Google Test]: <https://github.com/google/googletest/>

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
Available CMake options:
* `ESN_USE_SYSTEM_EIGEN` : Default is `OFF`.
if `ON`, ESN library uses system installed version of Eigen library;
if `OFF`, CMake downloads Eigen library during configuration step.
* `ESN_USE_SYSTEM_GTEST` : Default is `OFF`.
If `ON`, ESN library uses system installed version of Google Test library;
If `OFF`, CMake downloads Google Test library during configuration step.

Build and install
```sh
cmake --build . --target install
```

# TODO
* More tests
* More samples
* Saving/loading network

# License

BSD 2-clause license
