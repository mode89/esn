include(ExternalProject)

ExternalProject_Add(eigen-project
    URL "http://bitbucket.org/eigen/eigen/get/3.2.5.tar.bz2"
    PREFIX eigen
    CONFIGURE_COMMAND echo Skip configuration
    BUILD_COMMAND echo Skip building
    INSTALL_COMMAND echo Skip installation
)

set(EIGEN3_INCLUDE_DIR "${CMAKE_BINARY_DIR}/eigen/src/eigen-project")
