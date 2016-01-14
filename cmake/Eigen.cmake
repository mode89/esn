include(ExternalProject)

ExternalProject_Add(eigen
    URL "http://bitbucket.org/eigen/eigen/get/3.2.5.tar.bz2"
    CONFIGURE_COMMAND echo Skip configuration
    BUILD_COMMAND echo Skip building
    INSTALL_COMMAND echo Skip installation
)
