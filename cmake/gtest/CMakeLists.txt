cmake_minimum_required(VERSION 3.0)

include(ExternalProject)

ExternalProject_Add(gtest
    URL "https://github.com/google/googletest/archive/release-1.7.0.zip"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=ON
    INSTALL_COMMAND echo Skip installation
)
