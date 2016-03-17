include(ExternalProject)

ExternalProject_Add(gtest-project
    URL "https://github.com/google/googletest/archive/release-1.7.0.zip"
    PREFIX "${CMAKE_BINARY_DIR}/gtest"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        -DBUILD_SHARED_LIBS=ON
    INSTALL_COMMAND echo Skip installation
)
