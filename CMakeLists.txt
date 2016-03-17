project( ESN )
cmake_minimum_required( VERSION 3.0 )

# Options

    option(ESN_USE_SYSTEM_EIGEN "ESN library uses system Eigen library" ON)
    option(ESN_USE_SYSTEM_GTEST "ESN library uses system GTest library" OFF)

# CMake configuration

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Find dependencies

    if(NOT ESN_USE_SYSTEM_EIGEN)
        include(Eigen)
    endif()

    find_package(Eigen3)

# Source files

    file(GLOB SRC_FILES source/*.cpp)

# Includes

    include_directories(include)
    include_directories(source)
    include_directories(${EIGEN3_INCLUDE_DIR})

# Flags

    if (UNIX)
        add_compile_options(-std=c++11)
    endif()

# Target

    add_library(esn SHARED ${SRC_FILES})

# Install

    install( TARGETS esn LIBRARY DESTINATION lib )

# Subdirectoris

    add_subdirectory(python)
    add_subdirectory(samples)
    add_subdirectory(tests)
