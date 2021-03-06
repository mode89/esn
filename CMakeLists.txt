project( ESN )
cmake_minimum_required(VERSION 3.3)

# Options

    option(ESN_USE_SYSTEM_EIGEN "ESN library uses system Eigen library" OFF)
    option(ESN_USE_SYSTEM_GTEST "ESN library uses system GTest library" OFF)

# CMake configuration

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# External projects

    if(NOT ESN_USE_SYSTEM_EIGEN)
        include(Eigen)
    endif()

    if(NOT ESN_USE_SYSTEM_GTEST)
        include(GTest)
    endif()

# Find dependencies

    find_package(Eigen3)

# Source files

    file(GLOB SRC_FILES source/*.cpp)

# Includes

    include_directories(include)
    include_directories(source)

# Flags

    if (UNIX)
        add_compile_options(-std=c++11)
    endif()

# Target

    add_library(esn SHARED ${SRC_FILES})

# Linking

    target_link_libraries(esn ${EIGEN3_LIBRARY})

# Install

    install(TARGETS esn DESTINATION lib)

# Subdirectoris

    add_subdirectory(python)
    add_subdirectory(samples)
    add_subdirectory(tests)
