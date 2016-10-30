project( ESN )
cmake_minimum_required(VERSION 3.3)

# Options

    option(ESN_USE_SYSTEM_GTEST "ESN library uses system GTest library" OFF)
    option(ESN_MAKE_PYTHON_BINDING "Build Python binding module" OFF)
    option(ESN_BUILD_TESTS "Build ESN tests" ON)
    set(ESN_MATH "" CACHE STRING "ESN library math engine")

# CMake configuration

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# External projects

    if(ESN_BUILD_TESTS AND NOT ESN_USE_SYSTEM_GTEST)
        include(GTest)
    endif()

# Dependencies

    if(ESN_MATH STREQUAL "")
        message(FATAL_ERROR "ESN library math engine is not specified.")
    endif()
    add_subdirectory("source/esn/${ESN_MATH}")

# Includes

    include_directories(include)
    include_directories(source)

# Flags

    if (UNIX)
        add_compile_options(-std=c++11)
    endif()

# Target

    add_library(esn SHARED
        source/esn/adaptive_filter_rls.cpp
        source/esn/network_impl.cpp
        source/esn/trainer_impl.cpp
    )

# Linking

    target_link_libraries(esn
        esn-math
    )

# Install

    install(TARGETS esn DESTINATION lib)

# Subdirectoris

    if(ESN_MAKE_PYTHON_BINDING)
        add_subdirectory(python)
    endif()

    add_subdirectory(samples)

    if(ESN_BUILD_TESTS)
        add_subdirectory(tests)
    endif()
