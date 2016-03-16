message(STATUS "Building GTest...")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/gtest")
execute_process(
    COMMAND ${CMAKE_COMMAND} "${CMAKE_CURRENT_LIST_DIR}/gtest"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/gtest"
    OUTPUT_QUIET
)

execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}/gtest"
    OUTPUT_QUIET
)

set(GTEST_ROOT "${CMAKE_BINARY_DIR}/gtest/gtest-prefix/src/gtest")
list(APPEND CMAKE_LIBRARY_PATH
    "${CMAKE_BINARY_DIR}/gtest/gtest-prefix/src/gtest-build")
