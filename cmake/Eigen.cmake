message(STATUS "Building Eigen...")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/eigen")
execute_process(
    COMMAND ${CMAKE_COMMAND} "${CMAKE_CURRENT_LIST_DIR}/eigen"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/eigen"
    OUTPUT_QUIET
)

execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}/eigen"
    OUTPUT_QUIET
)

set(EIGEN3_INCLUDE_DIR "${CMAKE_BINARY_DIR}/eigen/eigen-prefix/src/eigen")
