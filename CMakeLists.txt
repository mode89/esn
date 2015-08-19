project( MASTER )
cmake_minimum_required( VERSION 3.0 )

set( CMAKE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}" )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_OUTPUT_DIRECTORY}/bin" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_OUTPUT_DIRECTORY}/lib" )

list( APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )

add_subdirectory( esn )
add_subdirectory( samples )
