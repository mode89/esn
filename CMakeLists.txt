project( MASTER )
cmake_minimum_required( VERSION 3.0 )

list( APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )

add_subdirectory( esn )
add_subdirectory( samples )
