project( TESTS )
cmake_minimum_required( VERSION 3.0 )

# Dependencies

    find_package(GTest)

file( GLOB SRC_FILES *.cpp )
add_executable( esn-tests ${SRC_FILES} )
target_link_libraries( esn-tests esn )

target_link_libraries( esn-tests ${GTEST_LIBRARIES} )
include_directories( ${GTEST_INCLUDE_DIRS} )
