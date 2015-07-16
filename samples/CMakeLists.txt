project( ESN_SAMPLES )
cmake_minimum_required( VERSION 3.0 )

if( UNIX )
    add_compile_options( -std=c++11 )
endif()

find_package( ESN REQUIRED )
include_directories( ${ESN_INCLUDE_DIRS} )

add_executable( pulse pulse.cpp )
target_link_libraries( pulse ${ESN_LIBRARIES} )

add_executable( sine sine.cpp )
target_link_libraries( sine ${ESN_LIBRARIES} )
