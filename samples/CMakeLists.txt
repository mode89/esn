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

if( NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY )
    set( OUTPUT_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}" )
else()
    set( OUTPUT_DIR
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" )
endif()

set( CONFIG_PY "${OUTPUT_DIR}/config.py" )
add_custom_command( OUTPUT ${CONFIG_PY}
    COMMAND ${CMAKE_COMMAND} -E
        echo "ESN_LIB = '$<TARGET_FILE:esn>'" > ${CONFIG_PY} )

set( SINE_PY "${OUTPUT_DIR}/sine.py" )
add_custom_command( OUTPUT ${SINE_PY}
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/sine.py"
        ${SINE_PY}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/sine.py" ${CONFIG_PY} )

add_custom_target( sine_py ALL DEPENDS ${SINE_PY} )
