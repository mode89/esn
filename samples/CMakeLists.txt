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

# Deploy all python samples

    file( GLOB PYTHON_FILES *.py )
    foreach( file IN ITEMS ${PYTHON_FILES} )
        get_filename_component( fileName ${file} NAME )
        set( outputFile "${OUTPUT_DIR}/${fileName}" )
        list( APPEND OUTPUT_PYTHON_FILES "${outputFile}" )
        add_custom_command( OUTPUT "${outputFile}"
            COMMAND ${CMAKE_COMMAND} -E copy "${file}" "${outputFile}"
            DEPENDS "${file}"
            COMMENT "Deploy ${fileName}"
        )
    endforeach()
    add_custom_target( samples-python ALL DEPENDS ${OUTPUT_PYTHON_FILES} )
