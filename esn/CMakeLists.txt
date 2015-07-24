project( ESN )
cmake_minimum_required( VERSION 3.0 )

# Dependencies

    find_package( Eigen3 )

# Source files

    file( GLOB SRC_FILES source/*.cpp )

# Includes

    include_directories( include )
    include_directories( source )
    include_directories( ${EIGEN3_INCLUDE_DIR} )

# Flags

    if ( UNIX )
        add_compile_options( -std=c++11 )
    endif()

# Target

    add_library( esn SHARED ${SRC_FILES} )

    if( NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY )
        set( OUTPUT_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}" )
    else()
        set( OUTPUT_DIR
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}" )
    endif()

    add_custom_command( TARGET esn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/python" "${OUTPUT_DIR}/esn" )

# Tests

    add_subdirectory( tests )
