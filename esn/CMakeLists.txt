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

    if( ESN_DEVELOP )
        set( ESN_PY_DLL_PATH \\"$<TARGET_FILE:esn>\\" )
    else()
        set( ESN_PY_DLL_PATH find_library( \\"esn\\" ) )
    endif()

    set( SRC_ESN_PY "${CMAKE_CURRENT_SOURCE_DIR}/source/esn.py" )
    set( DST_ESN_PY "${OUTPUT_DIR}/esn.py" )
    set( CONFIGURE_FILE_CMAKE
        "${CMAKE_CURRENT_SOURCE_DIR}/configure_file.cmake" )
    add_custom_command( OUTPUT "${DST_ESN_PY}"
        COMMAND ${CMAKE_COMMAND}
            -DSRC_FILE="${SRC_ESN_PY}" -DDST_FILE="${DST_ESN_PY}"
            -DESN_PY_DLL_PATH="${ESN_PY_DLL_PATH}"
            -P "${CONFIGURE_FILE_CMAKE}"
        DEPENDS
            "${SRC_ESN_PY}"
            "${CONFIGURE_FILE_CMAKE}"
        COMMENT "Deploy python module"
    )
    add_custom_target( esn-python ALL DEPENDS "${DST_ESN_PY}" )

# Tests

    add_subdirectory( tests )

# Install

    install( TARGETS esn
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib )
    install( FILES "source/esn.py"
        DESTINATION bin )
