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

# Tests

    add_subdirectory( tests )

# Install

    install( TARGETS esn
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib )
    install( FILES "source/esn.py"
        DESTINATION bin )
