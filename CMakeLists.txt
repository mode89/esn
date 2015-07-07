project( ESN )
cmake_minimum_required( VERSION 3.0 )

# Source files

    file( GLOB SRC_FILES source/*.cpp )

# Includes

    include_directories( include )
    include_directories( source )

# Flags

    if ( UNIX )
        add_compile_options( -std=c++11 )
    endif()

# Target

    add_library( esn ${SRC_FILES} )
