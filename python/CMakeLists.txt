# Dependencies

    find_package(PythonInterp REQUIRED)
    find_package(SWIG REQUIRED)

# Variables

    set(ESN_WRAP_CPP esn_wrap.cpp)
    get_property(ESN_LIBRARY_DIR
        TARGET esn PROPERTY LIBRARY_OUTPUT_DIRECTORY)

# Targets

    add_custom_command(
        OUTPUT ${ESN_WRAP_CPP}
        COMMAND ${SWIG_EXECUTABLE} -python -c++ -outcurrentdir
            -I"${CMAKE_SOURCE_DIR}/include"
            -o ${ESN_WRAP_CPP}
            "${CMAKE_CURRENT_SOURCE_DIR}/esn.i"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/esn.i"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    add_custom_target(esn-python ALL
        DEPENDS ${ESN_WRAP_CPP}
    )

# Installation

    configure_file(setup.py.in setup.py)
    install(CODE
        "execute_process(
            COMMAND \"${PYTHON_EXECUTABLE}\" setup.py build install
                --prefix=${CMAKE_INSTALL_PREFIX}
            WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\"
        )"
    )
