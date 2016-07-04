# Dependencies

    find_package(PythonLibs)
    find_package(PythonInterp)
    find_package(SWIG)
    include(UseSWIG)

# Targets

    set_property(SOURCE esn.i PROPERTY CPLUSPLUS ON)
    swig_add_module(esn python esn.i)

# Includes

    target_include_directories(${SWIG_MODULE_esn_REAL_NAME}
        PRIVATE
            ${PYTHON_INCLUDE_DIRS}
    )

# Linking

    swig_link_libraries(esn esn)

# Installation

    install(TARGETS ${SWIG_MODULE_esn_REAL_NAME} DESTINATION lib)
    install(CODE
        "execute_process(
            COMMAND \"${PYTHON_EXECUTABLE}\"
                \"${CMAKE_CURRENT_SOURCE_DIR}/setup.py\"
                    build -b \"${CMAKE_CURRENT_BINARY_DIR}\" install
                    --prefix=${CMAKE_INSTALL_PREFIX}
            WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\"
        )"
    )
