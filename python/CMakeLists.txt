find_package( PythonInterp )

install( CODE "execute_process(
        COMMAND \"${PYTHON_EXECUTABLE}\"
            \"${CMAKE_CURRENT_SOURCE_DIR}/setup.py\"
                build -b \"${CMAKE_CURRENT_BINARY_DIR}\" install
                --prefix=${CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\"
    )"
)
