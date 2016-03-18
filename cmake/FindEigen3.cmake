include(FindPackageHandleStandardArgs)

if(NOT EIGEN3_INCLUDE_DIR)
    find_path(EIGEN3_INCLUDE_DIR Eigen
        PATH_SUFFIXES eigen3
    )
endif()

add_library(eigen INTERFACE)

set(EIGEN3_LIBRARY eigen)
set_property(TARGET eigen APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${EIGEN3_INCLUDE_DIR})

if(NOT ESN_USE_SYSTEM_EIGEN)
    add_dependencies(eigen eigen-project)
endif()

find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR)
mark_as_advanced(EIGEN3_INCLUDE_DIR)
