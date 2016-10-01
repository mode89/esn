find_library(LAPACKE_LIBRARY lapacke)
find_path(LAPACKE_INCLUDE_DIR lapacke.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE DEFAULT_MSG
    LAPACKE_LIBRARY
    LAPACKE_INCLUDE_DIR)
mark_as_advanced(
    LAPACKE_LIBRARY
    LAPACKE_INCLUDE_DIR)
