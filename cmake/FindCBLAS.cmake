find_library(CBLAS_LIBRARY cblas)
find_path(CBLAS_INCLUDE_DIR cblas.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS DEFAULT_MSG
    CBLAS_LIBRARY
    CBLAS_INCLUDE_DIR)
mark_as_advanced(
    CBLAS_LIBRARY
    CBLAS_INCLUDE_DIR)
