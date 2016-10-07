# Dependencies

    find_package(CBLAS REQUIRED)
    find_package(CUDA REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(LAPACKE REQUIRED)

# Targets

    add_library(esn-math
        math.cpp
        pointer.cpp
    )

# Flags

    if (UNIX)
        target_compile_options(esn-math
            PRIVATE
                -fPIC
                -std=c++11
        )
    endif()

# Includes

    target_include_directories(esn-math
        PRIVATE
            ../../
            ${CBLAS_INCLUDE_DIR}
            ${CUDA_INCLUDE_DIRS}
            ${LAPACKE_INCLUDE_DIR}
    )

# Linking

    target_link_libraries(esn-math
        PRIVATE
            ${CBLAS_LIBRARY}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_CUDART_LIBRARY}
            ${CUDA_cusolver_LIBRARY}
            ${LAPACKE_LIBRARY}
            ${LAPACK_LIBRARIES}
    )
