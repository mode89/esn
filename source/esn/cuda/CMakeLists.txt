# Dependencies

    find_package(CUDA REQUIRED)

# Targets

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -Xcompiler -fPIC")
    cuda_compile(CUDA_COMPILED_FILES wrap.cu)

    add_library(esn-math
        ${CUDA_COMPILED_FILES}
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
            ${CUDA_INCLUDE_DIRS}
    )

# Linking

    target_link_libraries(esn-math
        PRIVATE
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_CUDART_LIBRARY}
            ${CUDA_curand_LIBRARY}
            ${CUDA_cusolver_LIBRARY}
    )
