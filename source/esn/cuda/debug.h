#ifndef __ESN_CUDA_DEBUG_H__
#define __ESN_CUDA_DEBUG_H__

#define DEBUG(...) { printf(__VA_ARGS__); printf("\n"); }

#define VCU(func, ...) { \
        if (func(__VA_ARGS__) != cudaSuccess) \
            DEBUG("Failed " #func "()"); \
    }

#define VCB(func, ...) { \
        if (func(__VA_ARGS__) != CUBLAS_STATUS_SUCCESS) \
            DEBUG("Failed " #func "()"); \
    }

#define VCS(func, ...) { \
        if (func(__VA_ARGS__) != CUSOLVER_STATUS_SUCCESS) \
            DEBUG("Failed " #func "()"); \
    }

#endif // __ESN_CUDA_DEBUG_H__
