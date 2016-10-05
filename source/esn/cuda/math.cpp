#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <esn/cuda/debug.h>
#include <esn/math.h>
#include <memory>
#include <random>

extern "C" {
    #include <cblas.h>
}

#include <lapacke.h>

namespace ESN {

    std::default_random_engine sRandomEngine;

    static cublasHandle_t & GetHandle()
    {
        static auto deleter = [] (cublasHandle_t * h) {
            DEBUG("Destroying cuBLAS context ...");
            if (cublasDestroy(*h) != CUBLAS_STATUS_SUCCESS)
                DEBUG("Failed to release cuBLAS");
            delete h;
        };

        static std::unique_ptr<cublasHandle_t, decltype(deleter) &>
            handle(nullptr, deleter);

        if (!handle)
        {
            DEBUG("Creating cuBLAS context ...");
            handle.reset(new cublasHandle_t);
            if (cublasCreate(handle.get()) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Failed to initialize cuBLAS");
        }

        return *handle;
    }

    static inline CBLAS_TRANSPOSE ToCblasTranspose(char trans)
    {
        switch (trans)
        {
        case 'N':
            return CblasNoTrans;
        case 'T':
            return CblasTrans;
        }
    }

    static inline CBLAS_UPLO ToCblasUplo(char uplo)
    {
        switch (uplo)
        {
        case 'U':
            return CblasUpper;
        case 'L':
            return CblasLower;
        }
    }

    void RandomUniform(float * v, int size, float a, float b)
    {
        std::uniform_real_distribution<float> dist(a, b);
        for (int i = 0; i < size; ++ i)
            v[i] = dist(sRandomEngine);
    }

    void Constant(float * v, int size, float value)
    {
        for (int i = 0; i < size; ++ i)
            v[i] = value;
    }

    void TanhEwise(float * v, int size)
    {
        for (int i = 0; i < size; ++ i)
            v[i] = std::tanh(v[i]);
    }

    void ProductEwise(float * out, const float * in, int size)
    {
        for (int i = 0; i < size; ++ i)
            out[i] *= in[i];
    }

    void SumEwise(float * out, const float * a, const float * b, int size)
    {
        for (int i = 0; i < size; ++ i)
            out[i] = a[i] + b[i];
    }

    void SCOPY(const int n, const float * x, const int incx, float * y,
        const int incy)
    {
        cblas_scopy(n, x, incx, y, incy);
    }

    void SAXPY(const int n, const float alpha, const float * x,
        const int incx, float * y, const int incy)
    {
        cblas_saxpy(n, alpha, x, incx, y, incy);
    }

    float SDOT(const int n, const float * x, const int incx,
        const float * y, const int incy)
    {
        float * cudaResult = nullptr;
        VCU(cudaMalloc, &cudaResult, sizeof(float));

        float * cudaX = nullptr;
        VCU(cudaMalloc, &cudaX, n * sizeof(float));
        VCB(cublasSetVector, n, sizeof(float), x, 1, cudaX, 1);

        float * cudaY = nullptr;
        VCU(cudaMalloc, &cudaY, n * sizeof(float));
        VCB(cublasSetVector, n, sizeof(float), y, 1, cudaY, 1);

        VCB(cublasSetPointerMode, GetHandle(), CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSdot, GetHandle(),
            n, cudaX, incx, cudaY, incy, cudaResult);

        float result = 0.0f;
        VCU(cudaMemcpy, &result, cudaResult,
            sizeof(float), cudaMemcpyDeviceToHost);

        VCU(cudaFree, cudaX);
        VCU(cudaFree, cudaY);
        VCU(cudaFree, cudaResult);

        return result;
    }

    void SGEMV(const char trans, const int m, const int n,
        const float alpha, const float * a, const int lda, const float * x,
        const int incx, const float beta, float * y, const int incy)
    {
        cblas_sgemv(CblasColMajor, ToCblasTranspose(trans), m, n, alpha,
            a, lda, x, incx, beta, y, incy);
    }

    void SSBMV(const char uplo, const int n, const int k,
        const float alpha, const float * a, const int lda, const float * x,
        const int incx, const float beta, float * y, const int incy)
    {
        cblas_ssbmv(CblasColMajor, ToCblasUplo(uplo), n, k, alpha, a, lda,
            x, incx, beta, y, incy);
    }

    void SGEMM(const char transa, const char transb, const int m,
        const int n, const int k, const float alpha, const float * a,
        const int lda, const float * b, const int ldb, const float beta,
        float * c, const int ldc)
    {
        cblas_sgemm(CblasColMajor, ToCblasTranspose(transa),
            ToCblasTranspose(transb), m, n, k, alpha, a, lda, b, ldb,
            beta, c, ldc);
    }

    int SGESDD(const char jobz, const int m, const int n, float * a,
        const int lda, float * s, float * u, const int ldu, float * vt,
        const int ldvt)
    {
        return LAPACKE_sgesdd(LAPACK_COL_MAJOR, jobz, m, n, a, lda, s, u,
            ldu, vt, ldvt);
    }

} // namespace ESN
