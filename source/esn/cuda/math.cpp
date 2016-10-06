#include <cublas_v2.h>
#include <esn/cuda/debug.h>
#include <esn/math.h>
#include <esn/pointer.h>
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

    static inline cublasOperation_t to_cublas_operation(char op)
    {
        switch (op)
        {
        case 'N':
            return CUBLAS_OP_N;
        case 'T':
            return CUBLAS_OP_T;
        case 'C':
            return CUBLAS_OP_C;
        default:
            DEBUG("Unknown cuBLAS operation");
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
        pointer deviceAlpha = make_pointer(sizeof(float));
        memcpy(deviceAlpha, &alpha, sizeof(float));

        pointer deviceX = make_pointer(n * sizeof(float));
        VCB(cublasSetVector, n, sizeof(float), x, incx, deviceX.get(), 1);

        pointer deviceY = make_pointer(n * sizeof(float));
        VCB(cublasSetVector, n, sizeof(float), y, incx, deviceY.get(), 1);

        VCB(cublasSetPointerMode, GetHandle(), CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSaxpy, GetHandle(),
            n, deviceAlpha.get(), deviceX.get(), 1, deviceY.get(), 1);

        VCB(cublasGetVector, n, sizeof(float), deviceY.get(), 1, y, incy);
    }

    void saxpy(const int n, const const_pointer & alpha,
        const const_pointer & x, const int incx, const pointer & y,
        const int incy)
    {
        VCB(cublasSetPointerMode, GetHandle(), CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSaxpy, GetHandle(),
            n, alpha.get(), x.get(), incx, y.get(), incy);
    }

    float SDOT(const int n, const float * x, const int incx,
        const float * y, const int incy)
    {
        pointer deviceResult = make_pointer(sizeof(float));

        pointer deviceX = make_pointer(n * sizeof(float));
        VCB(cublasSetVector, n, sizeof(float), x, incx, deviceX.get(), 1);

        pointer deviceY = make_pointer(n * sizeof(float));
        VCB(cublasSetVector, n, sizeof(float), y, incx, deviceY.get(), 1);

        VCB(cublasSetPointerMode, GetHandle(), CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSdot, GetHandle(), n, deviceX.get(), 1,
            deviceY.get(), 1, deviceResult.get());

        float result = 0.0f;
        memcpy(&result, deviceResult, sizeof(float));

        return result;
    }

    void sdot(const int n, const const_pointer & x, const int incx,
        const const_pointer & y, const int incy, const pointer & result)
    {
        VCB(cublasSetPointerMode, GetHandle(), CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSdot, GetHandle(),
            n, x.get(), incx, y.get(), incy, result.get());
    }

    void SGEMV(const char trans, const int m, const int n,
        const float alpha, const float * a, const int lda, const float * x,
        const int incx, const float beta, float * y, const int incy)
    {
        cblas_sgemv(CblasColMajor, ToCblasTranspose(trans), m, n, alpha,
            a, lda, x, incx, beta, y, incy);
    }

    void sgemv(const char trans, const int m, const int n,
        const const_pointer & alpha, const const_pointer & a, const int lda,
        const const_pointer & x, const int incx, const const_pointer & beta,
        const pointer & y, const int incy)
    {
        VCB(cublasSetPointerMode, GetHandle(), CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSgemv, GetHandle(), to_cublas_operation(trans), m, n,
            alpha.get(), a.get(), lda, x.get(), incx, beta.get(), y.get(),
            incy);
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

    void sgemm(const char transa, const char transb, const int m,
        const int n, const int k, const const_pointer & alpha,
        const const_pointer & a, const int lda, const const_pointer & b,
        const int ldb, const const_pointer & beta, const pointer & c,
        const int ldc)
    {
        VCB(cublasSetPointerMode, GetHandle(), CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSgemm, GetHandle(), to_cublas_operation(transa),
            to_cublas_operation(transb), m, n, k, alpha.get(), a.get(), lda,
            b.get(), ldb, beta.get(), c.get(), ldc);
    }

    int SGESDD(const char jobz, const int m, const int n, float * a,
        const int lda, float * s, float * u, const int ldu, float * vt,
        const int ldvt)
    {
        return LAPACKE_sgesdd(LAPACK_COL_MAJOR, jobz, m, n, a, lda, s, u,
            ldu, vt, ldvt);
    }

} // namespace ESN
