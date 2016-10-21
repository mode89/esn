// Copyright (c) 2016, Andrey Krainyak - All Rights Reserved
// You may use, distribute and modify this code under the terms of
// BSD 2-clause license.

#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <esn/cuda/debug.h>
#include <esn/cuda/wrap.h>
#include <esn/math.h>
#include <esn/pointer.h>
#include <memory>

extern "C" {
    #include <cblas.h>
}

#define PTR(val) ((val).ptr().get() + (val).off())

namespace ESN {

    static cublasHandle_t & get_cublas_handle()
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

    static cusolverDnHandle_t & get_cusolver_handle()
    {
        static auto deleter = [] (cusolverDnHandle_t * h) {
            DEBUG("Destroying cuSOLVER context ...");
            if (cusolverDnDestroy(*h) != CUSOLVER_STATUS_SUCCESS)
                DEBUG("Failed to release cuSOLVER");
            delete h;
        };

        static std::unique_ptr<cusolverDnHandle_t, decltype(deleter) &>
            handle(nullptr, deleter);

        if (!handle)
        {
            DEBUG("Creating cuSOLVER context ...");
            handle.reset(new cusolverDnHandle_t);
            if (cusolverDnCreate(handle.get()) != CUSOLVER_STATUS_SUCCESS)
                throw std::runtime_error("Failed to initialize cuSOLVER");
        }

        return *handle;
    }

    static curandGenerator_t & get_curand_handle()
    {
        static auto deleter = [] (curandGenerator_t * h) {
            DEBUG("Destroying cuRAND context ...");
            if (curandDestroyGenerator(*h) != CURAND_STATUS_SUCCESS)
                DEBUG("Failed to release cuRAND");
            delete h;
        };

        static std::unique_ptr<curandGenerator_t, decltype(deleter) &>
            handle(nullptr, deleter);

        if (!handle)
        {
            DEBUG("Create cuRAND context ...");
            handle.reset(new curandGenerator_t);
            if (curandCreateGenerator(handle.get(),
                CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
                throw std::runtime_error("Failed to initialize cuRAND");
            VCR(curandSetPseudoRandomGeneratorSeed, *handle, 0);
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
            DEBUG("Unknown operation");
        }
    }

    static inline cublasFillMode_t to_cublas_fill_mode(char mode)
    {
        switch (mode)
        {
        case 'L':
            return CUBLAS_FILL_MODE_LOWER;
        case 'U':
            return CUBLAS_FILL_MODE_UPPER;
        default:
            DEBUG("Unknown fill mode");
        }
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

    void sfillv(const int n, const const_pointer & alpha,
        const pointer & x)
    {
        wrap_sfillv(n, alpha.get(), x.get());
    }

    template <>
    void fillv(
        const scalar<float> & alpha,
        vector<float> & x)
    {
        if (x.inc() != 1)
            throw std::runtime_error(
                "fillv(): 'x' must have unity increment");
        wrap_sfillv(x.size(), PTR(alpha), PTR(x));
    }

    void srandv(const int n, const const_pointer & a,
        const const_pointer & b, const pointer & x)
    {
        VCR(curandGenerateUniform, get_curand_handle(), x.get(), n);
        wrap_srandv_helper(n, a.get(), b.get(), x.get());
    }

    template <>
    void randv(
        const scalar<float> & a,
        const scalar<float> & b,
        vector<float> & x)
    {
        if (x.size() <= 0)
            throw std::runtime_error(
                "randv(): 'x' must be not empty");
        if (x.inc() != 1)
            throw std::runtime_error(
                "randv(): 'x' must have unitary increment");
        VCR(curandGenerateUniform, get_curand_handle(), PTR(x), x.size());
        wrap_srandv_helper(x.size(), PTR(a), PTR(b), PTR(x));
    }

    template <>
    void randm(
        const scalar<float> & a,
        const scalar<float> & b,
        matrix<float> & x)
    {
        if (x.rows() != x.ld())
            throw std::runtime_error("randm(): x.rows() != x.ld()");
        int size = x.rows() * x.cols();
        VCR(curandGenerateUniform, get_curand_handle(), PTR(x), size);
        wrap_srandv_helper(size, PTR(a), PTR(b), PTR(x));
    }

    void srandspv(const int n, const const_pointer & a,
        const const_pointer & b,
        const const_pointer & sparsity, const pointer & x)
    {
        srandv(n, a, b, x);

        scalar<float> zero(0.0f);
        scalar<float> one(1.0f);
        vector<float> spx(n);
        srandv(n, zero.ptr(), one.ptr(), spx.ptr());

        wrap_srandspv_helper(n, sparsity.get(), PTR(spx), x.get());
    }

    template <>
    void randspm(
        const scalar<float> & a,
        const scalar<float> & b,
        const scalar<float> & sparsity,
        matrix<float> & x)
    {
        if (x.rows() != x.ld())
            throw std::runtime_error("randspm(): x.rows() != x.ld()");

        randm(a, b, x);

        scalar<float> zero(0.0f);
        scalar<float> one(1.0f);
        matrix<float> spx(x.rows(), x.cols());
        randm(zero, one, spx);

        wrap_srandspv_helper(x.rows() * x.cols(),
            PTR(sparsity), PTR(spx), PTR(x));
    }

    void srcp(const pointer & v)
    {
        wrap_srcp(v.get());
    }

    template <>
    void rcp(scalar<float> & x)
    {
        wrap_srcp(PTR(x));
    }

    void stanhv(const int n, const pointer & v)
    {
        wrap_stanhv(n, v.get());
    }

    template <>
    void tanhv(vector<float> & x)
    {
        if (x.inc() != 1)
            throw std::runtime_error(
                "tanhv(): vector must have unity increment");
        wrap_stanhv(x.size(), PTR(x));
    }

    void sprodvv(const int n, const const_pointer & x,
        const pointer & y)
    {
        wrap_sprodvv(n, x.get(), y.get());
    }

    template <>
    void prodvv(
        const vector<float> & x,
        vector<float> & y)
    {
        if (x.size() != y.size())
            throw std::runtime_error(
                "prodvv(): 'x' and 'y' must be the same size");
        if (x.inc() != 1)
            throw std::runtime_error(
                "prodvv(): 'x' must has unity increment");
        if (y.inc() != 1)
            throw std::runtime_error(
                "prodvv(): 'y' must has unity increment");
        wrap_sprodvv(x.size(), PTR(x), PTR(y));
    }

    template <>
    void divvv(
        vector<float> & x,
        const vector<float> & y)
    {
        if (x.size() != y.size())
            throw std::runtime_error(
                "divvv(): 'x' and 'y' must have the same size");
        if (x.inc() != 1)
            throw std::runtime_error(
                "divvv(): 'x' must have unity increment");
        if (y.inc() != 1)
            throw std::runtime_error(
                "divvv(): 'y' must have unity increment");
        wrap_sdivvv(x.size(), PTR(x), PTR(y));
    }

    void SCOPY(const int n, const float * x, const int incx, float * y,
        const int incy)
    {
        cblas_scopy(n, x, incx, y, incy);
    }

    template <>
    void copy(
        const vector<float> & x,
        vector<float> & y)
    {
        if (x.size() != y.size())
            throw std::runtime_error(
                "copy(): 'x' and 'y' must have the same size");
        VCB(cublasScopy, get_cublas_handle(), x.size(), PTR(x), x.inc(),
            PTR(y), y.inc());
    }

    void saxpy(const int n, const const_pointer & alpha,
        const const_pointer & x, const int incx,
        const pointer & y, const int incy)
    {
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSaxpy, get_cublas_handle(),
            n, alpha.get(), x.get(), incx, y.get(), incy);
    }

    template <>
    void axpy(
        const scalar<float> & alpha,
        const vector<float> & x,
        vector<float> & y)
    {
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSaxpy, get_cublas_handle(),
            x.size(), PTR(alpha), PTR(x), x.inc(), PTR(y), y.inc());
    }

    void sdot(const int n, const const_pointer & x, const int incx,
        const const_pointer & y, const int incy,
        const pointer & result)
    {
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSdot, get_cublas_handle(),
            n, x.get(), incx, y.get(), incy, result.get());
    }

    template <>
    void dot(
        const vector<float> & x,
        const vector<float> & y,
        scalar<float> & result)
    {
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSdot, get_cublas_handle(),
            x.size(), PTR(x), x.inc(), PTR(y), y.inc(), PTR(result));
    }

    void sgemv(const char trans, const int m, const int n,
        const const_pointer & alpha, const const_pointer & a,
        const int lda, const const_pointer & x, const int incx,
        const const_pointer & beta, const pointer & y,
        const int incy)
    {
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSgemv, get_cublas_handle(), to_cublas_operation(trans),
            m, n, alpha.get(), a.get(), lda, x.get(), incx, beta.get(),
            y.get(), incy);
    }

    template <>
    void gemv(
        const char trans,
        const scalar<float> & alpha,
        const matrix<float> & a,
        const vector<float> & x,
        const scalar<float> & beta,
        vector<float> & y)
    {
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSgemv, get_cublas_handle(), to_cublas_operation(trans),
            a.rows(), a.cols(), PTR(alpha), PTR(a), a.ld(),
            PTR(x), x.inc(), PTR(beta), PTR(y), y.inc());
    }

    void ssbmv(const char uplo, const int n, const int k,
        const const_pointer & alpha, const const_pointer & a,
        const int lda, const const_pointer & x, const int incx,
        const const_pointer & beta, const pointer & y,
        const int incy)
    {
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSsbmv, get_cublas_handle(), to_cublas_fill_mode(uplo),
            n, k, alpha.get(), a.get(), lda, x.get(), incx, beta.get(),
            y.get(), incy);
    }

    template <>
    void sbmv(
        const char uplo,
        const int n,
        const int k,
        const scalar<float> & alpha,
        const vector<float> & a,
        const int lda,
        const vector<float> & x,
        const scalar<float> & beta,
        vector<float> & y)
    {
        if (a.inc() != 1)
            throw std::runtime_error(
                "sbmv(): 'a' must has unity increment");
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSsbmv, get_cublas_handle(), to_cublas_fill_mode(uplo),
            n, k, PTR(alpha), PTR(a), lda, PTR(x), x.inc(), PTR(beta),
            PTR(y), y.inc());
    }

    void sgemm(const char transa, const char transb, const int m,
        const int n, const int k, const const_pointer & alpha,
        const const_pointer & a, const int lda,
        const const_pointer & b, const int ldb,
        const const_pointer & beta, const pointer & c,
        const int ldc)
    {
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSgemm, get_cublas_handle(), to_cublas_operation(transa),
            to_cublas_operation(transb), m, n, k, alpha.get(), a.get(), lda,
            b.get(), ldb, beta.get(), c.get(), ldc);
    }

    template <>
    void gemm(
        const char transa,
        const char transb,
        const scalar<float> & alpha,
        const matrix<float> & a,
        const matrix<float> & b,
        const scalar<float> & beta,
        matrix<float> & c)
    {
        int m = (transa == 'N') ? a.rows() : a.cols();
        int n = (transb == 'N') ? b.cols() : b.rows();
        int k = (transa == 'N') ? a.cols() : a.rows();
        VCB(cublasSetPointerMode, get_cublas_handle(),
            CUBLAS_POINTER_MODE_DEVICE);
        VCB(cublasSgemm, get_cublas_handle(), to_cublas_operation(transa),
            to_cublas_operation(transb), m, n, k, PTR(alpha), PTR(a),
            a.ld(), PTR(b), b.ld(), PTR(beta), PTR(c), c.ld());
    }

    int sgesvd(const char jobu, const char jobvt, const int m, const int n,
        const pointer & a, const int lda, const pointer & s,
        const pointer & u, const int ldu, const pointer & vt,
        const int ldvt)
    {
        int lwork = 0;
        VCS(cusolverDnSgesvd_bufferSize,
            get_cusolver_handle(), m, n, &lwork);
        vector<float> work(lwork);
        int * devInfo = nullptr;
        VCU(cudaMalloc, &devInfo, sizeof(int));
        VCS(cusolverDnSgesvd, get_cusolver_handle(), jobu, jobvt, m, n,
            a.get(), lda, s.get(), u.get(), ldu, vt.get(), ldvt, PTR(work),
            lwork, nullptr, devInfo);
        int hostInfo = 0;
        VCU(cudaMemcpy, &hostInfo, devInfo, sizeof(int),
            cudaMemcpyDeviceToHost);
        VCU(cudaFree, devInfo);
        return hostInfo;
    }

    template <>
    int gesvd(
        const char jobu,
        const char jobvt,
        matrix<float> & a,
        vector<float> & s,
        matrix<float> & u,
        matrix<float> & vt)
    {
        int lwork = 0;
        VCS(cusolverDnSgesvd_bufferSize,
            get_cusolver_handle(), a.rows(), a.cols(), &lwork);
        vector<float> work(lwork);
        scalar<int> devInfo(0);
        VCS(cusolverDnSgesvd, get_cusolver_handle(), jobu, jobvt,
            a.rows(), a.cols(), PTR(a), a.ld(), PTR(s), PTR(u), u.ld(),
            PTR(vt), vt.ld(), PTR(work), lwork, nullptr,
            // TODO replace with PTR(devInfo)
            reinterpret_cast<int*>(devInfo.ptr().get()));
        return static_cast<int>(devInfo);
    }

} // namespace ESN
