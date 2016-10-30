// Copyright (c) 2016, Andrey Krainyak - All Rights Reserved
// You may use, distribute and modify this code under the terms of
// BSD 2-clause license.

#include <esn/math.h>
#include <random>

extern "C" {
    #include <cblas.h>
}

#include <lapacke.h>

static inline float * cast_ptr(void * ptr)
{
    return static_cast<float*>(ptr);
}

static inline const float * cast_ptr(const void * ptr)
{
    return static_cast<const float*>(ptr);
}

#define PTR(val) (cast_ptr((val).ptr().get()) + (val).off())

namespace ESN {

    std::default_random_engine sRandomEngine;

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

    void SumEwise(float * out, const float * a, const float * b, int size)
    {
        for (int i = 0; i < size; ++ i)
            out[i] = a[i] + b[i];
    }

    template <>
    void fillv(
        const scalar<float> & alpha,
        vector<float> & x)
    {
        if (x.inc() != 1)
            throw std::runtime_error(
                "fillv(): 'x' must have unity increment");

        const float valAlpha = *PTR(alpha);
        float * const ptrX = PTR(x);
        const std::size_t n = x.size();
        for (std::size_t i = 0; i < n; ++ i)
            ptrX[i] = valAlpha;
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

        std::uniform_real_distribution<float> dist(*PTR(a), *PTR(b));
        float * const ptrX = PTR(x);
        const std::size_t n = x.size();
        for (std::size_t i = 0; i < n; ++ i)
            ptrX[i] = dist(sRandomEngine);
    }

    template <>
    void randm(
        const scalar<float> & a,
        const scalar<float> & b,
        matrix<float> & x)
    {
        if (x.rows() != x.ld())
            throw std::runtime_error("randm(): x.rows() != x.ld()");

        std::uniform_real_distribution<float> dist(*PTR(a), *PTR(b));
        float * const ptrX = PTR(x);
        const std::size_t n = x.rows() * x.cols();
        for (std::size_t i = 0; i < n; ++ i)
            ptrX[i] = dist(sRandomEngine);
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

        const float valSparsity = *PTR(sparsity);
        const float * const ptrSPX = PTR(spx);
        float * const ptrX = PTR(x);
        const std::size_t n = x.rows() * x.cols();
        for (std::size_t i = 0; i < n; ++ i)
            if (ptrSPX[i] < valSparsity)
                ptrX[i] = 0.0f;
    }

    template <>
    void rcp(scalar<float> & x)
    {
        float * const ptrX = PTR(x);
        *ptrX = 1.0f / *ptrX;
    }

    template <>
    void tanhv(vector<float> & x)
    {
        if (x.inc() != 1)
            throw std::runtime_error(
                "tanhv(): vector must have unity increment");

        float * const ptrX = PTR(x);
        const std::size_t n = x.size();
        for (std::size_t i = 0; i < n; ++ i)
            ptrX[i] = std::tanh(ptrX[i]);
    }

    template <>
    void atanhv(vector<float> & x)
    {
        if (x.inc() != 1)
            throw std::runtime_error(
                "atanhv(): vector must have unity increment");

        float * const ptrX = PTR(x);
        const std::size_t n = x.size();
        for (std::size_t i = 0; i < n; ++ i)
            ptrX[i] = std::atanh(ptrX[i]);
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

        const float * const ptrX = PTR(x);
        float * const ptrY = PTR(y);
        const std::size_t n = x.size();
        for (std::size_t i = 0; i < n; ++ i)
            ptrY[i] *= ptrX[i];
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

        float * const ptrX = PTR(x);
        const float * const ptrY = PTR(y);
        const std::size_t n = x.size();
        for (std::size_t i = 0; i < n; ++ i)
            ptrX[i] /= ptrY[i];
    }

    template <>
    void copy(
        const vector<float> & x,
        vector<float> & y)
    {
        cblas_scopy(x.size(), PTR(x), x.inc(), PTR(y), y.inc());
    }

    template <>
    void axpy(
        const scalar<float> & alpha,
        const vector<float> & x,
        vector<float> & y)
    {
        cblas_saxpy(x.size(), *PTR(alpha), PTR(x), x.inc(),
            PTR(y), y.inc());
    }

    template <>
    void dot(
        const vector<float> & x,
        const vector<float> & y,
        scalar<float> & result)
    {
        *PTR(result) = cblas_sdot(x.size(), PTR(x), x.inc(),
            PTR(y), y.inc());
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
