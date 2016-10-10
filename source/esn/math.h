#ifndef __ESN_SOURCE_ESN_MATH_H__
#define __ESN_SOURCE_ESN_MATH_H__

#include <esn/pointer.h>

namespace ESN {

    void RandomUniform(float * v, int size, float a, float b);
    void Constant(float * v, int size, float value);
    void TanhEwise(float * v, int size);
    void ProductEwise(float * out, const float * in, int size);
    void SumEwise(float * out, const float * a, const float * b, int size);

    void sfillv(const int n, const const_pointer & alpha,
        const pointer & x);
    void srandv(const int n, const const_pointer & a,
        const const_pointer & b, const pointer & x);
    void srandspv(const int n, const const_pointer & a,
        const const_pointer & b, const const_pointer & sparsity,
        const pointer & x);
    void srandspv(const int n, const float a, const float b,
        const float sparsity, float * x);
    void srcp(const pointer & v);
    void stanhv(const int n, const pointer & v);
    void sprodvv(const int n, const const_pointer & x, const pointer & y);
    void SCOPY(const int n, const float * x, const int incx, float * y,
        const int incy);
    void SAXPY(const int n, const float alpha, const float * x,
        const int incx, float * y, const int incy);
    void saxpy(const int h, const const_pointer & alpha,
        const const_pointer & x, const int incx, const pointer & y,
        const int incy);
    float SDOT(const int n, const float * x, const int incx,
        const float * y, const int incy);
    void sdot(const int n, const const_pointer & x, const int incx,
        const const_pointer & y, const int incy, const pointer & result);
    void SGEMV(const char trans, const int m, const int n,
        const float alpha, const float * a, const int lda, const float * x,
        const int incx, const float beta, float * y, const int incy);
    void sgemv(const char trans, const int m, const int n,
        const const_pointer & alpha, const const_pointer & a, const int lda,
        const const_pointer & x, const int incx, const const_pointer & beta,
        const pointer & y, const int incy);
    void SSBMV(const char uplo, const int n, const int k,
        const float alpha, const float * a, const int lda, const float * x,
        const int incx, const float beta, float * y, const int incy);
    void ssbmv(const char uplo, const int n, const int k,
        const const_pointer & alpha, const const_pointer & a, const int lda,
        const const_pointer & x, const int incx, const const_pointer & beta,
        const pointer & y, const int incy);
    void SGEMM(const char transa, const char transb, const int m,
        const int n, const int k, const float alpha, const float * a,
        const int lda, const float * b, const int ldb, const float beta,
        float * c, const int ldc);
    void sgemm(const char transa, const char transb, const int m,
        const int n, const int k, const const_pointer & alpha,
        const const_pointer & a, const int lda, const const_pointer & b,
        const int ldb, const const_pointer & beta, const pointer & c,
        const int ldc);
    int SGESDD(const char jobz, const int m, const int n, float * a,
        const int lda, float * s, float * u, const int ldu, float * vt,
        const int ldvt);
    int sgesvd(const char jobu, const char jobvt, const int m, const int n,
        const pointer & a, const int lda, const pointer & s,
        const pointer & u, const int ldu, const pointer & vt,
        const int ldvt);

} // namespace ESN

#endif // __ESN_SOURCE_ESN_MATH_H__
