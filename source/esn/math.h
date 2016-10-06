#ifndef __ESN_SOURCE_ESN_MATH_H__
#define __ESN_SOURCE_ESN_MATH_H__

#include <esn/pointer.h>

namespace ESN {

    void RandomUniform(float * v, int size, float a, float b);
    void Constant(float * v, int size, float value);
    void TanhEwise(float * v, int size);
    void ProductEwise(float * out, const float * in, int size);
    void SumEwise(float * out, const float * a, const float * b, int size);

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
    void SGEMM(const char transa, const char transb, const int m,
        const int n, const int k, const float alpha, const float * a,
        const int lda, const float * b, const int ldb, const float beta,
        float * c, const int ldc);
    int SGESDD(const char jobz, const int m, const int n, float * a,
        const int lda, float * s, float * u, const int ldu, float * vt,
        const int ldvt);

} // namespace ESN

#endif // __ESN_SOURCE_ESN_MATH_H__
