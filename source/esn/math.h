#ifndef __ESN_SOURCE_ESN_MATH_H__
#define __ESN_SOURCE_ESN_MATH_H__

#include <esn/scalar.h>
#include <esn/matrix.h>
#include <esn/vector.h>

namespace ESN {

    void Constant(float * v, int size, float value);
    void TanhEwise(float * v, int size);
    void ProductEwise(float * out, const float * in, int size);
    void SumEwise(float * out, const float * a, const float * b, int size);

    void sfillv(const int n, const const_pointer<float> & alpha,
        const pointer<float> & x);
    void srandv(const int n, const const_pointer<float> & a,
        const const_pointer<float> & b, const pointer<float> & x);
    void srandspv(const int n, const const_pointer<float> & a,
        const const_pointer<float> & b,
        const const_pointer<float> & sparsity,
        const pointer<float> & x);

    void srcp(const pointer<float> & v);
    template <class T>
    void rcp(scalar<T> & x);

    void stanhv(const int n, const pointer<float> & v);
    void sprodvv(const int n, const const_pointer<float> & x,
        const pointer<float> & y);
    void SCOPY(const int n, const float * x, const int incx, float * y,
        const int incy);
    void saxpy(const int h, const const_pointer<float> & alpha,
        const const_pointer<float> & x, const int incx,
        const pointer<float> & y, const int incy);
    template <class T>
    void axpy(
        const scalar<T> & alpha,
        const vector<T> & x,
        vector<T> & y);

    void sdot(const int n, const const_pointer<float> & x, const int incx,
        const const_pointer<float> & y, const int incy,
        const pointer<float> & result);
    template <class T>
    void dot(
        const vector<T> & x,
        const vector<T> & y,
        scalar<T> & result);

    void sgemv(const char trans, const int m, const int n,
        const const_pointer<float> & alpha, const const_pointer<float> & a,
        const int lda, const const_pointer<float> & x, const int incx,
        const const_pointer<float> & beta, const pointer<float> & y,
        const int incy);
    template <class T>
    void gemv(
        const char trans,
        const scalar<T> & alpha,
        const matrix<T> & a,
        const vector<T> & x,
        const scalar<T> & beta,
        vector<T> & y);

    void ssbmv(const char uplo, const int n, const int k,
        const const_pointer<float> & alpha, const const_pointer<float> & a,
        const int lda, const const_pointer<float> & x, const int incx,
        const const_pointer<float> & beta, const pointer<float> & y,
        const int incy);

    void sgemm(const char transa, const char transb, const int m,
        const int n, const int k, const const_pointer<float> & alpha,
        const const_pointer<float> & a, const int lda,
        const const_pointer<float> & b, const int ldb,
        const const_pointer<float> & beta, const pointer<float> & c,
        const int ldc);
    template <class T>
    void gemm(
        const char transa,
        const char transb,
        const scalar<T> & alpha,
        const matrix<T> & a,
        const matrix<T> & b,
        const scalar<T> & beta,
        matrix<T> & c);

    int sgesvd(const char jobu, const char jobvt, const int m, const int n,
        const pointer<float> & a, const int lda, const pointer<float> & s,
        const pointer<float> & u, const int ldu, const pointer<float> & vt,
        const int ldvt);

} // namespace ESN

#endif // __ESN_SOURCE_ESN_MATH_H__
