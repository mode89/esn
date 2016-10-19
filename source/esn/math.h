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

    void sfillv(const int n, const const_pointer & alpha,
        const pointer & x);
    template <class T>
    void fillv(
        const scalar<T> & alpha,
        vector<T> & x);

    void srandv(const int n, const const_pointer & a,
        const const_pointer & b, const pointer & x);
    template <class T>
    void randv(
        const scalar<T> & a,
        const scalar<T> & b,
        vector<T> & x);

    void srandspv(const int n, const const_pointer & a,
        const const_pointer & b,
        const const_pointer & sparsity,
        const pointer & x);

    void srcp(const pointer & v);
    template <class T>
    void rcp(scalar<T> & x);

    void stanhv(const int n, const pointer & v);
    void sprodvv(const int n, const const_pointer & x,
        const pointer & y);
    void SCOPY(const int n, const float * x, const int incx, float * y,
        const int incy);
    void saxpy(const int h, const const_pointer & alpha,
        const const_pointer & x, const int incx,
        const pointer & y, const int incy);
    template <class T>
    void axpy(
        const scalar<T> & alpha,
        const vector<T> & x,
        vector<T> & y);

    void sdot(const int n, const const_pointer & x, const int incx,
        const const_pointer & y, const int incy,
        const pointer & result);
    template <class T>
    void dot(
        const vector<T> & x,
        const vector<T> & y,
        scalar<T> & result);

    void sgemv(const char trans, const int m, const int n,
        const const_pointer & alpha, const const_pointer & a,
        const int lda, const const_pointer & x, const int incx,
        const const_pointer & beta, const pointer & y,
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
        const const_pointer & alpha, const const_pointer & a,
        const int lda, const const_pointer & x, const int incx,
        const const_pointer & beta, const pointer & y,
        const int incy);

    void sgemm(const char transa, const char transb, const int m,
        const int n, const int k, const const_pointer & alpha,
        const const_pointer & a, const int lda,
        const const_pointer & b, const int ldb,
        const const_pointer & beta, const pointer & c,
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
        const pointer & a, const int lda, const pointer & s,
        const pointer & u, const int ldu, const pointer & vt,
        const int ldvt);
    template <class T>
    int gesvd(
        const char jobu,
        const char jobvt,
        matrix<T> & a,
        vector<T> & s,
        matrix<T> & u,
        matrix<T> & vt);

} // namespace ESN

#endif // __ESN_SOURCE_ESN_MATH_H__
