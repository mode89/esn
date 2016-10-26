#ifndef __ESN_SOURCE_ESN_MATH_H__
#define __ESN_SOURCE_ESN_MATH_H__

#include <esn/scalar.h>
#include <esn/matrix.h>
#include <esn/vector.h>

namespace ESN {

    template <class T>
    void fillv(
        const scalar<T> & alpha,
        vector<T> & x);

    template <class T>
    void randv(
        const scalar<T> & a,
        const scalar<T> & b,
        vector<T> & x);

    template <class T>
    void randm(
        const scalar<T> & a,
        const scalar<T> & b,
        matrix<T> & x);

    template <class T>
    void randspm(
        const scalar<T> & a,
        const scalar<T> & b,
        const scalar<T> & sparsity,
        matrix<T> & x);

    template <class T>
    void rcp(scalar<T> & x);

    template <class T>
    void tanhv(vector<T> & x);

    template <class T>
    void atanhv(vector<T> & x);

    template <class T>
    void prodvv(
        const vector<T> & x,
        vector<T> & y);

    template <class T>
    void divvv(
        vector<T> & x,
        const vector<T> & y);

    template <class T>
    void copy(
        const vector<T> & x,
        vector<T> & y);

    template <class T>
    void axpy(
        const scalar<T> & alpha,
        const vector<T> & x,
        vector<T> & y);

    template <class T>
    void dot(
        const vector<T> & x,
        const vector<T> & y,
        scalar<T> & result);

    template <class T>
    void gemv(
        const char trans,
        const scalar<T> & alpha,
        const matrix<T> & a,
        const vector<T> & x,
        const scalar<T> & beta,
        vector<T> & y);

    template <class T>
    void sbmv(
        const char uplo,
        const int n,
        const int k,
        const scalar<T> & alpha,
        const vector<T> & a,
        const int lda,
        const vector<T> & x,
        const scalar<T> & beta,
        vector<T> & y);

    template <class T>
    void gemm(
        const char transa,
        const char transb,
        const scalar<T> & alpha,
        const matrix<T> & a,
        const matrix<T> & b,
        const scalar<T> & beta,
        matrix<T> & c);

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
