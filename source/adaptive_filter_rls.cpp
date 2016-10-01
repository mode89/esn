#include <adaptive_filter_rls.h>

extern "C" {
#include <cblas/cblas.h>
}

namespace ESN {

    AdaptiveFilterRLS::AdaptiveFilterRLS( unsigned inputCount,
        float forgettingFactor, float regularization )
        : mForgettingFactor( forgettingFactor )
        , mP( Eigen::MatrixXf::Identity(
            inputCount, inputCount ) * regularization )
        , mTemp(inputCount)
        , mK(inputCount)
    {
    }

    void AdaptiveFilterRLS::Train(
        Eigen::VectorXf & w,
        float actualOutput,
        float referenceOutput,
        const Eigen::VectorXf & input)
    {
        // mTemp = transpose(mP) * input
        int N = input.size();
        cblas_sgemv(CblasColMajor, CblasTrans, N, N, 1.0f, mP.data(), N,
            input.data(), 1, 0.0f, mTemp.data(), 1);

        // dot = mTemp * input
        float dot = cblas_sdot(N, mTemp.data(), 1, input.data(), 1);

        // mK = mP * input / (mForgettingFactor + dot)
        cblas_sgemv(CblasColMajor, CblasNoTrans, N, N,
            1.0f / (mForgettingFactor + dot), mP.data(), N,
            input.data(), 1, 0.0f, mK.data(), 1);

        // mP = 1 / mForgettingFactor * (mP - mK * mTemp.transpose())
        // and BLAS representation
        // mP = -1 / mForgettingFactor * mK * mTemp.transpose() +
        //      1 / mForgettingFactor * mP
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, N, 1,
            -1 / mForgettingFactor, mK.data(), N, mTemp.data(), N,
            1 / mForgettingFactor, mP.data(), N);

        // w = (referenceOutput - actualOutput) * mK + w
        cblas_saxpy(N, referenceOutput - actualOutput,
            mK.data(), 1, w.data(), 1);
    }

} // namespace ESN
