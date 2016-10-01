#include <adaptive_filter_rls.h>

extern "C" {
#include <cblas/cblas.h>
}

namespace ESN {

    AdaptiveFilterRLS::AdaptiveFilterRLS(
        unsigned inputCount,
        float forgettingFactor,
        float regularization)
        : mForgettingFactor(forgettingFactor)
        , mInputCount(inputCount)
        , mP(inputCount * inputCount)
        , mTemp(inputCount)
        , mK(inputCount)
    {
        // Initialize diagonal matrix
        std::fill(mP.begin(), mP.end(), 0.0f);
        for (int i = 0; i < inputCount; ++ i)
            mP[i + i * inputCount] = regularization;
    }

    void AdaptiveFilterRLS::Train(
        float * w,
        float actualOutput,
        float referenceOutput,
        const float * input)
    {
        // mTemp = transpose(mP) * input
        int N = mInputCount;
        cblas_sgemv(CblasColMajor, CblasTrans, N, N, 1.0f, mP.data(), N,
            input, 1, 0.0f, mTemp.data(), 1);

        // dot = mTemp * input
        float dot = cblas_sdot(N, mTemp.data(), 1, input, 1);

        // mK = mP * input / (mForgettingFactor + dot)
        cblas_sgemv(CblasColMajor, CblasNoTrans, N, N,
            1.0f / (mForgettingFactor + dot), mP.data(), N,
            input, 1, 0.0f, mK.data(), 1);

        // mP = 1 / mForgettingFactor * (mP - mK * mTemp.transpose())
        // and BLAS representation
        // mP = -1 / mForgettingFactor * mK * mTemp.transpose() +
        //      1 / mForgettingFactor * mP
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, N, 1,
            -1 / mForgettingFactor, mK.data(), N, mTemp.data(), N,
            1 / mForgettingFactor, mP.data(), N);

        // w = (referenceOutput - actualOutput) * mK + w
        cblas_saxpy(N, referenceOutput - actualOutput,
            mK.data(), 1, w, 1);
    }

} // namespace ESN
