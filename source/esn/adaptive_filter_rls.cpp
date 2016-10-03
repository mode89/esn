#include <esn/adaptive_filter_rls.h>
#include <esn/math.h>

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
        SGEMV('T', N, N, 1.0f, mP.data(), N, input, 1, 0.0f,
            mTemp.data(), 1);

        // dot = mTemp * input
        float dot = SDOT(N, mTemp.data(), 1, input, 1);

        // mK = mP * input / (mForgettingFactor + dot)
        SGEMV('N', N, N, 1.0f / (mForgettingFactor + dot), mP.data(), N,
            input, 1, 0.0f, mK.data(), 1);

        // mP = 1 / mForgettingFactor * (mP - mK * mTemp.transpose())
        // and BLAS representation
        // mP = -1 / mForgettingFactor * mK * mTemp.transpose() +
        //      1 / mForgettingFactor * mP
        SGEMM('N', 'T', N, N, 1, -1 / mForgettingFactor, mK.data(), N,
            mTemp.data(), N, 1 / mForgettingFactor, mP.data(), N);

        // w = (referenceOutput - actualOutput) * mK + w
        SAXPY(N, referenceOutput - actualOutput, mK.data(), 1, w, 1);
    }

} // namespace ESN
