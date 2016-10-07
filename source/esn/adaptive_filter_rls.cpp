#include <esn/adaptive_filter_rls.h>
#include <esn/math.h>

namespace ESN {

    AdaptiveFilterRLS::AdaptiveFilterRLS(
        unsigned inputCount,
        float forgettingFactor,
        float regularization)
        : mForgettingFactor(make_pointer(forgettingFactor))
        , mInputCount(inputCount)
        , kZero(make_pointer(0.0f))
        , kOne(make_pointer(1.0f))
        , kAlpha(make_pointer(-1.0f / forgettingFactor))
        , kBeta(make_pointer(1.0f / forgettingFactor))
        , mInput(make_pointer(sizeof(float) * inputCount))
        , mW(make_pointer(sizeof(float) * inputCount))
        , mP(make_pointer(sizeof(float) * inputCount * inputCount))
        , mTemp(make_pointer(sizeof(float) * inputCount))
        , mDot(make_pointer(0.0f))
        , mK(make_pointer(sizeof(float) * inputCount))
        , mDelta(make_pointer(0.0f))
    {
        // Initialize diagonal matrix
        std::vector<float> p(inputCount * inputCount);
        std::fill(p.begin(), p.end(), 0.0f);
        for (int i = 0; i < inputCount; ++ i)
            p[i + i * inputCount] = regularization;
        memcpy(mP, p);
    }

    void AdaptiveFilterRLS::Train(
        float * w,
        float actualOutput,
        float referenceOutput,
        const float * input)
    {
        int N = mInputCount;
        memcpy(mInput, input, N * sizeof(float));
        memcpy(mDelta, referenceOutput - actualOutput);
        memcpy(mW, w, N * sizeof(float));

        // mTemp = transpose(mP) * mInput
        sgemv('T', N, N, kOne, mP, N, mInput, 1, kZero, mTemp, 1);
        // SGEMV('T', N, N, 1.0f, mP.data(), N, mInput, 1, 0.0f,
        //     mTemp.data(), 1);

        // mDot = mTemp * mInput
        sdot(N, mTemp, 1, mInput, 1, mDot);

        // mDot = mForgettingFactor + mDot
        saxpy(1, kOne, mForgettingFactor, 1, mDot, 1);

        // mDot = 1 / mDot
        srcp(mDot);

        // mK = mDot * mP * mInput
        sgemv('N', N, N, mDot, mP, N, mInput, 1, kZero, mK, 1);
        // SGEMV('N', N, N, 1.0f / (mForgettingFactor + dot), mP.data(), N,
        //     mInput, 1, 0.0f, mK.data(), 1);

        // mP = 1 / mForgettingFactor * (mP - mK * mTemp.transpose())
        // and BLAS representation
        // mP = mAlpha * mK * mTemp.transpose() + mBeta * mP
        // where
        // mAlpha = -1.0f / mForgettingFactor
        // mBeta = 1.0f / mForgettingFactor
        sgemm('N', 'T', N, N, 1, kAlpha, mK, N, mTemp, N, kBeta, mP, N);
        // SGEMM('N', 'T', N, N, 1, -1 / mForgettingFactor, mK.data(), N,
        //     mTemp.data(), N, 1 / mForgettingFactor, mP.data(), N);

        // w = (referenceOutput - actualOutput) * mK + w
        saxpy(N, mDelta, mK, 1, mW, 1);

        memcpy(w, mW, N * sizeof(float));
    }

} // namespace ESN
