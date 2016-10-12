#include <esn/adaptive_filter_rls.h>
#include <esn/math.h>

namespace ESN {

    AdaptiveFilterRLS::AdaptiveFilterRLS(
        unsigned inputCount,
        float forgettingFactor,
        float regularization)
        : mForgettingFactor(make_pointer<float>(forgettingFactor))
        , mInputCount(inputCount)
        , kZero(make_pointer<float>(0.0f))
        , kOne(make_pointer<float>(1.0f))
        , kAlpha(make_pointer<float>(-1.0f / forgettingFactor))
        , kBeta(make_pointer<float>(1.0f / forgettingFactor))
        , mW(make_pointer<float>(inputCount))
        , mP(make_pointer<float>(inputCount * inputCount))
        , mTemp(make_pointer<float>(inputCount))
        , mDot(make_pointer<float>(0.0f))
        , mK(make_pointer<float>(inputCount))
        , mDelta(make_pointer<float>(0.0f))
    {
        // Initialize diagonal matrix
        std::vector<float> p(inputCount * inputCount);
        std::fill(p.begin(), p.end(), 0.0f);
        for (int i = 0; i < inputCount; ++ i)
            p[i + i * inputCount] = regularization;
        memcpy<float>(mP, p);
    }

    void AdaptiveFilterRLS::Train(
        float * w,
        float actualOutput,
        float referenceOutput,
        const pointer<float> & input)
    {
        int N = mInputCount;
        memcpy<float>(mDelta, referenceOutput - actualOutput);
        memcpy<float>(mW, w, N);

        // mTemp = transpose(mP) * input
        sgemv('T', N, N, kOne, mP, N, input, 1, kZero, mTemp, 1);
        // SGEMV('T', N, N, 1.0f, mP.data(), N, input, 1, 0.0f,
        //     mTemp.data(), 1);

        // mDot = mTemp * input
        sdot(N, mTemp, 1, input, 1, mDot);

        // mDot = mForgettingFactor + mDot
        saxpy(1, kOne, mForgettingFactor, 1, mDot, 1);

        // mDot = 1 / mDot
        srcp(mDot);

        // mK = mDot * mP * input
        sgemv('N', N, N, mDot, mP, N, input, 1, kZero, mK, 1);
        // SGEMV('N', N, N, 1.0f / (mForgettingFactor + dot), mP.data(), N,
        //     input, 1, 0.0f, mK.data(), 1);

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

        memcpy<float>(w, mW, N);
    }

} // namespace ESN
