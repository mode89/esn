#include <esn/adaptive_filter_rls.h>
#include <esn/math.h>

namespace ESN {

    AdaptiveFilterRLS::AdaptiveFilterRLS(
        unsigned inputCount,
        float forgettingFactor,
        float regularization)
        : mInputCount(inputCount)
        , mForgettingFactor(forgettingFactor)
        , kZero(0.0f)
        , kOne(1.0f)
        , kAlpha(-1.0f / forgettingFactor)
        , kBeta(1.0f / forgettingFactor)
        , mW(inputCount)
        , mP(inputCount, inputCount)
        , mTemp(inputCount)
        , mDot(0.0f)
        , mK(inputCount)
        , mDelta(0.0f)
    {
        // Initialize diagonal matrix
        std::vector<float> p(inputCount * inputCount);
        std::fill(p.begin(), p.end(), 0.0f);
        for (int i = 0; i < inputCount; ++ i)
            p[i + i * inputCount] = regularization;
        memcpy<float>(mP.ptr(), p);
    }

    void AdaptiveFilterRLS::Train(
        float * w,
        float actualOutput,
        float referenceOutput,
        const pointer<float> & input)
    {
        int N = mInputCount;
        memcpy<float>(mDelta.ptr(), referenceOutput - actualOutput);
        memcpy<float>(mW.ptr(), w, N);

        // mTemp = transpose(mP) * input
        sgemv('T', N, N, kOne.ptr(), mP.ptr(), N, input, 1,
            kZero.ptr(), mTemp.ptr(), 1);
        // SGEMV('T', N, N, 1.0f, mP.data(), N, input, 1, 0.0f,
        //     mTemp.data(), 1);

        // mDot = mTemp * input
        sdot(N, mTemp.ptr(), 1, input, 1, mDot.ptr());

        // mDot = mForgettingFactor + mDot
        saxpy(1, kOne.ptr(), mForgettingFactor.ptr(), 1, mDot.ptr(), 1);

        // mDot = 1 / mDot
        srcp(mDot.ptr());

        // mK = mDot * mP * input
        sgemv('N', N, N, mDot.ptr(), mP.ptr(), N, input, 1,
            kZero.ptr(), mK.ptr(), 1);
        // SGEMV('N', N, N, 1.0f / (mForgettingFactor + dot), mP.data(), N,
        //     input, 1, 0.0f, mK.data(), 1);

        // mP = 1 / mForgettingFactor * (mP - mK * mTemp.transpose())
        // and BLAS representation
        // mP = mAlpha * mK * mTemp.transpose() + mBeta * mP
        // where
        // mAlpha = -1.0f / mForgettingFactor
        // mBeta = 1.0f / mForgettingFactor
        sgemm('N', 'T', N, N, 1, kAlpha.ptr(), mK.ptr(), N, mTemp.ptr(),
            N, kBeta.ptr(), mP.ptr(), N);
        // SGEMM('N', 'T', N, N, 1, -1 / mForgettingFactor, mK.data(), N,
        //     mTemp.data(), N, 1 / mForgettingFactor, mP.data(), N);

        // w = (referenceOutput - actualOutput) * mK + w
        saxpy(N, mDelta.ptr(), mK.ptr(), 1, mW.ptr(), mW.inc());

        memcpy<float>(w, mW.ptr(), N);
    }

} // namespace ESN
