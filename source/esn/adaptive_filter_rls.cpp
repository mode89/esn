// Copyright (c) 2016, Andrey Krainyak - All Rights Reserved
// You may use, distribute and modify this code under the terms of
// BSD 2-clause license.

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
        mP = matrix<float>(p, inputCount, inputCount);
    }

    void AdaptiveFilterRLS::Train(
        vector<float> & w,
        float actualOutput,
        float referenceOutput,
        const pointer & input)
    {
        int N = mInputCount;
        mDelta = referenceOutput - actualOutput;
        vector<float> vecInput(input, N);

        // mTemp = transpose(mP) * input
        gemv('T', kOne, mP, vecInput, kZero, mTemp);

        // mDot = mTemp * input
        dot(mTemp, vecInput, mDot);

        // mDot = mForgettingFactor + mDot
        axpy(kOne, mForgettingFactor, mDot);

        // mDot = 1 / mDot
        rcp(mDot);

        // mK = mDot * mP * input
        gemv('N', mDot, mP, vecInput, kZero, mK);

        // mP = 1 / mForgettingFactor * (mP - mK * mTemp.transpose())
        // and BLAS representation
        // mP = mAlpha * mK * mTemp.transpose() + mBeta * mP
        // where
        // mAlpha = -1.0f / mForgettingFactor
        // mBeta = 1.0f / mForgettingFactor
        matrix<float> matK(mK.ptr(), N, 1, N);
        matrix<float> matTemp(mTemp.ptr(), N, 1, N);
        gemm('N', 'T', kAlpha, matK, matTemp, kBeta, mP);

        // w = (referenceOutput - actualOutput) * mK + w
        axpy(mDelta, mK, w);
    }

} // namespace ESN
