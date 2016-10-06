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
        pointer ptrAlpha = make_pointer(1.0f);
        pointer ptrP = make_pointer(mP);
        pointer ptrInput = make_pointer(N * sizeof(float));
        memcpy(ptrInput, input, N * sizeof(float));
        pointer ptrBeta = make_pointer(0.0f);
        pointer ptrTemp = make_pointer(mTemp);
        sgemv('T', N, N, ptrAlpha, ptrP, N, ptrInput, 1, ptrBeta,
            ptrTemp, 1);
        memcpy(mTemp, ptrTemp);
        // SGEMV('T', N, N, 1.0f, mP.data(), N, input, 1, 0.0f,
        //     mTemp.data(), 1);

        // dot = mTemp * input
        memcpy(ptrTemp, mTemp);
        pointer ptrDot = make_pointer(sizeof(float));
        sdot(N, ptrTemp, 1, ptrInput, 1, ptrDot);
        float dot = 0.0f;
        memcpy(&dot, ptrDot, sizeof(float));

        // mK = mP * input / (mForgettingFactor + dot)
        memcpy(ptrAlpha, 1.0f / (mForgettingFactor + dot));
        memcpy(ptrP, mP);
        memcpy(ptrBeta, 0.0f);
        pointer ptrK = make_pointer(mK);
        sgemv('N', N, N, ptrAlpha, ptrP, N, ptrInput, 1, ptrBeta, ptrK, 1);
        memcpy(mK, ptrK);
        // SGEMV('N', N, N, 1.0f / (mForgettingFactor + dot), mP.data(), N,
        //     input, 1, 0.0f, mK.data(), 1);

        // mP = 1 / mForgettingFactor * (mP - mK * mTemp.transpose())
        // and BLAS representation
        // mP = -1 / mForgettingFactor * mK * mTemp.transpose() +
        //      1 / mForgettingFactor * mP
        SGEMM('N', 'T', N, N, 1, -1 / mForgettingFactor, mK.data(), N,
            mTemp.data(), N, 1 / mForgettingFactor, mP.data(), N);

        // w = (referenceOutput - actualOutput) * mK + w
        memcpy(ptrAlpha, referenceOutput - actualOutput);
        memcpy(ptrK, mK);
        pointer ptrW = make_pointer(N * sizeof(float));
        memcpy(ptrW, w, N * sizeof(float));
        saxpy(N, ptrAlpha, ptrK, 1, ptrW, 1);
        memcpy(w, ptrW, N * sizeof(float));
    }

} // namespace ESN
