#ifndef __ESN_ADAPTIVE_FILTER_RLS_H__
#define __ESN_ADAPTIVE_FILTER_RLS_H__

#include <esn/export.h>
#include <esn/pointer.h>
#include <esn/matrix.h>
#include <esn/scalar.h>
#include <esn/vector.h>

namespace ESN {

    class AdaptiveFilterRLS
    {
    public:
        ESN_EXPORT AdaptiveFilterRLS(
            unsigned inputCount,
            float forgettingFactor = 0.99f,
            float regularization = 1000.0f );

        ESN_EXPORT void
        Train(
            vector<float> & w,
            const scalar<float> & actualOutput,
            const scalar<float> & referenceOutput,
            const pointer & input);

    private:
        const unsigned mInputCount;
        const scalar<float> mForgettingFactor;
        const scalar<float> kZero;
        const scalar<float> kOne;
        const scalar<float> kMinusOne;
        const scalar<float> kAlpha;
        const scalar<float> kBeta;
        matrix<float> mP;
        vector<float> mTemp;
        scalar<float> mDot;
        vector<float> mK;
        scalar<float> mDelta;
    };

} // namespace ESN

#endif // __ESN_ADAPTIVE_FILTER_RLS_H__
