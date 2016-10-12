#ifndef __ESN_ADAPTIVE_FILTER_RLS_H__
#define __ESN_ADAPTIVE_FILTER_RLS_H__

#include <esn/export.h>
#include <esn/pointer.h>
#include <vector>

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
            float * w,
            float actualOutput,
            float referenceOutput,
            const pointer<float> & input);

    private:
        const pointer<float> mForgettingFactor;
        const unsigned mInputCount;
        const const_pointer<float> kZero;
        const const_pointer<float> kOne;
        const const_pointer<float> kAlpha;
        const const_pointer<float> kBeta;
        const pointer<float> mW;
        const pointer<float> mP;
        const pointer<float> mTemp;
        const pointer<float> mDot;
        const pointer<float> mK;
        const pointer<float> mDelta;
    };

} // namespace ESN

#endif // __ESN_ADAPTIVE_FILTER_RLS_H__
