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
            const pointer & input);

    private:
        const pointer mForgettingFactor;
        const unsigned mInputCount;
        const const_pointer kZero;
        const const_pointer kOne;
        const const_pointer kAlpha;
        const const_pointer kBeta;
        const pointer mW;
        pointer mP;
        pointer mTemp;
        const pointer mDot;
        pointer mK;
        const pointer mDelta;
    };

} // namespace ESN

#endif // __ESN_ADAPTIVE_FILTER_RLS_H__
