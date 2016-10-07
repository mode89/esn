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
            const float * input);

    private:
        const float mForgettingFactor;
        const unsigned mInputCount;
        std::vector<float> mP;
        pointer mTemp;
        std::vector<float> mK;
    };

} // namespace ESN

#endif // __ESN_ADAPTIVE_FILTER_RLS_H__
