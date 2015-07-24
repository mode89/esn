#ifndef __ESN_ADAPTIVE_FILTER_RLS_H__
#define __ESN_ADAPTIVE_FILTER_RLS_H__

#include <Eigen/Dense>
#include <esn/export.h>

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
            Eigen::VectorXf & w,
            float actualOutput,
            float referenceOutput,
            Eigen::VectorXf input );

    private:
        const float mForgettingFactor;
        Eigen::MatrixXf mP;
    };

} // namespace ESN

#endif // __ESN_ADAPTIVE_FILTER_RLS_H__
