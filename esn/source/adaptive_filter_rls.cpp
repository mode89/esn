#include <adaptive_filter_rls.h>

namespace ESN {

    AdaptiveFilterRLS::AdaptiveFilterRLS( unsigned inputCount,
        float forgettingFactor, float regularization )
        : mForgettingFactor( forgettingFactor )
        , mP( Eigen::MatrixXf::Identity(
            inputCount, inputCount ) * regularization )
    {
    }

    void AdaptiveFilterRLS::Train(
        Eigen::VectorXf & w,
        float actualOutput,
        float referenceOutput,
        Eigen::VectorXf input )
    {
        auto inT_P = input.transpose() * mP;
        Eigen::VectorXf K = mP * input /
            ( mForgettingFactor + inT_P.dot( input ) );
        mP = 1 / mForgettingFactor * ( mP - K * inT_P );
        w += ( referenceOutput - actualOutput ) * K;
    }

} // namespace ESN
