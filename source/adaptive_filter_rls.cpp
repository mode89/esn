#include <adaptive_filter_rls.h>

namespace ESN {

    AdaptiveFilterRLS::AdaptiveFilterRLS( unsigned inputCount,
        float forgettingFactor, float regularization )
        : mForgettingFactor( forgettingFactor )
        , mP( Eigen::MatrixXf::Identity(
            inputCount, inputCount ) * regularization )
        , mTemp(inputCount)
        , mK(inputCount)
    {
    }

    void AdaptiveFilterRLS::Train(
        Eigen::VectorXf & w,
        float actualOutput,
        float referenceOutput,
        const Eigen::VectorXf & input)
    {
        mTemp = mP.transpose() * input;
        mK = mP * input / (mForgettingFactor + mTemp.dot(input));
        mP = 1 / mForgettingFactor * (mP - mK * mTemp.transpose());
        w += ( referenceOutput - actualOutput ) * mK;
    }

} // namespace ESN
