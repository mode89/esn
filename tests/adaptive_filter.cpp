#include <cmath>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <adaptive_filter_rls.h>

class ReferenceFilter
{
public:
    ReferenceFilter( unsigned inputCount )
        : mW( Eigen::VectorXf::Random( inputCount ) )
    {}

    float operator()( Eigen::VectorXf inputs )
    {
        return mW.dot( inputs );
    }

    Eigen::VectorXf mW;
};

class Model
{
    const unsigned kInputCount = 100;
    const float kMaxAmplitude = 1.0f;
    const float kMaxFrequency = 10.0f;
    const float kStep = 0.1f * 1.0f / kMaxFrequency;

public:
    Model()
        : mAmplitude( kMaxAmplitude / 2.0f *
            ( Eigen::VectorXf::Random( kInputCount ).array() + 1.0f ) )
        , mOmega( kMaxFrequency / 2.0f *
            ( Eigen::VectorXf::Random( kInputCount ).array() + 1.0f ) )
        , mInput( kInputCount )
        , mW( Eigen::VectorXf::Random( kInputCount ) )
        , mOutput( 0.0f )
        , mReferenceFilter( kInputCount )
        , mReferenceOutput( 0.0f )
        , mTime( 0.0f )
    {}

    void Update()
    {
        mTime += kStep;
        mInput = mAmplitude.array() * ( mOmega.array() * mTime ).unaryExpr(
            std::ptr_fun< float, float >( std::sin ) );
        mOutput = mW.dot( mInput );
        mReferenceOutput = mReferenceFilter( mInput );
    }

    Eigen::VectorXf mAmplitude;
    Eigen::VectorXf mOmega;
    Eigen::VectorXf mInput;
    Eigen::VectorXf mW;
    float mOutput;
    ReferenceFilter mReferenceFilter;
    float mReferenceOutput;
    float mTime;
};

TEST( AdaptiveFilter, NLMS )
{
    const unsigned kStepCount = 10000;
    const float kTrainStep = 0.1f;

    Model model;
    model.Update();
    float error = model.mReferenceOutput - model.mOutput;
    float initialError = std::fabs( error / model.mOutput );

    for ( int i = 0; i < kStepCount; ++ i )
    {
        model.Update();
        error = model.mReferenceOutput - model.mOutput;
        model.mW += ( kTrainStep * error * model.mInput.transpose() /
            model.mInput.squaredNorm() );
    }

    EXPECT_TRUE( std::fabs( error / model.mOutput ) < initialError );
}

TEST( AdaptiveFilter, RLS )
{
    const unsigned kStepCount = 1000;
    const float kRegularization = 1000.0f;
    const float kForgettingFactor = 0.99f;

    Model model;
    model.Update();
    float error = model.mReferenceOutput - model.mOutput;
    float initialError = std::fabs( error / model.mOutput );

    ESN::AdaptiveFilterRLS filter( model.mInput.size(),
        kForgettingFactor, kRegularization );

    for ( int i = 0; i < kStepCount; ++ i )
    {
        model.Update();
        error = model.mReferenceOutput - model.mOutput;
        filter.Train( model.mW, model.mOutput, model.mReferenceOutput,
            model.mInput );
    }

    EXPECT_LT( std::fabs( error / model.mOutput ), initialError );
}
