#include <cmath>
#include <Eigen/Dense>
#include <gtest/gtest.h>

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

    for ( int i = 0; i < kStepCount; ++ i )
    {
        model.Update();
        float error = model.mReferenceOutput - model.mOutput;
        model.mW += ( kTrainStep * error * model.mInput.transpose() /
            model.mInput.squaredNorm() );
    }
}

TEST( AdaptiveFilter, RLS )
{
    const unsigned kStepCount = 1000;
    const float kTrainStep = 0.1f;
    const float kDelta = 1000.0f;
    const float kGamma = 0.999f;

    Model model;

    Eigen::MatrixXf P = Eigen::MatrixXf::Identity(
        model.mInput.size(), model.mInput.size() ) * kDelta;

    for ( int i = 0; i < kStepCount; ++ i )
    {
        model.Update();
        float error = model.mReferenceOutput - model.mOutput;
        auto & u = model.mInput;
        auto uT_P = u.transpose() * P;
        Eigen::VectorXf K = P * u / ( kGamma + uT_P.dot( u ) );
        P = 1 / kGamma * ( P - K * uT_P );
        model.mW += error * K;
    }
}
