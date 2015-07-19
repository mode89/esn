#include <cmath>
#include <Eigen/Dense>
#include <gtest/gtest.h>

class ReferenceFilter
{
public:
    ReferenceFilter( unsigned inputCount, unsigned outputCount )
        : mW( Eigen::MatrixXf::Random( outputCount, inputCount ) )
    {}

    Eigen::VectorXf operator()( Eigen::VectorXf inputs )
    {
        return mW * inputs;
    }

    Eigen::MatrixXf mW;
};

class Model
{
    const unsigned kInputCount = 100;
    const unsigned kOutputCount = 3;
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
        , mW( Eigen::MatrixXf::Random( kOutputCount, kInputCount ) )
        , mOutput( kOutputCount )
        , mReferenceFilter( kInputCount, kOutputCount )
        , mReferenceOutput( kOutputCount )
        , mTime( 0.0f )
    {}

    void Update()
    {
        mTime += kStep;
        mInput = mAmplitude.array() * ( mOmega.array() * mTime ).unaryExpr(
            std::ptr_fun< float, float >( std::sin ) );
        mOutput = mW * mInput;
        mReferenceOutput = mReferenceFilter( mInput );
    }

    Eigen::VectorXf mAmplitude;
    Eigen::VectorXf mOmega;
    Eigen::VectorXf mInput;
    Eigen::MatrixXf mW;
    Eigen::VectorXf mOutput;
    ReferenceFilter mReferenceFilter;
    Eigen::VectorXf mReferenceOutput;
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
        Eigen::VectorXf error = model.mReferenceOutput - model.mOutput;
        model.mW += ( kTrainStep * error * model.mInput.transpose() /
            model.mInput.squaredNorm() );
    }
}

TEST( AdaptiveFilter, RLS )
{
}
