#include <cmath>
#include <Eigen/Dense>
#include <gtest/gtest.h>

class ReferenceFilter
{
public:
    ReferenceFilter( unsigned size )
        : mW( Eigen::VectorXf::Random( size ) )
    {}

    float operator()( Eigen::VectorXf inputs )
    {
        return mW.dot( inputs );
    }

    Eigen::VectorXf mW;
};

TEST( AdaptiveFilter, LMS )
{
    const unsigned kInputCount = 10;
    const unsigned kSampleCount = 100;
    const float kMaxAmplitude = 1.0f;
    const float kMaxFrequency = 10.0f;
    const float kStep = 0.1f * 1.0f / kMaxFrequency;

    ReferenceFilter referenceFilter( kInputCount );
    Eigen::VectorXf input( kInputCount );
    Eigen::VectorXf A = kMaxAmplitude / 2.0f *
        ( Eigen::VectorXf::Random( kInputCount ).array() + 1.0f );
    Eigen::VectorXf W = kMaxFrequency / 2.0f *
        ( Eigen::VectorXf::Random( kInputCount ).array() + 1.0f );

    for ( int i = 0; i < kSampleCount; ++ i )
    {
        float t = kStep * i;
        input = A.array() * ( W.array() * t ).unaryExpr(
            std::ptr_fun( std::sinf ) );
    }
}
