#include <cmath>
#include <esn/create_network_nsli.h>
#include <network_nsli.h>

namespace ESN {

    std::unique_ptr< Network > CreateNetwork(
        const NetworkParamsNSLI & params )
    {
        return std::unique_ptr< NetworkNSLI >( new NetworkNSLI( params ) );
    }

    NetworkNSLI::NetworkNSLI( const NetworkParamsNSLI & params )
        : mParams( params )
        , mIn( params.inputCount )
        , mWIn( params.neuronCount, params.inputCount )
        , mX( params.neuronCount )
        , mW( params.neuronCount, params.neuronCount )
        , mOut( params.outputCount )
        , mWOut( params.outputCount, params.neuronCount )
    {
        if ( params.inputCount <= 0 )
            throw std::invalid_argument(
                "NetworkParamsNSLI::inputCount must be not null" );
        if ( params.neuronCount <= 0 )
            throw std::invalid_argument(
                "NetworkParamsNSLI::neuronCount must be not null" );
        if ( params.outputCount <= 0 )
            throw std::invalid_argument(
                "NetworkParamsNSLI::outputCount must be not null" );
        if ( !( params.leakingRate > 0.0 && params.leakingRate <= 1.0 ) )
            throw std::invalid_argument(
                "NetworkParamsNSLI::leakingRate must be withing "
                "interval [0,1)" );

        mWIn = Eigen::MatrixXf::Random(
            params.neuronCount, params.inputCount ).sparseView();
        mW = Eigen::MatrixXf::Random(
            params.neuronCount, params.neuronCount ).sparseView();
    }

    NetworkNSLI::~NetworkNSLI()
    {
    }

    void NetworkNSLI::Step( float step )
    {
        mX = ( 1 - mParams.leakingRate ) * mX +
            mParams.leakingRate * ( mWIn * mIn + mW * mX ).unaryExpr(
                [] ( float x ) -> float { return std::tanh( x ); } );
        mOut = mWOut * mX;
    }

} // namespace ESN
