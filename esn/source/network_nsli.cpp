#include <cmath>
#include <Eigen/Eigenvalues>
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

        Eigen::MatrixXf randomWeights = Eigen::MatrixXf::Random(
            params.neuronCount, params.neuronCount );
        float spectralRadius =
            randomWeights.eigenvalues().cwiseAbs().maxCoeff();
        mW = ( randomWeights / spectralRadius ).sparseView() ;
    }

    NetworkNSLI::~NetworkNSLI()
    {
    }

    void NetworkNSLI::SetInputs( const std::vector< float > & inputs )
    {
        if ( inputs.size() != mIn.rows() )
            throw std::invalid_argument( "Wrong size of the input vector" );
        mIn = Eigen::Map< Eigen::VectorXf >(
            const_cast< float * >( inputs.data() ), inputs.size() );
    }

    void NetworkNSLI::Step( float step )
    {
        mX = ( 1 - mParams.leakingRate ) * mX +
            mParams.leakingRate * ( mWIn * mIn + mW * mX ).unaryExpr(
                [] ( float x ) -> float { return std::tanh( x ); } );
        mOut = mWOut * mX;
    }

    void NetworkNSLI::Train(
        const std::vector< std::vector< float > > & inputs,
        const std::vector< std::vector< float > > & outputs )
    {
        if ( inputs.size() == 0 )
            throw std::invalid_argument(
                "Number of samples must be not null" );
        if ( inputs.size() != outputs.size() )
            throw std::invalid_argument(
                "Number of input and output samples must be equal" );
    }

} // namespace ESN
