#include <cmath>
#include <cstring>
#include <Eigen/Eigenvalues>
#include <esn/network_nsli.h>
#include <esn/network_nsli.hpp>
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
        , mWFB( params.neuronCount, params.outputCount )
        , mAdaptiveFilter( params.neuronCount,
            params.onlineTrainingForgettingFactor,
            params.onlineTrainingInitialCovariance )
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
                "NetworkParamsNSLI::leakingRate must be within "
                "interval [0,1)" );
        if ( !( params.connectivity > 0.0f &&
                params.connectivity <= 1.0f ) )
            throw std::invalid_argument(
                "NetworkParamsNSLI::connectivity must be within "
                "interval (0,1]" );

        mWIn = Eigen::MatrixXf::Random(
            params.neuronCount, params.inputCount );

        Eigen::MatrixXf randomWeights =
            ( Eigen::MatrixXf::Random( params.neuronCount,
                params.neuronCount ).array().abs()
                    <= params.connectivity ).cast< float >() *
            Eigen::MatrixXf::Random( params.neuronCount,
                params.neuronCount ).array();
        float spectralRadius =
            randomWeights.eigenvalues().cwiseAbs().maxCoeff();
        mW = ( randomWeights / spectralRadius *
            params.spectralRadius ).sparseView() ;

        mWOut = Eigen::MatrixXf::Zero(
            params.outputCount, params.neuronCount );

        mWFB = Eigen::MatrixXf::Random(
            params.neuronCount, params.outputCount );

        mIn = Eigen::VectorXf::Zero( params.inputCount );
        mX = Eigen::VectorXf::Random( params.neuronCount );
        mOut = Eigen::VectorXf::Zero( params.outputCount );
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
        if ( step <= 0.0f )
            throw std::invalid_argument(
                "Step size must be positive value" );

        auto tanh = [] ( float x ) -> float { return std::tanh( x ); };

        if ( mParams.linearOutput )
        {
            mX = ( 1 - mParams.leakingRate ) * mX +
                ( mParams.leakingRate * ( mWIn * mIn + mW * mX +
                    mWFB * mOut.unaryExpr( tanh ) ) ).unaryExpr( tanh );

            mOut = mWOut * mX;
        }
        else
        {
            mX = ( 1 - mParams.leakingRate ) * mX + ( mParams.leakingRate *
                ( mWIn * mIn + mW * mX + mWFB * mOut ) ).unaryExpr( tanh );

            mOut = ( mWOut * mX ).unaryExpr( tanh );
        }
    }

    void NetworkNSLI::CaptureOutput( std::vector< float > & output )
    {
        if ( output.size() != mParams.outputCount )
            throw std::invalid_argument(
                "Size of the vector must be equal "
                "actual number of outputs" );

        for ( int i = 0; i < mParams.outputCount; ++ i )
            output[ i ] = mOut( i );
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
        const unsigned kSampleCount = inputs.size();

        Eigen::MatrixXf matX( mParams.neuronCount, kSampleCount );
        Eigen::MatrixXf matY( mParams.outputCount, kSampleCount );
        for ( int i = 0; i < kSampleCount; ++ i )
        {
            SetInputs( inputs[i] );
            Step( 0.1f );
            matX.col( i ) = mX;
            matY.col( i ) = Eigen::Map< Eigen::VectorXf >(
                const_cast< float * >( outputs[i].data() ),
                    mParams.outputCount );
        }

        Eigen::MatrixXf matXT = matX.transpose();

        mWOut = ( matY * matXT * ( matX * matXT ).inverse() );
    }

    void NetworkNSLI::TrainOnline( const std::vector< float > & output,
        bool forceOutput )
    {
        for ( unsigned i = 0; i < mParams.outputCount; ++ i )
        {
            Eigen::VectorXf w = mWOut.row( i ).transpose();
            if ( mParams.linearOutput )
                mAdaptiveFilter.Train( w, mOut( i ), output[i], mX );
            else
                mAdaptiveFilter.Train( w, std::atanh( mOut( i ) ),
                    std::atanh( output[i] ), mX );
            mWOut.row( i ) = w.transpose();
        }

        if ( forceOutput )
            mOut = Eigen::Map< Eigen::VectorXf >(
                const_cast< float * >( output.data() ),
                mParams.outputCount );
    }

} // namespace ESN

#define SIZEOF_MEMBER( structure, member ) \
    sizeof( ( ( structure * ) 0 )->member )

void * esnCreateNetworkNSLI( esnNetworkParamsNSLI * params )
{
    static_assert( ( sizeof( esnNetworkParamsNSLI ) -
        SIZEOF_MEMBER( esnNetworkParamsNSLI, structSize ) ) ==
        sizeof( ESN::NetworkParamsNSLI ),
        "Wrong size of esnNetworkParamsNSLI" );

    if ( params->structSize != sizeof( esnNetworkParamsNSLI ) )
        throw std::invalid_argument(
            "esnNetworkParamsNSLI::structSize must be equal the "
            "sizeof( esnNetworkParamsNSLI )" );

    ESN::NetworkParamsNSLI p;
    std::memcpy( &p, reinterpret_cast< char * >( params ) +
        SIZEOF_MEMBER( esnNetworkParamsNSLI, structSize ),
        sizeof( ESN::NetworkParamsNSLI ) );

    return new ESN::NetworkNSLI( p );
}

#undef SIZEOF_MEMBER
