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
        , mR( params.neuronCount )
        , mW( params.neuronCount, params.neuronCount )
        , mOut( params.outputCount )
        , mWOut( params.outputCount, params.neuronCount )
        , mWFB( params.neuronCount, params.outputCount )
        , mAdaptiveFilter( params.neuronCount )
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
            params.neuronCount, params.inputCount );

        Eigen::MatrixXf randomWeights = Eigen::MatrixXf::Random(
            params.neuronCount, params.neuronCount );
        float spectralRadius =
            randomWeights.eigenvalues().cwiseAbs().maxCoeff();
        mW = ( randomWeights / spectralRadius ).sparseView() ;

        mWOut = Eigen::MatrixXf::Random(
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

        mR = mX.unaryExpr(
            [] ( float x ) -> float { return std::tanh( x ); } );

        mX = ( 1 - mParams.leakingRate ) * mX +
            mParams.leakingRate * ( mW * mR + mWFB * mOut );

        mOut = mWOut * mR;
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

    void NetworkNSLI::TrainOnline( const std::vector< float > & output )
    {
        Eigen::VectorXf w = mWOut.row( 0 ).transpose();
        mAdaptiveFilter.Train( w, mOut( 0 ), output[0], mR );
        mWOut.row( 0 ) = w.transpose();
    }

} // namespace ESN
