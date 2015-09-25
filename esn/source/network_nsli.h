#ifndef __ESN_SOURCE_NETWORK_NSLI_H__
#define __ESN_SOURCE_NETWORK_NSLI_H__

#include <Eigen/Sparse>
#include <esn/network.hpp>
#include <adaptive_filter_rls.h>

namespace ESN {

    struct NetworkParamsNSLI;

    /**
     * Implementation of a network based on non-spiking linear integrator
     * neurons.
     */
    class NetworkNSLI : public Network
    {
    public:
        void
        SetInputs( const std::vector< float > & );

        void
        SetInputScalings( const std::vector< float > & );

        void
        SetInputBias( const std::vector< float > & );

        void
        SetFeedbackScalings( const std::vector< float > & );

        void
        Step( float step );

        void
        CaptureTransformedInput( std::vector< float > & input );

        void
        CaptureActivations( std::vector< float > & activations );

        void
        CaptureOutput( std::vector< float > & output );

        void
        Train(
            const std::vector< std::vector< float > > & inputs,
            const std::vector< std::vector< float > > & outputs );

        void
        TrainOnline(
            const std::vector< float > & output,
            bool forceOutput );

    public:
        NetworkNSLI( const NetworkParamsNSLI & );
        ~NetworkNSLI();

    private:
        NetworkParamsNSLI mParams;
        Eigen::VectorXf mIn;
        Eigen::MatrixXf mWIn;
        Eigen::VectorXf mWInScaling;
        Eigen::VectorXf mWInBias;
        Eigen::VectorXf mX;
        Eigen::SparseMatrix< float > mW;
        Eigen::VectorXf mLeakingRate;
        Eigen::VectorXf mOneMinusLeakingRate;
        Eigen::VectorXf mOut;
        Eigen::MatrixXf mWOut;
        Eigen::MatrixXf mWFB;
        Eigen::VectorXf mWFBScaling;
        AdaptiveFilterRLS mAdaptiveFilter;
    };

} // namespace ESN

#endif // __ESN_SOURCE_NETWORK_NSLI_H__
