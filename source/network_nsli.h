#ifndef __ESN_SOURCE_NETWORK_NSLI_H__
#define __ESN_SOURCE_NETWORK_NSLI_H__

#include <Eigen/Sparse>
#include <esn/network.h>
#include <esn/network_nsli.h>

namespace ESN {

    /**
     * Implementation of a network based on non-spiking linear integrator
     * neurons.
     */
    class NetworkNSLI : public Network
    {
        friend class TrainerImpl;

    public:
        void
        SetInputs( const std::vector< float > & );

        void
        SetInputScalings( const std::vector< float > & );

        void
        SetInputBias( const std::vector< float > & );

        void
        SetOutputScale(const std::vector<float> &);

        void
        SetOutputBias(const std::vector<float> &);

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
        Eigen::MatrixXf mW;
        Eigen::VectorXf mLeakingRate;
        Eigen::VectorXf mOneMinusLeakingRate;
        Eigen::VectorXf mOut;
        Eigen::VectorXf mOutScale;
        Eigen::VectorXf mOutBias;
        Eigen::MatrixXf mWOut;
        Eigen::MatrixXf mWFB;
        Eigen::VectorXf mWFBScaling;
        Eigen::VectorXf mTemp;
    };

} // namespace ESN

#endif // __ESN_SOURCE_NETWORK_NSLI_H__
