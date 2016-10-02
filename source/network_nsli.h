#ifndef __ESN_SOURCE_NETWORK_NSLI_H__
#define __ESN_SOURCE_NETWORK_NSLI_H__

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
        std::vector<float> mIn;
        std::vector<float> mWIn;
        std::vector<float> mWInScaling;
        std::vector<float> mWInBias;
        std::vector<float> mX;
        std::vector<float> mW;
        std::vector<float> mLeakingRate;
        std::vector<float> mOneMinusLeakingRate;
        std::vector<float> mOut;
        std::vector<float> mOutScale;
        std::vector<float> mOutBias;
        std::vector<float> mWOut;
        std::vector<float> mWFB;
        std::vector<float> mWFBScaling;
        std::vector<float> mTemp;
    };

} // namespace ESN

#endif // __ESN_SOURCE_NETWORK_NSLI_H__
