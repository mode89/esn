#ifndef __ESN_SOURCE_NETWORK_IMPL_H__
#define __ESN_SOURCE_NETWORK_IMPL_H__

#include <esn/network.h>

namespace ESN {

    /**
     * Implementation of a network based on non-spiking linear integrator
     * neurons.
     */
    class NetworkImpl : public Network
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
        NetworkImpl(const NetworkParams &);
        ~NetworkImpl();

    private:
        NetworkParams mParams;
        const scalar<float> kOne;
        const scalar<float> kMinusOne;
        const scalar<float> kZero;
        vector<float> mIn;
        matrix<float> mWIn;
        vector<float> mWInScaling;
        vector<float> mWInBias;
        vector<float> mX;
        matrix<float> mW;
        vector<float> mLeakingRate;
        vector<float> mOneMinusLeakingRate;
        std::vector<float> mOut;
        vector<float> mOutScale;
        vector<float> mOutBias;
        std::vector<float> mWOut;
        matrix<float> mWFB;
        vector<float> mWFBScaling;
        vector<float> mTemp;
    };

} // namespace ESN

#endif // __ESN_SOURCE_NETWORK_IMPL_H__
