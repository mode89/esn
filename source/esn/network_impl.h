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
        const const_pointer<float> kOne;
        const const_pointer<float> kMinusOne;
        const const_pointer<float> kZero;
        const pointer<float> mIn;
        const pointer<float> mWIn;
        const pointer<float> mWInScaling;
        const pointer<float> mWInBias;
        const pointer<float> mX;
        const pointer<float> mW;
        const pointer<float> mLeakingRate;
        const pointer<float> mOneMinusLeakingRate;
        std::vector<float> mOut;
        std::vector<float> mOutScale;
        std::vector<float> mOutBias;
        std::vector<float> mWOut;
        const pointer<float> mWFB;
        const pointer<float> mWFBScaling;
        const pointer<float> mTemp;
    };

} // namespace ESN

#endif // __ESN_SOURCE_NETWORK_IMPL_H__
