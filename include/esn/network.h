#ifndef __ESN_NETWORK_H__
#define __ESN_NETWORK_H__

#include <esn/export.h>
#include <vector>

namespace ESN {

    class Network
    {
    public:
        virtual ESN_EXPORT void
        SetInputs( const std::vector< float > & ) = 0;

        virtual ESN_EXPORT void
        SetInputScalings( const std::vector< float > & ) = 0;

        virtual ESN_EXPORT void
        SetInputBias( const std::vector< float > & ) = 0;

        virtual ESN_EXPORT void
        SetOutputScale(const std::vector<float> &) = 0;

        virtual ESN_EXPORT void
        SetOutputBias(const std::vector<float> &) = 0;

        virtual ESN_EXPORT void
        SetFeedbackScalings(const std::vector< float > &) = 0;

        virtual ESN_EXPORT void
        Step( float step ) = 0;

        virtual ESN_EXPORT void
        CaptureTransformedInput( std::vector< float > & input ) = 0;

        virtual ESN_EXPORT void
        CaptureActivations( std::vector< float > & activations ) = 0;

        virtual ESN_EXPORT void
        CaptureOutput( std::vector< float > & output ) = 0;

        virtual ESN_EXPORT ~Network() {}
    };

} // namespace ESN

#endif // __ESN_NETWORK_H__
