#ifndef __ESN_NETWORK_H__
#define __ESN_NETWORK_H__

#include <esn/export.h>
#include <memory>
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

    struct NetworkParams
    {
        unsigned inputCount;
        unsigned neuronCount;
        unsigned outputCount;
        float leakingRateMin;
        float leakingRateMax;
        float connectivity;
        bool linearOutput;
        bool hasOutputFeedback;

        NetworkParams()
            : inputCount(0)
            , neuronCount(0)
            , outputCount(0)
            , leakingRateMin(0.1f)
            , leakingRateMax(1.0f)
            , connectivity(1.0f)
            , linearOutput(false)
            , hasOutputFeedback(true)
        {}
    };

    ESN_EXPORT std::shared_ptr<Network> CreateNetwork(
        const NetworkParams &);

} // namespace ESN

#endif // __ESN_NETWORK_H__
