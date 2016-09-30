#ifndef __ESN_NETWORK_NSLI_H__
#define __ESN_NETWORK_NSLI_H__

#include <esn/export.h>
#include <memory>

namespace ESN {

    class Network;

    struct NetworkParamsNSLI
    {
        unsigned inputCount;
        unsigned neuronCount;
        unsigned outputCount;
        float leakingRateMin;
        float leakingRateMax;
        bool useOrthonormalMatrix;
        float spectralRadius;
        float connectivity;
        bool linearOutput;
        bool hasOutputFeedback;

        NetworkParamsNSLI()
            : inputCount( 0 )
            , neuronCount( 0 )
            , outputCount( 0 )
            , leakingRateMin( 0.1f )
            , leakingRateMax( 1.0f )
            , useOrthonormalMatrix( true )
            , spectralRadius( 1.0f )
            , connectivity( 1.0f )
            , linearOutput( false )
            , hasOutputFeedback(true)
        {}
    };

    ESN_EXPORT std::shared_ptr<Network>
    CreateNetwork(const NetworkParamsNSLI &);

} // namespace ESN

#endif // __ESN_NETWORK_NSLI_H__
