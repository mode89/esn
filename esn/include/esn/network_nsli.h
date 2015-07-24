#ifndef __ESN_NETWORK_NSLI_H__
#define __ESN_NETWORK_NSLI_H__

#include <esn/export.h>

extern "C" {

    struct esnNetworkParamsNSLI
    {
        unsigned inputCount;
        unsigned neuronCount;
        unsigned outputCount;
        float leakingRate;
        float spectralRadius;
        float onlineTrainingForgettingFactor;
        float onlineTrainingInitialCovariance;
    };

    ESN_EXPORT void *
    esnCreateNetworkNSLI( esnNetworkParamsNSLI * );

} // export "C"

#endif // __ESN_NETWORK_NSLI_H__
