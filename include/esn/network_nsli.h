#ifndef __ESN_NETWORK_NSLI_H__
#define __ESN_NETWORK_NSLI_H__

#include <esn/export.h>

extern "C" {

    struct esnNetworkParamsNSLI
    {
        unsigned structSize;
        unsigned inputCount;
        unsigned neuronCount;
        unsigned outputCount;
        float leakingRateMin;
        float leakingRateMax;
        bool useOrthonormalMatrix;
        float spectralRadius;
        float connectivity;
        bool linearOutput;
        float onlineTrainingForgettingFactor;
        float onlineTrainingInitialCovariance;
        bool hasOutputFeedback;
    };

    ESN_EXPORT void *
    esnCreateNetworkNSLI( esnNetworkParamsNSLI * );

} // export "C"

#endif // __ESN_NETWORK_NSLI_H__
