#ifndef __ESN_CREATE_NETWORK_NSLI_H__
#define __ESN_CREATE_NETWORK_NSLI_H__

#include <memory>

namespace ESN {

    class Network;

    struct NetworkParamsNSLI
    {
        unsigned inputCount;
        unsigned neuronCount;
        unsigned outputCount;
        float leakingRate;
        float spectralRadius;
        float onlineTrainingForgettingFactor;
        float onlineTrainingInitialCovariance;

        NetworkParamsNSLI()
            : inputCount( 0 )
            , neuronCount( 0 )
            , outputCount( 0 )
            , leakingRate( 1.0f )
            , spectralRadius( 1.0f )
            , onlineTrainingForgettingFactor( 0.999f )
            , onlineTrainingInitialCovariance( 1000.0f )
        {}
    };

    std::unique_ptr< Network >
    CreateNetwork( const NetworkParamsNSLI & );

} // namespace ESN

#endif // __ESN_CREATE_NETWORK_NSLI_H__
