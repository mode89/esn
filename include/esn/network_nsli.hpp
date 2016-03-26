#ifndef __ESN_NETWORK_NSLI_HPP__
#define __ESN_NETWORK_NSLI_HPP__

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
        float onlineTrainingForgettingFactor;
        float onlineTrainingInitialCovariance;
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
            , onlineTrainingForgettingFactor( 1.0f )
            , onlineTrainingInitialCovariance( 1000.0f )
            , hasOutputFeedback(true)
        {}
    };

    ESN_EXPORT std::unique_ptr< Network >
    CreateNetwork( const NetworkParamsNSLI & );

} // namespace ESN

#endif // __ESN_NETWORK_NSLI_HPP__
