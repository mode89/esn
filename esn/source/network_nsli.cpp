#include <esn/create_network_nsli.h>
#include <network_nsli.h>

namespace ESN {

    std::unique_ptr< Network > CreateNetwork(
        const NetworkParamsNSLI & params )
    {
        return std::unique_ptr< NetworkNSLI >( new NetworkNSLI( params ) );
    }

    NetworkNSLI::NetworkNSLI( const NetworkParamsNSLI & params )
        : mIn( params.inputCount )
        , mWIn( params.neuronCount, params.inputCount )
        , mX( params.neuronCount )
        , mW( params.neuronCount, params.neuronCount )
    {
        if ( params.inputCount <= 0 )
            throw std::invalid_argument(
                "NetworkParamsNSLI::inputCount must be not null" );
        if ( params.neuronCount <= 0 )
            throw std::invalid_argument(
                "NetworkParamsNSLI::neuronCount must be not null" );
    }

    NetworkNSLI::~NetworkNSLI()
    {
    }

    void NetworkNSLI::Step( float step )
    {
    }

} // namespace ESN
