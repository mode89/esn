#include <network_nsli.h>

namespace ESN {

    std::unique_ptr< Network > CreateNetwork( unsigned neuronCount )
    {
        return std::unique_ptr< NetworkNSLI >(
            new NetworkNSLI( neuronCount ) );
    }

    NetworkNSLI::NetworkNSLI( unsigned neuronCount )
    {
    }

    NetworkNSLI::~NetworkNSLI()
    {
    }

    void NetworkNSLI::Step( float step )
    {
    }

} // namespace ESN
