#ifndef __ESN_CREATE_NETWORK_NSLI_H__
#define __ESN_CREATE_NETWORK_NSLI_H__

#include <memory>

namespace ESN {

    class Network;

    struct NetworkParamsNSLI
    {
        unsigned neuronCount;
    };

    std::unique_ptr< Network >
    CreateNetwork( const NetworkParamsNSLI & );

} // namespace ESN

#endif // __ESN_CREATE_NETWORK_NSLI_H__
