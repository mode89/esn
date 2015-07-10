#ifndef __ESN_NETWORK_NSLI_H__
#define __ESN_NETWORK_NSLI_H__

#include <esn/network.h>

namespace ESN {

    struct NetworkParamsNSLI;

    /**
     * Implementation of a network based on non-spiking linear integrator
     * neurons.
     */
    class NetworkNSLI : public Network
    {
    public:
        void Step( float step );

    public:
        NetworkNSLI( const NetworkParamsNSLI & );
        ~NetworkNSLI();
    };

} // namespace ESN

#endif // __ESN_NETWORK_NSLI_H__
