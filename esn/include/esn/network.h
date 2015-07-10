#ifndef __ESN_NETWORK_H__
#define __ESN_NETWORK_H__

#include <memory>

namespace ESN {

    class Network
    {
    public:
        virtual void Step( float step ) = 0;

        virtual ~Network() {}
    };

    std::unique_ptr< Network > CreateNetwork( unsigned neuronCount );

} // namespace ESN

#endif // __ESN_NETWORK_H__
