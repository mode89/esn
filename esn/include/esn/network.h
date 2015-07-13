#ifndef __ESN_NETWORK_H__
#define __ESN_NETWORK_H__

#include <vector>

namespace ESN {

    class Network
    {
    public:
        virtual void
        SetInputs( const std::vector< float > & ) = 0;

        virtual void
        Step( float step ) = 0;

        virtual void
        Train(
            const std::vector< std::vector< float > > & inputs,
            const std::vector< std::vector< float > > & outputs ) = 0;

        virtual ~Network() {}
    };

} // namespace ESN

#endif // __ESN_NETWORK_H__
