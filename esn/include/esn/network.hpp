#ifndef __ESN_NETWORK_HPP__
#define __ESN_NETWORK_HPP__

#include <esn/export.h>
#include <vector>

namespace ESN {

    class Network
    {
    public:
        virtual ESN_EXPORT void
        SetInputs( const std::vector< float > & ) = 0;

        virtual ESN_EXPORT void
        Step( float step ) = 0;

        virtual ESN_EXPORT void
        CaptureOutput( std::vector< float > & output ) = 0;

        virtual ESN_EXPORT void
        Train(
            const std::vector< std::vector< float > > & inputs,
            const std::vector< std::vector< float > > & outputs ) = 0;

        virtual ESN_EXPORT void
        TrainOnline(
            const std::vector< float > & output,
            bool forceOutput = false ) = 0;

        virtual ESN_EXPORT ~Network() {}
    };

} // namespace ESN

#endif // __ESN_NETWORK_HPP__
