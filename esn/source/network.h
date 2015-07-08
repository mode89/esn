#ifndef __ESN_NETWORK_IMPL_H__
#define __ESN_NETWORK_IMPL_H__

#include <vector>
#include <esn/network.h>

namespace ESN {

    class NetworkImpl : public Network
    {
    public:
        void Step( float step );

    public:
        NetworkImpl( unsigned neuronCount );

    private:
        std::vector< float > mPotential;
        std::vector< float > mThreshold;
        std::vector< float > mResistance;
        std::vector< float > mMebraneTimeConstant;
        std::vector< float > mSpikeCurrent;
        std::vector< float > mSpikeTime;
        std::vector< std::vector< unsigned > > mConnection;
        std::vector< std::vector< float > > mConnectionWeight;
    };

} // namespace ESN

#endif // __ESN_NETWORK_IMPL_H__
