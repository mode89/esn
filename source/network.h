#ifndef __ESN_NETWORK_IMPL_H__
#define __ESN_NETWORK_IMPL_H__

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
        std::vector< float > mTimeConstant;
        std::vector< float > mOutputCurrent;
        std::vector< std::vector< unsigned > > mInputNeurons;
        std::vector< std::vector< float > > mInputWeights;
    };

} // namespace ESN

#endif // __ESN_NETWORK_IMPL_H__
