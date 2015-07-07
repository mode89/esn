#include <network.h>

namespace ESN {

    NetworkImpl::NetworkImpl( unsigned neuronCount )
    {
    }

    void NetworkImpl::Step( float step )
    {
        for ( int i = 0, n = mPotential.size(); i < n; ++ i )
        {
            float inputCurrent = 0.0f;
            for ( int j = 0, nj = mConnection[i].size(); j < nj; ++ j )
                inputCurrent +=
                    mConnection[i][j] * mConnectionWeight[i][j];

            float delta = step * ( inputCurrent * mResistance[i] -
                mPotential[i] ) / mTimeConstant[i];
            mPotential[i] += delta;
        }
    }

} // namespace ESN
