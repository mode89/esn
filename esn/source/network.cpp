#include <cmath>
#include <network.h>

namespace ESN {

    NetworkImpl::NetworkImpl( unsigned neuronCount )
        : mPotential( neuronCount )
        , mThreshold( neuronCount )
        , mResistance( neuronCount )
        , mMebraneTimeConstant( neuronCount )
        , mSpikeCurrent( neuronCount )
        , mSpikeTime( neuronCount )
        , mConnection( neuronCount )
        , mConnectionWeight( neuronCount )
    {
        for ( auto & connection : mConnection )
            connection.resize( neuronCount );

        for ( auto & connectionWeight : mConnectionWeight )
            connectionWeight.resize( neuronCount );
    }

    void NetworkImpl::Step( float step )
    {
        for ( int i = 0, n = mPotential.size(); i < n; ++ i )
        {
            float inputCurrent = 0.0f;
            for ( int j = 0, nj = mConnection[i].size(); j < nj; ++ j )
            {
                int inputNeuron = mConnection[i][j];
                inputCurrent += mSpikeCurrent[ inputNeuron ] *
                    mConnectionWeight[i][j];
            }

            float delta = step * ( inputCurrent * mResistance[i] -
                mPotential[i] ) / mMebraneTimeConstant[i];
            mPotential[i] += delta;
        }

        for ( int i = 0, n = mPotential.size(); i < n; ++ i )
        {
            if ( mPotential[i] > mThreshold[i] )
                mSpikeTime[i] = 0.0f;

            mSpikeCurrent[i] = mSpikeTime[i] *
                std::exp( 1 - mSpikeTime[i] );

            mSpikeTime[i] += step;
        }
    }

} // namespace ESN
