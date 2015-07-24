#include <esn/network.h>
#include <esn/network.hpp>

void esnNetworkSetInputs( void * network, float * inputs, int inputCount )
{
    static_cast< ESN::Network * >( network )->SetInputs(
        std::vector< float >( inputs, inputs + inputCount ) );
}

void esnNetworkStep( void * network, float step )
{
    static_cast< ESN::Network * >( network )->Step( step );
}

void esnNetworkCaptureOutput( void * network,
    float * outputs, int outputCount )
{
    std::vector< float > outputVector( outputCount );
    static_cast< ESN::Network * >( network )->CaptureOutput( outputVector );
    std::copy( outputVector.begin(), outputVector.end(), outputs );
}

void esnNetworkTrainOnline( void * network,
    float * outputs, int outputCount, bool forceOutpus )
{
    static_cast< ESN::Network * >( network )->TrainOnline(
        std::vector< float >( outputs, outputs + outputCount ),
        forceOutpus );
}

void esnNetworkDestruct( void * network )
{
    delete static_cast< ESN::Network * >( network );
}
