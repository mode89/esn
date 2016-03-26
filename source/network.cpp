#include <esn/errors.h>
#include <esn/exceptions.hpp>
#include <esn/network.h>
#include <esn/network.hpp>

void esnNetworkSetInputs( void * network, float * inputs, int inputCount )
{
    static_cast< ESN::Network * >( network )->SetInputs(
        std::vector< float >( inputs, inputs + inputCount ) );
}

void esnNetworkSetInputScalings( void * network,
    float * scalings, int count )
{
    static_cast< ESN::Network * >( network )->SetInputScalings(
        std::vector< float >( scalings, scalings + count ) );
}

void esnNetworkSetInputBias( void * network,
    float * bias, int count )
{
    static_cast< ESN::Network * >( network )->SetInputBias(
        std::vector< float >( bias, bias + count ) );
}

void esnNetworkSetOutputScale(void * network, float * scale, int count)
{
    static_cast<ESN::Network*>(network)->SetOutputScale(
        std::vector<float>(scale, scale + count));
}

void esnNetworkSetOutputBias(void * network, float * bias, int count)
{
    static_cast<ESN::Network*>(network)->SetOutputBias(
        std::vector<float>(bias, bias + count));
}

void esnNetworkSetFeedbackScalings( void * network,
    float * scalings, int count )
{
    static_cast< ESN::Network * >( network )->SetFeedbackScalings(
        std::vector< float >( scalings, scalings + count ) );
}

int esnNetworkStep( void * network, float step )
{
    try {
        static_cast< ESN::Network * >( network )->Step( step );
    } catch ( const ESN::OutputIsNotFinite & e ) {
        return ESN_OUTPUT_IS_NOT_FINITE;
    }
    return ESN_NO_ERROR;
}

void esnNetworkCaptureTransformedInput( void * network,
    float * input, int inputCount )
{
    std::vector< float > inputVector( inputCount );
    static_cast< ESN::Network * >( network )->CaptureTransformedInput(
        inputVector );
    std::copy( inputVector.begin(), inputVector.end(), input );
}

void esnNetworkCaptureActivations( void * network,
    float * activations, int neuronCount )
{
    std::vector< float > activationsVector( neuronCount );
    static_cast< ESN::Network * >( network )->CaptureActivations(
        activationsVector );
    std::copy( activationsVector.begin(), activationsVector.end(),
        activations );
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
