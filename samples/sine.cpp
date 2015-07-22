#define _USE_MATH_DEFINES
#include <esn/esn.h>
#include <iomanip>
#include <iostream>
#include <math.h>

static const unsigned kNeuronCount = 100;
static const float kLeakingRate = 0.1f;
static const float kSineFrequency = 1.0f;
static const float kSimulationStep = 0.01f;
static const float kTrainingTime = 25.0f * 1.0f / kSineFrequency;

int main()
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 1;
    params.neuronCount = kNeuronCount;
    params.outputCount = 1;
    params.leakingRate = kLeakingRate;
    std::unique_ptr< ESN::Network > network = ESN::CreateNetwork( params );

    std::vector< float > output( 1 );
    std::vector< float > actualOutput( 1 );
    for ( float time = 0.0f; true; time += kSimulationStep )
    {
        output[0] = sin( 2 * static_cast< float >( M_PI ) *
            kSineFrequency * time ) + 2.0f;

        network->Step( 0.1f );
        network->CaptureOutput( actualOutput );
        float error = fabs( ( actualOutput[0] - output[0] ) / output[0] );

        if ( time < kTrainingTime )
            network->TrainOnline( output, true );

        std::cout << std::setw( 14 ) << output[0];
        std::cout << std::setw( 14 ) << actualOutput[0];
        std::cout << std::setw( 14 ) << error << std::endl;
    }

    return 0;
}
