#define _USE_MATH_DEFINES
#include <esn/esn.h>
#include <math.h>

static const unsigned kNeuronCount = 100;
static const unsigned kTrainSteps = 1000;
static const float kSineFrequency = 1.0f;
static const float kSineStep = 0.01f;

int main()
{
    ESN::NetworkParamsNSLI params;
    params.inputCount = 1;
    params.neuronCount = kNeuronCount;
    params.outputCount = 1;
    std::unique_ptr< ESN::Network > network = ESN::CreateNetwork( params );

    std::vector< float > output( 1 );
    std::vector< float > actualOutput( 1 );
    for ( int i = 0; i < kTrainSteps; ++ i )
    {
        output[0] = std::sin( 2 * static_cast< float >( M_PI ) *
            kSineFrequency * i * kSineStep );
        network->Step( 0.1f );
        network->CaptureOutput( actualOutput );
        network->TrainOnline( output );
    }

    return 0;
}
